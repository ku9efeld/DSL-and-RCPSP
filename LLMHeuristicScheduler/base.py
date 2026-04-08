# CUSTOM
import os
from scripts.wg_converter import WorkGraphConverter, ProjectConverter
import json
from scripts.valid import validate_schedule_bool, interpter_solver
# SAMPO
from sampo.scheduler.genetic import GeneticScheduler
from sampo.scheduler.genetic.schedule_builder import build_schedules, build_schedules_with_cache
from sampo.schemas.schedule import Schedule
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.landscape import LandscapeConfiguration
from sampo.utilities.validation import validate_schedule
from sampo.schemas.time import Time


class LLMHeuristicScheduler(GeneticScheduler):
    def __init__(self,
                 solvers_by,
                 number_of_generation = 50,
                 mutate_order = None,
                 mutate_zones = None,
                 mutate_resources = None,
                 size_of_population = None
                 ):
        self.solvers_by_path = os.path.join('Heuristics', solvers_by)
        self.llm_heuristics = self.get_llm_heuristics()
        super().__init__(number_of_generation=number_of_generation, 
                        mutate_order=mutate_order, 
                        mutate_resources=mutate_resources, 
                        mutate_zones=mutate_zones,
                        size_of_population=size_of_population)

    def get_filter(self):
        with open(os.path.join(self.solvers_by_path, 'original_heuristics.json'), "r", encoding="utf-8") as f:
            filter = set(json.load(f))
        return filter
    
    def get_llm_heuristics(self):
        filter = self.get_filter()
        files = [file for file in os.listdir(self.solvers_by_path) if file not in ('Steps', 'original_heuristics.json')]
        heuristics = {}
        for file in files:
            if file.split(' ')[-1] not in filter:
                continue
            with open(os.path.join(self.solvers_by_path, file), "r", encoding="utf-8") as f:
                code = f.read()
            heuristics[file] = code
        return heuristics

    def to_input_solver(self, work_graph, contractors):
        return WorkGraphConverter().convert(work_graph, contractors)['rcpsp_data']
    
    def generate_llm_population(self, work_graph, contractors, spec = ScheduleSpec()):
        data = self.to_input_solver(work_graph, contractors)
        project_converter = ProjectConverter(work_graph, contractors)
        population = {}
        for method, code in self.llm_heuristics.items():
            schedule, order, _, job_usage,  makespan = interpter_solver(method, code, data)
            schedule_obj = project_converter.to_schedule(schedule, order, job_usage, makespan)
            graph_nodes = project_converter.get_list_graph_nodes(schedule_obj)
            if validate_schedule_bool(schedule_obj, work_graph, contractors, spec):
                population[method] = (schedule_obj, graph_nodes, spec, 1)  # Schedule, list(GraphNode), Spec, weight
        return population
    
    def schedule_with_cache(self, work_graph, 
                            contractors, spec = ScheduleSpec(), validate = False, 
                            assigned_parent_time = Time(0), timeline = None, landscape = LandscapeConfiguration()):
        
        
        
        basic_init_schedules = super().generate_first_population(work_graph, contractors, 
                                                                 landscape, spec,
                                                                 self.work_estimator,
                                                                 self._deadline, self._weights)
        
        
        init_schedules = basic_init_schedules | self.generate_llm_population(work_graph, contractors)
        mutate_order, mutate_resources, mutate_zones, size_of_population = self.get_params(work_graph.vertex_count)
        deadline = None if self._optimize_resources else self._deadline

        schedules = build_schedules(work_graph,
                                    contractors,
                                    size_of_population,
                                    self.number_of_generation,
                                    mutate_order,
                                    mutate_resources,
                                    mutate_zones,
                                    init_schedules,
                                    self.rand,
                                    spec,
                                    self._weights,
                                    None,
                                    landscape,
                                    self.fitness_constructor,
                                    self.fitness_weights,
                                    self.work_estimator,
                                    self.sgs_type,
                                    assigned_parent_time,
                                    timeline,
                                    self._time_border,
                                    self._max_plateau_steps,
                                    self._optimize_resources,
                                    deadline,
                                    self._only_lft_initialization,
                                    self._is_multiobjective)
        schedules = [
            (Schedule.from_scheduled_works(scheduled_works.values(), work_graph), schedule_start_time, timeline, order_nodes)
            for scheduled_works, schedule_start_time, timeline, order_nodes in schedules]

        if validate:
            for schedule, *_ in schedules:
                validate_schedule(schedule, work_graph, contractors, spec)

        return schedules