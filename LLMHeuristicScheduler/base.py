# CUSTOM
import os
from scripts.wg_converter import WorkGraphConverter, ProjectConverter
import json
import math
import random as rand
from scripts.valid import validate_schedule_bool, interpter_solver

from sampo_api import create_mvp_toolbox

# SAMPO
from sampo.scheduler.lft.base import RandomizedLFTScheduler
from sampo.schemas.time_estimator import WorkTimeEstimator, DefaultWorkEstimator
from sampo.scheduler.topological.base import RandomizedTopologicalScheduler
from sampo.scheduler.genetic import GeneticScheduler
from sampo.scheduler.genetic.operators import TimeFitness
from sampo.scheduler.genetic.schedule_builder import build_schedules, build_schedules_with_cache
from sampo.schemas.schedule import Schedule
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.landscape import LandscapeConfiguration
from sampo.utilities.validation import validate_schedule
from sampo.schemas.time import Time
from sampo.scheduler.genetic.converter import convert_schedule_to_chromosome
from sampo.scheduler.genetic.utils import prepare_optimized_data_structures

class Evaluator():
    def __init__(self, work_graph, contractors, f = TimeFitness()):
        self.toolbox = create_mvp_toolbox(work_graph, contractors)
        self.fitness_func = f
    def makespan(self, chromosome):
        makespan = self.fitness_func.evaluate(chromosome, evaluator=self.toolbox.evaluate_chromosome)[0]
        return makespan
    

class LLMHeuristicScheduler(GeneticScheduler):
    def __init__(self,
                 solvers_by,
                 number_of_generation = 50,
                 mutate_order = None,
                 mutate_zones = None,
                 mutate_resources = None,
                 size_of_population = None,
                 type_init_pop_structure = None,
                 max_plateau = 100,
                 imprortance = 10
                 ):
    
        self.solvers_by_path = os.path.join('Heuristics', solvers_by)
        self.llm_heuristics = self.get_llm_heuristics()
        self.imprt = imprortance
        super().__init__(number_of_generation=number_of_generation, 
                        mutate_order=mutate_order, 
                        mutate_resources=mutate_resources, 
                        mutate_zones=mutate_zones,
                        size_of_population=size_of_population,
                        )
        self.type_init_pop_structure = type_init_pop_structure
        self._max_plateau_steps  = max_plateau
        self.imprt = imprortance

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
    
    
    def generate_llm_first_population(self, work_graph, contractors, spec = ScheduleSpec()):
        data = self.to_input_solver(work_graph, contractors)
        project_converter, eval = ProjectConverter(work_graph, contractors), Evaluator(work_graph, contractors)
        population = {}
        for method, code in self.llm_heuristics.items():
            schedule, order, _, job_usage,  makespan = interpter_solver(method, code, data)
            schedule_obj = project_converter.to_schedule(schedule, order, job_usage, makespan)
            graph_nodes = project_converter.get_list_graph_nodes(schedule_obj)
            if validate_schedule_bool(schedule_obj, work_graph, contractors, spec):
                # Проверка, что TimeEst работ совпадает между SAMPO / внутренним подсчетом в эвристике, по makespan
                if eval.makespan(project_converter.to_chromosome(schedule_obj)) == makespan:
                    population[method] = (schedule_obj, graph_nodes, spec, self.imprt)  # Schedule, list(GraphNode), Spec, weight
        return population
    

    def sampo_fist_popualtion_chrms(self, work_graph, contractors,):
        project_converter = ProjectConverter(work_graph, contractors)
        sampo_population = super().generate_first_population(work_graph, contractors, 
                                                            landscape = LandscapeConfiguration(), 
                                                            spec = ScheduleSpec(),
                                                            work_estimator = self.work_estimator,
                                                            deadline = self._deadline, weights =self._weights)
        
        population = {name : (project_converter.to_chromosome(schedule), 
                           importance) for name, (schedule, _, _, importance) in sampo_population.items()}
        return population

    def generate_llm_first_population_chrms(self, work_graph, contractors, spec = ScheduleSpec()):
        data = self.to_input_solver(work_graph, contractors)
        project_converter, eval = ProjectConverter(work_graph, contractors), Evaluator(work_graph, contractors)
        population = {}
        for method, code in self.llm_heuristics.items():
            schedule, order, _, job_usage,  makespan = interpter_solver(method, code, data)
            schedule_obj = project_converter.to_schedule(schedule, order, job_usage, makespan)
            if validate_schedule_bool(schedule_obj, work_graph, contractors, spec):
                # Проверка, что TimeEst работ совпадает между SAMPO / внутренним подсчетом в эвристике, по makespan
                chromosome = project_converter.to_chromosome(schedule_obj)
                if eval.makespan(chromosome) == makespan:
                    population[method] = chromosome # ChromosomeType
        return population
    


    
    def generate_init_population(self, n, work_graph, contractors,
                                  landscape = LandscapeConfiguration(), 
                                  spec = ScheduleSpec()):
        type_init_pop_structure = self.type_init_pop_structure
        if not type_init_pop_structure == 'switch_ALL':
                _, _, _, work_id2index, worker_name2index, _, \
        _, contractor2index, contractor_borders, _, _, _, _, \
        _, _ = prepare_optimized_data_structures(work_graph, contractors,  landscape = landscape, spec = spec)


        def randomized_init(is_topological: bool = False):
            if is_topological:
                    seed = int(rand.random() * 1000000)
                    schedule, _, _, node_order = RandomizedTopologicalScheduler(self.work_estimator,
                                                                                seed).schedule_with_cache(work_graph, contractors, spec,
                                                                                                        landscape=landscape)[0]
            else:
                schedule, _, _, node_order = RandomizedLFTScheduler(work_estimator=self.work_estimator,
                                                                        rand=rand).schedule_with_cache(work_graph, contractors, spec,
                                                                                                    landscape=landscape)[0]
            return convert_schedule_to_chromosome(work_id2index, worker_name2index, 
                                                contractor2index, contractor_borders,
                                                schedule, spec, landscape, node_order)
        

        project_converter = ProjectConverter(work_graph, contractors)
        match type_init_pop_structure:
            case "switch_ALL":
                init_schedules = self.generate_llm_first_population_chrms(work_graph, contractors) 
                count_for_specified_types = math.ceil( n / len(init_schedules) )
                count_for_specified_types = count_for_specified_types if count_for_specified_types > 0 else 1
                counts = [count_for_specified_types] * len(init_schedules)
                chromosome_types = rand.sample(list(init_schedules.keys()), k=n, counts=counts)
                chromosomes = []
                for generated_type in chromosome_types:
                        chrm = init_schedules[generated_type]
                        chromosomes.append(chrm)
                return chromosomes[:n]
            
            case "switch_LFTrand":
                init_schedules = self.generate_llm_first_population_chrms(work_graph, contractors)
                sampo_init_schedules =  self.sampo_fist_popualtion_chrms(work_graph, contractors)
                
                count_for_specified_types = (n // 3) // len(sampo_init_schedules)
                count_for_specified_types = count_for_specified_types if count_for_specified_types > 0 else 1
                weights = [importance for _,  importance in sampo_init_schedules.values()]
    
                sum_of_weights = sum(weights)
                weights = [weight / sum_of_weights for weight in weights]

                counts = [math.ceil(count_for_specified_types * weight) for weight in weights]
                sum_counts_for_specified_types = sum(counts)

                count_for_topological = n // 2 - sum_counts_for_specified_types
                count_for_topological = count_for_topological if count_for_topological > 0 else 1
                counts.append(count_for_topological)

                # Counts for llm's heuristics
                count_for_llm = n - count_for_topological - sum_counts_for_specified_types
                count_for_specified_heuristics = math.ceil(count_for_llm / len(init_schedules))
                count_for_specified_heuristics = count_for_specified_heuristics if count_for_specified_heuristics > 0 else 1
                counts_for_llm = [ count_for_specified_heuristics] * len(init_schedules)
                counts += counts_for_llm
 
                chromosome_types = rand.sample( list(sampo_init_schedules.keys()) + ['topological'] + list(init_schedules.keys()), k=n, counts=counts)

                chromosomes = []

                for generated_type in chromosome_types:
                    match generated_type:
                        case 'topological':
                            ind = randomized_init(is_topological=True)
                        case _ if generated_type in sampo_init_schedules.keys():
                            ind = sampo_init_schedules[generated_type][0]
                        case _ if generated_type in init_schedules.keys():
                            ind = init_schedules[generated_type]
                    chromosomes.append(ind)
                return chromosomes[:n]
                
            case "switch_Topological":
                init_schedules = self.generate_llm_first_population_chrms(work_graph, contractors)
                
                sampo_init_schedules = self.sampo_fist_popualtion_chrms(work_graph, contractors)
                
                count_for_specified_types = (n // 3) // len(sampo_init_schedules)
                count_for_specified_types = count_for_specified_types if count_for_specified_types > 0 else 1
                weights = [importance for _, importance in sampo_init_schedules.values()]
                counts = [math.ceil(count_for_specified_types * weight) for weight in weights]
                sum_counts_for_specified_types = sum(counts)

                count_for_specified_heuristics = n // 2 - sum_counts_for_specified_types
                counts_for_llm = [count_for_specified_heuristics] * len(init_schedules)
                counts += counts_for_llm

                count_for_rand_lft = n - sum_counts_for_specified_types - sum(counts_for_llm)
                count_for_rand_lft = count_for_rand_lft if count_for_rand_lft > 0 else 1
                counts.append(count_for_rand_lft)

                chromosome_types = rand.sample( list(sampo_init_schedules.keys())  + list(init_schedules.keys()) + ['rand_lft'], k=n, counts=counts)
                chromosomes = []

                for generated_type in chromosome_types:
                    match generated_type:
                        case 'rand_lft':
                            ind = randomized_init(is_topological=False)
                        case _ if generated_type in sampo_init_schedules.keys():
                            ind = sampo_init_schedules[generated_type][0]
                        case _ if generated_type in init_schedules.keys():
                            ind = init_schedules[generated_type]
                    chromosomes.append(ind)
                return chromosomes[:n]


                pass

            case "50/50":
                pass
            
        
                
    
    
    def schedule_with_cache(self, work_graph, 
                            contractors, spec = ScheduleSpec(), validate = False, 
                            assigned_parent_time = Time(0), 
                            timeline = None, landscape = LandscapeConfiguration()): # validate change on TRUE !!!!
        
        
        
    
        # basic_init_schedules | 
        #print(len(init_schedules))
        
        basic_init_schedules = super().generate_first_population(work_graph, contractors, 
                                                                 landscape, spec,
                                                                 self.work_estimator,
                                                                 self._deadline, self._weights)
        init_schedules =  basic_init_schedules | self.generate_llm_first_population(work_graph, contractors)

        mutate_order, mutate_resources, mutate_zones, size_of_population = self.get_params(work_graph.vertex_count)
        deadline = None if self._optimize_resources else self._deadline
        


        # None -> pop = list[ChoromosomeType], n=len(pop), assert n = population_size
        pop = None
        if self.type_init_pop_structure:
             pop = self.generate_init_population(size_of_population, work_graph, contractors)
        # else:
        #         basic_init_schedules = super().generate_first_population(work_graph, contractors, 
        #                                                          landscape, spec,
        #                                                          self.work_estimator,
        #                                                          self._deadline, self._weights)
        #         init_schedules = basic_init_schedules | self.generate_llm_first_population(work_graph, contractors)
        
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
                                    pop,
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
    
    # [FROM] sampo.sheduler.genetic.operators.py

    # def generate_chromosomes(n: int,
    #                      wg: WorkGraph,
    #                      contractors: list[Contractor],
    #                      spec: ScheduleSpec,
    #                      work_id2index: dict[str, int],
    #                      worker_name2index: dict[str, int],
    #                      contractor2index: dict[str, int],
    #                      contractor_borders: np.ndarray,
    #                      init_chromosomes: dict[str, tuple[ChromosomeType, float, ScheduleSpec]],
    #                      rand: random.Random,
    #                      toolbox: Toolbox,
    #                      work_estimator: WorkTimeEstimator = None,
    #                      landscape: LandscapeConfiguration = LandscapeConfiguration(),
    #                      only_lft_initialization: bool = False) -> list[ChromosomeType]:
    # """
    # Generates n chromosomes.
    # Do not use `generate_chromosome` function.
    # """

    # def randomized_init(is_topological: bool = False) -> ChromosomeType:
    #     if is_topological:
    #         seed = int(rand.random() * 1000000)
    #         schedule, _, _, node_order = RandomizedTopologicalScheduler(work_estimator,
    #                                                                     seed).schedule_with_cache(wg, contractors, spec,
    #                                                                                               landscape=landscape)[0]
    #     else:
    #         schedule, _, _, node_order = RandomizedLFTScheduler(work_estimator=work_estimator,
    #                                                             rand=rand).schedule_with_cache(wg, contractors, spec,
    #                                                                                            landscape=landscape)[0]
    #     return convert_schedule_to_chromosome(work_id2index, worker_name2index, contractor2index, contractor_borders,
    #                                           schedule, spec, landscape, node_order)

    # if only_lft_initialization:
    #     chromosomes = [toolbox.Individual(randomized_init(is_topological=False)) for _ in range(n - 1)]
    #     chromosomes.append(toolbox.Individual(init_chromosomes['lft'][0]))
    #     return chromosomes

    # count_for_specified_types = (n // 3) // len(init_chromosomes)
    # count_for_specified_types = count_for_specified_types if count_for_specified_types > 0 else 1
    # weights = [importance for _, importance, _ in init_chromosomes.values()]
    # sum_of_weights = sum(weights)
    # weights = [weight / sum_of_weights for weight in weights]

    # counts = [math.ceil(count_for_specified_types * weight) for weight in weights]
    # sum_counts_for_specified_types = sum(counts)

    # count_for_topological = n // 2 - sum_counts_for_specified_types
    # count_for_topological = count_for_topological if count_for_topological > 0 else 1
    # counts.append(count_for_topological)

    # count_for_rand_lft = n - count_for_topological - sum_counts_for_specified_types
    # count_for_rand_lft = count_for_rand_lft if count_for_rand_lft > 0 else 1
    # counts.append(count_for_rand_lft)

    # chromosome_types = rand.sample(list(init_chromosomes.keys()) + ['topological', 'rand_lft'], k=n, counts=counts)

    # chromosomes = []

    # for generated_type in chromosome_types:
    #     match generated_type:
    #         case 'topological':
    #             ind = randomized_init(is_topological=True)
    #         case 'rand_lft':
    #             ind = randomized_init(is_topological=False)
    #         case _:
    #             ind = init_chromosomes[generated_type][0]

    #     ind = toolbox.Individual(ind)
    #     chromosomes.append(ind)

    # return chromosomes[:n]