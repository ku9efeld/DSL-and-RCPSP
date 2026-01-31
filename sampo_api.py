
from sampo.generator import SimpleSynthetic
from sampo.schemas.graph import WorkGraph
from sampo.api.genetic_api import ChromosomeType
from sampo.scheduler.genetic.base import GeneticScheduler
from sampo.schemas.time_estimator import DefaultWorkEstimator
from sampo.scheduler.genetic.utils import prepare_optimized_data_structures
from sampo.scheduler.genetic.converter import convert_schedule_to_chromosome
from sampo.scheduler.utils import get_head_nodes_with_connections_mappings


# Config, Enum etc
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.landscape import LandscapeConfiguration


# Toolbox
from deap import base
from deap.base import Toolbox
from sampo.schemas.schedule import Schedule
from sampo.scheduler.genetic.converter import (convert_schedule_to_chromosome, convert_chromosome_to_schedule, ScheduleGenerationScheme)
from sampo.scheduler.genetic.operators import is_chromosome_correct


def contractor(N = 10):
    ss = SimpleSynthetic(rand=1)
    contractors = [ss.contractor(x) for x in range(1, 1 + N)]
    return contractors

def create_first_population(wg, contractors , work_estimator=DefaultWorkEstimator()):
    return  GeneticScheduler()\
                .generate_first_population(wg, contractors=contractors,   work_estimator=work_estimator)

def first_population(wg, contractors):
    init_popul = create_first_population(wg, contractors)
    landscape = LandscapeConfiguration()
    spec = ScheduleSpec()
    _, _, _, work_id2index, worker_name2index, _, \
        _, contractor2index, contractor_borders, _, _, _, _, \
        _, _ = prepare_optimized_data_structures(wg, contractors, landscape, spec)
    init_chromosomes = \
            {name: (convert_schedule_to_chromosome(work_id2index, worker_name2index,
                                                contractor2index, contractor_borders, schedule, chromosome_spec,
                                                landscape, order),
                    importance, chromosome_spec)
            if schedule is not None else None
            for name, (schedule, order, chromosome_spec, importance) in init_popul.items()}
    return init_chromosomes
# TO DO
def check_chromosome(wg : WorkGraph, contractors, chrm : ChromosomeType, landscape=LandscapeConfiguration(), spec=ScheduleSpec()):
    worker_pool, index2node, _, work_id2index, worker_name2index, index2contractor_obj, \
        worker_pool_indices, contractor2index, contractor_borders, node_indices, priorities, parents, children, \
        _, _ = prepare_optimized_data_structures(wg, contractors, landscape, spec)
    return is_chromosome_correct(chrm, node_indices=node_indices, parents=parents,
                     contractor_borders=contractor_borders, 
                     index2node=index2node, index2contractor=index2contractor_obj)


### Toolbox for fitness functions 
from sampo.api.genetic_api import ChromosomeType, Individual
from sampo.schemas.time import Time
from sampo.schemas.graph import WorkGraph, GraphNode

def register_individual_constructor(fitness_weights: tuple[int | float, ...], toolbox: base.Toolbox):
    class IndividualFitness(base.Fitness):
        weights = fitness_weights
    toolbox.register('Individual', Individual.prepare(IndividualFitness))

def evaluate(chromosome: ChromosomeType, wg: WorkGraph, toolbox: Toolbox) -> Schedule | None:
    if toolbox.validate(chromosome):
        sworks = toolbox.chromosome_to_schedule(chromosome)[0]
        return Schedule.from_scheduled_works(sworks.values(), wg)
    else:
        return None


def create_mvp_toolbox(wg, contractors, landscape=LandscapeConfiguration(), spec=ScheduleSpec(), work_estimator=DefaultWorkEstimator()):
    toolbox = base.Toolbox()
    worker_pool, index2node, index2zone, _, worker_name2index, index2contractor_obj, \
        worker_pool_indices, contractor2index, contractor_borders, node_indices, _, parents, _, \
        _, _ = prepare_optimized_data_structures(wg, contractors, landscape, spec)
    toolbox.register('validate', is_chromosome_correct, node_indices=node_indices, parents=parents,
                     contractor_borders=contractor_borders, index2node=index2node, index2contractor=index2contractor_obj)
    toolbox.register('chromosome_to_schedule', convert_chromosome_to_schedule, worker_pool=worker_pool,
                     index2node=index2node, index2contractor=index2contractor_obj,
                     worker_pool_indices=worker_pool_indices, assigned_parent_time=Time(0),
                     work_estimator=work_estimator, worker_name2index=worker_name2index,
                     contractor2index=contractor2index, index2zone=index2zone,
                     landscape=landscape, sgs_type=ScheduleGenerationScheme.Parallel)
    toolbox.register('register_individual_constructor', register_individual_constructor, toolbox=toolbox)
    toolbox.register('evaluate_chromosome', evaluate, wg=wg, toolbox=toolbox)
    toolbox.register_individual_constructor((-1,))
    return toolbox

def take_children(wg):
    nodes, _, node_id2child_ids = get_head_nodes_with_connections_mappings(wg)
    index2node: dict[int, GraphNode] = {index: node for index, node in enumerate(nodes)}
    work_id2index: dict[str, int] = {node.id: index for index, node in index2node.items()}
    children = {work_id2index[node_id]: set(work_id2index[child_id] for child_id in child_ids)
                for node_id, child_ids in node_id2child_ids.items()}
    return children











#def # import json
# children_serial = {str(k): list(v) for k, v in children.items()}
# json_str = json.dumps(children_serial)
# with open('trash/graph_activity', 'w', encoding='utf-8') as f:
#        f.write(json_str)
# Для проекта 35_0