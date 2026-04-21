from scripts.metrics import StatsCollector, StatsHandler
from LLMHeuristicScheduler.base import LLMHeuristicScheduler




import os
from scripts.wg_converter import WorkGraphConverter, ProjectConverter
import json
import math
import random as rand
from scripts.valid import validate_schedule_bool, interpter_solver

from sampo_api import create_mvp_toolbox




from sampo.scheduler.genetic import GeneticScheduler
from sampo.base import SAMPO

from sampo_api import contractor
from sampo.schemas.graph import WorkGraph

import copy
from collections import defaultdict


def init_experiment(GA_params, 
                    model_name, 
                    structures = (
                                  'switch_Topological',), 
                    imprt = 10):
    solvers_dict = {}
    solvers_dict['genetic'] = GeneticScheduler(**GA_params)
    for structure in structures:
        print(f'Init {structure} for GA')
        model = LLMHeuristicScheduler(model_name, 
                                      **GA_params, 
                                      type_init_pop_structure=structure, 
                                      imprortance=imprt)
        # if structure is None:
        #     structure = 'x'
        solvers_dict[model_name + '_' + structure] = model
    return solvers_dict

# structure.py
from collections import defaultdict
from copy import deepcopy


def run_one_experiment(args):
    GA, size, n, structures, model_name = args
    wg , contractors  = WorkGraph.loadf('wgs/main', f'sampo_j{size}'), contractor(N=n)
    sc = StatsCollector()
    stats_handler = StatsHandler(sc)
    experiment_logs = defaultdict(list)
    solvers_dict = init_experiment(GA, model_name, structures)
    try:
        SAMPO.logger.addHandler(stats_handler)
        for solver, model in solvers_dict.items():
            model.schedule(wg, contractors)
            experiment_logs[solver].append(deepcopy(sc.items))
            sc.clear()
    finally:
        SAMPO.logger.removeHandler(stats_handler)
    return dict(experiment_logs)

