
import json
import numpy as np
import os
import sys
import csv
from pathlib import Path
from sampo.schemas.graph import WorkGraph
#
from sampo_api import contractor
#
from sampo.scheduler.heft import HEFTScheduler
from sampo.scheduler.genetic import GeneticScheduler
from sampo.scheduler.lft import LFTScheduler


def write(problems, path_dataset):
    fieldnames = ['FILENAME', 'HEFT', 'LFT', 'GA', 'B-makespan']
    with open(Path(path_dataset), 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for filename, p in problems:
            row = solve(p)
            row['FILENAME'] = filename
            row['B-makespan'] = min(list(row.values())[:-1])
            writer.writerow(row)
# TODO: Number увеличить обратно до 50-70
def solve(probem):
    schedulers = [HEFTScheduler(), LFTScheduler(), GeneticScheduler(mutate_order=0.1, mutate_resources=0.15, number_of_generation=10)]
    names = ['HEFT', 'LFT', 'GA']
    res = {}
    for sched_method, name in zip(schedulers, names):
            res[name] = sched_method.schedule(probem, contractor(N=5))[0].execution_time
    return res

def open_dataset(path_problems):
    return [(f, WorkGraph.loadf(path_problems, f[:-5])) for f in os.listdir(path_problems)]

def main(path_dataset, path_problems):
    problems = open_dataset(path_problems)
    write(problems, path_dataset)
    

# datasets/results.csv  //  wgs/small_synth/.
if __name__ == "__main__": 
    np.random.seed(42)
    path_dataset = sys.argv[1]
    path_problems = sys.argv[2]
    print(path_dataset, path_problems)
    main(path_dataset, path_problems)


# source .project_rcpsp/bin/activate  <--- 1) Способ подключения среды, после можно 2) создавать датасет, после 3) deactivate