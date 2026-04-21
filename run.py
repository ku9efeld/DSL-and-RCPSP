
from sampo.schemas.graph import WorkGraph
from sampo_api import contractor
from Experiments.structure import run_experiments_parallel
import pandas as pd

if __name__ == "__main__":
    GA  = {
    'number_of_generation' : 2,  
    'size_of_population' : 60,
    'mutate_order' : 0.05,
    'mutate_resources': 0.05,
    'mutate_zones': 0.05} 
    wg , contractors  = WorkGraph.loadf('wgs/small_synth', 'wg_32'), contractor(N=5)
    res_dfs = run_experiments_parallel(GA, wg, contractors, 'deepseek_chat', N_runs=2, max_workers=2)
    print(pd.concat(res_dfs).shape)