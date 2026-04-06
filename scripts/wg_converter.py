import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

ROLE_CODES = {
    "driver": "d",
    "fitter": "f",
    "manager": "m",
    "handyman": "h",
    "electrician": "e",
    "engineer": "g",
}

import pandas as pd
from typing import Dict, List, Any

class WorkGraphConverter:
    """
    Преобразует WorkGraph и список подрядчиков в универсальный формат
    """
    def __init__(self):
        pass
    
    def convert(self, work_graph, contractors: List[Any], description: str = "") -> Dict:
        df = work_graph.to_frame(save_req=True)
        return {
            'source_data': self._extract_source_data(work_graph, contractors, df),
            'rcpsp_data': self._convert_to_rcpsp(df, contractors),
        }
    
    def _extract_source_data(self, work_graph, contractors: List[Any], df: pd.DataFrame) -> Dict:
        return {
            'work_graph': work_graph._serialize(),
            'contractors': [
                {
                    'id': c.id,
                    'name': c.name,
                    'workers': dict(c.workers),
                    'equipments': dict(c.equipments)
                }
                for c in contractors
            ],
            'raw_dataframe': df.to_dict(orient='records')
        }
    
    def _convert_to_rcpsp(self, df: pd.DataFrame, contractors: List[Any]) -> Dict:
        return {
            'jobs': [self._convert_job(row) for _, row in df.iterrows()],
            'resources_detailed': self._extract_contractors_info(contractors)
        }
    def _convert_req(self, req):
        return {r:v for r, v in req.items()}
            

    
    def _convert_job(self, row: pd.Series) -> Dict:
        return {
            'id': row['activity_id'],
            'name': row['activity_name'],
            'demand_min': self._convert_req(row['min_req']) if pd.notna(row['min_req']) else {},
            'demand_max': self._convert_req(row['max_req']) if pd.notna(row['max_req']) else {},
            'work_volume': self._convert_req(row['req_volume']) if pd.notna(row['req_volume']) else {},
            'predecessors': self._parse_predecessors(row),
            'priority': int(row['priority']) if pd.notna(row['priority']) else 1
        }
    
    def _parse_predecessors(self, row: pd.Series) -> List[str]:
        if pd.isna(row['predecessor_ids']) or not row['predecessor_ids']:
            return []
        return [p.strip() for p in row['predecessor_ids'].split(',') if p.strip()]

    def _extract_contractors_info(self, contractors: List[Any]) -> List[Dict]:
        return [
        {
            'id': c.id,
            'name': c.name,
            'workers': {r: v.count for r, v in c.workers.items()},  # оригинал, TODO исправить ... 
            'equipments': dict(c.equipments)
        }
        for c in contractors
        ]



from sampo.schemas.resources import Worker
from sampo.schemas.schedule import Schedule, order_nodes_by_start_time
from sampo.schemas.time import Time
from sampo.schemas.resources import Worker
from sampo.schemas.scheduled_work import ScheduledWork

from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.landscape import LandscapeConfiguration
from sampo.scheduler.genetic.utils import prepare_optimized_data_structures
from sampo.scheduler.genetic.converter import convert_schedule_to_chromosome
from sampo.scheduler.genetic.operators import is_chromosome_correct

class ProjectConverter:
    def __init__(self, work_graph, contractors):
        self.work_graph = work_graph
        self.contractors = {contractor.id : contractor for contractor in contractors}
        self.work_units = {node.id: node.work_unit for node in work_graph.nodes}
        self.nodes = work_graph.dict_nodes
        self.landscape = LandscapeConfiguration()
        self.spec = ScheduleSpec()
        _, self.index2node, _, self.work_id2index, self.worker_name2index, self.index2contractor_obj, \
        _, self.contractor2index, self.contractor_borders, self.node_indices, _, self.parents, _, \
        _, _ = prepare_optimized_data_structures(work_graph, contractors, self.landscape, self.spec)
    
    def create_workers(self, assigment, contractor_id):
        workers = []
        for resource_type, count in assigment.items():
            worker = Worker(id=' ', name = resource_type, count = int(count), contractor_id=contractor_id)
            workers.append(worker)
        return workers
    
    def get_graph(self, order):
        return [self.nodes[node] for node in order]
    
    def get_list_graph_nodes(self, schedule_object):
        schedule_works = iter(schedule_object.to_schedule_work_dict.values())
        order_sampo = order_nodes_by_start_time(schedule_works, self.work_graph)
        return self.get_graph(order_sampo)
        
    def to_chromosome(self, schedule, order, job_usage, makespan):
        schedule_object = self.to_schedule(schedule, order, job_usage, makespan)
        schedule_works = iter(schedule_object.to_schedule_work_dict.values())
        order_sampo = order_nodes_by_start_time(schedule_works, self.work_graph)
        return convert_schedule_to_chromosome(self.work_id2index, self.worker_name2index,
                                               self.contractor2index, self.contractor_borders, 
                                               schedule_object, 
                                               self.spec, self.landscape, 
                                               self.get_graph(order_sampo)
                                               )

        
    def to_schedule(self, schedule, order, job_usage, makespan):
        scheduled_works = []
        start, end = self.work_graph.start.id, self.work_graph.finish.id
        contractors = iter(self.contractors.values())
        start_project = ScheduledWork(self.work_units[start],  start_end_time=(Time(0), Time(0)), 
                                    workers=[], contractor=next(contractors))
        scheduled_works.append(start_project)
        for job_id in order:
            contractor_id, start_time, end_time  = schedule[job_id]
            workers = self.create_workers(job_usage[job_id], contractor_id)
            scheduled_work = ScheduledWork(work_unit = self.work_units[job_id],
                                           start_end_time = (Time(start_time), Time(end_time)),
                                           workers = workers,
                                           contractor = self.contractors[contractor_id])
            scheduled_works.append(scheduled_work)
        end_project = ScheduledWork(self.work_units[end],  start_end_time=(Time(makespan), Time(makespan)), 
                                    workers=[], contractor=next(contractors))
        scheduled_works.append(end_project)
        return Schedule.from_scheduled_works(scheduled_works, self.work_graph)
    
    def is_valid_chromosome(self, chromosome):
        return is_chromosome_correct(chromosome, node_indices=self.node_indices, parents=self.parents,
                     contractor_borders=self.contractor_borders, 
                     index2node=self.index2node, index2contractor=self.index2contractor_obj)






# 'workers': {ROLE_CODES[r]: v.count for r, v in c.workers.items()},  # оригинал, TODO исправить ... 