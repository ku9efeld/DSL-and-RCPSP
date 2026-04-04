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



# 'workers': {ROLE_CODES[r]: v.count for r, v in c.workers.items()},  # оригинал, TODO исправить ... 