import numpy as np
import re
from sampo.api.genetic_api import ChromosomeType


TAGS = {
    'Activity List':'TA',
    'Resources for activity':'RA',
    'Boarders for resources':'R'
}

def array_to_string(arr, line_separator = '\n', column_separator=' ', with_brackets=False):
    result_lines = []
    for row in arr:
        # Форматируем каждое число в строке
        formatted_row = []
        for val in row:
            if np.isnan(val):
                formatted_row.append("nan")
            elif np.isinf(val):
                formatted_row.append("inf" if val > 0 else "-inf")
            else:
                if isinstance(val, (int, np.integer)):
                    formatted_row.append(str(int(val)))
        
        # Собираем строку
        row_str = column_separator.join(formatted_row)
        
        if with_brackets:
            row_str = f"[{row_str}]"
        
        result_lines.append(row_str)
    
    return line_separator.join(result_lines)

def chromosome_to_text(chrm : ChromosomeType):
    c1, c2, c3, _, _ = chrm
    task_activity = '<TA>'+ ','.join([str(e) for e in c1.tolist()]) + '</TA>\n'
    resources_for_activity = '<RA>' + array_to_string(c2) + '</RA>'
    resources_border = '<R>' + array_to_string(c3) + '</R>'
    return np.array(task_activity), resources_for_activity, resources_border

def str_to_numpy_array(data_str):
    return np.array([[int(x) for x in line.split()] 
                     for line in data_str.strip().split('\n') 
                     if line.strip()])

def parse_chromosome(llm_output, first=True):
    res = []
    for tag in ['TA','RA']:
        m = re.findall(f'<{tag}>(.+?)</{tag}>', llm_output, re.DOTALL)[0]
        if tag == 'TA':
            m = list(map(int, m.split(',')))
            res.append(m)
        elif tag == 'R' and not first:
                continue
        else:
            res.append(str_to_numpy_array(m))
    return res

