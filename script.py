import pandas as pd
import sys
import re
#
from sampo_api import *
from prompts import * 
from models import *
#
import os
from sampo.schemas.graph import WorkGraph
from sampo.scheduler.genetic.operators import TimeFitness
import json
from sampo.scheduler.genetic.operators import is_chromosome_correct

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY")

def dict2string(dict):
    return json.dumps({str(k): list(v) for k, v in dict.items()})

def llm_genetic_algorithm(
    initial_population, 
    toolbox,
    llm_model,
    wg,
    contractors,
    max_generations=4, # 4-6
    elite_size=14, # 8-10
    max_plateau=3,
    f = TimeFitness()):
    """
    Упрощенный генетический алгоритм с селекцией и отслеживанием плато
    """

    # Оцениваем начальную популяцию
    population = []
    for chrom in initial_population:
        fitness = f.evaluate(chrom, evaluator=toolbox.evaluate_chromosome)
        population.append((chrom, fitness, False))
        
    B, land, H = initial_population[0][2], initial_population[0][3], initial_population[0][4] 
    # Сортируем по fitness (чем выше, тем лучше)
    population.sort(key=lambda x: x[1], reverse=True)
    
    # Инициализация
    generation = 0
    plateau = 0
    best_fitness = population[0][1]
    bad_cnt = 0
    # Главный цикл
    while generation < max_generations and plateau < max_plateau:
        
        # Элитарный отбор
        elites = population[:elite_size]
        
        # Генерация потомков
        offsprings = []
        fitness = None
        # Случайные пары для кроссинговера
        idxs = np.arange(len(population))
        np.random.shuffle(idxs)
        print(f'{generation} Поколение генерируется')
        for i in range(0, len(idxs)-1, 2):
            # Берем двух случайных родителей
            idx1, idx2 = idxs[i], idxs[i+1]
            chrom1, _, _ = population[idx1]
            chrom2, _, _ = population[idx2]
            
            # Кроссинговер
      
            str1 = chromosome_to_text(chrom1)
            str2 = chromosome_to_text(chrom2)
            offspring_str = llm_model.generate(str1, str2) 
            TA, RA = parse_chromosome(offspring_str)
            offspring_chrom  = ChromosomeType((np.array(TA), np.array(RA), np.array(B), land, H))
            try:
                if check_chromosome(wg, contractors, offspring_chrom):
                    fitness = f.evaluate(offspring_chrom, evaluator=toolbox.evaluate_chromosome)
                    print(f'В {generation} поколении успешно сгенерирован допустимый кандидат')
                    offsprings.append((offspring_chrom, fitness, True))
                else:
                    bad_cnt += 1
                    print(f'В {generation} поколении ошибка генерации')
            except:
                bad_cnt +=1
                continue
        
        # Если потомков недостаточно, добавляем копии случайных родителей
        if len(offsprings) < len(population) - elite_size:
            needed = (len(population) - elite_size) - len(offsprings)
            for _ in range(needed):
                idx = np.random.randint(0, len(population))
                chrom, fit, is_gen = population[idx]
                offsprings.append((chrom, fit, is_gen))
        
        # Объединяем и сортируем
        combined = elites + offsprings
        combined.sort(key=lambda x: x[1], reverse=False)
        #new_population = combined[:len(population)]
        
        # Проверяем улучшение
        new_best = combined[0][1]
        if new_best > best_fitness:
            print(new_best)
            best_fitness = new_best
        #   plateau = 0
        #else:
        #    plateau += 1
        population = combined
        generation += 1
    
    # Финальная сортировка
    population.sort(key=lambda x: x[1], reverse=False)
    return population, bad_cnt

def create_model(generate_prompt, system_prompt):
    return DeepSeekSession(
        api_key=API_KEY,
        generate_prompt=generate_prompt,
        system_prompt=system_prompt,
        max_history=100
        )

def prompts(path_prompt):
    with open(path_prompt, 'r', encoding='utf-8') as file:
        content = file.read()
    system_prompt, generate_prompt = re.findall('<start>(.+?)</start>', content, re.DOTALL)[0],\
     re.findall('<push>(.+?)</push>', content, re.DOTALL)[0]
    return system_prompt, generate_prompt

def init_problem(path_problems, problem, path_prompt):
    system_prompt, generate_prompt = prompts(path_prompt)
    wg = WorkGraph.loadf(path_problems, problem[:-5])
    contractors = contractor(N = 5)
    children = take_children(wg)
    activity_graph = dict2string(children)
    init_choromosome = first_population(wg, contractors)
    #
    _, _, R = chromosome_to_text(init_choromosome['heft_end'][0])
    #
    system_prompt = system_prompt.format(R, activity_graph) 
    llm = create_model(generate_prompt, system_prompt)
    #
    toolbox = create_mvp_toolbox(wg, contractors)
    population = [chrom_d[0] for chrom_d in list(init_choromosome.values())]
    return llm, toolbox, population, wg, contractors
    
def run_experiment(problems, path_problems, path_prompt):
    exec_times = []
    feasibilities = []
    for n, problem in enumerate(problems):
        print(f'{n+1} problem to solve')
        model, toolbox, population, wg, contractors = init_problem(path_problems, problem, path_prompt)
        population_g, cnt = llm_genetic_algorithm(population, toolbox, model, wg, contractors)
        feasibility = 1 - cnt/model.request_counter
        is_search = False
        for (p, fit, b_f) in population_g:
            if b_f:
                exec_times.append(fit[0])
                is_search = True
                break
        if is_search == False:
            exec_times.append(0)
        feasibilities.append(feasibility)
    return exec_times, feasibilities




def gap(value):
    makespan, opt_makespan = value
    return max(0, (makespan/opt_makespan - 1) * 100)

def save(df, path, metrics): # Сократить строчку с сохранением результатов до функции
    pass

if __name__ == "__main__": 
    np.random.seed(42)
    path_dataset = sys.argv[1]
    result_df = pd.read_csv(path_dataset)[:10] # TODO убрать слайс для полноценного эксперимента
    problems = result_df.FILENAME.tolist()
    path_problems = 'wgs/small_synth'
    path_prompt = 'prompts/' + sys.argv[2] + '.txt'
    llm_execution_time, feasibility = run_experiment(problems, path_problems, path_prompt)
    result_df[f'llm_B_makespan'] = llm_execution_time
    result_df[f'feasibility'] = feasibility
    print(llm_execution_time)
    result_df['gap'] = result_df[['llm_B_makespan','B-makespan']].apply(gap, axis=1, result_type='expand')
    result_df.to_csv(f'datasets/llm_opt_by_prompt_v{sys.argv[2]}',index=False)
    print(f'Результаты эксперимента сохранены в datasets/llm_opt_by_prompt_v{sys.argv[2]}')

    #datasets/results.csv
    # Control + C - keyboard interputt


# STAART - 330 k tokens
# 21:36 STARTss