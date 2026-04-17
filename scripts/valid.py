import tempfile
import os
from collections import defaultdict

from sampo.utilities.validation import validate_schedule
from sampo.schemas.schedule_spec import ScheduleSpec



def interpter_solver(method, code, data):
    communication_coefficient = lambda n, m: 1.0 / (6 * m ** 2) * (-2 * n ** 3 + 3 * n ** 2 + (6 * m ** 2 - 1) * n) if m != 0 else 0.0
    with tempfile.NamedTemporaryFile(delete=False, suffix='.py') as temp_module:
        temp_module_name = os.path.basename(temp_module.name).split('.')[0]
        temp_module.write(code.encode('utf-8'))
        temp_module_path = temp_module.name
        jobs, resources_borders = data['jobs'], data['resources_detailed']
        try:
            # Import the module
            import importlib.util
            spec = importlib.util.spec_from_file_location(temp_module_name, temp_module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            # Check if 'rcpsp_solver' function exists
            if hasattr(module, 'rcpsp_solver'):
                schedule, order, resource_usage, job_usage, makespan= module.rcpsp_solver(jobs, resources_borders, communication_coefficient)
                if not schedule or not order:
                    print(f"rcpsp_solver did not return a valid solution for method {method}")
            else:
                print(f"No rcpsp_solver function found in the code for method {method}")

            return schedule, order, resource_usage, job_usage, makespan

        except Exception as e:
                print(f"Error executing code for method {method}: {e}")
        finally:
                # Clean up the temporary file
                os.remove(temp_module_path)


def check_predecessors(schedule, jobs):
    """
    Проверяет, что каждая работа стартует не раньше завершения всех своих предшественников.
    Возвращает (ok: bool, violations: list[str]) — список id работ, нарушающих ограничения.
    """
    job_by_id = {job["id"]: job for job in jobs}
    violations = []

    for job_id, (contractor_id, start_time, end_time) in schedule.items():
        job = job_by_id.get(job_id)
        if job is None:
            # В расписании есть работа, которой нет в описании jobs
            violations.append(job_id)
            continue
        
        preds = job.get("predecessors", [])
        if isinstance(preds, str):
            preds = [] if preds == "" else [preds]

        for pred_id in preds:
            if pred_id not in schedule:
                # Предшественник вообще не запланирован
                violations.append(job_id)
                break
            _, pred_start, pred_end = schedule[pred_id]
            if start_time < pred_end:
                # Работа началась раньше окончания предшественника
                violations.append(job_id)
                break

    return len(violations) == 0


def check_capacity(
    jobs,
    resources_borders,
    schedule,
    job_usage):
    """
    Найти все перегрузки capacity по подрядчикам, используя ФАКТИЧЕСКИЕ allocation из job_usage.

    Возвращает список словарей:
      {
        "contractor_id": str,
        "time": int,
        "resource": str,
        "capacity": int,
        "used": int,
        "jobs": [job_id_1, job_id_2, ...]  # работы, которые в этот момент создают перегрузку
      }
    """
    # Быстрый доступ к capacity
    contractor_caps = {c["id"]: c["workers"] for c in resources_borders}

    # contractor_id -> список (job_id, start, end)
    contractor_jobs: Dict[str, List[Tuple[str, int, int]]] = defaultdict(list)
    for job_id, (c_id, start, end) in schedule.items():
        contractor_jobs[c_id].append((job_id, start, end))

    violations: List[Dict[str, Any]] = []

    # Обрабатываем каждого подрядчика
    for contractor_id, cj in contractor_jobs.items():
        if contractor_id not in contractor_caps:
            continue
        capacity = contractor_caps[contractor_id]  # dict[resource -> max]

        if not cj:
            continue

        # Диапазон времени, который нужно сканировать
        min_t = min(start for _, start, _ in cj)
        max_t = max(end for _, _, end in cj)

        # Проходим по всем дискретным моментам времени
        for t in range(min_t, max_t):
            # Какие работы активны в момент t
            active_jobs = [
                job_id for job_id, start, end in cj
                if start <= t < end
            ]
            if not active_jobs:
                continue

            # Считаем использование ресурсов в этот момент по job_usage
            used = defaultdict(int)
            for job_id in active_jobs:
                # Берём уже посчитанный фактический allocation
                usage_for_job = job_usage.get(job_id, {})
                for r, alloc in usage_for_job.items():
                    used[r] += alloc

            # Фиксируем все ресурсы, где есть перегрузка
            for r, used_val in used.items():
                cap_val = capacity.get(r, 0)
                if used_val > cap_val:
                    violations.append({
                        "contractor_id": contractor_id,
                        "time": t,
                        "resource": r,
                        "capacity": cap_val,
                        "used": used_val,
                        "jobs": active_jobs[:],  # копия списка работ
                    })

    return len(violations) == 0



def check_resource_feasibility(resource_usage, contractors):
    """
    Проверяет, что в итоговом resource_usage:
      - ни один подрядчик не использует больше ресурса, чем у него есть в 'workers';
      - не используются ресурсы, которых у подрядчика нет.
    Вход:
      resource_usage: dict[contractor_id][t][resource_type] -> allocated_count
      contractors: list[dict], каждый dict содержит:
         - 'id': str
         - 'workers': dict[resource_type] -> capacity (int)

    Возвращает:
      (ok: bool, violations: list[(contractor_id, t, resource_type, used, capacity)])
    """
    contractor_by_id = {c["id"]: c for c in contractors}
    violations = []

    for contractor_id, timeline in resource_usage.items():
        contractor = contractor_by_id.get(contractor_id)
        if contractor is None:
            # Подрядчик, которого нет в списке contractors
            for t, res_dict in timeline.items():
                for r, used in res_dict.items():
                    violations.append((contractor_id, t, r, used, None))
            continue

        workers = contractor.get("workers", {})

        for t, res_dict in timeline.items():
            for r, used in res_dict.items():
                cap = workers.get(r, 0)

                # 1) ресурс вообще не объявлен у подрядчика
                # 2) или используется больше, чем capacity
                if cap == 0 or used > cap:
                    violations.append((contractor_id, t, r, used, cap))


    return len(violations) == 0

def check_feasibility(schedule, job_usage, resource_usage, data):
    return check_predecessors(schedule, data['jobs']) \
          and check_resource_feasibility(resource_usage, data['resources_detailed']) \
          and check_capacity(data['jobs'], data['resources_detailed'], schedule, job_usage)



def validate_schedule_bool(schedule_obj, work_graph, contractors, spec = ScheduleSpec()):
    try:
        validate_schedule(schedule_obj, work_graph,  contractors, ScheduleSpec())
        return True
    except AssertionError:
        return False



def resource_profiles(schedule, resource_usage):
    """
    Строит агрегированный ресурс-профиль для всех работ.
    """
    res_profile = {}

    for job_id, (contractor_id, start_time, end_time) in schedule.items():
        profile = defaultdict(int)
        for t in range(start_time, end_time):
            slot_usage = resource_usage.get(contractor_id, {}).get(t, {})
            for r, used in slot_usage.items():
                profile[r] += used
        res_profile[job_id] = dict(profile)

    return res_profile

def run_heuristic(base_path, name, data): 
    with open(os.path.join(base_path, name), "r", encoding="utf-8") as f:
                code = f.read()
    schedule, order, resource_usage, job_usage,  makespan = interpter_solver(name, code, data)
    return schedule, order, resource_usage, job_usage, makespan 
