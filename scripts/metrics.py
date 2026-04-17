import numpy as np
import re
import logging


def distance_matrices(solutions):
    """
    solutions: list of tuples (solver_name, res_profile, schedule)
      - solver_name: str
      - res_profile: dict[job_id] -> dict[resource_type] -> total_usage_over_job
      - schedule: dict[job_id] -> (contractor_id, start_time, end_time)

    Возвращает:
      solvers: list[str] в порядке индексов матриц
      D_contractor: np.ndarray [N, N] – доля различий по подрядчикам
      D_interval:   np.ndarray [N, N] – доля различий по интервалам (start,end)
      D_resource:   np.ndarray [N, N] – доля различий по ресурс-профилю
      D_order:      np.ndarray [N, N] – доля различий по относительному порядку работ
    """
    solvers = [s[0][1] for s in solutions]
    N = len(solvers)

    # Удобные словари по имени солвера
    schedules = {name[1] : sched for name, _, sched in  solutions}
    res_profiles = {name[1] : rprof for name, rprof, _ in solutions}

    D_contractor = np.zeros((N, N), dtype=float)
    D_interval   = np.zeros((N, N), dtype=float)
    D_resource   = np.zeros((N, N), dtype=float)
    D_order      = np.zeros((N, N), dtype=float)

    def sign(x: int) -> int:
        if x < 0:
            return -1
        if x > 0:
            return 1
        return 0

    for i in range(N):
        name_a = solvers[i]
        sched_a = schedules[name_a]
        rprof_a = res_profiles[name_a]

        for j in range(i, N):
            name_b = solvers[j]
            sched_b = schedules[name_b]
            rprof_b = res_profiles[name_b]

            if i == j:
                # По определению, расстояние до самого себя = 0
                D_contractor[i, j] = 0.0
                D_interval[i, j]   = 0.0
                D_resource[i, j]   = 0.0
                D_order[i, j]      = 0.0
                continue

            # Общие работы
            jobs_a = set(sched_a.keys())
            jobs_b = set(sched_b.keys())
            common_jobs = jobs_a & jobs_b
            if not common_jobs:
                # Нет общих работ – оставляем 0 или можно поставить np.nan
                continue
            # ---------- 1) Различия по подрядчикам ----------
            diff_contractor = 0
            for job_id in common_jobs:
                cid_a, _, _ = sched_a[job_id]
                cid_b, _, _ = sched_b[job_id]
                if cid_a != cid_b:
                    diff_contractor += 1
            d_contractor = diff_contractor / len(common_jobs)
            # ---------- 2) Различия по интервалам ----------
            diff_interval = 0
            for job_id in common_jobs:
                _, sa, ea = sched_a[job_id][0], sched_a[job_id][1], sched_a[job_id][2]
                _, sb, eb = sched_b[job_id][0], sched_b[job_id][1], sched_b[job_id][2]
                if sa != sb or ea != eb:
                    diff_interval += 1
            d_interval = diff_interval / len(common_jobs)

            # ---------- 3) Различия по ресурс-профилю ----------
            diff_resource = 0
            for job_id in common_jobs:
                pa = rprof_a.get(job_id, {})
                pb = rprof_b.get(job_id, {})
                if pa != pb:
                    diff_resource += 1
            d_resource = diff_resource / len(common_jobs)

            # ---------- 4) Различия по порядку работ ----------
            start_a = {job_id: sched_a[job_id][1] for job_id in common_jobs}
            start_b = {job_id: sched_b[job_id][1] for job_id in common_jobs}

            jobs_list = list(common_jobs)
            diff_order = 0
            pair_count = 0

            for idx1 in range(len(jobs_list)):
                for idx2 in range(idx1 + 1, len(jobs_list)):
                    j1 = jobs_list[idx1]
                    j2 = jobs_list[idx2]
                    pair_count += 1

                    s_a = sign(start_a[j1] - start_a[j2])
                    s_b = sign(start_b[j1] - start_b[j2])

                    if s_a != s_b:
                        diff_order += 1

            d_order = diff_order / pair_count if pair_count > 0 else 0.0
            D_contractor[i, j] = D_contractor[j, i] = d_contractor
            D_interval[i, j]   = D_interval[j, i]   = d_interval
            D_resource[i, j]   = D_resource[j, i]   = d_resource
            D_order[i, j]      = D_order[j, i]      = d_order

    return solvers, D_contractor, D_interval, D_resource, D_order

class StatsCollector:
    def __init__(self,):
        self.items = []

    def add(self, fitness):
        self.items.append(fitness)
    def clear(self):
        self.items = []

class StatsHandler(logging.Handler):
    pattern = re.compile(
        r"-- Generation (?P<generation>\d+), population=(?P<population>\d+), best fitness=\((?P<fitness>\d+)\.0,\) --"
    )
    def __init__(self, collector):
        super().__init__()
        self.collector = collector  # сюда кладём внешний объект
    def emit(self, record):
        msg = self.format(record)
        m = self.pattern.search(msg)
        if not m:
            return
        #generation = int(m.group("generation"))
        fitness = float(m.group("fitness"))
        self.collector.add(fitness)

