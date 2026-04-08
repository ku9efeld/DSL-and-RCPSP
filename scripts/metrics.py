import numpy as np


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

from math import inf

def summarize_runs_main(runs):
    """
    runs: список run-ов main-алгоритма (algs_history['genetic']),
          каждый run — список точек с 'best_fitness' и 'generation'.

    Возвращает по каждому run:
      {
        'main_best_fitness': float | None,
        'main_best_gen': int | None,
      }
    """
    summaries = []
    for run_history in runs:
        vals = [
            x for x in run_history
            if isinstance(x, dict)
            and 'best_fitness' in x
            and x['best_fitness'] != inf
        ]
        if not vals:
            summaries.append({
                'main_best_fitness': None,
                'main_best_gen': None,
            })
            continue

        best = min(vals, key=lambda x: x['best_fitness'])
        summaries.append({
            'main_best_fitness': best['best_fitness'],
            'main_best_gen': best['generation'],
        })
    return summaries


def metrics_by_init_population(
    main_summary,
    test_runs):
    """
    main_summary: результат summarize_runs_main(algs_history['genetic'])
    test_runs: сравниваемый ГА 

    Возвращает список per_run с полями:
      - main_best_fitness, main_best_gen
      - test_best_fitness, test_best_gen
      - reached_main_level, gen_to_main_level, delta_gen_to_main
      - found_better_than_main, gen_to_better, delta_gen_to_better
      - gain_improve
    """
    assert len(main_summary) == len(test_runs)
    num_runs = len(test_runs)

    per_run = []

    for i in range(num_runs):
        ms = main_summary[i]
        main_best = ms['main_best_fitness']
        main_gen  = ms['main_best_gen']

        vals = [
            x for x in test_runs[i]
            if isinstance(x, dict)
            and 'best_fitness' in x
            and x['best_fitness'] != inf
        ]

        # Сортируем точки по поколениям
        vals = sorted(vals, key=lambda x: x['generation'])

        # лучший теста
        best_test = min(vals, key=lambda x: x['best_fitness'])
        test_best_fitness = best_test['best_fitness']
        test_best_gen = best_test['generation']

        # 1) достиг ли уровня main (best_fitness <= main_best) и когда впервые
        gen_to_main = None
        for rec in vals:
            if rec['best_fitness'] <= main_best:
                gen_to_main = rec['generation']
                break

        if gen_to_main is not None:
            reached_main_level = True
            delta_gen_to_main = gen_to_main - main_gen  # <0 → тест достиг уровня main раньше
        else:
            reached_main_level = False
            delta_gen_to_main = None

        # 2) нашёл ли лучше, чем main (best_fitness < main_best) и когда впервые
        gen_to_better = None
        for rec in vals:
            if rec['best_fitness'] < main_best:
                gen_to_better = rec['generation']
                break

        if gen_to_better is not None:
            found_better_than_main = True
            delta_gen_to_better = gen_to_better - main_gen  # <0 → тест раньше нашёл лучшую точку, чем main дошёл до своей
        else:
            found_better_than_main = False
            delta_gen_to_better = None

        # 3) gain_improve = max((main_best - test_best_fitness) / main_best, 0)
        if main_best is not None and main_best != 0:
            raw_gain = (main_best - test_best_fitness) * 100/ main_best
            gain_improve = max(raw_gain, 0.0)
        else:
            gain_improve = 0.0

        per_run.append({
            'run_index': i,
            'main_best_fitness': main_best,
            'main_best_gen': main_gen,
            'test_best_fitness': test_best_fitness,
            'test_best_gen': test_best_gen,
            'reached_main_level': reached_main_level,
            'gen_to_main_level': gen_to_main,
            'delta_gen_to_main': delta_gen_to_main,
            'found_better_than_main': found_better_than_main,
            'gen_to_better': gen_to_better,
            'delta_gen_to_better': delta_gen_to_better,
            'gain_improve': gain_improve
        })

    return per_run

def aggregate_metrics(all_runs):
    reached_flags = [r['reached_main_level'] for r in all_runs]
    better_flags = [r['found_better_than_main'] for r in all_runs]
    gains = [r['gain_improve'] for r in all_runs]

    delta_main = [r['delta_gen_to_main'] for r in all_runs if r['delta_gen_to_main'] is not None]
    delta_better = [r['delta_gen_to_better'] for r in all_runs if r['delta_gen_to_better'] is not None]

    return {
        'success_rate': float(100 * np.mean([1.0 if f else 0.0 for f in reached_flags])),
        'found_better_rate': float(100 * np.mean([1.0 if f else 0.0 for f in better_flags])),
        'mean_delta_gen_to_main': float(np.mean(delta_main)) if delta_main else None,
        'mean_delta_gen_to_better': float(np.mean(delta_better)) if delta_better else None,
        'mean_gain_improve': round(float(np.mean(gains)) if gains else 0.0, 1),
        'count_gain_improve': np.sum([1 if g > 0 else 0 for g in gains]) if gains else 0.0,
    }