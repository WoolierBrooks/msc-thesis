"""
Runner paralelo do exp.py (few-shot), sem alterar a lógica do experimento.

- N_REPETITIONS sobrescrito (padrão 10).
- Cada (dataset, repetição) vira uma tarefa independente num pool de
  processos (uma por núcleo lógico). Como split, scaler e rng dependem só
  de rep_seed = SEED + rep, os resultados são idênticos aos do loop
  sequencial do exp.py — muda só a ordem das linhas no CSV.
- Dentro de cada processo tudo roda single-thread (BLAS e N_JOBS=1) para
  não haver oversubscription.
- Checkpoint/resume reaproveita load_completed_reps() do exp.py.

Uso: python run_parallel.py [--reps 10] [--workers 32]
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys
import time
import argparse
import importlib.util
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))


def load_exp(n_reps):
    spec = importlib.util.spec_from_file_location("exp045", os.path.join(HERE, "exp.py"))
    exp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp)
    exp.N_REPETITIONS = n_reps
    exp.N_JOBS = 1
    exp.OUTPUT_CSV = os.path.join(HERE, f"results_few_shot_quantification_{n_reps}reps.csv")
    return exp


_exp = None
_moss = None


def _init_worker(n_reps):
    global _exp, _moss
    _exp = load_exp(n_reps)
    _exp.print = lambda *a, **k: None  # silencia prints internos dos workers
    _moss = _exp.load_moss_base_binary()


def run_task(ds_name, rep):
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    exp = _exp
    t0 = time.perf_counter()

    dataset = exp.qp.datasets.fetch_UCIBinaryDataset(ds_name, verbose=False)
    train, test = dataset.train_test
    X_all = np.vstack([train.instances, test.instances])
    y_all = np.concatenate([train.labels, test.labels])

    rep_seed = exp.SEED + rep
    Xtr_full, Xte, ytr_full, yte = train_test_split(
        X_all, y_all, test_size=0.3, random_state=rep_seed, stratify=y_all
    )
    scaler = StandardScaler().fit(Xtr_full)
    Xtr_full = scaler.transform(Xtr_full)
    Xte = scaler.transform(Xte)

    all_rows = []
    for n_per_class in exp.N_SHOTS:
        rows = exp.run_one_dataset_one_rep_one_nshot(
            ds_name, Xtr_full, ytr_full, Xte, yte, n_per_class, rep_seed, _moss,
        )
        for r in rows:
            r["repeticao"] = rep + 1
        all_rows.extend(rows)
    return ds_name, rep, all_rows, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    exp = load_exp(args.reps)
    moss = exp.load_moss_base_binary()

    datasets = exp.get_active_dataset_list()
    completed = exp.load_completed_reps()

    # tarefas maiores primeiro, para balancear o pool
    sizes = {}
    for ds in datasets:
        d = exp.qp.datasets.fetch_UCIBinaryDataset(ds, verbose=False)
        tr, te = d.train_test
        sizes[ds] = (len(tr) + len(te)) * tr.instances.shape[1]
    tasks = [(ds, r) for ds in datasets for r in range(args.reps)
             if (ds, r + 1) not in completed]
    tasks.sort(key=lambda t: -sizes[t[0]])

    print(f"Datasets: {len(datasets)} | Repetições: {args.reps} | N-shots: {exp.N_SHOTS}")
    print(f"SMQ habilitado: {moss is not None}")
    print(f"Tarefas pendentes: {len(tasks)} (já concluídas: {len(completed)}) | workers: {args.workers}")
    print(f"Saída: {exp.OUTPUT_CSV}")

    header_written = os.path.exists(exp.OUTPUT_CSV) and os.path.getsize(exp.OUTPUT_CSV) > 0
    timing_path = os.path.join(HERE, f"timing_{args.reps}reps.csv")
    t_start = time.perf_counter()
    sum_task = 0.0
    done = 0

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(args.reps,)) as pool:
        futures = {pool.submit(run_task, ds, r): (ds, r) for ds, r in tasks}
        for fut in as_completed(futures):
            ds, r = futures[fut]
            try:
                ds, r, rows, dt = fut.result()
            except Exception as e:
                print(f"❌ {ds} rep {r + 1}: {e!r}")
                continue
            pd.DataFrame(rows).to_csv(exp.OUTPUT_CSV, mode="a",
                                      header=not header_written, index=False)
            header_written = True
            with open(timing_path, "a") as f:
                f.write(f"{ds},{r + 1},{dt:.2f}\n")
            sum_task += dt
            done += 1
            el = time.perf_counter() - t_start
            print(f"✅ [{done}/{len(tasks)}] {ds} rep {r + 1} em {dt:.1f}s "
                  f"| decorrido {el / 60:.1f} min")

    wall = time.perf_counter() - t_start
    print(f"\n🏁 Concluído em {wall / 60:.1f} min de parede; soma dos tempos das "
          f"tarefas = {sum_task / 60:.1f} min (≈ tempo sequencial), "
          f"speedup ≈ {sum_task / max(wall, 1e-9):.1f}x")


if __name__ == "__main__":
    main()
