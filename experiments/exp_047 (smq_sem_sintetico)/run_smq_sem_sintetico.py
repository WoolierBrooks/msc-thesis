"""
Experimento 47 — SMQ sem as curvas sintéticas (ablação do MoSS).

Hipótese
--------
No SMQ, cada calibrador (rf, et, gbr) recebe como entrada
    [prevalência prevista pelo MoSS (treinado em curvas SINTÉTICAS),
     8 metafeatures dos scores do batch (média, variância, assimetria e
     curtose dos scores de cada classe)]
e aprende a prevalência real a partir de batches de calibração gerados do
treino REAL. A pergunta é: quanto o MoSS sintético contribui? Para isso este
script treina os mesmos regressores SÓ com as 8 metafeatures das curvas reais
(sem a prevalência do MoSS) e compara com o SMQ completo.

Métodos gerados (por dataset, repetição, n-shot):
  nosyn_rf, nosyn_et, nosyn_gbr   regressor só com as metafeatures (NOVO)
  smq_base, smq_rf, smq_et,       SMQ original, recalculado no mesmo run para a
  smq_gbr, smq_iso                comparação ser pareada (requer o MoSS)

Tudo é idêntico entre nosyn_* e smq_* exceto a entrada do MoSS: mesmo
classificador (RandomForest, 300 árvores), mesmos scores out-of-fold, mesmos
1008 batches de calibração, mesmo StandardScaler na entrada, mesmos
regressores com as mesmas sementes e mesmos 1008 batches de teste.
(smq_iso não tem versão nosyn: ele calibra a saída 1-D do MoSS, que não
existe sem o sintético.)

Protocolo (idêntico ao exp_045 few-shot, importado de lá)
---------------------------------------------------------
30 datasets binários UCI (QuaPy) x N repetições x n-shot. Por padrão roda SÓ
n = full (treino completo), para ser rápido; --nshots inclui outros n. Como o
exp.py recria rng(rep_seed) a cada n-shot, rodar só o full não muda nenhum
batch. Split 70/30 estratificado com semente 42+rep; amostra few-shot; batches APP
(grade de 21 prevalências, 500 instâncias). O rng(rep_seed) de cada
(repetição, n-shot) é consumido exatamente na mesma ordem do exp.py do
exp_045 (amostra few-shot -> batches de teste de cada baseline viável ->
FSNN -> FSMM -> FSGMM/FSGPEM/FSPL -> calibração do SMQ -> teste do SMQ), então
os batches são os mesmos do exp_045 e os resultados ficam pareados com
results_few_shot_quantification_10reps.csv e results_smq_10reps.csv.

Verificação (coluna "pareado"), quando os CSVs do exp_045 existem:
  - AE do CC recalculado sobre os batches do CC == CSV do exp_045;
  - AE do smq_et recalculado == results_smq_10reps.csv do exp_045.

Requisitos
----------
- Repositório com ../exp_045 (few_shot)/exp.py.
- MoSS em ../exp_046 (llp)/moss_binario_lite.pkl (vem pelo Git LFS: rode
  `git lfs pull` no servidor). Sem ele, só os nosyn_* rodam.
- Para a verificação: ../exp_045 (few_shot)/results_few_shot_quantification_10reps.csv
  e results_smq_10reps.csv (opcionais).
- Pacotes: numpy, pandas, scipy, scikit-learn, quapy==0.2.2.

Uso
---
    python run_smq_sem_sintetico.py                      # n=full, 10 repetições, 1 worker por núcleo
    python run_smq_sem_sintetico.py --workers 64
    python run_smq_sem_sintetico.py --nshots 1 2 3 5 10 100 full   # todos os n
    python run_smq_sem_sintetico.py --datasets iris.1 --reps 1     # teste rápido

Saída: results_sem_sintetico_<N>reps_n<lista de n>.csv (ex.: ..._10reps_nfull.csv;
uma linha por dataset, repetição,
n e método; mesmas colunas do exp_045 + "pareado"). Tem resume.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import importlib.util
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

HERE = os.path.dirname(os.path.abspath(__file__))
EXP45_DIR = os.path.normpath(os.path.join(HERE, "..", "exp_045 (few_shot)"))
EXP45_SCRIPT = os.path.join(EXP45_DIR, "exp.py")
MOSS_PKL = os.path.normpath(os.path.join(HERE, "..", "exp_046 (llp)", "moss_binario_lite.pkl"))
TOL = 1e-9


def _instalar_compat_read_csv():
    """pandas 3 removeu delim_whitespace, que o QuaPy 0.2.2 usa ao baixar o german."""
    from pandas.io.parsers import read_csv as read_csv_original

    def read_csv_compat(*args, **kwargs):
        if kwargs.pop("delim_whitespace", False):
            kwargs["sep"] = r"\s+"
        return read_csv_original(*args, **kwargs)

    pd.read_csv = read_csv_compat


def load_exp():
    _instalar_compat_read_csv()
    spec = importlib.util.spec_from_file_location("exp045", EXP45_SCRIPT)
    exp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp)
    exp.N_JOBS = 1
    exp.MOSS_BINARY_PKL = MOSS_PKL
    return exp


_exp = None
_moss = None


def _init_worker():
    global _exp, _moss
    _exp = load_exp()
    _exp.print = lambda *a, **k: None
    _moss = _exp.load_moss_base_binary() if os.path.exists(MOSS_PKL) else None


# ---------------- utilidades vetorizadas (mesmas fórmulas do exp.py) ----------------

def batch_features(scores, batches_idx):
    """(B batches) -> (B, 8): por classe [média, variância, assimetria, curtose],
    igual a baseline_features do exp.py."""
    if len({len(i) for i in batches_idx}) == 1:
        S = scores[np.stack(batches_idx)]
        cols = []
        for c in range(S.shape[2]):
            s = S[:, :, c]
            cols += [s.mean(axis=1), s.var(axis=1), skew(s, axis=1), kurtosis(s, axis=1)]
        return np.nan_to_num(np.column_stack(cols), nan=0.0, posinf=0.0, neginf=0.0)
    return np.array([_exp.baseline_features(scores[i], 2) for i in batches_idx])


def safe_rows(P):
    P = np.nan_to_num(np.asarray(P, dtype=float)[:, :2], nan=0.0, posinf=0.0, neginf=0.0)
    P = np.clip(P, 0, 1)
    s = P.sum(axis=1)
    out = np.full_like(P, 0.5)
    ok = s > 0
    out[ok] = P[ok] / s[ok, None]
    return out


def summarize(P_hat, P_real, n_train):
    ae = np.abs(P_hat - P_real).mean(axis=1)
    eps = 1.0 / (2 * n_train)
    ph = (P_hat + eps) / (P_hat.sum(axis=1, keepdims=True) + 2 * eps)
    pt = (P_real + eps) / (P_real.sum(axis=1, keepdims=True) + 2 * eps)
    rae = (np.abs(ph - pt) / pt).mean(axis=1)
    return _exp.summarize_method_results(ae, rae)


# ---------------- uma (dataset, repetição, n-shot) ----------------

def one_nshot(Xtr_full, ytr_full, Xte, yte, n_per_class, rep_seed):
    from sklearn.preprocessing import StandardScaler
    exp = _exp
    rng = np.random.default_rng(rep_seed)
    Xtr, ytr = exp.sample_few_shot(Xtr_full, ytr_full, n_per_class, rng)

    def test_batches():
        return list(exp.artificial_prevalence_batches(Xte, yte, 2, exp.N_TEST_BATCHES,
                                                      exp.BATCH_SIZE, rng))

    # replay do consumo do rng pelos métodos que vêm antes do SMQ no exp.py
    viaveis = [nm for nm in exp.BASELINE_CLASSES
               if not (nm in exp.NEEDS_VAL_SPLIT and exp.adaptive_val_split(n_per_class) is None)]
    cc_test = test_batches()                 # CC
    for _ in viaveis[1:]:
        test_batches()                       # PCC, ACC, PACC, EMQ, DyS, KDEyML
    for _ in range(3):
        test_batches()                       # FSNN, FSMM, FSGMM/FSGPEM/FSPL

    lab = exp.make_lr().fit(Xtr, ytr).predict(Xte).astype(float)
    cc_ae = float(np.mean([exp.ae(np.array([1 - lab[i].mean(), lab[i].mean()]), p)
                           for i, p in cc_test]))

    # ---- calibração (mesma do fit_smq) ----
    clf = exp.make_rf_smq(rep_seed).fit(Xtr, ytr)
    oof = exp.get_oof_scores_adaptive(Xtr, ytr, rep_seed)
    cal = list(exp.artificial_prevalence_batches(Xtr, ytr, 2, exp.N_CALIB_BATCHES,
                                                 exp.BATCH_SIZE, rng))
    F_cal = batch_features(oof, [i for i, _ in cal])
    real = np.array([p for _, p in cal])

    test = test_batches()
    test_idx = [i for i, _ in test]
    P_real = np.array([p for _, p in test])
    F_te = batch_features(clf.predict_proba(Xte), test_idx)

    preds = {}

    # ---- NOVO: regressores só com as metafeatures (sem MoSS) ----
    sc_ns = StandardScaler().fit(F_cal)
    Xs_cal, Xs_te = sc_ns.transform(F_cal), sc_ns.transform(F_te)
    for nm, factory in exp.REGRESSOR_FACTORIES_SMQ.items():
        m = factory(rep_seed).fit(Xs_cal, real)
        preds[f"nosyn_{nm}"] = safe_rows(m.predict(Xs_te))

    # ---- SMQ original (com MoSS), para comparação pareada ----
    if _moss is not None:
        pred_cal = safe_rows(_moss.predict(F_cal))
        X_in = np.nan_to_num(np.hstack([pred_cal, F_cal]), nan=0.0, posinf=0.0, neginf=0.0)
        sc = StandardScaler().fit(X_in)
        Xs = sc.transform(X_in)
        base = safe_rows(_moss.predict(F_te))
        Xs_t = sc.transform(np.hstack([base, F_te]))
        preds["smq_base"] = base
        for nm, factory in exp.REGRESSOR_FACTORIES_SMQ.items():
            preds[f"smq_{nm}"] = safe_rows(factory(rep_seed).fit(Xs, real).predict(Xs_t))
        iso = exp.train_isotonic_calibrator(pred_cal, real, 2)
        preds["smq_iso"] = safe_rows(np.column_stack([iso[c].predict(base[:, c]) for c in range(2)]))

    rows = [dict(n=n_per_class, metodo=nm, **summarize(P, P_real, len(ytr)))
            for nm, P in preds.items()]
    return rows, cc_ae


def parse_nshots(values):
    return [v if v == "full" else int(v) for v in values]


def run_task(ds_name, rep, refs, nshots):
    """refs: {(n_str, 'CC'|'smq_et'): ae_mean} vindos dos CSVs do exp_045."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    exp = _exp
    t0 = time.perf_counter()
    d = exp.qp.datasets.fetch_UCIBinaryDataset(ds_name, verbose=False)
    train, test = d.train_test
    X_all = np.vstack([train.instances, test.instances])
    y_all = np.concatenate([train.labels, test.labels])
    rep_seed = exp.SEED + rep
    Xtr_full, Xte, ytr_full, yte = train_test_split(
        X_all, y_all, test_size=0.3, random_state=rep_seed, stratify=y_all)
    sc = StandardScaler().fit(Xtr_full)
    Xtr_full, Xte = sc.transform(Xtr_full), sc.transform(Xte)

    all_rows, bad = [], []
    for n in nshots:
        rows, cc_ae = one_nshot(Xtr_full, ytr_full, Xte, yte, n, rep_seed)
        checks = []
        if (str(n), "CC") in refs:
            checks.append(abs(cc_ae - refs[(str(n), "CC")]) <= TOL)
        et = next((r["ae_mean"] for r in rows if r["metodo"] == "smq_et"), None)
        if et is not None and (str(n), "smq_et") in refs:
            checks.append(abs(et - refs[(str(n), "smq_et")]) <= TOL)
        ok = all(checks) if checks else None   # None = sem CSV de referência
        if ok is False:
            bad.append(n)
        for r in rows:
            r.update(dataset=ds_name, repeticao=rep + 1, pareado=ok)
        all_rows.extend(rows)
    return ds_name, rep, all_rows, bad, time.perf_counter() - t0


def _load_refs(n_reps):
    refs = {}
    for fname, metodo in ((f"results_few_shot_quantification_{n_reps}reps.csv", "CC"),
                          (f"results_smq_{n_reps}reps.csv", "smq_et")):
        p = os.path.join(EXP45_DIR, fname)
        if not os.path.exists(p):
            print(f"⚠️  Sem referência para verificação: {p}")
            continue
        df = pd.read_csv(p, dtype={"n": str})
        df = df[df["metodo"] == metodo]
        for d, rp, n, a in zip(df["dataset"], df["repeticao"], df["n"], df["ae_mean"]):
            refs.setdefault((d, int(rp)), {})[(n, metodo)] = float(a)
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--nshots", nargs="*", default=["full"],
                    help="n-shots a rodar (padrão: só full). Ex.: --nshots 1 2 3 5 10 100 full")
    args = ap.parse_args()
    nshots = parse_nshots(args.nshots)

    os.chdir(HERE)
    exp = load_exp()
    tem_moss = os.path.exists(MOSS_PKL)
    tag = "n" + "-".join(str(n) for n in nshots)
    out_csv = os.path.join(HERE, f"results_sem_sintetico_{args.reps}reps_{tag}.csv")
    refs = _load_refs(args.reps)

    datasets = [d for d in exp.get_active_dataset_list()
                if not args.datasets or d in args.datasets]
    done = set()
    if os.path.exists(out_csv):
        o = pd.read_csv(out_csv, usecols=["dataset", "repeticao"])
        done = set(zip(o["dataset"], o["repeticao"].astype(int)))
    pend = [(d, r) for d in datasets for r in range(1, args.reps + 1) if (d, r) not in done]

    # baixa/carrega todos os datasets antes do pool (evita downloads concorrentes)
    sizes = {}
    for ds in {d for d, _ in pend}:
        dd = exp.qp.datasets.fetch_UCIBinaryDataset(ds, verbose=False)
        tr, te = dd.train_test
        sizes[ds] = (len(tr) + len(te)) * tr.instances.shape[1]
    pend.sort(key=lambda k: (-sizes[k[0]], k[0], k[1]))

    cols = ["dataset", "n", "metodo", "ae_mean", "ae_std", "ae_median", "rae_mean",
            "rae_std", "rae_median", "n_batches", "repeticao", "pareado"]
    print(f"MoSS: {MOSS_PKL if tem_moss else 'NÃO encontrado — só nosyn_* serão gerados'}")
    print(f"Datasets: {len(datasets)} | Repetições: {args.reps} | N-shots: {nshots}")
    print(f"Pares pendentes: {len(pend)} (já feitos: {len(done)}) | workers: {args.workers}")
    print(f"Saída: {out_csv}")

    t_start, n_done, n_bad = time.perf_counter(), 0, 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        futs = {pool.submit(run_task, d, rp - 1, refs.get((d, rp), {}), nshots): (d, rp)
                for d, rp in pend}
        for fut in as_completed(futs):
            d, rp = futs[fut]
            try:
                ds, rep, rows, bad, dt = fut.result()
            except Exception as e:
                print(f"❌ {d} rep {rp}: {e!r}")
                continue
            pd.DataFrame(rows).reindex(columns=cols).to_csv(
                out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
            n_done += 1
            n_bad += bool(bad)
            el = time.perf_counter() - t_start
            msg = f" ⚠️ NÃO pareado em n={bad}" if bad else ""
            print(f"✅ [{n_done}/{len(pend)}] {ds} rep {rep + 1} em {dt:.1f}s | decorrido "
                  f"{el / 60:.1f} min, faltam ~{(len(pend) - n_done) * el / n_done / 60:.1f} min{msg}")

    print(f"\n🏁 Concluído em {(time.perf_counter() - t_start) / 60:.1f} min; "
          f"repetições não pareadas: {n_bad}")


if __name__ == "__main__":
    main()
