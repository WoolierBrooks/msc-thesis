"""
Complementa o exp_045 (few-shot) com os métodos SMQ_* (smq_base, smq_rf,
smq_et, smq_gbr, smq_iso), PAREADOS com o CSV já gerado pelo run_parallel.py.

O run original não encontrou o MoSS (caminho de servidor Linux fixo no
exp.py), então só os outros 12 métodos rodaram. Este script usa o MoSS em
../exp_046 (llp)/moss_binario_lite.pkl.

Como a reprodução funciona
--------------------------
No exp.py, para cada (dataset, repetição, n-shot) é criado rng(rep_seed), que
é consumido nesta ordem: sample_few_shot; um gerador de batches de teste por
baseline viável (CC, PCC, ACC, PACC, EMQ, DyS, KDEyML — no 1-shot só CC e
PCC); FSNN; FSMM; o laço FSGMM/FSGPEM/FSPL; e por fim o SMQ (batches de
calibração e depois batches de teste). Os modelos em si não consomem esse rng.
Este script repete exatamente esse consumo sem treinar os outros métodos e
então treina e avalia o SMQ — é o resultado que o exp.py teria gerado se o
MoSS estivesse no caminho.

Verificação: o AE médio do CC é recalculado sobre os batches de teste do CC e
comparado com o do CSV (confere split, amostra few-shot e batches de ponta a
ponta). Se bater, as linhas saem com pareado=True.

Desempenho: scores do teste calculados uma vez e indexados por batch, e
features/predições do MoSS e dos calibradores em uma chamada para todos os
batches (mesmos números do apply_smq, sem as ~5000 chamadas de predict).

Uso: python run_smq_complementar.py [--reps 10] [--workers 32]
Saída: results_smq_<N>reps.csv (mesmas colunas do CSV principal + pareado).
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

import run_parallel as RP

HERE = RP.HERE
MOSS_PKL = os.path.normpath(os.path.join(HERE, "..", "exp_046 (llp)", "moss_binario_lite.pkl"))
TOL_CC = 1e-9

_exp = None
_moss = None


def _init_worker(n_reps):
    global _exp, _moss
    _exp = RP.load_exp(n_reps)
    _exp.print = lambda *a, **k: None
    _exp.MOSS_BINARY_PKL = MOSS_PKL
    _moss = _exp.load_moss_base_binary()


# ---------------- SMQ vetorizado (mesmas fórmulas do exp.py) ----------------

def batch_features(scores, batches_idx):
    if len({len(i) for i in batches_idx}) == 1:
        S = scores[np.stack(batches_idx)]          # (B, n, 2)
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


def smq_one_nshot(Xtr_full, ytr_full, Xte, yte, n_per_class, rep_seed):
    exp = _exp
    rng = np.random.default_rng(rep_seed)
    Xtr, ytr = exp.sample_few_shot(Xtr_full, ytr_full, n_per_class, rng)

    def test_batches():
        return list(exp.artificial_prevalence_batches(Xte, yte, 2, exp.N_TEST_BATCHES,
                                                      exp.BATCH_SIZE, rng))

    # baselines viáveis (mesma regra do fit_baseline: None quando falta val_split)
    viaveis = [nm for nm in exp.BASELINE_CLASSES
               if not (nm in exp.NEEDS_VAL_SPLIT and exp.adaptive_val_split(n_per_class) is None)]
    assert viaveis[0] == "CC"
    cc_test = test_batches()                       # batches do CC
    for _ in viaveis[1:]:
        test_batches()                             # PCC, ACC, ...
    for _ in range(3):
        test_batches()                             # FSNN, FSMM, FSGMM/FSGPEM/FSPL

    # checagem: CC (= LR classificando e contando)
    lab = exp.make_lr().fit(Xtr, ytr).predict(Xte).astype(float)
    cc_ae = float(np.mean([exp.ae(np.array([1 - lab[i].mean(), lab[i].mean()]), p)
                           for i, p in cc_test]))

    # ---- fit_smq ----
    clf = exp.make_rf_smq(rep_seed).fit(Xtr, ytr)
    oof = exp.get_oof_scores_adaptive(Xtr, ytr, rep_seed)
    cal = list(exp.artificial_prevalence_batches(Xtr, ytr, 2, exp.N_CALIB_BATCHES,
                                                 exp.BATCH_SIZE, rng))
    F_cal = batch_features(oof, [i for i, _ in cal])
    real = np.array([p for _, p in cal])
    pred_cal = safe_rows(_moss.predict(F_cal))
    X_in = np.nan_to_num(np.hstack([pred_cal, F_cal]), nan=0.0, posinf=0.0, neginf=0.0)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_in)
    Xs = scaler.transform(X_in)
    calibs = {nm: f(rep_seed).fit(Xs, real) for nm, f in exp.REGRESSOR_FACTORIES_SMQ.items()}
    iso = exp.train_isotonic_calibrator(pred_cal, real, 2)

    # ---- apply_smq nos batches de teste do SMQ ----
    test = test_batches()
    test_idx = [i for i, _ in test]
    P_real = np.array([p for _, p in test])
    F_te = batch_features(clf.predict_proba(Xte), test_idx)
    base = safe_rows(_moss.predict(F_te))
    Xs_te = scaler.transform(np.hstack([base, F_te]))
    preds = {"smq_base": base}
    for nm, m in calibs.items():
        preds[f"smq_{nm}"] = safe_rows(m.predict(Xs_te))
    preds["smq_iso"] = safe_rows(np.column_stack([iso[c].predict(base[:, c]) for c in range(2)]))

    rows = [dict(dataset=None, n=n_per_class, metodo=nm, **summarize(P, P_real, len(ytr)))
            for nm, P in preds.items()]
    return rows, cc_ae


def run_task(ds_name, rep, cc_ref):
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
    for n in exp.N_SHOTS:
        rows, cc_ae = smq_one_nshot(Xtr_full, ytr_full, Xte, yte, n, rep_seed)
        ref = cc_ref.get(str(n))
        ok = ref is not None and abs(cc_ae - ref) <= TOL_CC
        if not ok:
            bad.append((n, cc_ae, ref))
        for r in rows:
            r.update(dataset=ds_name, repeticao=rep + 1, pareado=ok)
        all_rows.extend(rows)
    return ds_name, rep, all_rows, bad, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    if not os.path.exists(MOSS_PKL):
        print(f"❌ MoSS não encontrado: {MOSS_PKL}")
        sys.exit(1)

    exp = RP.load_exp(args.reps)
    base_csv = exp.OUTPUT_CSV
    out_csv = os.path.join(HERE, f"results_smq_{args.reps}reps.csv")
    base = pd.read_csv(base_csv, dtype={"n": str})
    columns = list(base.columns) + ["pareado"]
    cc = base[base["metodo"] == "CC"]
    cc_ref = {}
    for d, rp, n, a in zip(cc["dataset"], cc["repeticao"], cc["n"], cc["ae_mean"]):
        cc_ref.setdefault((d, int(rp)), {})[n] = float(a)
    todo = sorted(cc_ref)

    done = set()
    if os.path.exists(out_csv):
        o = pd.read_csv(out_csv, usecols=["dataset", "repeticao"])
        done = set(zip(o["dataset"], o["repeticao"].astype(int)))
    pend = [k for k in todo if k not in done]

    sizes = {}
    for ds in {d for d, _ in pend}:
        dd = exp.qp.datasets.fetch_UCIBinaryDataset(ds, verbose=False)
        tr, te = dd.train_test
        sizes[ds] = (len(tr) + len(te)) * tr.instances.shape[1]
    pend.sort(key=lambda k: -sizes[k[0]])

    print(f"MoSS: {MOSS_PKL}")
    print(f"Pares (dataset, repetição) pendentes: {len(pend)} (já feitos: {len(done)}) | workers: {args.workers}")
    print(f"Saída: {out_csv}")

    t_start, n_done, n_bad = time.perf_counter(), 0, 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(args.reps,)) as pool:
        futs = {pool.submit(run_task, d, rp - 1, cc_ref[(d, rp)]): (d, rp) for d, rp in pend}
        for fut in as_completed(futs):
            d, rp = futs[fut]
            try:
                ds, rep, rows, bad, dt = fut.result()
            except Exception as e:
                print(f"❌ {d} rep {rp}: {e!r}")
                continue
            pd.DataFrame(rows).reindex(columns=columns).to_csv(
                out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
            n_done += 1
            n_bad += bool(bad)
            msg = "" if not bad else f" ⚠️ NÃO pareado em n={[b[0] for b in bad]}"
            print(f"✅ [{n_done}/{len(pend)}] {ds} rep {rep + 1} em {dt:.1f}s "
                  f"| decorrido {(time.perf_counter() - t_start) / 60:.1f} min{msg}")

    print(f"\n🏁 Concluído em {(time.perf_counter() - t_start) / 60:.1f} min; "
          f"repetições não pareadas: {n_bad}")


if __name__ == "__main__":
    main()
