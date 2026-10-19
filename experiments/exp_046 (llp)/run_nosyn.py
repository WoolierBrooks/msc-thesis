"""
Script complementar: acrescenta ao experimento LLP vs quantificação os
regressores SEM o MoSS sintético (ablação do exp_047), PAREADOS com
results_smq_complementar.csv.

Métodos
-------
  [oraculo]  nosyn_rf, nosyn_et, nosyn_gbr
  [smq_llp]  nosyn_llp_rf, nosyn_llp_et, nosyn_llp_gbr

Mesma ideia do exp_047: o regressor recebe SÓ as 8 metafeatures dos scores do
batch (média, variância, assimetria e curtose por classe), sem a prevalência
prevista pelo MoSS. Tudo o mais é idêntico ao smq_* / smq_llp_* do
run_smq_complementar.py: mesmo classificador, mesmos scores OOF, mesmos 1000
batches de calibração (oráculo) / uniões de bags (LLP), mesmos regressores com
as mesmas sementes e mesmos 1008 batches de teste.

Como a reprodução funciona
--------------------------
Reaproveita o run_rep do run_smq_complementar.py, só trocando
fit_calibration_fast / predict_all por versões que, além do SMQ, treinam e
aplicam os regressores nosyn com as mesmas features de calibração. O fluxo dos
dois rngs (rng(rep_seed) e rng_smq) fica intacto, então os batches são os
mesmos. Verificação, por configuração (grupo, regime, bag_size): os AEs do SMQ
pareado=True quando as checagens do próprio run_rep passam: AE do CC do
oráculo igual ao do CSV base (dados, split e batches de teste) e estatísticas
das bags iguais às gravadas.

Atenção: o SMQ recalculado aqui NÃO bate bit a bit com
results_smq_complementar.csv (nem o smq_base, que só depende do MoSS e do RF
classificador): os RandomForest saem diferentes entre máquinas/versões do
sklearn. As diferenças são pequenas (~1e-4 de AE), mas por isso o SMQ deste
mesmo run também é salvo, em results_nosyn_smq_rerun.csv: para a ablação
nosyn vs smq use esse arquivo (pareamento estrito, mesmo MoSS e mesmo RF); para
comparar com os outros métodos do exp_046 os batches são os mesmos. A coluna
smq_drift guarda o maior |AE recalculado - AE do CSV| do SMQ na configuração.

Uso
---
    python run_nosyn.py                    # tudo, 1 worker por núcleo
    python run_nosyn.py --workers 16
    python run_nosyn.py --datasets iris.1 --reps 1   # teste rápido

Saída: results_nosyn.csv (linhas nosyn_*) e results_nosyn_smq_rerun.csv (linhas
smq_* recalculadas), com as colunas de results_smq_complementar.csv +
smq_drift. Tem resume.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import contextlib
import io
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SMQ_CSV = "results_smq_complementar.csv"
BASE_CSV = "results_llp_vs_quantification.csv"
OUT_CSV = "results_nosyn.csv"
RERUN_CSV = "results_nosyn_smq_rerun.csv"
CFG_KEYS = ["grupo", "regime", "bag_size"]

C = None      # módulo run_smq_complementar (que carrega o principal como C.E)
_moss = None


def load_complementar():
    """Importa run_smq_complementar.py sem rodar o main (silencia os prints).
    O arquivo veio de um notebook com todo o código indentado, daí o dedent."""
    import textwrap
    import types
    os.chdir(HERE)
    path = os.path.join(HERE, "run_smq_complementar.py")
    with open(path, encoding="utf-8") as f:
        src = textwrap.dedent(f.read())
    mod = types.ModuleType("run_smq_complementar")
    mod.__file__ = path
    sys.modules[mod.__name__] = mod
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, path, "exec"), mod.__dict__)
    mod.E.N_JOBS = 1
    return mod


def _install_nosyn():
    """Envolve fit_calibration_fast / predict_all do complementar para treinar e
    aplicar também os regressores só com metafeatures."""
    fit_orig, predict_orig = C.fit_calibration_fast, C.predict_all

    def fit_calibration_fast(moss, F_cal, real, seed):
        scaler, cal, iso = fit_orig(moss, F_cal, real, seed)
        sc_ns = C.StandardScaler().fit(F_cal)
        Xs = sc_ns.transform(F_cal)
        cal["__nosyn__"] = (sc_ns, {name: factory(seed).fit(Xs, real)
                                    for name, factory in C.E.REGRESSOR_FACTORIES_SMQ.items()})
        return scaler, cal, iso

    def predict_all(moss, scaler, cal, iso, F, prefix):
        sc_ns, models = cal.pop("__nosyn__")
        out = predict_orig(moss, scaler, cal, iso, F, prefix)
        ns_prefix = prefix.replace("smq", "nosyn", 1)
        Xs = sc_ns.transform(F)
        for name, m in models.items():
            out[f"{ns_prefix}_{name}"] = C.safe_rows(m.predict(Xs))
        return out

    C.fit_calibration_fast = fit_calibration_fast
    C.predict_all = predict_all


def _init_worker():
    global C, _moss
    C = load_complementar()
    C.print = lambda *a, **k: None
    C.E.print = C.print
    _install_nosyn()
    _moss = C.E.load_moss_base_binary()


_DATA = {}


def _load_dataset(ds):
    if ds not in _DATA:
        d = C.E.qp.datasets.fetch_UCIBinaryDataset(ds, verbose=False)
        tr, te = d.train_test
        _DATA.clear()
        _DATA[ds] = (np.vstack([tr.instances, te.instances]),
                     np.concatenate([tr.labels, te.labels]).astype(int))
    return _DATA[ds]


def _cfg_key(r):
    bs = r["bag_size"]
    return (r["grupo"], r["regime"], None if pd.isna(bs) else int(bs))


def run_task(ds, rep, base_cfg, base_cc, smq_ref):
    """smq_ref: {(grupo, regime, bag_size, metodo): ae_mean} desta (ds, rep)."""
    t0 = time.perf_counter()
    X_all, y_all = _load_dataset(ds)
    rows = C.run_rep(ds, X_all, y_all, rep, _moss, base_cfg, base_cc)

    drift = {}
    for r in rows:
        if r["metodo"].startswith("smq"):
            k = _cfg_key(r)
            ref = smq_ref.get(k + (r["metodo"],))
            d = np.inf if ref is None else abs(r["ae_mean"] - ref)
            drift[k] = max(drift.get(k, 0.0), d)

    out, rerun = [], []
    for r in rows:
        r["pareado"] = bool(r["pareado"])
        r["smq_drift"] = drift.get(_cfg_key(r), np.inf)
        (out if r["metodo"].startswith("nosyn") else rerun).append(r)
    return ds, rep, out, rerun, time.perf_counter() - t0


def main():
    global C
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--reps", type=int, default=None, help="só as N primeiras repetições")
    args = ap.parse_args()

    C = load_complementar()
    E = C.E
    os.chdir(HERE)

    base = pd.read_csv(BASE_CSV)
    llp = base[base["grupo"] == "llp"]
    base_cfg = {}
    for (d, rp, rg, r), (nb, sd, pu) in (llp.groupby(["dataset", "repeticao", "regime", "bag_size"])
                                         [["n_bags", "bag_prop_std", "bag_purity_mean"]]
                                         .first().iterrows()):
        base_cfg.setdefault((d, int(rp)), {})[(d, int(rp), rg, int(r))] = (int(nb), float(sd), float(pu))
    cc = base[(base["grupo"] == "oraculo") & (base["metodo"] == "CC")]
    base_cc = {(d, int(rp)): float(a) for d, rp, a in zip(cc["dataset"], cc["repeticao"], cc["ae_mean"])}
    del base, llp, cc

    smq = pd.read_csv(SMQ_CSV)
    columns = list(smq.columns) + ["smq_drift"]
    smq_ref = {}
    for rec in smq[["dataset", "repeticao", "metodo", "ae_mean"] + CFG_KEYS].to_dict("records"):
        smq_ref.setdefault((rec["dataset"], int(rec["repeticao"])), {})[
            _cfg_key(rec) + (rec["metodo"],)] = rec["ae_mean"]
    todo = set(zip(smq["dataset"], smq["repeticao"].astype(int)))
    del smq

    done = set()
    if os.path.exists(OUT_CSV):
        o = pd.read_csv(OUT_CSV, usecols=["dataset", "repeticao"])
        done = set(zip(o["dataset"], o["repeticao"].astype(int)))

    datasets = [d for d in E.get_active_dataset_list() if not args.datasets or d in args.datasets]
    pend = [(d, rp) for d, rp in todo - done
            if d in datasets and (args.reps is None or rp <= args.reps)]

    # baixa/carrega os datasets antes do pool e ordena do maior para o menor
    sizes = {}
    for ds in {d for d, _ in pend}:
        dd = E.qp.datasets.fetch_UCIBinaryDataset(ds, verbose=False)
        tr, te = dd.train_test
        sizes[ds] = (len(tr) + len(te)) * tr.instances.shape[1]
    pend.sort(key=lambda k: (-sizes[k[0]], k[0], k[1]))

    print(f"Pares pendentes: {len(pend)} (já feitos: {len(done)}) | workers: {args.workers}")
    print(f"Saída: {os.path.join(HERE, OUT_CSV)}", flush=True)

    t_start, n_done, n_bad = time.perf_counter(), 0, 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        futs = {pool.submit(run_task, d, rp - 1, base_cfg.get((d, rp), {}),
                            base_cc, smq_ref.get((d, rp), {})): (d, rp)
                for d, rp in pend}
        for fut in as_completed(futs):
            d, rp = futs[fut]
            try:
                ds, rep, rows, rerun, dt = fut.result()
            except Exception as e:
                print(f"❌ {d} rep {rp}: {e!r}", flush=True)
                continue
            pd.DataFrame(rerun).reindex(columns=columns).to_csv(
                RERUN_CSV, mode="a", header=not os.path.exists(RERUN_CSV), index=False)
            df = pd.DataFrame(rows).reindex(columns=columns)
            df.to_csv(OUT_CSV, mode="a", header=not os.path.exists(OUT_CSV), index=False)
            n_done += 1
            pareado = bool(df["pareado"].all())
            n_bad += not pareado
            el = time.perf_counter() - t_start
            print(f"✅ [{n_done}/{len(pend)}] {ds} rep {rep + 1} ({len(rows)} linhas, "
                  f"pareado={pareado}, drift SMQ {df['smq_drift'].max():.1e}, {dt:.0f}s) | decorrido {el / 60:.1f} min, "
                  f"faltam ~{(len(pend) - n_done) * el / n_done / 60:.1f} min", flush=True)

    print(f"\n🏁 Concluído em {(time.perf_counter() - t_start) / 60:.1f} min; "
          f"repetições não pareadas: {n_bad}")


if __name__ == "__main__":
    main()
