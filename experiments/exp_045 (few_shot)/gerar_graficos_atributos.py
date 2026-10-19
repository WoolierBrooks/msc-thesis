"""
SMQ x número de atributos: o SMQ ganha dos rivais em datasets com mais
atributos e perde nos de poucos atributos?

Gera em graficos_atributos/:
  delta_ae_vs_atributos.png  log2(AE_SMQ / AE_rival) por dataset vs nº de
                             atributos (abaixo de 0 = SMQ melhor), contra o
                             melhor rival e contra o KDEy-ML
  rank_vs_atributos.png      rank médio de cada método por faixa de atributos
O rank é calculado entre o SMQ escolhido e os 12 métodos do
results_few_shot_quantification_10reps.csv (as outras variantes de SMQ ficam
de fora para não dividirem o ranking).
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quapy as qp

CSV_BASE = "results_few_shot_quantification_10reps.csv"
CSV_SMQ = "results_smq_10reps.csv"
FEATURES_CSV = "atributos_datasets.csv"
OUT_DIR = "graficos_atributos"
SMQ = "smq_gbr"
N_VALUES = ["100", "full"]
LIMIAR = 10   # "mais de 10 atributos"
YCLIP = 2.5   # pontos além de ±YCLIP aparecem na borda como triângulo
# faixas com ~7-8 datasets cada
FAIXAS = [(3, 5), (8, 10), (11, 21), (24, 256)]

BLUE, RED, ORANGE, GRAY, INK = "#2a78d6", "#d6402a", "#eb6834", "#b5b4ad", "#3d3d3a"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.linewidth": 0.4, "axes.edgecolor": "#aaaaaa",
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.minor.size": 0, "ytick.minor.size": 0,
    "axes.grid": True, "grid.linewidth": 0.4, "grid.alpha": 0.5, "grid.color": "#aaaaaa",
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})
os.makedirs(OUT_DIR, exist_ok=True)

LABEL = {
    "smq_gbr": r"$\mathrm{SMQ}_{\mathrm{GBR}}$", "smq_et": r"$\mathrm{SMQ}_{\mathrm{ET}}$",
    "smq_rf": r"$\mathrm{SMQ}_{\mathrm{RF}}$", "KDEyML": r"$\mathrm{KDEy}_{\mathrm{ML}}$",
}
TITLE_N = {"full": "Full Training Set"}

# ============================================================
# DADOS
# ============================================================

df = pd.concat([pd.read_csv(CSV_BASE),
                pd.read_csv(CSV_SMQ).query("metodo == @SMQ")], ignore_index=True)
df["n"] = df["n"].astype(str)

if not os.path.exists(FEATURES_CSV):
    rows = []
    for d in sorted(df["dataset"].unique()):
        X = qp.datasets.fetch_UCIBinaryDataset(d, verbose=False).training.X
        rows.append((d, X.shape[1]))
    pd.DataFrame(rows, columns=["dataset", "n_atributos"]).to_csv(FEATURES_CSV, index=False)
feats = pd.read_csv(FEATURES_CSV)

m = df.groupby(["n", "dataset", "metodo"])["ae_mean"].mean().reset_index()
m["rank"] = m.groupby(["n", "dataset"])["ae_mean"].rank(method="average")
m = m.merge(feats, on="dataset")

# ============================================================
# FIGURA 1: log2(AE_SMQ / AE_rival) vs nº de atributos
# ============================================================

rng = np.random.default_rng(0)
jitter = {d: 2 ** rng.uniform(-0.12, 0.12) for d in feats["dataset"]}

fig, axes = plt.subplots(2, len(N_VALUES), figsize=(7.2, 5.4), sharex=True, sharey=True)
for j, n in enumerate(N_VALUES):
    x = m[m["n"] == n]
    smq = x[x["metodo"] == SMQ].set_index("dataset")
    rivais = {
        "best rival": x[x["metodo"] != SMQ].groupby("dataset")["ae_mean"].min(),
        LABEL["KDEyML"]: x[x["metodo"] == "KDEyML"].set_index("dataset")["ae_mean"],
    }
    for i, (nome, ae_riv) in enumerate(rivais.items()):
        ax = axes[i, j]
        r = smq.join(ae_riv.rename("riv"), how="inner")
        r["delta"] = np.log2(r["ae_mean"] / r["riv"])
        r["xj"] = r["n_atributos"] * r.index.map(jitter)
        ganha = r["delta"] < 0
        r["yp"] = r["delta"].clip(-YCLIP * 0.96, YCLIP * 0.96)
        fora = r["delta"].abs() > YCLIP

        ax.axhline(0, color=INK, lw=0.8)
        ax.axvline(LIMIAR + 0.5, color=INK, lw=0.8, ls=(0, (4, 3)))
        for sel, cor, rot in ((ganha, BLUE, "better"), (~ganha, RED, "worse")):
            ax.scatter(r.loc[sel & ~fora, "xj"], r.loc[sel & ~fora, "yp"], s=24, color=cor,
                       edgecolor="white", lw=0.6, zorder=3, label=f"{LABEL[SMQ]} {rot}")
            ax.scatter(r.loc[sel & fora, "xj"], r.loc[sel & fora, "yp"], s=34, color=cor,
                       marker="^" if rot == "worse" else "v", edgecolor="white", lw=0.6, zorder=3)
        if fora.any():
            txt = ", ".join(f"{d} ({v:+.1f})" if np.isfinite(v) else f"{d} (rival AE = 0)"
                            for d, v in r.loc[fora, "delta"].items())
            ax.text(0.03, 0.97, f"off scale: {txt}", transform=ax.transAxes, ha="left",
                    va="top", fontsize=7.5, color=INK)

        # tendência: mediana móvel em log(atributos)
        r = r.sort_values("n_atributos")
        lx = np.log(r["n_atributos"].to_numpy())
        grid = np.linspace(lx.min(), np.log(60), 80)   # acima de 60 só há o semeion
        med = [np.median(r["delta"].to_numpy()[np.abs(lx - g) <= 0.7]) for g in grid]
        ax.plot(np.exp(grid), med, color=INK, lw=1.6, zorder=2, label="rolling median")

        for lado, sel, xpos, ha in (("≤", r["n_atributos"] <= LIMIAR, 0.03, "left"),
                                    (">", r["n_atributos"] > LIMIAR, 0.97, "right")):
            w, t = int((r.loc[sel, "delta"] < 0).sum()), int(sel.sum())
            ax.text(xpos, 0.04, f"{lado}{LIMIAR} attrs: wins {w}/{t}", transform=ax.transAxes,
                    ha=ha, va="bottom", fontsize=8.5, color=INK,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))

        ax.set_xscale("log")
        ax.set_xticks([3, 5, 10, 20, 50, 100, 256])
        ax.set_xticklabels(["3", "5", "10", "20", "50", "100", "256"])
        ax.grid(axis="x", visible=False)
        if i == 0:
            ax.set_title(TITLE_N.get(n, f"n = {n} per class"), fontsize=11, pad=8)
        if j == 0:
            ax.set_ylabel(f"log$_2$(AE {LABEL[SMQ]} / AE {nome})")
        if i == 1:
            ax.set_xlabel("Number of attributes (log scale)")

for ax in axes.flat:
    ax.set_ylim(-YCLIP - 0.6, YCLIP)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(os.path.join(OUT_DIR, "delta_ae_vs_atributos.png"))
plt.close(fig)

# ============================================================
# FIGURA 2: rank médio por faixa de atributos
# ============================================================

def faixa(k):
    for a, b in FAIXAS:
        if a <= k <= b:
            return f"{a}–{b}"
    raise ValueError(k)

m["faixa"] = m["n_atributos"].map(faixa)
ordem_faixas = [f"{a}–{b}" for a, b in FAIXAS]
n_ds = m.drop_duplicates("dataset").groupby("faixa").size()

fig, axes = plt.subplots(1, len(N_VALUES), figsize=(7.2, 3.6), sharey=True)
for ax, n in zip(axes, N_VALUES):
    x = m[m["n"] == n]
    tab = x.groupby(["metodo", "faixa"])["rank"].mean().unstack()[ordem_faixas]
    xs = np.arange(len(ordem_faixas))
    for metodo, row in tab.iterrows():
        if metodo in (SMQ, "KDEyML"):
            continue
        ax.plot(xs, row.values, color=GRAY, lw=0.9, zorder=1,
                label=f"other {tab.shape[0] - 2} methods")
    for metodo, cor in ((("KDEyML", ORANGE)), (SMQ, BLUE)):
        row = tab.loc[metodo]
        ax.plot(xs, row.values, color=cor, lw=2.2, marker="o", ms=5,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3, label=LABEL[metodo])
        ax.text(xs[-1] + 0.08, row.values[-1], LABEL[metodo], fontsize=8.5,
                color=INK, va="center", fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{f}\n({n_ds[f]} datasets)" for f in ordem_faixas], fontsize=8)
    ax.set_xlim(-0.3, len(xs) - 1 + 0.9)
    ax.set_title(TITLE_N.get(n, f"n = {n} per class"), fontsize=11, pad=8)
    ax.set_xlabel("Number of attributes")
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel(f"Mean rank (AE) among {tab.shape[0]} methods")
h, l = axes[0].get_legend_handles_labels()
uniq = dict(zip(l, h))
fig.legend([uniq[k] for k in (LABEL[SMQ], LABEL["KDEyML"], f"other {tab.shape[0] - 2} methods")],
           [LABEL[SMQ], LABEL["KDEyML"], f"other {tab.shape[0] - 2} methods"],
           loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
axes[0].invert_yaxis()
axes[0].set_ylim(tab.shape[0] + 0.5, 0.5)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(os.path.join(OUT_DIR, "rank_vs_atributos.png"))
plt.close(fig)

# ============================================================
# RESUMO NO TERMINAL
# ============================================================

for n in m["n"].unique():
    x = m[m["n"] == n]
    s = x[x["metodo"] == SMQ].set_index("dataset")
    best = x[x["metodo"] != SMQ].groupby("dataset")["ae_mean"].min()
    s["vence"] = s["ae_mean"] < best
    hi = s["n_atributos"] > LIMIAR
    print(f"n={n:>4}: {SMQ} vence todos os rivais em {s.loc[hi, 'vence'].sum()}/{hi.sum()} "
          f"(>{LIMIAR} atrib.) e {s.loc[~hi, 'vence'].sum()}/{(~hi).sum()} (≤{LIMIAR}); "
          f"rank médio {s.loc[hi, 'rank'].mean():.1f} vs {s.loc[~hi, 'rank'].mean():.1f}")
