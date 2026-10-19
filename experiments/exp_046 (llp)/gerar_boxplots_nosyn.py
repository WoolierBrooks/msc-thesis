"""
Boxplots de ranking (AE) do exp_046 com os modelos nosyn, no mesmo estilo do
exp_047 (gerar_boxplots.py): média do ae_mean nas repetições por (bloco,
modelo), rank por bloco (menor AE = rank 1) e um boxplot dos ranks por
modelo, ordenado pela mediana.

Blocos: no oráculo, cada dataset; no LLP, cada (dataset, regime, r), com
r em BAG_SIZES (r=1 é o caso supervisionado, fora do rank como no
analise_ranks_llp.ipynb).

O SMQ vem de results_nosyn_smq_rerun.csv (recalculado no mesmo run do nosyn,
pareamento estrito), não de results_smq_complementar.csv. Ver run_nosyn.py.

Figuras (em OS_DIR):
  ablacao_oraculo     smq_* x nosyn_*
  ablacao_llp         smq_llp_* x nosyn_llp_*  (+ _iid, _hier)
  todos_oraculo       todos os métodos do grupo oráculo
  todos_llp           todos os métodos treinados só com proporções (+ _iid, _hier)
"""
import os
import shutil

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns

# ============================================================
# CONFIG
# ============================================================

CSV_PATHS = ["results_llp_vs_quantification.csv", "results_sota_nn.csv",
             "results_nosyn_smq_rerun.csv", "results_nosyn.csv"]
OS_DIR = "boxplots_nosyn"
METRIC = "ae_mean"
BAG_SIZES = [5, 10, 30, 100]
REGIMES = ["iid", "hier"]

shutil.rmtree(OS_DIR, ignore_errors=True)
os.makedirs(OS_DIR, exist_ok=True)

# ============================================================
# PUBLICATION STYLE (igual ao exp_045 / exp_047)
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.4,
    "xtick.major.size": 0.0,
    "ytick.major.size": 0.0,
    "xtick.minor.size": 0.0,
    "ytick.minor.size": 0.0,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.linewidth": 0.4,
    "grid.alpha": 0.5,
    "grid.color": "#aaaaaa",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# ============================================================
# LOAD
# ============================================================

df_master = pd.concat([pd.read_csv(p) for p in CSV_PATHS], ignore_index=True)
df_master[METRIC] = pd.to_numeric(df_master[METRIC], errors="coerce")
df_master = df_master.dropna(subset=["dataset", "metodo", METRIC])
df_master["modelo"] = df_master["metodo"]


def _sub(name, idx):
    return rf"$\mathrm{{{name}}}_{{\mathrm{{{idx}}}}}$"


label_map = {
    "smq_base": r"$\mathrm{SMQ}$",
    "smq_iso": _sub("SMQ", "ISO"), "smq_rf": _sub("SMQ", "RF"),
    "smq_et": _sub("SMQ", "ET"), "smq_gbr": _sub("SMQ", "GBR"),
    "nosyn_rf": _sub("NoSyn", "RF"), "nosyn_et": _sub("NoSyn", "ET"),
    "nosyn_gbr": _sub("NoSyn", "GBR"),
    "smq_llp_base": r"$\mathrm{SMQ\text{-}LLP}$",
    "smq_llp_iso": _sub("SMQ\\text{-}LLP", "ISO"), "smq_llp_rf": _sub("SMQ\\text{-}LLP", "RF"),
    "smq_llp_et": _sub("SMQ\\text{-}LLP", "ET"), "smq_llp_gbr": _sub("SMQ\\text{-}LLP", "GBR"),
    "nosyn_llp_rf": _sub("NoSyn\\text{-}LLP", "RF"), "nosyn_llp_et": _sub("NoSyn\\text{-}LLP", "ET"),
    "nosyn_llp_gbr": _sub("NoSyn\\text{-}LLP", "GBR"),
}

oraculo = df_master[df_master["grupo"] == "oraculo"].copy()
oraculo["bloco"] = oraculo["dataset"]
llp = df_master[df_master["grupo"].isin(["llp", "smq_llp"])
                & df_master["bag_size"].isin(BAG_SIZES)].copy()
llp["bloco"] = (llp["dataset"] + "|" + llp["regime"] + "|"
                + llp["bag_size"].astype(int).astype(str))

print(f"Métrica: {METRIC.upper()} | datasets: {df_master['dataset'].nunique()} | "
      f"repetições: {df_master['repeticao'].nunique()}\n")


# ============================================================
# RANK + BOXPLOT
# ============================================================

def ranks_por_bloco(df):
    """Média nas repetições por (bloco, modelo) e rank por bloco. Descarta
    blocos onde falta algum modelo."""
    media = df.groupby(["bloco", "modelo"])[METRIC].mean().unstack()
    faltando = int(media.isna().any(axis=1).sum())
    if faltando:
        print(f"⚠️  {faltando} blocos com modelo faltando descartados")
    media = media.dropna()
    rank = media.rank(axis=1, method="average", ascending=True)
    out = rank.stack().rename("rank_dataset").reset_index()
    out[METRIC] = media.stack().values
    return out


def boxplot(df, nome, titulo):
    dataset_mean = ranks_por_bloco(df)
    ordem_rank = (dataset_mean.groupby("modelo")["rank_dataset"]
                  .median().sort_values().index.tolist())
    n_models = len(ordem_rank)
    display_labels = [label_map.get(m, m.replace("_", " ")) for m in ordem_rank]
    palette = sns.color_palette("Spectral", n_colors=n_models)

    fig, ax = plt.subplots(figsize=(max(7.2, n_models * 0.55), 3.5))
    sns.boxplot(
        data=dataset_mean,
        x="modelo", y="rank_dataset",
        hue="modelo", order=ordem_rank, hue_order=ordem_rank,
        palette=palette, dodge=False, legend=False,
        width=0.55, linewidth=0.8,
        flierprops=dict(marker="o", markerfacecolor="none",
                        markeredgecolor="#555555", markeredgewidth=0.6,
                        markersize=3.5),
        medianprops=dict(color="black", linewidth=1.4),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
        boxprops=dict(linewidth=0.8),
        ax=ax,
    )
    ax.set_title(titulo, fontsize=11, pad=10)
    ax.set_xlabel("Methods", labelpad=6)
    ax.set_ylabel("Rank (AE)", labelpad=6)
    ax.set_ylim(0, n_models + 1)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1 if n_models <= 16 else 2))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
        spine.set_color("#aaaaaa")
    ax.tick_params(which="both", length=0)
    ax.set_xticks(range(n_models))
    ax.set_xticklabels(display_labels, rotation=40, ha="right", rotation_mode="anchor")

    plt.tight_layout()
    output_path = os.path.join(OS_DIR, f"rank_boxplot_{nome}.png")
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Salvo: {output_path} ({dataset_mean['bloco'].nunique()} blocos)")

    print(f"Ranking — {titulo}")
    for i, modelo in enumerate(ordem_rank, start=1):
        sel = dataset_mean[dataset_mean["modelo"] == modelo]
        print(f"{i:2d}. {modelo:<22} "
              f"| Median Rank = {sel['rank_dataset'].median():.2f} "
              f"| Mean Rank = {sel['rank_dataset'].mean():.2f} "
              f"| Mean AE = {sel[METRIC].mean():.6f}")
    print()


is_ablacao = lambda m: m.str.startswith(("smq", "nosyn"))

boxplot(oraculo[is_ablacao(oraculo["modelo"])], "ablacao_oraculo",
        "Oracle (individual labels) — SMQ vs NoSyn")
boxplot(llp[is_ablacao(llp["modelo"])], "ablacao_llp",
        f"LLP (bag proportions only), r ∈ {BAG_SIZES} — SMQ vs NoSyn")
boxplot(oraculo, "todos_oraculo", "Oracle (individual labels) — all methods")
boxplot(llp, "todos_llp", f"LLP (bag proportions only), r ∈ {BAG_SIZES} — all methods")
for reg in REGIMES:
    sel = llp[llp["regime"] == reg]
    boxplot(sel[is_ablacao(sel["modelo"])], f"ablacao_llp_{reg}",
            f"LLP, regime {reg}, r ∈ {BAG_SIZES} — SMQ vs NoSyn")
    boxplot(sel, f"todos_llp_{reg}", f"LLP, regime {reg}, r ∈ {BAG_SIZES} — all methods")
