"""
Boxplots de ranking (AE) dos modelos de SMQ do exp_047, no mesmo estilo dos
boxplots do exp_045 (note.ipynb): por n-shot, média do ae_mean nas repetições
por (dataset, modelo), rank por dataset (menor AE = rank 1) e um boxplot dos
ranks por modelo, ordenado pela mediana.
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

CSV_PATHS = ["results_sem_sintetico_10reps_nfull.csv"]
OS_DIR = "boxplots_binario"
METRIC = "ae_mean"

shutil.rmtree(OS_DIR, ignore_errors=True)
os.makedirs(OS_DIR, exist_ok=True)

# ============================================================
# PUBLICATION STYLE (igual ao exp_045)
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
df_master = df_master.dropna(subset=["dataset", "metodo", "n", METRIC])

# nosyn_*: regressor só com as 8 metafeatures (sem a prevalência do MoSS)
label_map = {
    "smq_base": r"$\mathrm{SMQ}$",
    "smq_iso": r"$\mathrm{SMQ}_{\mathrm{ISO}}$",
    "smq_rf": r"$\mathrm{SMQ}_{\mathrm{RF}}$",
    "smq_et": r"$\mathrm{SMQ}_{\mathrm{ET}}$",
    "smq_gbr": r"$\mathrm{SMQ}_{\mathrm{GBR}}$",
    "nosyn_rf": r"$\mathrm{NoSyn}_{\mathrm{RF}}$",
    "nosyn_et": r"$\mathrm{NoSyn}_{\mathrm{ET}}$",
    "nosyn_gbr": r"$\mathrm{NoSyn}_{\mathrm{GBR}}$",
}
df_master["modelo"] = df_master["metodo"]

df_master["n_str"] = df_master["n"].astype(str)
n_order_preferred = ["1", "2", "3", "5", "10", "100", "full"]
n_values = [n for n in n_order_preferred if n in set(df_master["n_str"])]

print(f"Métrica: {METRIC.upper()} | n: {n_values} | "
      f"datasets: {df_master['dataset'].nunique()} | "
      f"modelos: {df_master['modelo'].nunique()}\n")

# ============================================================
# LOOP: UM GRÁFICO PARA CADA N
# ============================================================

for n_value in n_values:
    df_n = df_master[df_master["n_str"] == n_value]

    dataset_mean = (
        df_n.groupby(["dataset", "modelo"])[METRIC].mean().reset_index()
    )
    dataset_mean["rank_dataset"] = (
        dataset_mean.groupby("dataset")[METRIC]
        .rank(method="average", ascending=True)
    )

    ordem_rank = (
        dataset_mean.groupby("modelo")["rank_dataset"]
        .median().sort_values().index.tolist()
    )
    n_models = len(ordem_rank)
    display_labels = [label_map.get(m, m) for m in ordem_rank]
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

    title_n = ("Full Training Set" if n_value == "full"
               else f"n = {n_value} per class")
    ax.set_title(title_n, fontsize=11, pad=10)
    ax.set_xlabel("Methods", labelpad=6)
    ax.set_ylabel("Rank (AE)", labelpad=6)
    ax.set_ylim(0, n_models + 1)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_minor_locator(ticker.NullLocator())

    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
        spine.set_color("#aaaaaa")

    ax.tick_params(which="both", length=0)
    ax.set_xticks(range(n_models))
    ax.set_xticklabels(display_labels, rotation=40, ha="right",
                       rotation_mode="anchor")

    plt.tight_layout()
    output_path = os.path.join(OS_DIR, f"rank_boxplot_smq_n_{n_value}.png")
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Salvo: {output_path}")

    print(f"Ranking para n = {n_value}")
    for i, modelo in enumerate(ordem_rank, start=1):
        sel = dataset_mean[dataset_mean["modelo"] == modelo]
        print(f"{i:2d}. {modelo:<10} "
              f"| Median Rank = {sel['rank_dataset'].median():.2f} "
              f"| Mean Rank = {sel['rank_dataset'].mean():.2f} "
              f"| Mean AE = {sel[METRIC].mean():.6f}")
    print()
