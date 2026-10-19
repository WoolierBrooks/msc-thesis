"""
Diagramas de diferença crítica (CD) do exp_046 com os modelos nosyn, nos
mesmos cenários dos boxplots (gerar_boxplots_nosyn.py): ae_mean médio nas 30
repetições por (bloco, método), rank por bloco (1 = menor AE).

Duas versões de cada diagrama (aeon.visualisation.plot_critical_difference):
  nemenyi   Demšar (2006) clássico: Friedman + pós-teste de Nemenyi; as
            barras ligam métodos cuja diferença de rank médio é < CD.
  wilcoxon  Benavoli et al. (2016): cliques por Wilcoxon pareado com Holm;
            não depende de quais outros métodos estão no pool.

Blocos: oráculo = dataset (30); LLP = (dataset, regime, r), r em BAG_SIZES.
Nos cenários LLP os blocos repetem datasets (não são independentes), então
os diagramas por regime/agregados são indicativos.

Saídas: cd_nosyn/cd_<cenario>_<teste>.png e cd_nosyn/friedman.csv.
"""
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aeon.visualisation import plot_critical_difference
from scipy.stats import friedmanchisquare, studentized_range

# ============================================================
# CONFIG
# ============================================================

CSV_PATHS = ["results_llp_vs_quantification.csv", "results_sota_nn.csv",
             "results_nosyn_smq_rerun.csv", "results_nosyn.csv"]
OS_DIR = "cd_nosyn"
METRIC = "ae_mean"
BAG_SIZES = [5, 10, 30, 100]
REGIMES = ["iid", "hier"]
ALPHA = 0.05
TESTES = ["nemenyi", "wilcoxon"]

shutil.rmtree(OS_DIR, ignore_errors=True)
os.makedirs(OS_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# ============================================================
# LOAD
# ============================================================

df = pd.concat([pd.read_csv(p) for p in CSV_PATHS], ignore_index=True)
df[METRIC] = pd.to_numeric(df[METRIC], errors="coerce")
df = df.dropna(subset=["dataset", "metodo", METRIC])

oraculo = df[df["grupo"] == "oraculo"]
llp = df[df["grupo"].isin(["llp", "smq_llp"]) & df["bag_size"].isin(BAG_SIZES)]

label_map = {
    "smq_base": "SMQ", "smq_iso": "SMQ-ISO", "smq_rf": "SMQ-RF",
    "smq_et": "SMQ-ET", "smq_gbr": "SMQ-GBR",
    "nosyn_rf": "NoSyn-RF", "nosyn_et": "NoSyn-ET", "nosyn_gbr": "NoSyn-GBR",
    "smq_llp_base": "SMQ-LLP", "smq_llp_iso": "SMQ-LLP-ISO", "smq_llp_rf": "SMQ-LLP-RF",
    "smq_llp_et": "SMQ-LLP-ET", "smq_llp_gbr": "SMQ-LLP-GBR",
    "nosyn_llp_rf": "NoSyn-LLP-RF", "nosyn_llp_et": "NoSyn-LLP-ET",
    "nosyn_llp_gbr": "NoSyn-LLP-GBR",
}


def tabela(d, bloco):
    """(bloco x método) com a média nas repetições."""
    return d.groupby(bloco + ["metodo"])[METRIC].mean().unstack().dropna()


def eh_ablacao(t):
    return t[[c for c in t.columns if c.startswith(("smq", "nosyn"))]]


t_or = tabela(oraculo, ["dataset"])
t_llp = tabela(llp, ["dataset", "regime", "bag_size"])
cenarios = [
    ("ablacao_oraculo", "Oracle — SMQ vs NoSyn", eh_ablacao(t_or)),
    ("ablacao_llp", f"LLP, r ∈ {BAG_SIZES} — SMQ vs NoSyn", eh_ablacao(t_llp)),
    ("todos_oraculo", "Oracle — all methods", t_or),
    ("todos_llp", f"LLP, r ∈ {BAG_SIZES} — all methods", t_llp),
]
for reg in REGIMES:
    t_reg = t_llp.xs(reg, level="regime")
    cenarios += [(f"ablacao_llp_{reg}", f"LLP {reg}, r ∈ {BAG_SIZES} — SMQ vs NoSyn",
                  eh_ablacao(t_reg)),
                 (f"todos_llp_{reg}", f"LLP {reg}, r ∈ {BAG_SIZES} — all methods", t_reg)]

# ============================================================
# FRIEDMAN + CD
# ============================================================

resumo = []
for nome, titulo, t in cenarios:
    n, k = t.shape
    p_fried = friedmanchisquare(*[t[c].values for c in t.columns]).pvalue
    cd = studentized_range.ppf(1 - ALPHA, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * n))
    resumo.append(dict(cenario=nome, blocos=n, metodos=k, p_friedman=p_fried, cd_nemenyi=cd))
    print(f"{nome:<22} blocos={n:<4} métodos={k:<3} Friedman p={p_fried:.3g}  CD(Nemenyi)={cd:.3f}")

    labels = [label_map.get(c, c.replace("_", " ")) for c in t.columns]
    for teste in TESTES:
        fig, ax = plot_critical_difference(
            t.values, labels, lower_better=True, test=teste, correction="holm",
            alpha=ALPHA, width=max(6, 0.45 * k), textspace=1.6)
        sub = (f"Friedman + Nemenyi, CD = {cd:.2f}" if teste == "nemenyi"
               else "pairwise Wilcoxon + Holm")
        fig.suptitle(f"{titulo}\n{sub}, α = {ALPHA}, N = {n}", fontsize=10, y=1.02)
        fig.savefig(os.path.join(OS_DIR, f"cd_{nome}_{teste}.png"))
        plt.close(fig)

pd.DataFrame(resumo).to_csv(os.path.join(OS_DIR, "friedman.csv"), index=False)
print(f"\nFiguras em {OS_DIR}/")
