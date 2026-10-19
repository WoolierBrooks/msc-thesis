"""
Teste de Wilcoxon (signed-rank, bicaudal) para os modelos nosyn do exp_046.

Mesma unidade dos boxplots (gerar_boxplots_nosyn.py): ae_mean médio nas 30
repetições por (bloco, método); oráculo: bloco = dataset; LLP: bloco =
(dataset, regime, r), r em BAG_SIZES.

1) Ablação: nosyn_X vs smq_X (X = rf, et, gbr), pareados por bloco.
   Oráculo (30 datasets), LLP por (regime, r) (30 datasets cada), LLP por
   regime e LLP agregado. Holm sobre os 3 regressores em cada cenário.
   Atenção: nos cenários agregados os blocos repetem datasets, então não são
   independentes; o teste por (regime, r) é o mais correto.
2) Melhor nosyn (menor rank médio) vs cada um dos outros métodos do grupo,
   com Holm sobre as comparações.

Colunas: n (blocos), vitorias/empates/derrotas do nosyn (menor AE = vitória),
mediana da diferença (nosyn - outro; negativo = nosyn melhor), p, p_holm e
sig (p_holm < ALPHA).

Saídas: wilcoxon_nosyn_ablacao.csv, wilcoxon_nosyn_vs_todos.csv.
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

CSV_PATHS = ["results_llp_vs_quantification.csv", "results_sota_nn.csv",
             "results_nosyn_smq_rerun.csv", "results_nosyn.csv"]
METRIC = "ae_mean"
BAG_SIZES = [5, 10, 30, 100]
REGIMES = ["iid", "hier"]
REGRESSORES = ["rf", "et", "gbr"]
ALPHA = 0.05

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)

df = pd.concat([pd.read_csv(p) for p in CSV_PATHS], ignore_index=True)
oraculo = df[df["grupo"] == "oraculo"]
llp = df[df["grupo"].isin(["llp", "smq_llp"]) & df["bag_size"].isin(BAG_SIZES)]


def tabela(d, bloco):
    """(bloco x método) com a média nas repetições."""
    return d.groupby(bloco + ["metodo"])[METRIC].mean().unstack().dropna()


def teste(a, b):
    diff = a - b
    p = wilcoxon(a, b, zero_method="wilcox").pvalue if (diff != 0).any() else 1.0
    return dict(n=len(diff), vitorias=int((diff < 0).sum()), empates=int((diff == 0).sum()),
                derrotas=int((diff > 0).sum()), mediana_diff=float(np.median(diff)),
                ae_nosyn=float(a.mean()), ae_outro=float(b.mean()), p=p)


def holm(linhas):
    out = pd.DataFrame(linhas)
    out["p_holm"] = multipletests(out["p"], method="holm")[1]
    out["sig"] = out["p_holm"] < ALPHA
    return out


# ============================================================
# 1) ABLAÇÃO: nosyn_X vs smq_X
# ============================================================

cenarios = [("oraculo", tabela(oraculo, ["dataset"]), "nosyn_{}", "smq_{}")]
t_llp = tabela(llp, ["dataset", "regime", "bag_size"])
cenarios.append(("llp (todos)", t_llp, "nosyn_llp_{}", "smq_llp_{}"))
for reg in REGIMES:
    t_reg = t_llp.xs(reg, level="regime")
    cenarios.append((f"llp {reg}", t_reg, "nosyn_llp_{}", "smq_llp_{}"))
    for r in BAG_SIZES:
        cenarios.append((f"llp {reg} r={r}", t_reg.xs(r, level="bag_size"),
                         "nosyn_llp_{}", "smq_llp_{}"))

ablacao = []
for nome, t, f_ns, f_smq in cenarios:
    linhas = [dict(cenario=nome, nosyn=f_ns.format(x), outro=f_smq.format(x),
                   **teste(t[f_ns.format(x)], t[f_smq.format(x)])) for x in REGRESSORES]
    ablacao.append(holm(linhas))
ablacao = pd.concat(ablacao, ignore_index=True)
ablacao.to_csv("wilcoxon_nosyn_ablacao.csv", index=False)

print("=" * 100)
print("1) ABLAÇÃO — nosyn vs SMQ (mesmo regressor). diff < 0: nosyn melhor")
print("=" * 100)
print(ablacao.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

# ============================================================
# 2) MELHOR NOSYN vs TODOS
# ============================================================

vs_todos = []
for nome, t, prefixo in [("oraculo", tabela(oraculo, ["dataset"]), "nosyn_"),
                         ("llp (todos)", t_llp, "nosyn_llp_")]:
    rank_medio = t.rank(axis=1).mean()
    melhor = rank_medio[[c for c in t.columns if c.startswith(prefixo)]].idxmin()
    linhas = [dict(cenario=nome, nosyn=melhor, outro=m, rank_medio_outro=rank_medio[m],
                   **teste(t[melhor], t[m]))
              for m in rank_medio.sort_values().index if m != melhor]
    vs_todos.append(holm(linhas))
vs_todos = pd.concat(vs_todos, ignore_index=True)
vs_todos.to_csv("wilcoxon_nosyn_vs_todos.csv", index=False)

print()
print("=" * 100)
print("2) MELHOR NOSYN vs CADA MÉTODO (Holm por cenário). diff < 0: nosyn melhor")
print("=" * 100)
print(vs_todos.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
