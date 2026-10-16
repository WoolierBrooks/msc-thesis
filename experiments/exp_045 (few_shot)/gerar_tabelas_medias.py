"""
Gera as tabelas por dataset usadas nos boxplots de ranking do note.ipynb.

Para cada n-shot, um CSV com 30 linhas (uma por dataset) e, para cada método:
  <metodo>_media   média do ae_mean nas 10 repetições  (valor usado no ranking)
  <metodo>_mediana mediana do ae_mean nas 10 repetições
  <metodo>_rank    posição do método naquele dataset (1 = menor AE, empates
                   recebem a média das posições, como no notebook)
Também gera um arquivo único com todas as linhas e a coluna "n".

Com --metodos, só os métodos listados entram (e o rank é recalculado entre
eles); --colunas escolhe quais estatísticas sair (media, mediana, rank);
--saida muda a pasta de saída.

Uso:
    python gerar_tabelas_medias.py
    python gerar_tabelas_medias.py --metodos CC PCC ACC PACC EMQ DyS KDEyML --saida tabelas_medias_baselines
"""

import argparse
import os

import pandas as pd

CSV_PATHS = [
    "results_few_shot_quantification_10reps.csv",
    "results_smq_10reps.csv",
]
N_ORDER = ["1", "2", "3", "5", "10", "100", "full"]
METRIC = "ae_mean"

ap = argparse.ArgumentParser()
ap.add_argument("--metodos", nargs="*", default=None, help="métodos a manter (padrão: todos)")
ap.add_argument("--colunas", nargs="*", default=["media", "mediana", "rank"],
                choices=["media", "mediana", "rank"], help="estatísticas a incluir")
ap.add_argument("--saida", default="tabelas_medias", help="pasta de saída")
args = ap.parse_args()
OUT_DIR = args.saida

df = pd.concat([pd.read_csv(p, dtype={"n": str}) for p in CSV_PATHS], ignore_index=True)
if args.metodos:
    faltando = set(args.metodos) - set(df["metodo"])
    if faltando:
        raise SystemExit(f"Métodos não encontrados nos CSVs: {sorted(faltando)}")
    df = df[df["metodo"].isin(args.metodos)]
os.makedirs(OUT_DIR, exist_ok=True)

todas = []
for n in N_ORDER:
    g = df[df["n"] == n].groupby(["dataset", "metodo"])[METRIC]
    media = g.mean().unstack()
    mediana = g.median().unstack()
    rank = media.rank(axis=1, method="average", ascending=True)

    # ordem das colunas: métodos do melhor para o pior pela mediana do rank (como no gráfico)
    metodos = rank.median().sort_values().index.tolist()
    cols = {}
    stats = {"media": media, "mediana": mediana, "rank": rank}
    for m in metodos:
        for s in args.colunas:
            cols[f"{m}_{s}"] = stats[s][m]
    tab = pd.DataFrame(cols).sort_index()
    tab.index.name = "dataset"

    tab.to_csv(os.path.join(OUT_DIR, f"medias_por_dataset_n_{n}.csv"), float_format="%.6f")
    todas.append(tab.reset_index().assign(n=n))
    print(f"n = {n}: {tab.shape[0]} datasets x {len(metodos)} métodos")

pd.concat(todas, ignore_index=True).set_index(["n", "dataset"]).to_csv(
    os.path.join(OUT_DIR, "medias_por_dataset_todos_n.csv"), float_format="%.6f")
print(f"Salvo em {OUT_DIR}/")
