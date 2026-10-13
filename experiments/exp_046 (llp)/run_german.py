"""
Roda SÓ o dataset german para completar o experimento LLP vs quantificação.

Por que o german falhou: o QuaPy 0.2.2 lê o arquivo do german com
pd.read_csv(..., delim_whitespace=True), e esse argumento foi REMOVIDO no
pandas 3.0. O TypeError era capturado pelo try/except do run_experiment,
que imprimia "❌ Erro carregando german" e seguia para o próximo dataset.
Os outros datasets não usam esse argumento, por isso só o german caiu.

Correção: durante a execução, delim_whitespace=True é traduzido para
sep=r"\\s+" (equivalente). O resto do pipeline do QuaPy (binarização,
classe positiva = 1, cache em ~/quapy_data) fica intacto, e o protocolo
do experimento é o MESMO do script principal (importado como módulo).

Uso: na pasta do experimento precisam estar o script principal
(llp_vs_quantification.py) e o CSV. Funciona como script
(python run_german.py) ou colado numa célula de notebook; no notebook,
se a pasta atual não for a do experimento, defina PASTA_DO_EXPERIMENTO.
Tem resume: se for interrompido, rode de novo que ele continua.
"""

import gc
import os
import sys
import traceback

import numpy as np
import pandas as pd

MAIN_SCRIPT_FILE = "llp_vs_quantification.py"  # arquivo do script principal
PASTA_DO_EXPERIMENTO = None  # ex.: r"C:\Users\LACA\smq_joylan"; None = pasta atual
DATASET = "german"

# Os outros 29 datasets rodaram SEM o MoSS (77 linhas por repetição). Por
# padrão o german roda igual, para ficar no mesmo regime e no mesmo fluxo de
# números aleatórios (mesmas bags que seriam geradas no run original).
INCLUIR_SMQ = False


def carregar_script_principal(arquivo):
    """Carrega o script principal pelo caminho do arquivo (funciona em .py e
    em notebook, sem depender do sys.path). Não executa o bloco
    if __name__ == "__main__" dele, só as definições."""
    import importlib.util
    caminho = os.path.abspath(arquivo)
    if not os.path.exists(caminho):
        py_aqui = sorted(f for f in os.listdir(".") if f.endswith(".py"))
        raise FileNotFoundError(
            f"Não encontrei o script principal: {caminho}\n"
            f"Pasta atual: {os.getcwd()}\n"
            f"Arquivos .py nesta pasta: {py_aqui}\n"
            "Salve o código do experimento principal com esse nome nesta pasta, "
            "ajuste MAIN_SCRIPT_FILE ou defina PASTA_DO_EXPERIMENTO.")
    nome = os.path.splitext(os.path.basename(caminho))[0]
    spec = importlib.util.spec_from_file_location(nome, caminho)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[nome] = modulo
    spec.loader.exec_module(modulo)
    return modulo


if PASTA_DO_EXPERIMENTO:
    os.chdir(PASTA_DO_EXPERIMENTO)

# ---- compatibilidade com pandas 3: delim_whitespace foi removido ----
# O remendo guarda o read_csv original numa closure (não numa variável global
# do script) e sempre o busca direto em pandas.io.parsers. Assim ele continua
# funcionando quando o notebook roda o arquivo de novo com %run (o %run limpa
# as variáveis da execução anterior) e substitui qualquer remendo antigo.
def _instalar_compat_read_csv():
    from pandas.io.parsers import read_csv as read_csv_original

    def read_csv_compat(*args, **kwargs):
        if kwargs.pop("delim_whitespace", False):
            kwargs["sep"] = r"\s+"
        return read_csv_original(*args, **kwargs)

    pd.read_csv = read_csv_compat


_instalar_compat_read_csv()

E = carregar_script_principal(MAIN_SCRIPT_FILE)

from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def main():
    try:
        dataset = E.qp.datasets.fetch_UCIBinaryDataset(DATASET, verbose=False)
    except Exception:
        print(f"❌ Ainda não foi possível carregar {DATASET}:")
        print(traceback.format_exc())
        sys.exit(1)

    train, test = dataset.train_test
    X_all = np.vstack([train.instances, test.instances])
    y_all = np.concatenate([train.labels, test.labels]).astype(int)
    print(f"📂 {DATASET}: {X_all.shape[0]} instâncias, {X_all.shape[1]} atributos, "
          f"prevalência de positivos = {y_all.mean():.3f}")

    moss = E.load_moss_base_binary() if INCLUIR_SMQ else None
    print(f"SMQ habilitado: {moss is not None}")

    csv = E.OUTPUT_CSV
    print(f"CSV de saída: {os.path.abspath(csv)}")
    done = E._done_keys()
    header = list(pd.read_csv(csv, nrows=0).columns) if os.path.exists(csv) else None

    for rep in range(E.N_REPETITIONS):
        if (DATASET, rep + 1) in done:
            continue
        rep_seed = E.SEED + rep
        Xtr, Xte, ytr, yte = train_test_split(
            X_all, y_all, test_size=0.3, random_state=rep_seed, stratify=y_all)
        scaler = StandardScaler().fit(Xtr)
        Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)

        rows = E.run_one_rep(DATASET, Xtr, ytr, Xte, yte, rep_seed, moss)
        for r in rows:
            r["repeticao"] = rep + 1
        df = pd.DataFrame(rows)

        if header is None:
            df.to_csv(csv, index=False)
            header = list(df.columns)
        else:
            extra = set(df.columns) - set(header)
            if extra:
                print(f"   ⚠️  colunas fora do cabeçalho do CSV serão descartadas: {extra}")
            # mesma ordem de colunas do CSV existente
            df.reindex(columns=header).to_csv(csv, mode="a", header=False, index=False)

        print(f"   ✅ rep {rep + 1}/{E.N_REPETITIONS} salva ({len(rows)} linhas)")
        gc.collect()

    print("🏁 german concluído.")


if __name__ == "__main__":
    main()
