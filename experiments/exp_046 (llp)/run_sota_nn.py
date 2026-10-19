"""
Script complementar: acrescenta ao experimento LLP vs quantificação os rivais
neurais / estado da arte, PAREADOS com o CSV que já existe (mesmos splits,
mesmas bags, mesmos 1008 batches de teste).

Métodos de CONJUNTO (recebem a bag e devolvem a prevalência)
------------------------------------------------------------
- HistNetQ (Pérez-Mon, Moreo, del Coz & González, "Quantification using
  Permutation-Invariant Networks based on Histograms", Neural Computing and
  Applications, 2025; arXiv:2403.15123): extrator de features por instância
  -> camada de histograma "hard" diferenciável (centros e larguras de bin
  aprendíveis por feature) -> MLP -> softmax com a prevalência da bag.
- DeepSets (Zaheer et al. 2017), exatamente como no baseline do paper do
  HistNetQ: mesma rede, trocando a camada de histograma por um pooling
  (média, mediana ou máximo).

As duas redes recebem um CONJUNTO de instâncias e devolvem direto a
prevalência; o treino só precisa de (conjunto, prevalência). Por isso entram
nos dois grupos do experimento:

  [oraculo] treinadas no cenário "assimétrico" do paper: amostras de 500
            instâncias geradas por APP a partir do split de treino ROTULADO
            (prevalência-alvo uniforme em [0, 1] a cada amostra). Nomes:
            HistNetQ, DeepSets_avg, DeepSets_median, DeepSets_max.

  [llp]     treinadas no cenário "simétrico" do paper, só com as bags do LLP
            e suas proporções (nenhum rótulo individual). Como as bags têm
            r instâncias e os batches de teste têm 500, cada amostra de
            treino é uma UNIÃO de ~500/r bags inteiras (prevalência conhecida
            pela média das proporções), sorteada para cobrir o intervalo de
            prevalências que as bags permitem atingir. É o mesmo "APP sobre
            bags" do SMQ_LLP e generaliza o Bag Mixer do paper (que une
            metades de 2 bags). Nomes: LLP_HistNetQ, LLP_DeepSets_avg, ...

Métodos LLP de INSTÂNCIA com MLP (só grupo [llp])
-------------------------------------------------
Aprendem um classificador por instância a partir das bags + proporções; viram
quantificadores por _CC (contagem com limiar 0,5) e _PCC (média das
probabilidades), igual aos LLP_* lineares do script principal. Todos usam a
mesma MLP (MLP_HIDDEN) e só diferem na perda:
- DLLP (Ardehaly & Culotta 2017): BCE entre a proporção da bag e a média das
  probabilidades previstas na bag. É o baseline de referência do LLP-Bench
  (KDD/CIKM 2024). Nome: LLP_DLLP_MLP_{CC,PCC}.
- EasyLLP (Busa-Fekete et al., NeurIPS 2023): perda de instância com rótulo
  suave ỹ = k(α_b − p) + p, onde k é o tamanho da bag, α_b a proporção da bag
  e p a prevalência global estimada pelas bags; estimador não viesado do
  risco supervisionado. Nome: LLP_EasyLLP_{CC,PCC}.
- GeneralUPM (Applebaum, Dick, Gentile, Kaplan & Koren, "Optimal Learning
  from Label Proportions with General Loss Functions", arXiv:2509.15145,
  2025): escrevendo ℓ(h, y) = f1(h) + y·f2(h), a perda da bag é
      E_x[f1 + p·f2] + (α_b − p)·(Σ_{i∈b} f2(h(x_i)) − k·E_x[f2])
  (versão centrada, de variância que não cresce com k). Com BCE,
  f1 = −log(1 − σ(z)) e f2 = −z. As esperanças E_x são estimadas no minibatch.
  Nome: LLP_GeneralUPM_{CC,PCC}.
EasyLLP e GeneralUPM supõem bags formadas ao acaso (independentes do rótulo):
valem no regime "iid"; no regime "hier" a suposição é violada (o resultado
mostra o quanto isso pesa). Treino: Adam, minibatches de ~INST_PER_STEP
instâncias (bags inteiras), early stopping pelo erro de proporção nas bags de
validação.

TabLLP-BDC NÃO foi incluído: usa um transformer tabular (SAINT) com
pré-treino contrastivo e busca de hiperparâmetros por dataset, foi proposto
para datasets com >= 15 mil instâncias e bags de 64 a 1024, e não tem código
público referenciado no paper — uma reimplementação aqui não seria fiel.

Validação / early stopping (como no paper: para após PATIENCE épocas sem
melhorar): no [llp] separa VAL_FRACTION das BAGS (só proporções, legítimo em
LLP); no [oraculo] separa VAL_FRACTION do split de treino (estratificado).
O conjunto de teste nunca é usado no treino nem na seleção.

Pareamento com o CSV base
-------------------------
Mesmo esquema do run_smq_complementar.py: no run original (sem MoSS) o rng
de cada repetição só foi consumido pelos batches de teste e depois pelas bags
de cada (regime, r), nessa ordem. Este script consome o rng na mesma ordem e
usa rngs SEPARADOS para tudo que é das redes. Por repetição, confere:
  (a) n_bags, bag_prop_std e bag_purity_mean de cada configuração contra o CSV;
  (b) o AE médio do CC do oráculo contra o CSV (confere dados, split e
      batches de teste de ponta a ponta).
Se tudo bater, as linhas saem com pareado=True.

Saída
-----
results_sota_nn.csv, com as mesmas colunas do CSV base + "pareado",
"epocas" e "tempo_treino_s". inst_error/inst_bal_error ficam vazios nos
métodos de conjunto (não classificam instâncias) e preenchidos nos de
instância. Para analisar, junte com os outros:
    pd.concat([pd.read_csv("results_llp_vs_quantification.csv"),
               pd.read_csv("results_smq_complementar.csv"),
               pd.read_csv("results_sota_nn.csv")])
Tem resume: se for interrompido, rode de novo que ele continua.

Requisitos
----------
Na MESMA pasta: llp_vs_quantification.py e results_llp_vs_quantification.csv.
Pacotes: os do experimento + PyTorch (pip install torch).

Uso
---
    python run_sota_nn.py                       # tudo, 1 worker por núcleo
    python run_sota_nn.py --workers 8
    python run_sota_nn.py --datasets iris.1 --reps 1   # teste rápido
    python run_sota_nn.py --device cuda --workers 1     # GPU
"""

import os

# cada worker roda single-thread; o paralelismo é entre repetições
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import copy
import gc
import importlib.util
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ============================================================
# CONFIG
# ============================================================

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_SCRIPT_FILE = "llp_vs_quantification.py"
BASE_CSV = "results_llp_vs_quantification.csv"
OUT_CSV = "results_sota_nn.csv"

RUN_ORACLE = True        # redes de conjunto treinadas com rótulos individuais (APP)
RUN_LLP = True           # métodos treinados só com bags + proporções

# métodos de conjunto: "HistNetQ" e "DeepSets_<avg|median|max>" (os 3 poolings do paper)
METHODS = ["HistNetQ", "DeepSets_avg", "DeepSets_median", "DeepSets_max"]

# métodos LLP de instância (só grupo [llp]): "DLLP_MLP", "EasyLLP", "GeneralUPM"
INSTANCE_METHODS = ["DLLP_MLP", "EasyLLP", "GeneralUPM"]
MLP_HIDDEN = [64, 64]     # MLP compartilhada pelos três
INST_LR = 1e-3
INST_PER_STEP = 512       # ~instâncias por minibatch (bags inteiras)

NN_SEED_OFFSET = 2_000_003   # rngs das redes = f(rep_seed + isto), separados do rng do protocolo

# amostras de treino (tamanho igual ao dos batches de teste, como no paper)
SAMPLE_SIZE = 500
TRAIN_SAMPLES_PER_EPOCH = 100
VAL_SAMPLES = 100
VAL_FRACTION = 0.2

# arquitetura (o paper não tem experimento tabular; tamanhos reduzidos para
# UCI, mantendo a estrutura: extrator -> camada invariante -> MLP -> softmax)
FE_HIDDEN = [64]          # camadas ocultas do extrator por instância
FE_OUT = 32               # nº de features que entram na camada invariante
HIST_BINS = 16            # bins por feature no HistNetQ
HEAD_HIDDEN = [256]       # MLP de quantificação
DROPOUT = 0.2
STE_SHARPNESS = 50.0      # inclinação da sigmoide usada só no gradiente do histograma

# otimização (AdamW + early stopping por paciência, como no paper)
LOSS = "ae"               # "ae" ou "rae"
LR = 3e-4
WEIGHT_DECAY = 1e-5
SETS_PER_STEP = 10        # nº de amostras (conjuntos) por passo do otimizador
MAX_EPOCHS = 300
PATIENCE = 20
PRED_CHUNK = 64           # batches de teste por chamada da rede na avaliação

TOL_BAGS = 1e-9
TOL_CC = 1e-3

# ============================================================
# SCRIPT PRINCIPAL (importado como módulo, sem executar o __main__)
# ============================================================


def _instalar_compat_read_csv():
    """pandas 3 removeu delim_whitespace, que o QuaPy 0.2.2 usa no german."""
    from pandas.io.parsers import read_csv as read_csv_original

    def read_csv_compat(*args, **kwargs):
        if kwargs.pop("delim_whitespace", False):
            kwargs["sep"] = r"\s+"
        return read_csv_original(*args, **kwargs)

    pd.read_csv = read_csv_compat


def carregar_script_principal():
    caminho = os.path.join(HERE, MAIN_SCRIPT_FILE)
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Não encontrei o script principal: {caminho}")
    nome = os.path.splitext(os.path.basename(caminho))[0]
    spec = importlib.util.spec_from_file_location(nome, caminho)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[nome] = modulo
    spec.loader.exec_module(modulo)
    return modulo


_instalar_compat_read_csv()
E = carregar_script_principal()
print = E.print  # print com flush

from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

DEVICE = "cpu"

# ============================================================
# REDES
# ============================================================


def _mlp(sizes, dropout, last_act=True):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_act:
            layers += [nn.LeakyReLU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)


class HardHistogram(nn.Module):
    """Camada de histograma do HistNetQ. Para cada feature k e bin b:
        H_b^k = (1/n) Σ_i 1[ |f_k(x_i) - mu_b^k| < w_b^k ]
    com centros mu e larguras w aprendíveis. A contagem "hard" é usada no
    forward; no backward o gradiente passa por uma sigmoide da mesma margem
    (straight-through), já que o indicador tem derivada zero."""

    def __init__(self, n_features, n_bins):
        super().__init__()
        centers = (torch.arange(n_bins, dtype=torch.float32) + 0.5) / n_bins
        self.mu = nn.Parameter(centers.repeat(n_features, 1))                  # (k, b)
        self.w = nn.Parameter(torch.full((n_features, n_bins), 0.5 / n_bins))  # meia largura

    def forward(self, f):                       # f: (B, n, k), valores em [0, 1]
        margin = self.w.abs() - (f.unsqueeze(-1) - self.mu).abs()   # (B, n, k, b)
        hard = (margin > 0).float()
        soft = torch.sigmoid(STE_SHARPNESS * margin)
        h = (hard + soft - soft.detach()).mean(dim=1)               # (B, k, b)
        return h.flatten(1)


class SetQuantifier(nn.Module):
    """extrator por instância -> camada invariante a permutação -> MLP -> softmax."""

    def __init__(self, d_in, pooling):
        super().__init__()
        self.pooling = pooling
        self.fe = _mlp([d_in] + FE_HIDDEN + [FE_OUT], DROPOUT, last_act=(pooling != "hist"))
        if pooling == "hist":
            self.hist = HardHistogram(FE_OUT, HIST_BINS)
            d_set = FE_OUT * HIST_BINS
        else:
            d_set = FE_OUT
        self.head = nn.Sequential(_mlp([d_set] + HEAD_HIDDEN, DROPOUT),
                                  nn.Linear(HEAD_HIDDEN[-1] if HEAD_HIDDEN else d_set, 2))

    def forward(self, X):                       # X: (B, n, d)
        f = self.fe(X)
        if self.pooling == "hist":
            z = self.hist(torch.sigmoid(f))     # features em [0, 1] para os bins
        elif self.pooling == "avg":
            z = f.mean(dim=1)
        elif self.pooling == "max":
            z = f.max(dim=1).values
        elif self.pooling == "median":
            z = f.median(dim=1).values
        else:
            raise ValueError(self.pooling)
        return torch.softmax(self.head(z), dim=-1)


def make_net(method, d_in):
    pooling = "hist" if method == "HistNetQ" else method.split("_", 1)[1]
    return SetQuantifier(d_in, pooling)


def quant_loss(p_hat, p_true, n):
    if LOSS == "ae":
        return (p_hat - p_true).abs().mean()
    eps = 1.0 / (2 * n)
    ph = (p_hat + eps) / (1 + 2 * eps)
    pt = (p_true + eps) / (1 + 2 * eps)
    return ((ph - pt).abs() / pt).mean()


# ============================================================
# AMOSTRAS DE TREINO
# ============================================================


def app_sampler(X, y, rng):
    """[oraculo] amostras de SAMPLE_SIZE com prevalência-alvo ~ U(0, 1),
    reamostragem com reposição por classe (APP do paper do HistNetQ)."""
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]

    def sample(n_samples):
        idx, props = [], []
        for _ in range(n_samples):
            p1 = rng.uniform()
            n_pos = int(round(p1 * SAMPLE_SIZE))
            if len(pos) == 0:
                n_pos = 0
            if len(neg) == 0:
                n_pos = SAMPLE_SIZE
            i = np.concatenate([rng.choice(pos, n_pos, replace=True) if n_pos else [],
                                rng.choice(neg, SAMPLE_SIZE - n_pos, replace=True)
                                if n_pos < SAMPLE_SIZE else []]).astype(int)
            idx.append(i)
            props.append(n_pos / SAMPLE_SIZE)
        return X[np.stack(idx)], np.array(props)
    return sample


def bag_union_sampler(bags, rng):
    """[llp] cada amostra = união de k ≈ SAMPLE_SIZE/r bags inteiras (com
    reposição entre bags), prevalência = média ponderada das proporções.
    A prevalência-alvo é sorteada em [min, max] das proporções das bags e
    atingida misturando bags acima e abaixo do alvo (mesma ideia do SMQ_LLP).
    Nenhum rótulo individual é usado."""
    r_typ = max(1, int(np.median(bags.sizes)))
    k = max(1, int(round(SAMPLE_SIZE / r_typ)))
    p_lo, p_hi = bags.props.min(), bags.props.max()
    same_size = len(np.unique(bags.sizes)) == 1
    ranges = [np.arange(bags.offsets[b], bags.offsets[b + 1]) for b in range(bags.n_bags)]

    def sample(n_samples):
        idx, props = [], []
        for _ in range(n_samples):
            p_t = rng.uniform(p_lo, p_hi)
            hi = np.where(bags.props >= p_t)[0]
            lo = np.where(bags.props < p_t)[0]
            if len(hi) == 0 or len(lo) == 0:
                sel = rng.choice(bags.n_bags, size=k, replace=True)
            else:
                m_hi, m_lo = bags.props[hi].mean(), bags.props[lo].mean()
                q = np.clip((p_t - m_lo) / max(m_hi - m_lo, 1e-12), 0, 1)
                k_hi = int(rng.binomial(k, q))
                sel = np.concatenate([rng.choice(hi, k_hi, replace=True),
                                      rng.choice(lo, k - k_hi, replace=True)]).astype(int)
            if same_size:   # caso do experimento (todas as bags com r instâncias): vetorizado
                idx.append((bags.offsets[sel][:, None] + np.arange(r_typ)).ravel())
            else:
                idx.append(np.concatenate([ranges[b] for b in sel]))
            props.append(float((bags.props[sel] * bags.sizes[sel]).sum() / bags.sizes[sel].sum()))
        if len({len(i) for i in idx}) == 1:
            return bags.X[np.stack(idx)], np.array(props)
        m = min(len(i) for i in idx)  # bags de tamanhos diferentes: corta no menor
        return bags.X[np.stack([i[:m] for i in idx])], np.array(props)
    return sample


# ============================================================
# TREINO E PREDIÇÃO
# ============================================================


def _t(a):
    return torch.as_tensor(a, dtype=torch.float32, device=DEVICE)


def train_net(method, d_in, train_sampler, val_sampler, seed):
    torch.manual_seed(seed)
    net = make_net(method, d_in).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    Xv, pv = val_sampler(VAL_SAMPLES)
    Xv, Pv = _t(Xv), _t(np.column_stack([1 - pv, pv]))
    n_v = Xv.shape[1]

    best, best_state, wait, epoch = np.inf, copy.deepcopy(net.state_dict()), 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        net.train()
        Xs, ps = train_sampler(TRAIN_SAMPLES_PER_EPOCH)
        Xs, Ps = _t(Xs), _t(np.column_stack([1 - ps, ps]))
        for i in range(0, len(Xs), SETS_PER_STEP):
            opt.zero_grad()
            loss = quant_loss(net(Xs[i:i + SETS_PER_STEP]), Ps[i:i + SETS_PER_STEP], Xs.shape[1])
            loss.backward()
            opt.step()

        net.eval()
        with torch.no_grad():
            v = float(quant_loss(torch.cat([net(Xv[i:i + PRED_CHUNK])
                                            for i in range(0, len(Xv), PRED_CHUNK)]), Pv, n_v))
        if v < best - 1e-7:
            best, best_state, wait = v, copy.deepcopy(net.state_dict()), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, epoch


def predict_test(net, Xte, test_idx):
    out = []
    with torch.no_grad():
        for i in range(0, len(test_idx), PRED_CHUNK):
            out.append(net(_t(Xte[np.stack(test_idx[i:i + PRED_CHUNK])])).cpu().numpy())
    return np.concatenate(out)


def summarize_preds(P_hat, P_real, n_train):
    """AE e RAE por batch, idênticos a E.ae / E.rae, depois resumidos."""
    P_hat = np.clip(np.nan_to_num(P_hat, nan=0.5), 0, 1)
    P_hat = P_hat / P_hat.sum(axis=1, keepdims=True)
    ae = np.abs(P_hat - P_real).mean(axis=1)
    eps = 1.0 / (2 * n_train)
    ph = (P_hat + eps) / (P_hat.sum(axis=1, keepdims=True) + 2 * eps)
    pt = (P_real + eps) / (P_real.sum(axis=1, keepdims=True) + 2 * eps)
    rae = (np.abs(ph - pt) / pt).mean(axis=1)
    return E.summarize_method_results(ae, rae)


def _split_bags(bags, rng):
    n_val = int(round(VAL_FRACTION * bags.n_bags))
    if bags.n_bags < 5 or n_val < 1:
        return bags, bags
    perm = rng.permutation(bags.n_bags)
    return bags.subset(np.sort(perm[n_val:])), bags.subset(np.sort(perm[:n_val]))


def _split_labeled(Xtr, ytr, seed):
    if np.bincount(ytr, minlength=2).min() < 2:
        return Xtr, ytr, Xtr, ytr
    Xa, Xb, ya, yb = train_test_split(Xtr, ytr, test_size=VAL_FRACTION,
                                      random_state=seed, stratify=ytr)
    return Xa, ya, Xb, yb


# ============================================================
# MÉTODOS LLP DE INSTÂNCIA COM MLP (DLLP / EasyLLP / GeneralUPM)
# ============================================================


def make_instance_mlp(d_in):
    return nn.Sequential(_mlp([d_in] + MLP_HIDDEN, DROPOUT),
                         nn.Linear(MLP_HIDDEN[-1] if MLP_HIDDEN else d_in, 1))


def instance_llp_loss(method, z, bid, alpha, k, p):
    """z: logits das instâncias do minibatch (n,); bid: índice LOCAL da bag de
    cada instância (n,); alpha, k: proporção e tamanho de cada bag (n_b,);
    p: prevalência global estimada pelas bags de treino."""
    n_b = len(alpha)
    if method == "DLLP_MLP":
        s = torch.zeros(n_b, device=z.device).index_add_(0, bid, torch.sigmoid(z))
        pbar = (s / k).clamp(1e-6, 1 - 1e-6)
        return -(alpha * torch.log(pbar) + (1 - alpha) * torch.log(1 - pbar)).mean()

    l0 = nn.functional.softplus(z)    # ℓ(h, 0) = −log(1 − σ(z))
    l1 = nn.functional.softplus(-z)   # ℓ(h, 1) = −log σ(z)
    if method == "EasyLLP":
        y_soft = k[bid] * (alpha[bid] - p) + p
        return (y_soft * l1 + (1 - y_soft) * l0).mean()
    if method == "GeneralUPM":
        f1, f2 = l0, l1 - l0              # ℓ(h, y) = f1 + y·f2  (f2 = −z)
        e_f1, e_f2 = f1.mean(), f2.mean()
        s2 = torch.zeros(n_b, device=z.device).index_add_(0, bid, f2)
        return (e_f1 + p * e_f2 + (alpha - p) * (s2 - k * e_f2)).mean()
    raise ValueError(method)


def train_instance_llp(method, d_in, b_tr, b_va, seed):
    torch.manual_seed(seed)
    g = np.random.default_rng(seed)
    net = make_instance_mlp(d_in).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=INST_LR, weight_decay=WEIGHT_DECAY)

    p = float(b_tr.global_prevalence())
    X, alpha_all, k_all = _t(b_tr.X), _t(b_tr.props), _t(b_tr.sizes)
    ranges = [np.arange(b_tr.offsets[b], b_tr.offsets[b + 1]) for b in range(b_tr.n_bags)]
    bags_per_step = max(1, INST_PER_STEP // max(1, int(np.median(b_tr.sizes))))

    Xv, alpha_v = _t(b_va.X), b_va.props
    bid_v = b_va.bag_id

    best, best_state, wait, epoch = np.inf, copy.deepcopy(net.state_dict()), 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        net.train()
        perm = g.permutation(b_tr.n_bags)
        for i in range(0, len(perm), bags_per_step):
            sel = perm[i:i + bags_per_step]
            idx = np.concatenate([ranges[b] for b in sel])
            bid = torch.as_tensor(np.repeat(np.arange(len(sel)), b_tr.sizes[sel]), device=DEVICE)
            opt.zero_grad()
            z = net(X[idx]).squeeze(-1)
            loss = instance_llp_loss(method, z, bid, alpha_all[sel], k_all[sel], p)
            loss.backward()
            opt.step()

        # validação: erro de proporção nas bags de validação (só proporções)
        net.eval()
        with torch.no_grad():
            pr = torch.sigmoid(net(Xv).squeeze(-1)).cpu().numpy()
        pbar = np.bincount(bid_v, weights=pr, minlength=b_va.n_bags) / b_va.sizes
        v = float(np.abs(pbar - alpha_v).mean())
        if v < best - 1e-7:
            best, best_state, wait = v, copy.deepcopy(net.state_dict()), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, epoch


# ============================================================
# UMA REPETIÇÃO (roda dentro de um worker)
# ============================================================

_DATA = {}


def _load_dataset(ds):
    if ds not in _DATA:
        d = E.qp.datasets.fetch_UCIBinaryDataset(ds, verbose=False)
        train, test = d.train_test
        _DATA[ds] = (np.vstack([train.instances, test.instances]),
                     np.concatenate([train.labels, test.labels]).astype(int))
    return _DATA[ds]


def _init_worker(device, gpu_slots=None, n_gpu=0):
    """Com --gpu-workers, os n_gpu primeiros workers usam cuda e o resto, cpu."""
    global DEVICE
    DEVICE = device
    if gpu_slots is not None:
        with gpu_slots.get_lock():
            idx = gpu_slots.value
            gpu_slots.value += 1
        DEVICE = "cuda" if idx < n_gpu else "cpu"
    torch.set_num_threads(1)


def run_rep(ds, rep, base_cfg, cc_ref):
    """base_cfg: {(regime, r): (n_bags, std, purity)} do CSV base;
    cc_ref: AE médio do CC do oráculo no CSV base."""
    t_rep = time.time()
    X_all, y_all = _load_dataset(ds)
    rep_seed = E.SEED + rep
    Xtr, Xte, ytr, yte = train_test_split(
        X_all, y_all, test_size=0.3, random_state=rep_seed, stratify=y_all)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr).astype(np.float32), sc.transform(Xte).astype(np.float32)
    d_in = Xtr.shape[1]

    rng = np.random.default_rng(rep_seed)   # mesmo fluxo do run original
    test = list(E.artificial_prevalence_batches(Xte, yte, 2, E.N_TEST_BATCHES, E.BATCH_SIZE, rng))
    test_idx = [i for i, _ in test]
    P_real = np.array([p for _, p in test])

    # (b) checagem ponta a ponta: CC do oráculo
    cc_pred = E.make_lr().fit(Xtr, ytr).predict(Xte).astype(float)
    p1 = np.array([cc_pred[i].mean() for i in test_idx])
    cc_ae = float(np.abs(np.column_stack([1 - p1, p1]) - P_real).mean(axis=1).mean())
    cc_ok = cc_ref is not None and abs(cc_ae - cc_ref) <= TOL_CC

    # bags na MESMA ordem do run original
    llp_parts, checks = [], []
    for regime in E.BAG_REGIMES:
        for r in E.BAG_SIZES:
            n_bags = max(E.MIN_TRAIN_BAGS, math.ceil(len(ytr) / r))
            bag_idx, props = E.make_llp_bags(ytr, r, n_bags, regime, rng)
            cfg = dict(regime=regime, bag_size=r, n_bags=n_bags,
                       bag_prop_std=float(props.std()),
                       bag_purity_mean=float(np.maximum(props, 1 - props).mean()))
            ref = base_cfg.get((regime, r))
            ok = (ref is not None and ref[0] == n_bags
                  and abs(ref[1] - cfg["bag_prop_std"]) <= TOL_BAGS
                  and abs(ref[2] - cfg["bag_purity_mean"]) <= TOL_BAGS)
            checks.append(ok)
            llp_parts.append((cfg, bag_idx, props, ok))
    rep_ok = cc_ok and all(checks)

    rows = []
    nan_inst = dict(inst_error=np.nan, inst_bal_error=np.nan)

    def emit(name, grupo, cfg, P_hat, n_train, pareado, epocas, tempo, inst=None):
        rows.append(dict(dataset=ds, grupo=grupo, metodo=name, **cfg,
                         **summarize_preds(P_hat, P_real, n_train), **(inst or nan_inst),
                         repeticao=rep + 1, pareado=pareado,
                         epocas=epocas, tempo_treino_s=round(tempo, 2)))

    S_te = np.stack(test_idx)   # (1008, 500)

    def fit_and_eval_instance(method, cfg, bags, cfg_id, pareado):
        seed = rep_seed + NN_SEED_OFFSET + 1000 * cfg_id + 7
        # mesma divisão treino/validação de bags para os três métodos (pareado)
        b_tr, b_va = _split_bags(bags, np.random.default_rng(seed))
        t0 = time.time()
        try:
            net, ep = train_instance_llp(method, d_in, b_tr, b_va, seed)
            with torch.no_grad():
                prob = torch.sigmoid(net(_t(Xte)).squeeze(-1)).cpu().numpy().astype(float)
            hard = (prob > 0.5).astype(float)
            inst = E.instance_errors(yte, hard)
            dt = time.time() - t0
            for suf, v in (("CC", hard), ("PCC", prob)):
                p1 = v[S_te].mean(axis=1)
                emit(f"LLP_{method}_{suf}", "llp", cfg, np.column_stack([1 - p1, p1]),
                     bags.n_instances, pareado, ep, dt, inst)
        except Exception as e:
            print(f"   ⚠️  {ds} rep {rep + 1} {method} ({cfg['regime']}, r={cfg['bag_size']}) falhou: {e!r}")

    def fit_and_eval(method, grupo, name, cfg, make_samplers, cfg_id, n_train, pareado):
        seed = rep_seed + NN_SEED_OFFSET + 1000 * cfg_id
        # mesmo fluxo de amostras para todos os métodos da configuração (pareado)
        tr_s, va_s = make_samplers(np.random.default_rng(seed))
        t0 = time.time()
        try:
            net, ep = train_net(method, d_in, tr_s, va_s, seed)
            emit(name, grupo, cfg, predict_test(net, Xte, test_idx), n_train, pareado,
                 ep, time.time() - t0)
        except Exception as e:
            print(f"   ⚠️  {ds} rep {rep + 1} {name} ({cfg['regime']}, r={cfg['bag_size']}) falhou: {e!r}")

    # --- [oraculo] ---
    if RUN_ORACLE:
        oracle_cfg = dict(regime="oraculo", bag_size=np.nan, n_bags=np.nan,
                          bag_prop_std=np.nan, bag_purity_mean=np.nan)
        Xa, ya, Xb, yb = _split_labeled(Xtr, ytr, rep_seed)

        def samplers_oracle(g):
            return app_sampler(Xa, ya, g), app_sampler(Xb, yb, g)

        for m in METHODS:
            fit_and_eval(m, "oraculo", m, oracle_cfg, samplers_oracle, 0, len(ytr), rep_ok)

    # --- [llp] ---
    if RUN_LLP:
        for cfg_id, (cfg, bag_idx, props, bag_ok) in enumerate(llp_parts, start=1):
            bags = E.LLPBags([Xtr[i] for i in bag_idx], props)

            def samplers_llp(g, bags=bags):
                b_tr, b_va = _split_bags(bags, g)
                return bag_union_sampler(b_tr, g), bag_union_sampler(b_va, g)

            for m in METHODS:
                fit_and_eval(m, "llp", f"LLP_{m}", cfg, samplers_llp, cfg_id,
                             bags.n_instances, cc_ok and bag_ok)

            for m in INSTANCE_METHODS:
                fit_and_eval_instance(m, cfg, bags, cfg_id, cc_ok and bag_ok)

    gc.collect()
    return ds, rep, rows, rep_ok, cc_ae, cc_ref, sum(checks), len(checks), time.time() - t_rep


# ============================================================
# LOOP PRINCIPAL
# ============================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--device", default="cpu", help="cpu ou cuda")
    ap.add_argument("--gpu-workers", type=int, default=0,
                    help="quantos dos --workers rodam em cuda (o resto em cpu); ignora --device")
    ap.add_argument("--datasets", nargs="*", default=None, help="subconjunto de datasets")
    ap.add_argument("--reps", type=int, default=None, help="só as N primeiras repetições")
    args = ap.parse_args()

    os.chdir(HERE)
    if not os.path.exists(BASE_CSV):
        print(f"❌ Não encontrei {BASE_CSV} em {HERE}.")
        sys.exit(1)

    base = pd.read_csv(BASE_CSV)
    llp = base[base["grupo"] == "llp"]
    base_cfg = {}
    for (d, rp, rg, r), (nb, sd, pu) in (llp.groupby(["dataset", "repeticao", "regime", "bag_size"])
                                         [["n_bags", "bag_prop_std", "bag_purity_mean"]]
                                         .first().iterrows()):
        base_cfg.setdefault((d, int(rp)), {})[(rg, int(r))] = (int(nb), float(sd), float(pu))
    cc = base[(base["grupo"] == "oraculo") & (base["metodo"] == "CC")]
    base_cc = {(d, int(rp)): float(a) for d, rp, a in zip(cc["dataset"], cc["repeticao"], cc["ae_mean"])}
    columns = list(base.columns) + ["pareado", "epocas", "tempo_treino_s"]
    todo = set(zip(base["dataset"], base["repeticao"].astype(int)))
    del base, llp, cc

    done = set()
    if os.path.exists(OUT_CSV):
        o = pd.read_csv(OUT_CSV, usecols=["dataset", "repeticao"])
        done = set(zip(o["dataset"], o["repeticao"].astype(int)))

    datasets = [d for d in E.get_active_dataset_list() if not args.datasets or d in args.datasets]
    pend = [(d, rp) for d, rp in todo - done if d in datasets
            and (args.reps is None or rp <= args.reps)]

    # baixa/carrega todos os datasets ANTES do pool (evita downloads concorrentes)
    sizes = {}
    for d in sorted({d for d, _ in pend}):
        X, _ = _load_dataset(d)
        sizes[d] = X.shape[0] * X.shape[1]
    pend.sort(key=lambda t: (-sizes[t[0]], t[0], t[1]))   # maiores primeiro

    n_llp = len(E.BAG_REGIMES) * len(E.BAG_SIZES) if RUN_LLP else 0
    n_treinos = ((1 if RUN_ORACLE else 0) + n_llp) * len(METHODS) + n_llp * len(INSTANCE_METHODS)
    print(f"Conjunto: {METHODS} | Instância: {INSTANCE_METHODS} | "
          f"oráculo={RUN_ORACLE} llp={RUN_LLP} | perda conjunto={LOSS}")
    print(f"Repetições pendentes: {len(pend)} (já feitas: {len(done)}) | "
          f"treinos por repetição: {n_treinos} | workers: {args.workers} | device: {args.device}")
    print(f"Saída: {os.path.abspath(OUT_CSV)}")

    t_start, n_done = time.time(), 0
    if args.gpu_workers:
        import multiprocessing as mp
        initargs = (args.device, mp.Value("i", 0), args.gpu_workers)
        print(f"Workers: {args.gpu_workers} em cuda + {args.workers - args.gpu_workers} em cpu")
    else:
        initargs = (args.device,)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=initargs) as pool:
        futs = {pool.submit(run_rep, d, rp - 1, base_cfg.get((d, rp), {}), base_cc.get((d, rp))): (d, rp)
                for d, rp in pend}
        for fut in as_completed(futs):
            d, rp = futs[fut]
            try:
                ds, rep, rows, ok, cc_ae, cc_ref, n_ok, n_chk, dt = fut.result()
            except Exception as e:
                print(f"❌ {d} rep {rp}: {e!r}")
                continue
            df = pd.DataFrame(rows).reindex(columns=columns)
            df.to_csv(OUT_CSV, mode="a", header=not os.path.exists(OUT_CSV), index=False)
            n_done += 1
            el = time.time() - t_start
            resta = (len(pend) - n_done) * el / n_done / 3600
            msg = "" if ok else (f" ⚠️ NÃO pareado (CC: calculado={cc_ae:.6f}, CSV={cc_ref}; "
                                 f"bags ok: {n_ok}/{n_chk})")
            print(f"✅ [{n_done}/{len(pend)}] {ds} rep {rep + 1} ({len(rows)} linhas, {dt:.0f}s) "
                  f"| decorrido {el / 3600:.2f} h, faltam ~{resta:.1f} h{msg}")

    print("\n🏁 Concluído.")


if __name__ == "__main__":
    main()
