"""
LLP vs Quantificação — seus quantificadores (CC, PCC, ACC, PACC, EMQ, DyS,
KDEyML, SMQ_*) contra métodos de Learning from Label Proportions, seguindo
Yu, Choromanski, Kumar, Jebara & Chang, "On Learning from Label
Proportions" (arXiv:1402.5902v2).

Removidos em relação ao script few-shot: FSNN, FSMM, FSGMM, FSGPEM, FSPL
(métodos do Esuli) e toda a amostragem n-shot.

------------------------------------------------------------------------
POR QUE ESSA COMPARAÇÃO FAZ SENTIDO
------------------------------------------------------------------------
O Teorema 1 do paper é um resultado sobre ERRO DE PROPORÇÃO EM BAGS NÃO
VISTAS — ou seja, é literalmente um problema de quantificação. O operador
phi_r^f(h) do paper (Def. 1) é exatamente o Classify & Count aplicado ao
classificador h aprendido via LLP. Então dá pra colocar tudo na mesma régua:
todos os métodos estimam a prevalência dos MESMOS batches de teste (APP,
grade de 21 pontos, 500 instâncias, idêntico ao seu script anterior) e são
medidos por AE/RAE.

A diferença está no que cada grupo vê no treino:

  [oraculo]  Seus quantificadores supervisionados (CC, PCC, ACC, PACC, EMQ,
             DyS, KDEyML, smq_base/rf/et/gbr/iso). Veem o rótulo de CADA
             instância do split de treino. É o limite superior de informação.

  [llp]      Métodos LLP citados/usados no paper. Veem as MESMAS instâncias,
             mas agrupadas em bags e só com a proporção de positivos de cada
             bag (os rótulos individuais nunca chegam no fit):
               - LLP_MeanMap   (Quadrianto et al. 2009, ref. [18])
               - LLP_InvCal    (Rüeping 2010, ref. [19])
               - LLP_AlterSVM  (alter-∝SVM, Yu et al. 2013, ref. [25] — o
                                método usado no estudo de caso da Seção 9,
                                inicializado por [18] e [19] como no paper)
               - LLP_EPRM      (minimização direta da Eq. 2 do paper com um
                                classificador logístico; perda BCE na
                                proporção da bag, que é a versão suave da
                                perda 1-Lipschitz do paper — mesma ideia do
                                DLLP)
             Cada classificador LLP vira quantificador por _CC (o phi do
             paper) e, quando é probabilístico, também por _PCC.

  [smq_llp]  (extra, opcional via RUN_SMQ_LLP) SEU SMQ treinado em regime
             LLP puro. Isso é natural porque a tabela de calibração do SMQ
             é exatamente (features dos scores de um batch, prevalência do
             batch) — e em LLP as bags de treino SÃO batches com prevalência
             conhecida. Scorer = LLP_EPRM com scores out-of-fold por bag;
             batches de calibração = uniões aleatórias de bags inteiras
             (a prevalência de uma união de bags é conhecida sem precisar de
             nenhum rótulo individual). Não usa rótulos individuais.

Além de AE/RAE, também registramos o ERRO DE INSTÂNCIA no split de teste
(inst_error, inst_bal_error), que é a métrica principal do paper (Seção 6
e Fig. 2), para os classificadores LLP e para os classificadores
supervisionados de referência.

------------------------------------------------------------------------
PROTOCOLO
------------------------------------------------------------------------
  for dataset in 30 datasets binários UCI (QuaPy):
    for repetição in 1..30:
      split treino/teste 70/30 estratificado + StandardScaler
      gera 1008 batches de teste APP UMA VEZ (pareados entre métodos)
      [oraculo] treina 1x no split de treino rotulado, avalia
      for regime in {"iid", "hier"}:
        for r (tamanho da bag) in {1, 5, 10, 30, 100}:
          monta m = max(MIN_TRAIN_BAGS, ceil(n_treino / r)) bags de treino
          [llp] + [smq_llp] treinam só com (bags, proporções), avaliam

Regimes de geração de bags (Seção 9 do paper):
  "iid"  -> cada bag sorteia r instâncias com reposição do treino inteiro
            (Fig. 2a). Proporções concentram no prior global: bags pouco
            "puras", que é o caso difícil previsto pelo Lema 2.
  "hier" -> geração hierárquica, instâncias condicionalmente independentes
            dada a bag (Seção 6.2 / Fig. 2b-d): cada bag tem seu próprio
            prior pi_b ~ U(0,1) e r instâncias sorteadas com reposição.
            Como os UCI não têm atributo de agrupamento, o "grupo" é
            simulado pelo prior da bag.
r = 1 é o caso supervisionado (Corolário 1) e serve de sanity check: os
métodos LLP devem ficar próximos do oráculo.

Saída: uma linha por (dataset, repetição, regime, r, método) com
média/desvio/mediana de AE e RAE sobre os batches de teste (mesmo formato
resumido do seu último script) + erro de instância. Resume automático:
(dataset, repetição) já presentes no CSV são pulados.
"""

import os
import sys
import math
import functools
import traceback
import pickle
import gc
import warnings

import numpy as np
import pandas as pd

from scipy.stats import skew, kurtosis
from scipy.optimize import minimize
from scipy.special import expit, logit

import quapy as qp
from quapy.method.aggregative import CC, PCC, ACC, PACC, EMQ, DyS, KDEyML

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_predict

print = functools.partial(print, flush=True)
warnings.filterwarnings("ignore")

# ============================================================
# JOBLIB TMPDIR (mesmo padrão dos scripts anteriores)
# ============================================================

_home_dir = os.path.expanduser("~")
_joblib_tmp = os.path.join(_home_dir, "tmp_joblib")
try:
    os.makedirs(_joblib_tmp, exist_ok=True)
    _test_path = os.path.join(_joblib_tmp, ".write_test")
    with open(_test_path, "w") as _f:
        _f.write("ok")
    os.remove(_test_path)
except (PermissionError, OSError) as _e:
    import tempfile
    _joblib_tmp = os.path.join(tempfile.gettempdir(), "julio_joblib_tmp")
    os.makedirs(_joblib_tmp, exist_ok=True)
    print(f"⚠️  Fallback joblib tmp: {_joblib_tmp} ({_e})")
os.environ["JOBLIB_TEMP_FOLDER"] = _joblib_tmp
print(f"📁 JOBLIB_TEMP_FOLDER = {_joblib_tmp}")

# ============================================================
# CONFIG
# ============================================================

SEED = 42
N_REPETITIONS = 30

BAG_SIZES = [1, 5, 10, 30, 100]      # r do paper (Fig. 2 usa 1, 5, 30, 100)
BAG_REGIMES = ["iid", "hier"]
MIN_TRAIN_BAGS = 20                  # piso de bags p/ datasets pequenos com r grande

N_CALIB_BATCHES = 1000               # calibração do SMQ (oráculo e LLP)
N_TEST_BATCHES = 1000                # -> 1008 com a grade de 21 pontos
BATCH_SIZE = 500

N_JOBS = 4

MOSS_BINARY_PKL = "/var/new_homes/julio/mestrado/mestrado-dyssyn/datasets/moss/lite/moss_binario_lite.pkl"
QUANTIFIER_N_ESTIMATORS = 300

RUN_SMQ_LLP = True                   # SMQ treinado em regime LLP puro (extra)

# alter-∝SVM. O paper ajusta C ∈ {0.1,1,10}, Cp ∈ {0.01,0.1,1} por CV no
# erro de proporção das bags. Isso multiplica o custo por ~27x; por padrão
# fica desligado e usa C=1, Cp=1.
TUNE_ALTERSVM = False
ALTERSVM_C = 1.0
ALTERSVM_CP = 1.0
ALTERSVM_GRID_C = [0.1, 1.0, 10.0]
ALTERSVM_GRID_CP = [0.01, 0.1, 1.0]
ALTERSVM_RESTARTS = 3
ALTERSVM_MAX_ITER = 20

OUTPUT_CSV = "results_llp_vs_quantification.csv"

# ============================================================
# DATASETS BINÁRIOS UCI (mesma lista confirmada no script anterior)
# ============================================================

UCI_BINARY_DATASETS = [
    "balance.1", "balance.3", "breast-cancer", "cmc.1", "cmc.2", "cmc.3",
    "ctg.1", "ctg.2", "ctg.3", "german", "haberman", "ionosphere",
    "iris.1", "iris.2", "iris.3", "mammographic", "pageblocks.5",
    "semeion", "sonar", "spambase", "spectf", "tictactoe", "transfusion",
    "wdbc", "wine.1", "wine.2", "wine.3", "wine-q-red", "wine-q-white",
    "yeast",
]
DATASETS_TO_EXCLUDE = set()


def get_active_dataset_list():
    return [d for d in UCI_BINARY_DATASETS if d not in DATASETS_TO_EXCLUDE]


# ============================================================
# BATCHES ARTIFICIAIS APP (idêntico ao script anterior)
# ============================================================

def artificial_prevalence_batches(X, y, n_classes, n_batches, batch_size, rng,
                                   n_prevalences=21, min_prev=0.0, max_prev=1.0):
    classes = np.arange(n_classes)
    idx_by_class = [np.where(y == c)[0] for c in classes]

    grid = np.linspace(min_prev, max_prev, n_prevalences)
    repeats_per_bin = max(1, round(n_batches / n_prevalences))

    for p1 in grid:
        for _ in range(repeats_per_bin):
            if n_classes == 2:
                p_target = np.array([1 - p1, p1])
            else:
                p_target = rng.dirichlet(np.ones(n_classes))

            counts = np.floor(p_target * batch_size).astype(int)
            counts[np.argmax(counts)] += batch_size - counts.sum()

            batch_idx = []
            for c, cnt in zip(classes, counts):
                if cnt <= 0 or len(idx_by_class[c]) == 0:
                    continue
                batch_idx.append(rng.choice(idx_by_class[c], size=cnt, replace=True))
            if not batch_idx:
                continue

            batch_idx = np.concatenate(batch_idx)
            rng.shuffle(batch_idx)
            yield batch_idx, counts / counts.sum()


# ============================================================
# MÉTRICAS
# ============================================================

def ae(p_hat, p_true):
    return np.mean(np.abs(p_hat - p_true))


def rae(p_hat, p_true, n_train):
    eps = 1.0 / (2 * n_train)
    n_classes = len(p_true)
    p_hat_s = (p_hat + eps) / (p_hat.sum() + n_classes * eps)
    p_true_s = (p_true + eps) / (p_true.sum() + n_classes * eps)
    return np.mean(np.abs(p_hat_s - p_true_s) / p_true_s)


def summarize_method_results(ae_list, rae_list):
    ae_arr = np.asarray(ae_list, dtype=float)
    rae_arr = np.asarray(rae_list, dtype=float)
    return dict(
        ae_mean=ae_arr.mean(), ae_std=ae_arr.std(), ae_median=np.median(ae_arr),
        rae_mean=rae_arr.mean(), rae_std=rae_arr.std(), rae_median=np.median(rae_arr),
        n_batches=len(ae_arr),
    )


def instance_errors(y_true, y_pred):
    """Erro de instância (métrica principal do paper) + erro balanceado."""
    y_pred = np.asarray(y_pred).astype(int)
    err = float(np.mean(y_pred != y_true))
    tpr = np.mean(y_pred[y_true == 1] == 1) if np.any(y_true == 1) else np.nan
    tnr = np.mean(y_pred[y_true == 0] == 0) if np.any(y_true == 0) else np.nan
    return dict(inst_error=err, inst_bal_error=float(1 - 0.5 * (tpr + tnr)))


def safe_prevalence_vector(p, n_classes):
    p = np.array(p, dtype=float)
    if len(p) < n_classes:
        padded = np.zeros(n_classes)
        padded[: len(p)] = p
        p = padded
    p = p[:n_classes]
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, 0, 1)
    s = p.sum()
    return p / s if s > 0 else np.ones(n_classes) / n_classes


def evaluate_on_batches(test_batches, predict_fn, n_train):
    """predict_fn(idx) -> {nome_metodo: p_hat}. Todos os métodos recebem
    exatamente os mesmos batches (comparação pareada)."""
    acc = {}
    for idx, p_real in test_batches:
        for name, p_hat in predict_fn(idx).items():
            p_hat = safe_prevalence_vector(p_hat, 2)
            e = acc.setdefault(name, {"ae": [], "rae": []})
            e["ae"].append(ae(p_hat, p_real))
            e["rae"].append(rae(p_hat, p_real, n_train))
    return {name: summarize_method_results(v["ae"], v["rae"]) for name, v in acc.items()}


def mean_prevalence_fn(per_instance):
    """per_instance: {nome: vetor (n_test,) com 0/1 (CC) ou P(y=1|x) (PCC)}.
    A prevalência de um batch é a média do vetor nos índices do batch."""
    def fn(idx):
        out = {}
        for name, v in per_instance.items():
            p1 = float(v[idx].mean())
            out[name] = np.array([1 - p1, p1])
        return out
    return fn


# ============================================================
# CLASSIFICADORES
# ============================================================

def make_lr(seed=None):
    return LogisticRegression(C=1.0, penalty="l2", class_weight="balanced",
                               max_iter=1000, random_state=seed)


def make_rf_smq(seed):
    return RandomForestClassifier(n_estimators=QUANTIFIER_N_ESTIMATORS,
                                   n_jobs=N_JOBS, random_state=seed)


# ============================================================
# [ORÁCULO] BASELINES QuaPy — treinados com rótulos individuais
# ============================================================

BASELINE_CLASSES = {
    "CC": CC, "PCC": PCC, "ACC": ACC, "PACC": PACC,
    "EMQ": EMQ, "DyS": DyS, "KDEyML": KDEyML,
}
NEEDS_VAL_SPLIT = {"ACC", "PACC", "DyS", "KDEyML"}  # EMQ padrão não usa


def fit_baseline(name, Xtr, ytr):
    cls = BASELINE_CLASSES[name]
    if name in NEEDS_VAL_SPLIT:
        vs = int(max(2, min(5, np.bincount(ytr).min())))
        model = cls(classifier=make_lr(), val_split=vs)
    else:
        model = cls(classifier=make_lr())
    model.fit(Xtr, ytr)
    return model


# ============================================================
# SMQ (compartilhado entre oráculo e LLP)
# ============================================================

def baseline_features(scores_matrix, target_n_classes):
    feats = []
    for c in range(scores_matrix.shape[1]):
        s = scores_matrix[:, c]
        feats.extend([np.mean(s), np.var(s), skew(s), kurtosis(s)])
    total_expected = 4 * target_n_classes
    if len(feats) < total_expected:
        feats.extend([0] * (total_expected - len(feats)))
    feats = np.array(feats[:total_expected])
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def get_oof_scores_adaptive(Xtr, ytr, rep_seed):
    min_class_count = np.bincount(ytr).min()
    if min_class_count >= 2:
        return cross_val_predict(make_rf_smq(rep_seed), Xtr, ytr,
                                 cv=min(5, min_class_count),
                                 method="predict_proba", n_jobs=N_JOBS)
    clf = make_rf_smq(rep_seed).fit(Xtr, ytr)
    return clf.predict_proba(Xtr)


REGRESSOR_FACTORIES_SMQ = {
    "rf": lambda seed: RandomForestRegressor(
        n_estimators=QUANTIFIER_N_ESTIMATORS, n_jobs=N_JOBS,
        random_state=seed, min_samples_leaf=2),
    "et": lambda seed: ExtraTreesRegressor(
        n_estimators=QUANTIFIER_N_ESTIMATORS, n_jobs=N_JOBS,
        random_state=seed, min_samples_leaf=2),
    "gbr": lambda seed: MultiOutputRegressor(
        GradientBoostingRegressor(n_estimators=200, random_state=seed),
        n_jobs=N_JOBS),
}


def load_moss_base_binary():
    if not os.path.exists(MOSS_BINARY_PKL):
        print(f"⚠️  MoSS base binário não encontrado em {MOSS_BINARY_PKL} — "
              f"métodos SMQ serão pulados.")
        return None
    with open(MOSS_BINARY_PKL, "rb") as f:
        synthetic_distributions = pickle.load(f)

    X_m, y_m = [], []
    for (alpha_prev, _), curves in synthetic_distributions.items():
        for scores_matrix in curves:
            X_m.append(baseline_features(scores_matrix, 2))
            y_m.append(list(alpha_prev))
    model = RandomForestRegressor(n_estimators=QUANTIFIER_N_ESTIMATORS,
                                   n_jobs=N_JOBS, random_state=SEED,
                                   min_samples_leaf=2)
    model.fit(np.array(X_m), np.array(y_m))
    return model


def train_isotonic_calibrator(pred_train, real_train, n_classes):
    calibrators = []
    for c in range(n_classes):
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing="auto",
                                  out_of_bounds="clip")
        iso.fit(pred_train[:, c], real_train[:, c])
        calibrators.append(iso)
    return calibrators


def _fit_smq_calibration(moss_base_model, score_batches, real_props, rep_seed):
    """Treina calibradores rf/et/gbr/iso a partir de uma lista de batches
    (matriz de scores) com prevalência conhecida. É a mesma lógica do seu
    fit_smq; só a ORIGEM dos batches muda entre oráculo e LLP."""
    n_classes = 2
    pred, feat = [], []
    for scores in score_batches:
        f = baseline_features(scores, n_classes)
        p = moss_base_model.predict(f.reshape(1, -1))[0][:n_classes]
        pred.append(safe_prevalence_vector(p, n_classes))
        feat.append(f)
    pred, feat, real = np.array(pred), np.array(feat), np.array(real_props)

    X_input = np.nan_to_num(np.hstack([pred, feat]), nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler().fit(X_input)
    X_scaled = scaler.transform(X_input)

    calibrators = {}
    for name, factory in REGRESSOR_FACTORIES_SMQ.items():
        m = factory(rep_seed)
        m.fit(X_scaled, real)
        calibrators[name] = {"model": m, "scaler": scaler}
    iso = train_isotonic_calibrator(pred, real, n_classes)
    return calibrators, iso


def apply_smq_scores(smq_state, scores_batch, prefix):
    """Aplica SMQ treinado a um batch já pontuado (scores pré-calculados
    para todo o teste uma única vez, e aqui só indexados)."""
    n_classes = 2
    moss = smq_state["moss_base_model"]
    f_vec = baseline_features(scores_batch, n_classes)
    p_base = safe_prevalence_vector(moss.predict(f_vec.reshape(1, -1))[0][:n_classes], n_classes)

    results = {f"{prefix}_base": p_base}
    X_in = np.hstack([p_base.reshape(1, -1), f_vec.reshape(1, -1)])
    for name, entry in smq_state["calibrators"].items():
        p_cal = np.array(entry["model"].predict(entry["scaler"].transform(X_in))[0])[:n_classes]
        results[f"{prefix}_{name}"] = safe_prevalence_vector(p_cal, n_classes)

    p_iso = np.array([smq_state["iso_calibrators"][c].predict([p_base[c]])[0]
                      for c in range(n_classes)])
    results[f"{prefix}_iso"] = safe_prevalence_vector(p_iso, n_classes)
    return results


def fit_smq_oracle(moss_base_model, Xtr, ytr, rep_seed, rng):
    """SMQ supervisionado — igual ao seu fit_smq, com rótulos individuais."""
    clf = make_rf_smq(rep_seed).fit(Xtr, ytr)
    oof = get_oof_scores_adaptive(Xtr, ytr, rep_seed)
    score_batches, real = [], []
    for idx, p_real in artificial_prevalence_batches(Xtr, ytr, 2, N_CALIB_BATCHES, BATCH_SIZE, rng):
        score_batches.append(oof[idx])
        real.append(p_real)
    cal, iso = _fit_smq_calibration(moss_base_model, score_batches, real, rep_seed)
    return {"scorer": clf, "moss_base_model": moss_base_model,
            "calibrators": cal, "iso_calibrators": iso}


# ============================================================
# BAGS LLP
# ============================================================

def make_llp_bags(y, bag_size, n_bags, regime, rng):
    """Gera índices das bags e suas proporções (calculadas a partir do
    ground truth, como na Seção 9 do paper). Os rótulos individuais NÃO
    saem desta função: só índices e proporções."""
    n = len(y)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    bags, props = [], []
    for _ in range(n_bags):
        if regime == "iid":
            idx = rng.choice(n, size=bag_size, replace=True)
        elif regime == "hier":
            n_pos = rng.binomial(bag_size, rng.uniform())
            idx = np.concatenate([
                rng.choice(idx_pos, size=n_pos, replace=True),
                rng.choice(idx_neg, size=bag_size - n_pos, replace=True),
            ]).astype(int)
            rng.shuffle(idx)
        else:
            raise ValueError(regime)
        bags.append(idx)
        props.append(float(y[idx].mean()))
    return bags, np.array(props)


class LLPBags:
    """Dados de treino LLP: lista de matrizes (uma por bag) + proporções.
    Instâncias ficam empilhadas em ordem de bag (contíguas)."""

    def __init__(self, bags_X, props):
        self.bags_X = list(bags_X)
        self.props = np.asarray(props, dtype=float)
        self.n_bags = len(self.bags_X)
        self.sizes = np.array([len(b) for b in self.bags_X])
        self.X = np.vstack(self.bags_X)
        self.bag_id = np.repeat(np.arange(self.n_bags), self.sizes)
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)])
        self.bag_means = np.array([b.mean(axis=0) for b in self.bags_X])

    @property
    def n_instances(self):
        return len(self.X)

    def global_prevalence(self):
        return float((self.props * self.sizes).sum() / self.sizes.sum())

    def bag_slice(self, b):
        return slice(self.offsets[b], self.offsets[b + 1])

    def subset(self, sel):
        return LLPBags([self.bags_X[b] for b in sel], self.props[sel])


def _augment(X):
    return np.hstack([X, np.ones((len(X), 1))])


# ============================================================
# [LLP] MÉTODOS DE LEARNING FROM LABEL PROPORTIONS
# ============================================================

class _LinearProbLLP:
    """Base: modelo linear w (bias na última posição), p = sigmoid(x·w)."""
    probabilistic = True

    def decision_function(self, X):
        return _augment(X) @ self.w_

    def predict_proba(self, X):
        p1 = expit(self.decision_function(X))
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)


def _l2_reg_mask(d):
    reg = np.ones(d)
    reg[-1] = 0.0  # não regulariza o bias
    return reg


class LLP_EPRM(_LinearProbLLP):
    """EPRM (Eq. 2 do paper) com h logístico:
        min_w  (1/N) Σ_b n_b · BCE(pi_b, mean_{i∈b} σ(x_i·w)) + ½λ||w||²
    λ = 1/(C·N) reproduz a escala da LogisticRegression(C) do sklearn.
    Com r=1 recai exatamente na regressão logística supervisionada."""

    def __init__(self, C=1.0, eps=1e-6, max_iter=500):
        self.C, self.eps, self.max_iter = C, eps, max_iter

    def fit(self, bags):
        Xa = _augment(bags.X)
        N, d = Xa.shape
        bid, pi, nb = bags.bag_id, bags.props, bags.sizes
        lam = 1.0 / (self.C * N)
        reg = _l2_reg_mask(d)

        def f(w):
            p = expit(Xa @ w)
            pbar = np.bincount(bid, weights=p, minlength=bags.n_bags) / nb
            pbar = np.clip(pbar, self.eps, 1 - self.eps)
            loss = -np.sum(nb * (pi * np.log(pbar) + (1 - pi) * np.log(1 - pbar))) / N
            loss += 0.5 * lam * np.sum(reg * w * w)
            g_b = (pbar - pi) / (pbar * (1 - pbar))
            grad = Xa.T @ (g_b[bid] * p * (1 - p)) / N + lam * reg * w
            return loss, grad

        w0 = np.zeros(d)
        w0[-1] = logit(np.clip(bags.global_prevalence(), 1e-3, 1 - 1e-3))
        res = minimize(f, w0, jac=True, method="L-BFGS-B",
                       options={"maxiter": self.max_iter})
        self.w_ = res.x
        return self


class LLP_MeanMap(_LinearProbLLP):
    """Mean Map (Quadrianto et al. 2009) binário.
    1) Estima as médias de classe mu+, mu- por mínimos quadrados a partir
       das médias das bags:  mean_b ≈ pi_b·mu+ + (1-pi_b)·mu-.
    2) Substitui a estatística suficiente Σ_i y_i·x_i da log-verossimilhança
       logística por N+·mu+ (N+ = Σ_b n_b·pi_b) e maximiza (convexo).
    Falha, como esperado, quando as proporções quase não variam entre bags
    (sistema mal condicionado) — o regime "iid" com r grande."""

    def __init__(self, C=1.0, max_iter=500):
        self.C, self.max_iter = C, max_iter

    def fit(self, bags):
        P = np.column_stack([bags.props, 1 - bags.props])
        W = np.sqrt(bags.sizes)[:, None]
        mu, *_ = np.linalg.lstsq(P * W, bags.bag_means * W, rcond=None)
        n_pos = float((bags.props * bags.sizes).sum())
        s = n_pos * np.append(mu[0], 1.0)

        Xa = _augment(bags.X)
        N, d = Xa.shape
        lam = 1.0 / (self.C * N)
        reg = _l2_reg_mask(d)

        def f(w):
            z = Xa @ w
            loss = (-(w @ s) + np.sum(np.logaddexp(0.0, z))) / N + 0.5 * lam * np.sum(reg * w * w)
            grad = (-s + Xa.T @ expit(z)) / N + lam * reg * w
            return loss, grad

        res = minimize(f, np.zeros(d), jac=True, method="L-BFGS-B",
                       options={"maxiter": self.max_iter})
        self.w_ = res.x
        return self


class LLP_InvCal(_LinearProbLLP):
    """Inverse Calibration (Rüeping 2010): SVR linear ε-insensível que
    regride logit(pi_b) a partir da média da bag; a instância recebe
    P(y=1|x) = σ(w·x + b). Proporções 0/1 são cortadas em [clip, 1-clip]."""

    def __init__(self, C=1.0, epsilon=0.0, clip=0.01):
        self.C, self.epsilon, self.clip = C, epsilon, clip

    def fit(self, bags):
        t = logit(np.clip(bags.props, self.clip, 1 - self.clip))
        svr = LinearSVR(C=self.C, epsilon=self.epsilon, loss="epsilon_insensitive",
                        dual=True, max_iter=20000, random_state=0)
        svr.fit(bags.bag_means, t, sample_weight=bags.sizes.astype(float))
        self.w_ = np.append(svr.coef_, np.ravel(svr.intercept_)[0])
        return self


class LLP_AlterSVM:
    """alter-∝SVM (Yu et al. 2013), o método da Seção 9 do paper:
        min_{w, y}  ½||w||² + C Σ_i hinge(y_i, w·x_i) + Cp Σ_b |p̂_b(y) - pi_b|
    Alterna: (a) fixa y e treina LinearSVC; (b) fixa w e, em cada bag,
    escolhe o nº k de positivos (top-k por score) que minimiza o custo —
    ótimo exato para esse subproblema. Inicializado por MeanMap e InvCal
    (como no paper) + restarts aleatórios; fica a solução de menor
    objetivo. Sem o annealing de C do paper original.
    Não é probabilístico -> só vira quantificador via CC."""
    probabilistic = False

    def __init__(self, C=1.0, Cp=1.0, n_restarts=3, max_iter=20, seed=0):
        self.C, self.Cp = C, Cp
        self.n_restarts, self.max_iter, self.seed = n_restarts, max_iter, seed

    def _optimize_labels(self, scores, bags):
        y = np.zeros(len(scores), dtype=int)
        for b in range(bags.n_bags):
            sl = bags.bag_slice(b)
            s = scores[sl]
            nb = len(s)
            order = np.argsort(-s)
            s_sorted = s[order]
            cum_pos = np.concatenate([[0.0], np.cumsum(np.maximum(0, 1 - s_sorted))])
            suf_neg = np.concatenate([np.cumsum(np.maximum(0, 1 + s_sorted)[::-1])[::-1], [0.0]])
            k = np.arange(nb + 1)
            cost = self.C * (cum_pos + suf_neg) + self.Cp * np.abs(k / nb - bags.props[b])
            k_best = int(np.argmin(cost))
            yb = np.zeros(nb, dtype=int)
            yb[order[:k_best]] = 1
            y[sl] = yb
        return y

    def _fit_svm(self, X, y):
        if len(np.unique(y)) < 2:
            return None
        return LinearSVC(C=self.C, loss="hinge", dual=True, max_iter=20000,
                         random_state=self.seed).fit(X, y)

    def _objective(self, svm, X, y, bags):
        s = svm.decision_function(X)
        hinge = np.maximum(0, 1 - (2 * y - 1) * s).sum()
        p_hat = np.bincount(bags.bag_id, weights=y, minlength=bags.n_bags) / bags.sizes
        return 0.5 * np.sum(svm.coef_ ** 2) + self.C * hinge + self.Cp * np.abs(p_hat - bags.props).sum()

    def fit(self, bags):
        rng = np.random.default_rng(self.seed)
        X = bags.X
        inits = []
        for init_model in (LLP_MeanMap(), LLP_InvCal()):
            try:
                inits.append(self._optimize_labels(init_model.fit(bags).decision_function(X), bags))
            except Exception:
                pass
        for _ in range(self.n_restarts):
            inits.append(self._optimize_labels(rng.normal(size=len(X)), bags))

        best_obj, best_svm = np.inf, None
        for y in inits:
            svm = None
            for _ in range(self.max_iter):
                svm = self._fit_svm(X, y)
                if svm is None:
                    break
                y_new = self._optimize_labels(svm.decision_function(X), bags)
                if np.array_equal(y_new, y):
                    break
                y = y_new
            if svm is None:
                continue
            obj = self._objective(svm, X, y, bags)
            if obj < best_obj:
                best_obj, best_svm = obj, svm

        self.svm_ = best_svm
        self.const_ = int(bags.global_prevalence() >= 0.5)  # fallback degenerado
        return self

    def predict(self, X):
        if self.svm_ is None:
            return np.full(len(X), self.const_, dtype=int)
        return self.svm_.predict(X).astype(int)


def bag_cv_proportion_error(make_model, bags, rng, n_folds=3):
    """Erro de proporção em bags held-out (critério de seleção do paper —
    só usa proporções, legítimo em LLP)."""
    n_folds = min(n_folds, bags.n_bags // 2)
    if n_folds < 2:
        return np.nan
    folds = np.array_split(rng.permutation(bags.n_bags), n_folds)
    errs = []
    for k in range(n_folds):
        tr = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        m = make_model().fit(bags.subset(tr))
        for b in folds[k]:
            errs.append(abs(m.predict(bags.bags_X[b]).mean() - bags.props[b]))
    return float(np.mean(errs))


def build_altersvm(seed, bags, rng):
    C, Cp = ALTERSVM_C, ALTERSVM_CP
    if TUNE_ALTERSVM and bags.n_bags >= 4:
        best = np.inf
        for c in ALTERSVM_GRID_C:
            for cp in ALTERSVM_GRID_CP:
                err = bag_cv_proportion_error(
                    lambda: LLP_AlterSVM(c, cp, ALTERSVM_RESTARTS, ALTERSVM_MAX_ITER, seed),
                    bags, rng)
                if err < best:
                    best, C, Cp = err, c, cp
    return LLP_AlterSVM(C, Cp, ALTERSVM_RESTARTS, ALTERSVM_MAX_ITER, seed)


LLP_METHODS = {
    "LLP_EPRM": lambda seed, bags, rng: LLP_EPRM(),
    "LLP_MeanMap": lambda seed, bags, rng: LLP_MeanMap(),
    "LLP_InvCal": lambda seed, bags, rng: LLP_InvCal(),
    "LLP_AlterSVM": build_altersvm,
}


# ============================================================
# [SMQ_LLP] SMQ treinado só com bags + proporções
# ============================================================

def fit_smq_llp(moss_base_model, bags, rep_seed, rng):
    n_classes = 2

    # scores out-of-fold POR BAG (nenhuma bag é pontuada por um modelo
    # que a viu no treino). Obs.: como as bags são amostradas com
    # reposição, a mesma instância pode aparecer em bags de folds
    # diferentes — mesmo vazamento que existiria com dados reais de LLP.
    oof = np.zeros((bags.n_instances, n_classes))
    n_folds = min(5, bags.n_bags)
    if n_folds >= 2:
        folds = np.array_split(rng.permutation(bags.n_bags), n_folds)
        for k in range(n_folds):
            tr = np.concatenate([folds[j] for j in range(n_folds) if j != k])
            m = LLP_EPRM().fit(bags.subset(tr))
            for b in folds[k]:
                oof[bags.bag_slice(b)] = m.predict_proba(bags.bags_X[b])
    scorer = LLP_EPRM().fit(bags)
    if n_folds < 2:
        oof = scorer.predict_proba(bags.X)

    # batches de calibração = uniões de k bags inteiras (prevalência
    # conhecida pela média ponderada das proporções). "APP sobre bags":
    # sorteia uma prevalência-alvo p e mistura bags do grupo "alto"
    # (pi_b >= p) e "baixo" (pi_b < p) na razão que leva a média a ~p.
    # Assim a tabela de calibração cobre todo o intervalo que as bags
    # permitem atingir (em r=1 cobre [0,1]; no regime iid com r grande
    # fica naturalmente estreito, que é a limitação real do LLP).
    r_typ = max(1, int(np.median(bags.sizes)))
    k_max = max(1, min(bags.n_bags, BATCH_SIZE // r_typ))
    p_lo, p_hi = bags.props.min(), bags.props.max()
    score_batches, real = [], []
    for _ in range(N_CALIB_BATCHES):
        k = int(rng.integers(1, k_max + 1))
        p_t = rng.uniform(p_lo, p_hi)
        hi = np.where(bags.props >= p_t)[0]
        lo = np.where(bags.props < p_t)[0]
        if len(lo) == 0 or len(hi) == 0:
            sel = rng.choice(bags.n_bags, size=k, replace=False)
        else:
            m_hi, m_lo = bags.props[hi].mean(), bags.props[lo].mean()
            q = np.clip((p_t - m_lo) / max(m_hi - m_lo, 1e-12), 0, 1)
            k_hi = int(rng.binomial(k, q))
            k_hi, k_lo = min(k_hi, len(hi)), min(k - k_hi, len(lo))
            sel = np.concatenate([rng.choice(hi, size=k_hi, replace=False),
                                  rng.choice(lo, size=k_lo, replace=False)]).astype(int)
            if len(sel) == 0:
                sel = rng.choice(bags.n_bags, size=1)
        idx = np.concatenate([np.arange(bags.offsets[b], bags.offsets[b + 1]) for b in sel])
        p1 = float((bags.props[sel] * bags.sizes[sel]).sum() / bags.sizes[sel].sum())
        score_batches.append(oof[idx])
        real.append([1 - p1, p1])

    cal, iso = _fit_smq_calibration(moss_base_model, score_batches, real, rep_seed)
    return {"scorer": scorer, "moss_base_model": moss_base_model,
            "calibrators": cal, "iso_calibrators": iso}


# ============================================================
# UMA REPETIÇÃO DE UM DATASET
# ============================================================

def run_one_rep(ds_name, Xtr, ytr, Xte, yte, rep_seed, moss_base_model):
    rng = np.random.default_rng(rep_seed)
    test_batches = list(artificial_prevalence_batches(Xte, yte, 2, N_TEST_BATCHES, BATCH_SIZE, rng))
    rows = []

    def emit(results, grupo, cfg, inst=None):
        for name, summ in results.items():
            rows.append(dict(dataset=ds_name, grupo=grupo, metodo=name, **cfg,
                             **summ, **(inst or {})))

    oracle_cfg = dict(regime="oraculo", bag_size=np.nan, n_bags=np.nan,
                      bag_prop_std=np.nan, bag_purity_mean=np.nan)

    # ---------------- [oraculo] ----------------
    lr_inst = instance_errors(yte, make_lr().fit(Xtr, ytr).predict(Xte))
    for name in BASELINE_CLASSES:
        try:
            model = fit_baseline(name, Xtr, ytr)
        except Exception as e:
            print(f"   ⚠️  {name} falhou: {e}")
            continue
        res = evaluate_on_batches(test_batches,
                                  lambda idx, m=model, n=name: {n: m.quantify(Xte[idx])},
                                  len(ytr))
        emit(res, "oraculo", oracle_cfg, lr_inst)

    if moss_base_model is not None:
        st = fit_smq_oracle(moss_base_model, Xtr, ytr, rep_seed, rng)
        te_scores = st["scorer"].predict_proba(Xte)
        rf_inst = instance_errors(yte, te_scores.argmax(axis=1))
        res = evaluate_on_batches(test_batches,
                                  lambda idx: apply_smq_scores(st, te_scores[idx], "smq"),
                                  len(ytr))
        emit(res, "oraculo", oracle_cfg, rf_inst)

    # ---------------- [llp] + [smq_llp] ----------------
    for regime in BAG_REGIMES:
        for r in BAG_SIZES:
            n_bags = max(MIN_TRAIN_BAGS, math.ceil(len(ytr) / r))
            bag_idx, props = make_llp_bags(ytr, r, n_bags, regime, rng)
            bags = LLPBags([Xtr[i] for i in bag_idx], props)
            # a partir daqui ytr não é usado por nenhum método LLP
            cfg = dict(regime=regime, bag_size=r, n_bags=n_bags,
                       bag_prop_std=float(props.std()),
                       bag_purity_mean=float(np.maximum(props, 1 - props).mean()))
            n_rae = bags.n_instances

            for mname, factory in LLP_METHODS.items():
                try:
                    model = factory(rep_seed, bags, rng).fit(bags)
                except Exception as e:
                    print(f"   ⚠️  {mname} ({regime}, r={r}) falhou: {e}")
                    continue
                hard = model.predict(Xte)
                per_inst = {f"{mname}_CC": hard.astype(float)}
                if getattr(model, "probabilistic", False):
                    per_inst[f"{mname}_PCC"] = model.predict_proba(Xte)[:, 1]
                res = evaluate_on_batches(test_batches, mean_prevalence_fn(per_inst), n_rae)
                emit(res, "llp", cfg, instance_errors(yte, hard))

            if RUN_SMQ_LLP and moss_base_model is not None:
                try:
                    st = fit_smq_llp(moss_base_model, bags, rep_seed, rng)
                    te_scores = st["scorer"].predict_proba(Xte)
                    res = evaluate_on_batches(
                        test_batches,
                        lambda idx, s=st, sc=te_scores: apply_smq_scores(s, sc[idx], "smq_llp"),
                        n_rae)
                    emit(res, "smq_llp", cfg, instance_errors(yte, te_scores.argmax(axis=1)))
                except Exception as e:
                    print(f"   ⚠️  SMQ_LLP ({regime}, r={r}) falhou: {e}")

            print(f"      · {regime:4s} r={r:<3d} bags={n_bags:<5d} "
                  f"std(pi_b)={cfg['bag_prop_std']:.3f}")
    return rows


# ============================================================
# LOOP PRINCIPAL (com resume)
# ============================================================

def _done_keys():
    if not os.path.exists(OUTPUT_CSV):
        return set()
    try:
        df = pd.read_csv(OUTPUT_CSV, usecols=["dataset", "repeticao"])
        return set(zip(df["dataset"], df["repeticao"]))
    except Exception:
        return set()


def run_experiment():
    datasets = get_active_dataset_list()
    moss_base_model = load_moss_base_binary()
    done = _done_keys()

    print(f"Datasets ativos: {len(datasets)} | Repetições: {N_REPETITIONS}")
    print(f"Regimes: {BAG_REGIMES} | Tamanhos de bag r: {BAG_SIZES}")
    print(f"SMQ habilitado: {moss_base_model is not None} | SMQ_LLP: {RUN_SMQ_LLP}")
    print(f"Já concluídos no CSV: {len(done)} (dataset, repetição)")

    header_written = os.path.exists(OUTPUT_CSV)

    for ds_name in datasets:
        print(f"\n📂 Dataset: {ds_name}")
        try:
            dataset = qp.datasets.fetch_UCIBinaryDataset(ds_name, verbose=False)
        except Exception as e:
            print(f"❌ Erro carregando {ds_name}: {e}")
            continue

        train, test = dataset.train_test
        X_all = np.vstack([train.instances, test.instances])
        y_all = np.concatenate([train.labels, test.labels]).astype(int)

        for rep in range(N_REPETITIONS):
            if (ds_name, rep + 1) in done:
                continue
            rep_seed = SEED + rep
            Xtr, Xte, ytr, yte = train_test_split(
                X_all, y_all, test_size=0.3, random_state=rep_seed, stratify=y_all)
            scaler = StandardScaler().fit(Xtr)
            Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)

            rows = run_one_rep(ds_name, Xtr, ytr, Xte, yte, rep_seed, moss_base_model)
            for r in rows:
                r["repeticao"] = rep + 1

            pd.DataFrame(rows).to_csv(OUTPUT_CSV, mode="a",
                                      header=not header_written, index=False)
            header_written = True
            print(f"   ✅ rep {rep + 1}/{N_REPETITIONS} salva ({len(rows)} linhas)")
            gc.collect()


if __name__ == "__main__":
    try:
        run_experiment()
    except Exception:
        print("\n❌ EXPERIMENTO ABORTADO")
        print(traceback.format_exc())
        sys.exit(1)
