    """
    Script complementar: acrescenta as linhas do SMQ (oráculo smq_* e smq_llp_*)
    ao experimento LLP vs quantificação, PAREADAS com o CSV que já existe.

    Por que um script separado
    --------------------------
    O run original não encontrou o MoSS, então nenhuma linha de SMQ foi gerada
    (77 linhas por repetição em vez de 132). Em vez de refazer tudo, este script
    reproduz exatamente os mesmos batches de teste e as mesmas bags de cada
    (dataset, repetição) e só treina e avalia o SMQ.

    Como a reprodução funciona
    --------------------------
    - Mesmo carregamento, mesmo split, mesmo StandardScaler e mesmo
    rng(rep_seed) do script principal (importado como módulo).
    - No run sem MoSS, o rng foi consumido só pelos batches de teste e depois
    pelas bags de cada (regime, r), nessa ordem. Este script consome o rng na
    mesma ordem e usa um rng SEPARADO para tudo que é do SMQ (batches de
    calibração e folds), então as bags saem idênticas.
    - Verificação, por repetição:
        (a) bag_prop_std, bag_purity_mean e n_bags de cada configuração são
            recalculados e comparados com os valores gravados no CSV original;
        (b) o AE médio do CC do oráculo é recalculado e comparado com o do CSV
            (confere dados, split e batches de teste de ponta a ponta).
    Se tudo bater, as linhas saem com pareado=True.

    Saída
    -----
    CSV separado (results_smq_complementar.csv), com as mesmas colunas do
    original + "pareado". Para analisar, junte os dois:
        pd.concat([pd.read_csv("results_llp_vs_quantification.csv"),
                pd.read_csv("results_smq_complementar.csv")])
    Tem resume: se for interrompido, rode de novo que ele continua.

    Desempenho
    ----------
    As features dos batches e as predições do MoSS e dos calibradores são
    calculadas de uma vez para todos os batches (uma chamada de predict em vez
    de ~1000 por método), que é onde estava quase todo o custo do SMQ.

    Requisitos na MESMA pasta: o script principal (llp_vs_quantification.py),
    moss_binario_lite.pkl e results_llp_vs_quantification.csv. Rode antes o
    run_german.py para o german entrar também. Funciona como script
    (python run_smq_complementar.py) ou colado numa célula de notebook; no
    notebook, se a pasta atual não for a do experimento, defina
    PASTA_DO_EXPERIMENTO.
    """

    import gc
    import math
    import os
    import pickle
    import sys
    import time

    # evita UnicodeEncodeError com emojis no Windows quando a saída é redirecionada
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    import numpy as np
    import pandas as pd
    from scipy.stats import skew, kurtosis

    # ============================================================
    # CONFIG
    # ============================================================

    MAIN_SCRIPT_FILE = "llp_vs_quantification.py"  # arquivo do script principal
    PASTA_DO_EXPERIMENTO = None  # ex.: r"C:\Users\LACA\smq_joylan"; None = pasta atual
    BASE_CSV = "results_llp_vs_quantification.csv"
    OUT_CSV = "results_smq_complementar.csv"
    MOSS_PKL = "moss_binario_lite.pkl"

    N_JOBS = max(1, (os.cpu_count() or 2) - 1)   # deixa 1 núcleo livre
    DATASETS_ONLY = None        # ex.: ["iris.1"] para um teste rápido; None = todos
    SMQ_SEED_OFFSET = 1_000_003  # semente do rng separado do SMQ = rep_seed + isto

    TOL_BAGS = 1e-9   # tolerância na comparação das estatísticas das bags
    TOL_CC = 1e-3     # tolerância no AE do CC (absorve diferenças numéricas mínimas)


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
    E.N_JOBS = N_JOBS
    E.MOSS_BINARY_PKL = MOSS_PKL

    from sklearn.model_selection import train_test_split  # noqa: E402
    from sklearn.preprocessing import StandardScaler  # noqa: E402

    print = E.print  # print com flush


    # ============================================================
    # CHECAGEM DO PKL DO MoSS
    # ============================================================

    def check_moss_orientation(path):
        """A coluna 1 das curvas deve ser o score da classe 1, e alpha_prev[1] a
        prevalência da classe 1 (mesma convenção do predict_proba). Então a
        média da coluna 1 tem que crescer com alpha_prev[1]."""
        with open(path, "rb") as f:
            d = pickle.load(f)
        p1 = np.array([float(k[0][1]) for k in d])
        m1 = np.array([np.mean([np.asarray(c)[:, 1].mean() for c in v]) for v in d.values()])
        return float(np.corrcoef(p1, m1)[0, 1])


    # ============================================================
    # SMQ VETORIZADO (mesmas fórmulas do script principal)
    # ============================================================

    def batch_features_fixed(S):
        """S: (B, n, 2) -> (B, 8). Mesma ordem e mesmas fórmulas de
        baseline_features: por classe, [média, variância, assimetria, curtose]."""
        cols = []
        for c in range(S.shape[2]):
            s = S[:, :, c]
            cols += [s.mean(axis=1), s.var(axis=1), skew(s, axis=1), kurtosis(s, axis=1)]
        return np.nan_to_num(np.column_stack(cols), nan=0.0, posinf=0.0, neginf=0.0)


    def features_for_batches(scores, batches_idx):
        if len({len(i) for i in batches_idx}) == 1:
            return batch_features_fixed(scores[np.stack(batches_idx)])
        return np.array([E.baseline_features(scores[i], 2) for i in batches_idx])


    def safe_rows(P):
        """safe_prevalence_vector aplicado linha a linha (2 classes)."""
        P = np.nan_to_num(np.asarray(P, dtype=float)[:, :2], nan=0.0, posinf=0.0, neginf=0.0)
        P = np.clip(P, 0, 1)
        s = P.sum(axis=1)
        out = np.full_like(P, 0.5)
        ok = s > 0
        out[ok] = P[ok] / s[ok, None]
        return out


    def fit_calibration_fast(moss, F_cal, real, seed):
        pred = safe_rows(moss.predict(F_cal))
        X_in = np.nan_to_num(np.hstack([pred, F_cal]), nan=0.0, posinf=0.0, neginf=0.0)
        scaler = StandardScaler().fit(X_in)
        Xs = scaler.transform(X_in)
        cal = {name: factory(seed).fit(Xs, real)
            for name, factory in E.REGRESSOR_FACTORIES_SMQ.items()}
        iso = E.train_isotonic_calibrator(pred, real, 2)
        return scaler, cal, iso


    def predict_all(moss, scaler, cal, iso, F, prefix):
        base = safe_rows(moss.predict(F))
        Xs = scaler.transform(np.hstack([base, F]))
        out = {f"{prefix}_base": base}
        for name, m in cal.items():
            out[f"{prefix}_{name}"] = safe_rows(m.predict(Xs))
        out[f"{prefix}_iso"] = safe_rows(
            np.column_stack([iso[c].predict(base[:, c]) for c in range(2)]))
        return out


    def summarize_preds(P_hat, P_real, n_train):
        """AE e RAE por batch, idênticos a E.ae / E.rae, depois resumidos."""
        ae = np.abs(P_hat - P_real).mean(axis=1)
        eps = 1.0 / (2 * n_train)
        ph = (P_hat + eps) / (P_hat.sum(axis=1, keepdims=True) + 2 * eps)
        pt = (P_real + eps) / (P_real.sum(axis=1, keepdims=True) + 2 * eps)
        rae = (np.abs(ph - pt) / pt).mean(axis=1)
        return E.summarize_method_results(ae, rae)


    def smq_oracle_fast(moss, Xtr, ytr, Xte, rep_seed, rng_smq, test_idx):
        clf = E.make_rf_smq(rep_seed).fit(Xtr, ytr)
        oof = E.get_oof_scores_adaptive(Xtr, ytr, rep_seed)
        cal_idx, cal_real = [], []
        for idx, p in E.artificial_prevalence_batches(Xtr, ytr, 2, E.N_CALIB_BATCHES,
                                                    E.BATCH_SIZE, rng_smq):
            cal_idx.append(idx)
            cal_real.append(p)
        scaler, cal, iso = fit_calibration_fast(
            moss, features_for_batches(oof, cal_idx), np.array(cal_real), rep_seed)
        te_scores = clf.predict_proba(Xte)
        preds = predict_all(moss, scaler, cal, iso,
                            features_for_batches(te_scores, test_idx), "smq")
        return preds, te_scores


    def smq_llp_fast(moss, bags, rep_seed, rng_smq, Xte, test_idx):
        """Mesma lógica de fit_smq_llp do script principal (scores OOF por bag
        com LLP_EPRM + batches de calibração = uniões de bags), vetorizada."""
        oof = np.zeros((bags.n_instances, 2))
        n_folds = min(5, bags.n_bags)
        if n_folds >= 2:
            folds = np.array_split(rng_smq.permutation(bags.n_bags), n_folds)
            for k in range(n_folds):
                tr = np.concatenate([folds[j] for j in range(n_folds) if j != k])
                m = E.LLP_EPRM().fit(bags.subset(tr))
                for b in folds[k]:
                    oof[bags.bag_slice(b)] = m.predict_proba(bags.bags_X[b])
        scorer = E.LLP_EPRM().fit(bags)
        if n_folds < 2:
            oof = scorer.predict_proba(bags.X)

        r_typ = max(1, int(np.median(bags.sizes)))
        k_max = max(1, min(bags.n_bags, E.BATCH_SIZE // r_typ))
        p_lo, p_hi = bags.props.min(), bags.props.max()
        cal_idx, cal_real = [], []
        for _ in range(E.N_CALIB_BATCHES):
            k = int(rng_smq.integers(1, k_max + 1))
            p_t = rng_smq.uniform(p_lo, p_hi)
            hi = np.where(bags.props >= p_t)[0]
            lo = np.where(bags.props < p_t)[0]
            if len(lo) == 0 or len(hi) == 0:
                sel = rng_smq.choice(bags.n_bags, size=k, replace=False)
            else:
                m_hi, m_lo = bags.props[hi].mean(), bags.props[lo].mean()
                q = np.clip((p_t - m_lo) / max(m_hi - m_lo, 1e-12), 0, 1)
                k_hi = int(rng_smq.binomial(k, q))
                k_hi, k_lo = min(k_hi, len(hi)), min(k - k_hi, len(lo))
                sel = np.concatenate([rng_smq.choice(hi, size=k_hi, replace=False),
                                    rng_smq.choice(lo, size=k_lo, replace=False)]).astype(int)
                if len(sel) == 0:
                    sel = rng_smq.choice(bags.n_bags, size=1)
            cal_idx.append(np.concatenate(
                [np.arange(bags.offsets[b], bags.offsets[b + 1]) for b in sel]))
            p1 = float((bags.props[sel] * bags.sizes[sel]).sum() / bags.sizes[sel].sum())
            cal_real.append([1 - p1, p1])

        scaler, cal, iso = fit_calibration_fast(
            moss, features_for_batches(oof, cal_idx), np.array(cal_real), rep_seed)
        te_scores = scorer.predict_proba(Xte)
        preds = predict_all(moss, scaler, cal, iso,
                            features_for_batches(te_scores, test_idx), "smq_llp")
        return preds, te_scores


    # ============================================================
    # UMA REPETIÇÃO
    # ============================================================

    def run_rep(ds, X_all, y_all, rep, moss, base_cfg, base_cc):
        rep_seed = E.SEED + rep
        Xtr, Xte, ytr, yte = train_test_split(
            X_all, y_all, test_size=0.3, random_state=rep_seed, stratify=y_all)
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

        rng = np.random.default_rng(rep_seed)            # mesmo fluxo do run original
        rng_smq = np.random.default_rng(rep_seed + SMQ_SEED_OFFSET)

        test = list(E.artificial_prevalence_batches(Xte, yte, 2, E.N_TEST_BATCHES,
                                                    E.BATCH_SIZE, rng))
        test_idx = [i for i, _ in test]
        P_real = np.array([p for _, p in test])

        # (b) checagem ponta a ponta: CC do oráculo
        cc_pred = E.make_lr().fit(Xtr, ytr).predict(Xte).astype(float)
        p1 = np.array([cc_pred[i].mean() for i in test_idx])
        cc_ae = float(np.abs(np.column_stack([1 - p1, p1]) - P_real).mean(axis=1).mean())
        cc_ref = base_cc.get((ds, rep + 1))
        cc_ok = cc_ref is not None and abs(cc_ae - cc_ref) <= TOL_CC

        rows, checks = [], []

        def emit(preds, grupo, cfg, inst, n_train, pareado):
            for name, P_hat in preds.items():
                rows.append(dict(dataset=ds, grupo=grupo, metodo=name, **cfg,
                                **summarize_preds(P_hat, P_real, n_train), **inst,
                                repeticao=rep + 1, pareado=pareado))

        # --- configurações LLP: gera as bags na MESMA ordem do run original ---
        llp_parts = []
        for regime in E.BAG_REGIMES:
            for r in E.BAG_SIZES:
                n_bags = max(E.MIN_TRAIN_BAGS, math.ceil(len(ytr) / r))
                bag_idx, props = E.make_llp_bags(ytr, r, n_bags, regime, rng)
                cfg = dict(regime=regime, bag_size=r, n_bags=n_bags,
                        bag_prop_std=float(props.std()),
                        bag_purity_mean=float(np.maximum(props, 1 - props).mean()))
                bag_ref = base_cfg.get((ds, rep + 1, regime, r))
                bag_ok = (bag_ref is not None and bag_ref[0] == n_bags
                        and abs(bag_ref[1] - cfg["bag_prop_std"]) <= TOL_BAGS
                        and abs(bag_ref[2] - cfg["bag_purity_mean"]) <= TOL_BAGS)
                checks.append(bag_ok)
                llp_parts.append((cfg, bag_idx, props, bag_ok))

        rep_ok = cc_ok and all(checks)

        # --- [oraculo] SMQ ---
        oracle_cfg = dict(regime="oraculo", bag_size=np.nan, n_bags=np.nan,
                        bag_prop_std=np.nan, bag_purity_mean=np.nan)
        preds, te_scores = smq_oracle_fast(moss, Xtr, ytr, Xte, rep_seed, rng_smq, test_idx)
        emit(preds, "oraculo", oracle_cfg, E.instance_errors(yte, te_scores.argmax(axis=1)),
            len(ytr), rep_ok)

        # --- [smq_llp] ---
        for cfg, bag_idx, props, bag_ok in llp_parts:
            bags = E.LLPBags([Xtr[i] for i in bag_idx], props)
            try:
                preds, te_scores = smq_llp_fast(moss, bags, rep_seed, rng_smq, Xte, test_idx)
                emit(preds, "smq_llp", cfg, E.instance_errors(yte, te_scores.argmax(axis=1)),
                    bags.n_instances, cc_ok and bag_ok)
            except Exception as e:
                print(f"   ⚠️  SMQ_LLP ({cfg['regime']}, r={cfg['bag_size']}) falhou: {e}")

        if not rep_ok:
            print(f"   ⚠️  {ds} rep {rep + 1}: NÃO pareado "
                f"(CC: calculado={cc_ae:.6f}, CSV={cc_ref}; "
                f"bags ok: {sum(checks)}/{len(checks)})")
        return rows


    # ============================================================
    # LOOP PRINCIPAL
    # ============================================================

    def main():
        for f in (BASE_CSV, MOSS_PKL):
            if not os.path.exists(f):
                print(f"❌ Não encontrei {f} na pasta atual ({os.getcwd()}).")
                sys.exit(1)

        corr = check_moss_orientation(MOSS_PKL)
        print(f"MoSS: correlação (prevalência da classe 1 x média da coluna 1) = {corr:.3f}")
        if not corr > 0:
            print("❌ Colunas do MoSS parecem invertidas em relação ao predict_proba. Abortando.")
            sys.exit(1)

        moss = E.load_moss_base_binary()

        base = pd.read_csv(BASE_CSV)
        llp = base[base["grupo"] == "llp"]
        base_cfg = {
            (d, int(rp), rg, int(r)): (int(nb), float(sd), float(pu))
            for (d, rp, rg, r), (nb, sd, pu) in
            llp.groupby(["dataset", "repeticao", "regime", "bag_size"])
            [["n_bags", "bag_prop_std", "bag_purity_mean"]].first().iterrows()
        }
        cc = base[(base["grupo"] == "oraculo") & (base["metodo"] == "CC")]
        base_cc = {(d, int(rp)): float(a) for d, rp, a in
                zip(cc["dataset"], cc["repeticao"], cc["ae_mean"])}
        columns = list(base.columns) + ["pareado"]
        todo_base = set(zip(base["dataset"], base["repeticao"].astype(int)))
        del base, llp, cc

        done = set()
        if os.path.exists(OUT_CSV):
            o = pd.read_csv(OUT_CSV, usecols=["dataset", "repeticao"])
            done = set(zip(o["dataset"], o["repeticao"].astype(int)))

        datasets = [d for d in E.get_active_dataset_list()
                    if DATASETS_ONLY is None or d in DATASETS_ONLY]
        faltando = [d for d in datasets if not any(k[0] == d for k in todo_base)]
        if faltando:
            print(f"⚠️  Sem linhas no CSV base (rode o script base antes): {faltando}")

        pendentes = sorted(k for k in todo_base - done if k[0] in datasets)
        print(f"Núcleos: {N_JOBS} | Repetições a processar: {len(pendentes)} "
            f"| Já feitas: {len(done)}")

        t_start, n_feitas = time.time(), 0
        for ds in datasets:
            reps = sorted(rp for d, rp in pendentes if d == ds)
            if not reps:
                continue
            print(f"\n📂 {ds}: {len(reps)} repetições")
            dataset = E.qp.datasets.fetch_UCIBinaryDataset(ds, verbose=False)
            train, test = dataset.train_test
            X_all = np.vstack([train.instances, test.instances])
            y_all = np.concatenate([train.labels, test.labels]).astype(int)

            for rp in reps:
                t0 = time.time()
                rows = run_rep(ds, X_all, y_all, rp - 1, moss, base_cfg, base_cc)
                df = pd.DataFrame(rows).reindex(columns=columns)
                df.to_csv(OUT_CSV, mode="a", header=not os.path.exists(OUT_CSV), index=False)

                n_feitas += 1
                media = (time.time() - t_start) / n_feitas
                resta = (len(pendentes) - n_feitas) * media / 3600
                print(f"   ✅ rep {rp}/{E.N_REPETITIONS} ({len(rows)} linhas, "
                    f"pareado={bool(df['pareado'].all())}, {time.time() - t0:.0f}s) "
                    f"| faltam ~{resta:.1f} h")
                gc.collect()

        print("\n🏁 Concluído.")


    if __name__ == "__main__":
        main()
