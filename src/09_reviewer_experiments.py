import importlib.util
import sys
import time
import platform
import multiprocessing
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from config import DATA_DIR, STAGE06_OUT, STAGE07_OUT, STAGE08_OUT, STAGE09_OUT
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    AdaBoostClassifier,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    cohen_kappa_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from scipy import stats

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier

    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")


class RevisionConfig:
    RANDOM_STATE = 42
    TEST_SIZE = 0.2

    BALANCED_DATA_PATH = STAGE07_OUT / "Final_Dataset_Balanced.csv"
    IMBALANCED_DATA_PATH = STAGE06_OUT / "somatic_variant_Cleaned.csv"

    # point at the fixed script so that
    # build_full_feature_matrix picks up the preprocessor.
    # Resolved relative to this file, not cwd, so it works regardless of
    # where the script is launched from. If you rename 08_tda_fuzzy_ensemble.py,
    # update the filename below.
    STEP8_SCRIPT_PATH = Path(__file__).resolve().parent / "08_tda_fuzzy_ensemble.py"

    GENOME_PATH = DATA_DIR / "hg19.fa"
    TARGET_COL = "LABEL_PATHOGENIC"
    LEAKAGE_COLS = ["chr", "pos", "ref", "alt", "CONSENSUS_SCORE", "TIER"]

    USE_FULL_FEATURES = True

    OUTPUT_DIR = STAGE09_OUT

    N_JOBS = 1

    KNORA_K = 11
    MIN_COMPETENCE_THRESHOLD = 0.7
    TDA_N_NEIGHBORS = 75
    SEQUENCE_WINDOW = 75
    FCGR_K_VALUES = [3, 4]
    TDA_HOMOLOGY_DIMS = (0, 1)

    # expected FCGR size after PCA compression.
    FCGR_PCA_COMPONENTS = 20

    # expected TDA feature count after expansion.
    # script 08 produces 10 features × 2 homology dims = 20 TDA features.
    TDA_N_FEATURES_EXPECTED = 20

    # selective prediction parameters (for table)
    SELECTIVE_ABSTAIN_THRESHOLD = 0.60
    AMBIGUOUS_MARGIN = 0.10
    AMBIGUOUS_SAMPLE_WEIGHT = 0.30

    DPI = 300
    SINGLE_COL = 85 / 25.4
    DOUBLE_COL = 170 / 25.4


RevisionConfig.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(RevisionConfig.RANDOM_STATE)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.dpi": RevisionConfig.DPI,
        "savefig.dpi": RevisionConfig.DPI,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# Module loader


def load_step8_module(script_path):
    path = Path(script_path)
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("step8_tfdfe", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules["step8_tfdfe"] = module
    spec.loader.exec_module(module)
    return module


# build_full_feature_matrix


def build_full_feature_matrix(variant_df):
    """Load the fitted stage-08 preprocessor and build the full feature matrix.

    The preprocessor applies PCA(n_components=20) to the FCGR block and
    computes ~20 TDA features per sample over the FCGR/sequence space, so
    n_fcgr = 20 and n_tda ≈ 20. All downstream slicing in run_feature_ablation
    and audit_feature_counts reflects these counts.
    """
    module = load_step8_module(RevisionConfig.STEP8_SCRIPT_PATH)
    if module is None or not hasattr(module, "TFDFEPreprocessor"):
        return None

    preprocessor = module.TFDFEPreprocessor(
        use_fcgr=True, use_tda=True, genome_path=RevisionConfig.GENOME_PATH
    )

    # Patch TDA extractor to use threading backend
    if hasattr(preprocessor, "tda_extractor"):
        import joblib

        def _patched_extract_global_tda(X, n_jobs=-1):
            from sklearn.neighbors import NearestNeighbors
            import multiprocessing as mp

            extractor = preprocessor.tda_extractor
            n_samples = X.shape[0]
            n_neighbors = min(extractor.n_neighbors, n_samples)

            print(
                f"Finding k-NN neighbors (k={n_neighbors}) for {n_samples} "
                "samples in a single optimized pass..."
            )
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
            nbrs.fit(X)
            all_indices = nbrs.kneighbors(X, return_distance=False)

            actual_jobs = mp.cpu_count() if n_jobs == -1 else max(1, n_jobs)
            print(
                f"Extracting TDA features using {actual_jobs} CPU cores (threading)..."
            )
            tda_features_list = joblib.Parallel(
                n_jobs=actual_jobs, backend="threading", verbose=10
            )(
                joblib.delayed(extractor.extract_local_tda)(X, all_indices[i])
                for i in range(n_samples)
            )
            tda_df = pd.DataFrame(tda_features_list)
            return tda_df.values

        preprocessor.tda_extractor.extract_global_tda = _patched_extract_global_tda

    X, y = preprocessor.fit_transform(variant_df)

    n_standard = len(preprocessor.feature_names)
    n_fcgr = len(preprocessor.fcgr_feature_names)  # PCA-compressed
    n_tda = len(preprocessor.tda_feature_names)  # expanded topological block

    print(
        f"\nbuild_full_feature_matrix: n_standard={n_standard}, "
        f"n_fcgr={n_fcgr} (PCA-compressed), n_tda={n_tda}"
    )

    return {
        "X": np.asarray(X),
        "y": np.asarray(y),
        "n_standard": n_standard,
        "n_fcgr": n_fcgr,
        "n_tda": n_tda,
        "feature_names": preprocessor.all_feature_names,
        "preprocessor": preprocessor,
    }


# Tabular fallback builder (unchanged)


def build_tabular_features(variant_df, target_col, leakage_cols, fitted=None):
    df_clean = variant_df.drop(columns=leakage_cols, errors="ignore")
    y = df_clean[target_col].values
    X_tab = df_clean.drop(columns=[target_col], errors="ignore")

    num_cols = X_tab.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X_tab.select_dtypes(include=["object"]).columns.tolist()

    if fitted is None:
        imputer = SimpleImputer(strategy="median")
        X_num = pd.DataFrame(
            imputer.fit_transform(X_tab[num_cols]),
            columns=num_cols,
            index=X_tab.index,
        )
        encoders = {}
        X_cat = X_tab[cat_cols].copy()
        for col in cat_cols:
            encoders[col] = LabelEncoder()
            X_cat[col] = encoders[col].fit_transform(X_cat[col].astype(str))
        X_combined = pd.concat([X_num, X_cat], axis=1)
        feature_names = X_combined.columns.tolist()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_combined)
        fitted = {
            "imputer": imputer,
            "encoders": encoders,
            "scaler": scaler,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "feature_names": feature_names,
        }
        return X_scaled, y, fitted

    X_num = pd.DataFrame(
        fitted["imputer"].transform(X_tab[fitted["num_cols"]]),
        columns=fitted["num_cols"],
        index=X_tab.index,
    )
    X_cat = X_tab[fitted["cat_cols"]].copy()
    for col in fitted["cat_cols"]:
        enc = fitted["encoders"][col]
        known = set(enc.classes_)
        X_cat[col] = (
            X_cat[col].astype(str).apply(lambda v: v if v in known else enc.classes_[0])
        )
        X_cat[col] = enc.transform(X_cat[col])
    X_combined = pd.concat([X_num, X_cat], axis=1)[fitted["feature_names"]]
    X_scaled = fitted["scaler"].transform(X_combined)
    return X_scaled, y, fitted


# Metrics helper (unchanged)


def compute_extended_metrics(y_true, y_pred, y_pred_proba=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "fpr": fpr,
        "fnr": fnr,
        "npv": npv,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if y_pred_proba is not None:
        metrics["auroc"] = roc_auc_score(y_true, y_pred_proba)
        metrics["auprc"] = average_precision_score(y_true, y_pred_proba)
        metrics["brier"] = brier_score_loss(y_true, y_pred_proba)
    return metrics


# Baseline comparison models (unchanged)


def build_comparison_models(random_state):
    models = {}
    models["LogisticRegression"] = LogisticRegression(
        max_iter=2000, C=1.0, n_jobs=RevisionConfig.N_JOBS, random_state=random_state
    )
    models["ExtraTrees"] = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=15,
        max_features="sqrt",
        n_jobs=RevisionConfig.N_JOBS,
        random_state=random_state,
    )
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        max_features="sqrt",
        n_jobs=RevisionConfig.N_JOBS,
        random_state=random_state,
    )
    models["GBDT"] = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=random_state,
    )
    models["HistGradientBoosting"] = HistGradientBoostingClassifier(
        max_iter=200, max_depth=6, learning_rate=0.05, random_state=random_state
    )
    models["AdaBoost"] = AdaBoostClassifier(
        n_estimators=200, learning_rate=0.5, random_state=random_state
    )
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=RevisionConfig.N_JOBS,
            verbosity=0,
            random_state=random_state,
            eval_metric="logloss",
        )
    if LGBM_AVAILABLE:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300,
            num_leaves=63,
            learning_rate=0.05,
            feature_fraction=0.8,
            n_jobs=RevisionConfig.N_JOBS,
            verbose=-1,
            random_state=random_state,
        )
    if CATBOOST_AVAILABLE:
        models["CatBoost"] = CatBoostClassifier(
            iterations=300,
            depth=5,
            learning_rate=0.05,
            verbose=False,
            random_state=random_state,
        )
    return models


def predict_proba_safe(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    score = model.decision_function(X)
    return (score - score.min()) / (score.max() - score.min() + 1e-12)


def run_baseline_comparison(X_train, y_train, X_test, y_test, output_dir):
    models = build_comparison_models(RevisionConfig.RANDOM_STATE)
    import gc

    rows, curve_data, fitted_models = [], {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        proba = predict_proba_safe(model, X_test)
        pred = (proba >= 0.5).astype(int)
        m = compute_extended_metrics(y_test, pred, proba)
        m["model"] = name
        rows.append(m)
        curve_data[name] = (np.array(y_test), proba)
        fitted_models[name] = model
        gc.collect()
    df = pd.DataFrame(rows)
    ordered = [
        "model",
        "accuracy",
        "mcc",
        "precision",
        "recall",
        "specificity",
        "f1",
        "kappa",
        "auroc",
        "auprc",
        "brier",
        "fpr",
        "fnr",
        "npv",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    df = df[[c for c in ordered if c in df.columns]]
    df = df.sort_values("mcc", ascending=False).reset_index(drop=True)
    df.to_csv(output_dir / "baseline_model_comparison.csv", index=False)
    return df, curve_data, fitted_models


# Plotting helpers (unchanged)


def plot_combined_roc_pr(curve_data, output_dir, filename="Fig_All_Models_ROC_PR"):
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(RevisionConfig.DOUBLE_COL, RevisionConfig.DOUBLE_COL / 2.4)
    )
    palette = plt.cm.tab10(np.linspace(0, 1, max(10, len(curve_data))))
    for i, (name, (y_true, proba)) in enumerate(curve_data.items()):
        fpr, tpr, _ = roc_curve(y_true, proba)
        auroc = roc_auc_score(y_true, proba)
        ax1.plot(
            fpr, tpr, linewidth=1.3, color=palette[i], label=f"{name} ({auroc:.3f})"
        )
    ax1.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.6)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("(A) ROC Curves", fontweight="bold", loc="left")
    ax1.legend(loc="best", fontsize=6, frameon=True, edgecolor="black")
    ax1.grid(False)
    for i, (name, (y_true, proba)) in enumerate(curve_data.items()):
        precision, recall, _ = precision_recall_curve(y_true, proba)
        auprc = average_precision_score(y_true, proba)
        ax2.plot(
            recall,
            precision,
            linewidth=1.3,
            color=palette[i],
            label=f"{name} ({auprc:.3f})",
        )
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("(B) Precision-Recall Curves", fontweight="bold", loc="left")
    ax2.legend(loc="lower left", fontsize=6, frameon=True, edgecolor="black")
    ax2.grid(False)
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{filename}.png", dpi=RevisionConfig.DPI, bbox_inches="tight"
    )
    plt.savefig(
        output_dir / f"{filename}.tiff", dpi=RevisionConfig.DPI, bbox_inches="tight"
    )
    plt.close()


# McNemar (unchanged)


def mcnemar_test(y_true, ref_pred, other_pred):
    y_true = np.asarray(y_true)
    ref_correct = ref_pred == y_true
    other_correct = other_pred == y_true
    b = int(np.sum(ref_correct & ~other_correct))
    c = int(np.sum(~ref_correct & other_correct))
    if b + c == 0:
        return np.nan, np.nan
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    if b + c < 25:
        p_value = 2 * min(
            stats.binom.cdf(min(b, c), b + c, 0.5),
            1 - stats.binom.cdf(max(b, c) - 1, b + c, 0.5),
        )
    else:
        p_value = 1 - stats.chi2.cdf(statistic, df=1)
    return statistic, p_value


def run_mcnemar_matrix(fitted_models, X_test, y_test, reference_name, output_dir):
    preds = {}
    for name, model in fitted_models.items():
        proba = predict_proba_safe(model, X_test)
        preds[name] = (proba >= 0.5).astype(int)
    if reference_name not in preds:
        reference_name = max(preds, key=lambda n: matthews_corrcoef(y_test, preds[n]))
    ref_pred = preds[reference_name]
    rows = []
    for name, pred in preds.items():
        if name == reference_name:
            continue
        stat, pval = mcnemar_test(y_test, ref_pred, pred)
        sig = "ns"
        if pval is not None and not np.isnan(pval):
            if pval < 0.001:
                sig = "***"
            elif pval < 0.01:
                sig = "**"
            elif pval < 0.05:
                sig = "*"
        rows.append(
            {
                "reference": reference_name,
                "compared_model": name,
                "mcnemar_statistic": stat,
                "p_value": pval,
                "significance": sig,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "mcnemar_vs_reference.csv", index=False)
    return df


# run_diversity_analysis


def run_diversity_analysis(output_dir):
    """Read the pairwise diversity CSVs and produce a summary report.

    The stage-08 model writes pairwise_q_statistic.csv and
    pairwise_disagreement.csv (via compute_pairwise_diversity) after training.
    This function:
      1. Reads those CSVs (non-blocking: skips if absent).
      2. Computes summary statistics (mean Q-stat, mean disagreement).
      3. Saves diversity_summary.csv and a heatmap figure.
      4. Prints a pass/fail verdict against the target thresholds
         (mean Q < 0.50 and mean disagreement > 0.05).

    Returns a summary dict (or None if the files are not present).
    """
    stage08_out = Path(STAGE08_OUT)
    q_path = stage08_out / "pairwise_q_statistic.csv"
    dis_path = stage08_out / "pairwise_disagreement.csv"

    if not q_path.exists() or not dis_path.exists():
        print(
            "\n  [Diversity Analysis] "
            "pairwise_q_statistic.csv / pairwise_disagreement.csv not found "
            f"in '{stage08_out}/'. Run 08_tda_fuzzy_ensemble.py first to generate them.\n"
            "  Skipping diversity analysis."
        )
        pd.DataFrame(
            [
                {
                    "status": "skipped",
                    "reason": "08 outputs not found",
                }
            ]
        ).to_csv(output_dir / "diversity_summary.csv", index=False)
        return None

    q_df = pd.read_csv(q_path, index_col=0)
    dis_df = pd.read_csv(dis_path, index_col=0)
    names = q_df.index.tolist()
    n = len(names)

    # Upper-triangle values (excluding diagonal)
    iu = np.triu_indices(n, k=1)
    q_vals = q_df.values[iu]
    dis_vals = dis_df.values[iu]

    mean_q = float(np.mean(q_vals))
    mean_dis = float(np.mean(dis_vals))
    max_q = float(np.max(q_vals))
    min_dis = float(np.min(dis_vals))

    q_target_pass = mean_q < 0.50
    dis_target_pass = mean_dis > 0.05

    print("\n  [Diversity Analysis Summary]")
    print(f"    Pool members: {names}")
    print(
        f"    Mean Q-statistic:    {mean_q:.4f}  "
        f"({'PASS' if q_target_pass else 'FAIL'} — target < 0.50)"
    )
    print(f"    Max Q-statistic:     {max_q:.4f}")
    print(
        f"    Mean Disagreement:   {mean_dis:.4f}  "
        f"({'PASS' if dis_target_pass else 'FAIL'} — target > 0.05)"
    )
    print(f"    Min Disagreement:    {min_dis:.4f}")

    summary = {
        "n_models": n,
        "mean_q_statistic": mean_q,
        "max_q_statistic": max_q,
        "mean_disagreement": mean_dis,
        "min_disagreement": min_dis,
        "q_target_pass": q_target_pass,
        "dis_target_pass": dis_target_pass,
        "overall_diversity_ok": q_target_pass and dis_target_pass,
    }
    pd.DataFrame([summary]).to_csv(output_dir / "diversity_summary.csv", index=False)

    # ---- Plot heatmaps ----
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(RevisionConfig.DOUBLE_COL, RevisionConfig.DOUBLE_COL / 1.9),
    )
    sns.heatmap(
        q_df,
        ax=axes[0],
        cmap="RdYlGn_r",
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.2,
        annot_kws={"size": 4},
    )
    axes[0].set_title(
        f"(A) Q-Statistic  (mean={mean_q:.3f})\n"
        f"{'PASS' if q_target_pass else 'FAIL'}: target < 0.50",
        fontweight="bold",
        loc="left",
        fontsize=9,
    )
    plt.setp(
        axes[0].get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=7,
    )
    axes[0].tick_params(axis="y", rotation=0, labelsize=7)

    sns.heatmap(
        dis_df,
        ax=axes[1],
        cmap="Blues",
        vmin=0,
        vmax=0.30,
        annot=True,
        fmt=".2f",
        linewidths=0.2,
        annot_kws={"size": 4},
    )
    axes[1].set_title(
        f"(B) Disagreement Rate  (mean={mean_dis:.3f})\n"
        f"{'PASS' if dis_target_pass else 'FAIL'}: target > 0.05",
        fontweight="bold",
        loc="left",
        fontsize=9,
    )
    plt.setp(
        axes[1].get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=7,
    )
    axes[1].tick_params(axis="y", rotation=0, labelsize=7)

    plt.tight_layout()
    plt.savefig(
        output_dir / "Fig_Diversity_Summary.png",
        dpi=RevisionConfig.DPI,
        bbox_inches="tight",
    )
    plt.savefig(
        output_dir / "Fig_Diversity_Summary.tiff",
        dpi=RevisionConfig.DPI,
        bbox_inches="tight",
    )
    plt.close()
    print("  Saved: diversity_summary.csv, Fig_Diversity_Summary.png")
    return summary


# run_feature_ablation


def run_feature_ablation(
    X_train, y_train, X_test, y_test, n_standard, n_fcgr, n_tda, output_dir
):
    """Feature ablation study.

    n_tda is ~20; the TDA slice is sized accordingly. When
    n_fcgr <= FCGR_PCA_COMPONENTS (default 20) the FCGR block is already
    PCA-compressed, so the redundant inner-PCA loop is skipped.
    """
    std_slice = slice(0, n_standard)
    fcgr_slice = slice(n_standard, n_standard + n_fcgr)
    tda_slice = slice(n_standard + n_fcgr, n_standard + n_fcgr + n_tda)

    def fit_eval(Xtr, Xte, tag):
        model = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=6,
            learning_rate=0.05,
            random_state=RevisionConfig.RANDOM_STATE,
            early_stopping=False,
        )
        model.fit(Xtr, y_train)
        proba = model.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        m = compute_extended_metrics(y_test, pred, proba)
        m["configuration"] = tag
        m["n_features"] = Xtr.shape[1]
        return m, (np.array(y_test), proba)

    configs, curves = {}, {}

    configs["Biological_only"], curves["Biological_only"] = fit_eval(
        X_train[:, std_slice], X_test[:, std_slice], "Biological_only"
    )

    if n_fcgr > 0:
        Xtr = np.hstack([X_train[:, std_slice], X_train[:, fcgr_slice]])
        Xte = np.hstack([X_test[:, std_slice], X_test[:, fcgr_slice]])
        configs["Bio_plus_FCGR_compressed"], curves["Bio_plus_FCGR_compressed"] = (
            fit_eval(Xtr, Xte, "Bio_plus_FCGR_compressed")
        )

    if n_tda > 0:
        # TDA slice is now ~20 features (over FCGR space)
        Xtr = np.hstack([X_train[:, std_slice], X_train[:, tda_slice]])
        Xte = np.hstack([X_test[:, std_slice], X_test[:, tda_slice]])
        configs["Bio_plus_TDA_FCGR"], curves["Bio_plus_TDA_FCGR"] = fit_eval(
            Xtr, Xte, "Bio_plus_TDA_FCGR"
        )

    configs["All_features"], curves["All_features"] = fit_eval(
        X_train, X_test, "All_features"
    )

    # If FCGR is already compressed, test sub-compression
    already_compressed = n_fcgr <= RevisionConfig.FCGR_PCA_COMPONENTS

    if n_fcgr > 0 and not already_compressed:
        for n_comp in [10, 20, 50]:
            n_comp_eff = min(n_comp, n_fcgr)
            pca = PCA(n_components=n_comp_eff, random_state=RevisionConfig.RANDOM_STATE)
            fcgr_tr_pca = pca.fit_transform(X_train[:, fcgr_slice])
            fcgr_te_pca = pca.transform(X_test[:, fcgr_slice])
            evr = float(np.sum(pca.explained_variance_ratio_))
            parts_tr = [X_train[:, std_slice], fcgr_tr_pca]
            parts_te = [X_test[:, std_slice], fcgr_te_pca]
            if n_tda > 0:
                parts_tr.append(X_train[:, tda_slice])
                parts_te.append(X_test[:, tda_slice])
            Xtr = np.hstack(parts_tr)
            Xte = np.hstack(parts_te)
            tag = f"Bio_TDA_FCGR-PCA{n_comp_eff}"
            m, c = fit_eval(Xtr, Xte, tag)
            m["fcgr_pca_explained_variance"] = evr
            configs[tag] = m
            curves[tag] = c

    elif n_fcgr > 0 and already_compressed:
        print(
            f"  FCGR already PCA-compressed ({n_fcgr}D). "
            f"Testing sub-compression to 5 and 10 components..."
        )
        for n_sub in [5, 10]:
            n_sub_eff = min(n_sub, n_fcgr)
            pca = PCA(n_components=n_sub_eff, random_state=RevisionConfig.RANDOM_STATE)
            fcgr_tr_sub = pca.fit_transform(X_train[:, fcgr_slice])
            fcgr_te_sub = pca.transform(X_test[:, fcgr_slice])
            evr = float(np.sum(pca.explained_variance_ratio_))
            parts_tr = [X_train[:, std_slice], fcgr_tr_sub]
            parts_te = [X_test[:, std_slice], fcgr_te_sub]
            if n_tda > 0:
                parts_tr.append(X_train[:, tda_slice])
                parts_te.append(X_test[:, tda_slice])
            Xtr = np.hstack(parts_tr)
            Xte = np.hstack(parts_te)
            tag = f"Bio_TDA_FCGR-sub{n_sub_eff}"
            m, c = fit_eval(Xtr, Xte, tag)
            m["fcgr_pca_explained_variance"] = evr
            configs[tag] = m
            curves[tag] = c

    df = pd.DataFrame(list(configs.values()))
    front = [
        "configuration",
        "n_features",
        "mcc",
        "accuracy",
        "f1",
        "auroc",
        "auprc",
        "fpr",
        "fnr",
    ]
    df = df[
        [c for c in front if c in df.columns]
        + [c for c in df.columns if c not in front]
    ]
    df = df.sort_values("mcc", ascending=False).reset_index(drop=True)
    df.to_csv(output_dir / "fcgr_ablation_results.csv", index=False)
    plot_combined_roc_pr(curves, output_dir, filename="Fig_Ablation_ROC_PR")
    return df


# Threshold sweep (unchanged)


def run_threshold_sweep(model, X_test, y_test, output_dir):
    proba = predict_proba_safe(model, X_test)
    rows = []
    for thr in np.round(np.arange(0.1, 0.95, 0.05), 2):
        pred = (proba >= thr).astype(int)
        m = compute_extended_metrics(y_test, pred, proba)
        m["threshold"] = float(thr)
        rows.append(m)
    df = pd.DataFrame(rows)
    front = [
        "threshold",
        "recall",
        "specificity",
        "precision",
        "fpr",
        "fnr",
        "f1",
        "mcc",
    ]
    df = df[
        [c for c in front if c in df.columns]
        + [c for c in df.columns if c not in front]
    ]
    df.to_csv(output_dir / "threshold_sensitivity_sweep.csv", index=False)
    return df


# Imbalanced evaluation (unchanged)


def run_imbalanced_evaluation(bio_model, n_standard, train_fitted, output_dir):
    path = Path(RevisionConfig.IMBALANCED_DATA_PATH)
    if not path.exists():
        empty = pd.DataFrame(
            [{"status": "imbalanced_source_not_found", "path": str(path)}]
        )
        empty.to_csv(output_dir / "imbalanced_fp_evaluation.csv", index=False)
        return empty
    try:
        sample_head = pd.read_csv(path, nrows=5, low_memory=False)
        all_cols = sample_head.columns.tolist()
    except Exception as e:
        empty = pd.DataFrame([{"status": f"read_error: {e}"}])
        empty.to_csv(output_dir / "imbalanced_fp_evaluation.csv", index=False)
        return empty

    print("  Reading imbalanced dataset in chunks...")
    pathogenic_chunks, benign_chunks = [], []
    chunk_size = 20000
    max_benign_retained = 500000
    n_benign_retained = 0

    def _downcast(frame):
        for c in frame.select_dtypes(include=["float64"]).columns:
            frame[c] = pd.to_numeric(frame[c], downcast="float")
        for c in frame.select_dtypes(include=["int64"]).columns:
            frame[c] = pd.to_numeric(frame[c], downcast="integer")
        return frame

    reader = pd.read_csv(path, chunksize=chunk_size, low_memory=False, memory_map=True)
    for chunk in reader:
        if RevisionConfig.TARGET_COL not in chunk.columns:
            continue
        chunk = _downcast(chunk)
        path_rows = chunk[chunk[RevisionConfig.TARGET_COL] == 1]
        ben_rows = chunk[chunk[RevisionConfig.TARGET_COL] == 0]
        if len(path_rows) > 0:
            pathogenic_chunks.append(path_rows)
        if len(ben_rows) > 0 and n_benign_retained < max_benign_retained:
            benign_chunks.append(ben_rows)
            n_benign_retained += len(ben_rows)

    if not pathogenic_chunks or not benign_chunks:
        empty = pd.DataFrame([{"status": "no_pathogenic_or_benign_found"}])
        empty.to_csv(output_dir / "imbalanced_fp_evaluation.csv", index=False)
        return empty

    pathogenic = pd.concat(pathogenic_chunks, ignore_index=True)
    benign = pd.concat(benign_chunks, ignore_index=True)
    del pathogenic_chunks, benign_chunks

    n_path_test = max(1, int(len(pathogenic) * RevisionConfig.TEST_SIZE))
    path_test = pathogenic.sample(
        n=min(n_path_test, len(pathogenic)), random_state=RevisionConfig.RANDOM_STATE
    )
    del pathogenic

    benign_test = benign.sample(
        n=min(len(benign), n_path_test * 50), random_state=RevisionConfig.RANDOM_STATE
    )
    del benign

    imb_test = (
        pd.concat([path_test, benign_test])
        .sample(frac=1, random_state=RevisionConfig.RANDOM_STATE)
        .reset_index(drop=True)
    )
    del path_test, benign_test

    X_imb, y_imb, _ = build_tabular_features(
        imb_test,
        RevisionConfig.TARGET_COL,
        RevisionConfig.LEAKAGE_COLS,
        fitted=train_fitted,
    )
    del imb_test
    X_imb = X_imb[:, :n_standard]

    rows = []
    proba = bio_model.predict_proba(X_imb)[:, 1]
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
        pred = (proba >= thr).astype(int)
        m = compute_extended_metrics(y_imb, pred, proba)
        m["threshold"] = thr
        m["n_test"] = len(y_imb)
        m["benign_pathogenic_ratio"] = (
            f"{int((y_imb == 0).sum())}:{int((y_imb == 1).sum())}"
        )
        rows.append(m)

    df = pd.DataFrame(rows)
    front = [
        "threshold",
        "benign_pathogenic_ratio",
        "n_test",
        "fpr",
        "recall",
        "specificity",
        "precision",
        "mcc",
        "auroc",
        "auprc",
    ]
    df = df[
        [c for c in front if c in df.columns]
        + [c for c in df.columns if c not in front]
    ]
    df.to_csv(output_dir / "imbalanced_fp_evaluation.csv", index=False)
    return df


# audit_feature_counts


def audit_feature_counts(n_standard, n_fcgr, n_tda, output_dir):
    """Audit feature counts.

    The expected TDA count is TDA_N_FEATURES_EXPECTED (20): 10 descriptors x
    2 homology dimensions over the FCGR space. The expected FCGR count is
    FCGR_PCA_COMPONENTS (20).
    """
    expected_fcgr = RevisionConfig.FCGR_PCA_COMPONENTS
    expected_tda = RevisionConfig.TDA_N_FEATURES_EXPECTED
    rows = [
        {
            "category": "Standard/Biological",
            "actual": n_standard,
            "expected": n_standard,
            "note": "18 tabular biological/annotation features",
        },
        {
            "category": "Fractal (FCGR) [PCA-compressed]",
            "actual": n_fcgr,
            "expected": expected_fcgr,
            "note": f"PCA({expected_fcgr}) of {sum(4**k for k in RevisionConfig.FCGR_K_VALUES)} raw FCGR dims (FIX 3)",
        },
        {
            "category": "Topological (TDA) [over FCGR space]",
            "actual": n_tda,
            "expected": expected_tda,
            "note": "10 descriptors × 2 homology dims, computed over FCGR repr (CHANGE 2)",
        },
        {
            "category": "Total",
            "actual": n_standard + n_fcgr + n_tda,
            "expected": n_standard + expected_fcgr + expected_tda,
            "note": "",
        },
    ]
    df = pd.DataFrame(rows)
    df["match"] = df["actual"] == df["expected"]
    df.to_csv(output_dir / "feature_count_audit.csv", index=False)
    print("\nFeature count audit:")
    print(df[["category", "actual", "expected", "match"]].to_string(index=False))
    return df


# Consensus score verification (unchanged)


def verify_consensus_score(variant_df, output_dir):
    needed = ["REVEL_score", "SIFT_score", "Polyphen2_HDIV_score"]
    rows = []
    if (
        all(c in variant_df.columns for c in needed)
        and "CONSENSUS_SCORE" in variant_df.columns
    ):
        sub = variant_df[needed + ["CONSENSUS_SCORE"]].dropna().head(100000).copy()
        recomputed = (
            sub["REVEL_score"].fillna(0.5) * 0.6
            + (1 - sub["SIFT_score"].fillna(0.5)) * 0.2
            + sub["Polyphen2_HDIV_score"].fillna(0.5) * 0.2
        )
        diff = np.abs(recomputed - sub["CONSENSUS_SCORE"])
        rows.append(
            {
                "formula": "0.6*REVEL + 0.2*(1-SIFT) + 0.2*PolyPhen2",
                "n_checked": int(len(sub)),
                "mean_abs_diff": float(diff.mean()),
                "max_abs_diff": float(diff.max()),
                "sift_inverted": True,
            }
        )
    else:
        rows.append(
            {
                "formula": "0.6*REVEL + 0.2*(1-SIFT) + 0.2*PolyPhen2",
                "n_checked": 0,
                "mean_abs_diff": None,
                "max_abs_diff": None,
                "sift_inverted": True,
                "note": "CONSENSUS_SCORE or component columns not present",
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "consensus_score_verification.csv", index=False)
    return df


# Overfitting analysis (unchanged)


def run_overfitting_analysis(X, y, n_standard, output_dir):
    train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9]
    rows = []
    for ts in train_sizes:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X[:, :n_standard],
            y,
            train_size=ts,
            stratify=y,
            random_state=RevisionConfig.RANDOM_STATE,
        )
        model = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=6,
            learning_rate=0.05,
            random_state=RevisionConfig.RANDOM_STATE,
            early_stopping=False,
        )
        model.fit(X_tr, y_tr)
        train_pred = model.predict(X_tr)
        test_pred = model.predict(X_te)
        rows.append(
            {
                "train_fraction": ts,
                "n_train": len(y_tr),
                "train_mcc": matthews_corrcoef(y_tr, train_pred),
                "test_mcc": matthews_corrcoef(y_te, test_pred),
                "train_test_gap": matthews_corrcoef(y_tr, train_pred)
                - matthews_corrcoef(y_te, test_pred),
                "train_accuracy": accuracy_score(y_tr, train_pred),
                "test_accuracy": accuracy_score(y_te, test_pred),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "overfitting_learning_curve.csv", index=False)

    fig, ax = plt.subplots(
        figsize=(RevisionConfig.SINGLE_COL * 1.4, RevisionConfig.SINGLE_COL)
    )
    ax.plot(df["n_train"], df["train_mcc"], "o-", color="#0077BB", label="Train MCC")
    ax.plot(df["n_train"], df["test_mcc"], "s--", color="#CC3311", label="Test MCC")
    ax.fill_between(
        df["n_train"], df["test_mcc"], df["train_mcc"], alpha=0.15, color="gray"
    )
    ax.set_xlabel("Number of training samples")
    ax.set_ylabel("MCC")
    ax.set_title("Train vs Test MCC (overfitting check)", fontweight="bold", loc="left")
    ax.legend(frameon=True, edgecolor="black")
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(
        output_dir / "Fig_Overfitting_Check.png",
        dpi=RevisionConfig.DPI,
        bbox_inches="tight",
    )
    plt.savefig(
        output_dir / "Fig_Overfitting_Check.tiff",
        dpi=RevisionConfig.DPI,
        bbox_inches="tight",
    )
    plt.close()
    return df


# Parameter sensitivity (unchanged)


def run_parameter_sensitivity(
    X_train, y_train, X_test, y_test, n_standard, n_fcgr, n_tda, output_dir
):
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        return pd.DataFrame()

    base = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=RevisionConfig.RANDOM_STATE,
        early_stopping=False,
    )
    base.fit(X_train, y_train)
    proba = base.predict_proba(X_test)[:, 1]

    rows = []
    for k in [5, 7, 9, 11, 15, 21]:
        for min_comp in [0.5, 0.6, 0.7, 0.8]:
            nn = NearestNeighbors(n_neighbors=k, n_jobs=RevisionConfig.N_JOBS)
            nn.fit(X_train)
            _, idx = nn.kneighbors(X_test)
            neighbor_labels = y_train[idx]
            local_purity = np.mean(neighbor_labels == y_test[:, None], axis=1)
            confident = local_purity >= min_comp
            adj_pred = (proba >= 0.5).astype(int)
            m = compute_extended_metrics(y_test, adj_pred, proba)
            rows.append(
                {
                    "knora_k": k,
                    "min_competence": min_comp,
                    "fraction_confident_neighborhoods": float(np.mean(confident)),
                    "mcc": m["mcc"],
                    "f1": m["f1"],
                    "auprc": m["auprc"],
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "parameter_sensitivity.csv", index=False)

    pivot = df.pivot(
        index="knora_k",
        columns="min_competence",
        values="fraction_confident_neighborhoods",
    )
    fig, ax = plt.subplots(
        figsize=(RevisionConfig.SINGLE_COL * 1.3, RevisionConfig.SINGLE_COL)
    )
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("min_competence")
    ax.set_ylabel("knora_k")
    ax.set_title("Confident neighborhood fraction", fontweight="bold", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(
        output_dir / "Fig_Parameter_Sensitivity.png",
        dpi=RevisionConfig.DPI,
        bbox_inches="tight",
    )
    plt.close()
    return df


# Best-params table


def save_best_params_table(output_dir):
    rows = [
        # Core KNORA / neighbourhood parameters
        {
            "hyperparameter": "knora_k",
            "value": RevisionConfig.KNORA_K,
            "change": "FIX 3",
        },
        {
            "hyperparameter": "min_competence",
            "value": RevisionConfig.MIN_COMPETENCE_THRESHOLD,
            "change": "",
        },
        {
            "hyperparameter": "tda_neighbors",
            "value": RevisionConfig.TDA_N_NEIGHBORS,
            "change": "",
        },
        {
            "hyperparameter": "sequence_window",
            "value": RevisionConfig.SEQUENCE_WINDOW,
            "change": "",
        },
        {
            "hyperparameter": "fcgr_k_values",
            "value": str(RevisionConfig.FCGR_K_VALUES),
            "change": "",
        },
        {
            "hyperparameter": "tda_homology_dims",
            "value": str(RevisionConfig.TDA_HOMOLOGY_DIMS),
            "change": "",
        },
        {
            "hyperparameter": "test_size",
            "value": RevisionConfig.TEST_SIZE,
            "change": "",
        },
        {
            "hyperparameter": "random_state",
            "value": RevisionConfig.RANDOM_STATE,
            "change": "",
        },
        # Feature sizes
        {
            "hyperparameter": "n_fcgr_features_raw",
            "value": sum(4**k for k in RevisionConfig.FCGR_K_VALUES),
            "change": "FIX 3",
        },
        {
            "hyperparameter": "n_fcgr_features_pca",
            "value": RevisionConfig.FCGR_PCA_COMPONENTS,
            "change": "FIX 3",
        },
        {
            "hyperparameter": "n_tda_features",
            "value": RevisionConfig.TDA_N_FEATURES_EXPECTED,
            "change": "CHANGE 2 (was 6)",
        },
        {"hyperparameter": "weak_learner_margin", "value": 0.10, "change": "FIX 2"},
        {"hyperparameter": "threshold_search_low", "value": 0.45, "change": "FIX 5"},
        {"hyperparameter": "threshold_search_high", "value": 0.65, "change": "FIX 5"},
        # diversity pool
        {
            "hyperparameter": "diversity_pool_additions",
            "value": "MLP_BioTDA, kNN_SML, ElasticNet_Bio",
            "change": "CHANGE 1",
        },
        # label ceiling parameters
        {
            "hyperparameter": "selective_abstain_threshold",
            "value": RevisionConfig.SELECTIVE_ABSTAIN_THRESHOLD,
            "change": "CHANGE 3",
        },
        {
            "hyperparameter": "ambiguous_margin",
            "value": RevisionConfig.AMBIGUOUS_MARGIN,
            "change": "CHANGE 3",
        },
        {
            "hyperparameter": "ambiguous_sample_weight",
            "value": RevisionConfig.AMBIGUOUS_SAMPLE_WEIGHT,
            "change": "CHANGE 3",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "best_hyperparameters.csv", index=False)
    return df


# Runtime profile (unchanged)


def save_runtime_profile(output_dir, timings, n_samples, n_features):
    try:
        import psutil

        mem_gb = round(psutil.virtual_memory().total / (1024**3), 2)
    except ImportError:
        mem_gb = None
    rows = [
        {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": multiprocessing.cpu_count(),
            "total_memory_gb": mem_gb,
            "n_samples": n_samples,
            "n_features": n_features,
        }
    ]
    for step, secs in timings.items():
        rows[0][f"time_{step}_sec"] = round(secs, 2)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "computational_requirements.csv", index=False)
    return df


# main —


def run_lime_analysis(model, X_train, X_test, y_test, output_dir, feature_names=None):
    try:
        import lime
        import lime.lime_tabular
    except ImportError:
        print("LIME package not installed. Skipping LIME analysis.")
        return

    print("\n--- Running LIME Analysis on 4 Random Samples ---")

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        mode="classification",
        feature_names=feature_names,
        class_names=["Benign", "Pathogenic"],
        random_state=42,
    )

    import numpy as np

    np.random.seed(42)
    sample_indices = np.random.choice(len(X_test), 4, replace=False)

    y_pred = (model.predict_proba(X_test[sample_indices])[:, 1] >= 0.5).astype(int)
    y_test_arr = np.array(y_test)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for i, idx in enumerate(sample_indices):
        true_label = "Pathogenic" if y_test_arr[idx] == 1 else "Benign"
        pred_label = "Pathogenic" if y_pred[i] == 1 else "Benign"
        case_name = f"Sample_{idx}_True_{true_label}_Pred_{pred_label}"

        print(f"  Generating LIME explanation for {case_name}")
        exp = explainer.explain_instance(
            X_test[idx], model.predict_proba, num_features=10
        )

        # Save as HTML
        exp.save_to_file(str(output_dir / f"lime_explanation_{case_name}.html"))

        # Draw on subplot
        ax = axes[i]
        exp_list = exp.as_list()
        vals = [x[1] for x in exp_list]
        names = [x[0] for x in exp_list]
        vals.reverse()
        names.reverse()
        colors = ["green" if x > 0 else "red" for x in vals]
        pos = np.arange(len(exp_list)) + 0.5

        ax.barh(pos, vals, align="center", color=colors)
        ax.set_yticks(pos)
        ax.set_yticklabels(names)
        ax.set_title(
            f"Sample {idx}: True={true_label}, Pred={pred_label}", fontweight="bold"
        )
        ax.axvline(x=0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_dir / "Fig_LIME_Combined_2x2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    out = RevisionConfig.OUTPUT_DIR
    timings = {}
    t_start = time.time()

    save_best_params_table(out)

    balanced_path = Path(RevisionConfig.BALANCED_DATA_PATH)
    if not balanced_path.exists():
        raise FileNotFoundError(
            f"Balanced dataset not found at {balanced_path}. "
            "Set RevisionConfig.BALANCED_DATA_PATH to the Final_Dataset.csv location."
        )

    variant_df = pd.read_csv(balanced_path, low_memory=False)

    full = None
    if RevisionConfig.USE_FULL_FEATURES:
        full = build_full_feature_matrix(variant_df)

    if full is not None:
        X = full["X"]
        y = full["y"]
        n_standard = full["n_standard"]
        n_fcgr = full["n_fcgr"]
        n_tda = full["n_tda"]
        tab_fitted = None
        print(
            f"Using full feature matrix: {X.shape[1]} features "
            f"(standard={n_standard}, fcgr_pca={n_fcgr}, tda={n_tda})"
        )
    else:
        X, y, tab_fitted = build_tabular_features(
            variant_df, RevisionConfig.TARGET_COL, RevisionConfig.LEAKAGE_COLS
        )
        n_standard = X.shape[1]
        n_fcgr = 0
        n_tda = 0
        print(
            f"Using tabular feature matrix only: {X.shape[1]} features. "
            "Full FCGR/TDA ablation requires Step 8 preprocessor and genome file."
        )

    X = np.asarray(X, dtype=np.float32)
    n_total_features = X.shape[1]
    print(f"X dtype forced to float32. Memory: {X.nbytes / 1e9:.2f} GB")

    audit_feature_counts(n_standard, n_fcgr, n_tda, out)
    verify_consensus_score(variant_df, out)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=RevisionConfig.TEST_SIZE,
        stratify=y,
        random_state=RevisionConfig.RANDOM_STATE,
    )

    X_train_bio = X_train[:, :n_standard]
    X_test_bio = X_test[:, :n_standard]

    # Diversity analysis — read script 08 output CSVs
    print("\nStep: Diversity Analysis")
    t0 = time.time()
    diversity_summary = run_diversity_analysis(out)
    timings["diversity_analysis"] = time.time() - t0

    t0 = time.time()
    comparison_df, curve_data, fitted_models = run_baseline_comparison(
        X_train_bio, y_train, X_test_bio, y_test, out
    )
    timings["baseline_comparison"] = time.time() - t0

    plot_combined_roc_pr(curve_data, out)
    run_mcnemar_matrix(fitted_models, X_test_bio, y_test, "XGBoost", out)

    if "XGBoost" in fitted_models:
        run_lime_analysis(
            fitted_models["XGBoost"],
            X_train_bio,
            X_test_bio,
            y_test,
            out,
            feature_names=(
                full["feature_names"][:n_standard]
                if full is not None
                else tab_fitted["feature_names"]
            ),
        )

    import gc

    t0 = time.time()
    run_overfitting_analysis(X, y, n_standard, out)
    timings["overfitting_analysis"] = time.time() - t0

    del X
    gc.collect()

    t0 = time.time()
    run_parameter_sensitivity(
        X_train, y_train, X_test, y_test, n_standard, n_fcgr, n_tda, out
    )
    timings["parameter_sensitivity"] = time.time() - t0

    t0 = time.time()
    run_feature_ablation(
        X_train, y_train, X_test, y_test, n_standard, n_fcgr, n_tda, out
    )
    timings["feature_ablation"] = time.time() - t0

    best_name = max(
        fitted_models,
        key=lambda n: matthews_corrcoef(
            y_test,
            (predict_proba_safe(fitted_models[n], X_test_bio) >= 0.5).astype(int),
        ),
    )
    run_threshold_sweep(fitted_models[best_name], X_test_bio, y_test, out)

    bio_model = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=RevisionConfig.RANDOM_STATE,
        early_stopping=False,
    )
    bio_model.fit(X_train_bio, y_train)

    if tab_fitted is None:
        _, _, tab_fitted = build_tabular_features(
            variant_df, RevisionConfig.TARGET_COL, RevisionConfig.LEAKAGE_COLS
        )

    try:
        run_imbalanced_evaluation(bio_model, n_standard, tab_fitted, out)
    except Exception as e:
        print(f"WARNING: Imbalanced evaluation failed: {e}")
        pd.DataFrame([{"status": f"failed: {e}"}]).to_csv(
            out / "imbalanced_fp_evaluation.csv", index=False
        )

    timings["total"] = time.time() - t_start
    save_runtime_profile(out, timings, len(y), n_total_features)

    print("Revision experiments complete. Outputs saved to:", out.resolve())
    print("\nBaseline comparison (sorted by MCC):")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
