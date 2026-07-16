from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from config import (
    STAGE01_OUT,
    STAGE04_OUT,
    STAGE06_OUT,
    STAGE07_OUT,
    STAGE09_OUT,
    STAGE10_OUT,
    STAGE12_OUT,
    STAGE13_OUT,
    OUTPUT_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stage13_figures")

FIG_DIR = STAGE13_OUT
FIG_DIR.mkdir(parents=True, exist_ok=True)

# PNG only (no pdf/svg), no caption files.
SAVE_FORMATS = ("png",)
DPI = 300

LABEL_COL = "LABEL_PATHOGENIC"

# Colorblind-safe (Okabe-Ito subset)
C_BENIGN = "#0072B2"  # blue
C_PATH = "#D55E00"    # vermillion
MODEL_COLORS = {
    "LightGBM": "#009E73",
    "CrossAttention": "#CC79A7",
    "TF-DFE": "#E69F00",
    "cross_attention": "#CC79A7",
    "lightgbm": "#009E73",
    "tf_dfe": "#E69F00",
}

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "lines.linewidth": 1.8,
    }
)

# ---------------------------------------------------------------------------
# Paths (auto-detected)
# ---------------------------------------------------------------------------
ENRICHED_CSV = STAGE09_OUT / "Final_Dataset_Balanced_with_ESM_Score.csv"
EMB_NPY = STAGE09_OUT / "esm_residue_embeddings.npy"
STAGE01_CSV = STAGE01_OUT / "somatic_variant_dbNSFP.csv"
STAGE01_META = STAGE01_OUT / "somatic_variant_dbNSFP.json"
RESULTS_JSON = STAGE10_OUT / "results.json"
EXT_JSON = STAGE12_OUT / "external_validation.json"
OOF_NPZ = STAGE10_OUT / "oof_predictions.npz"
EXT_PRED_NPZ = STAGE12_OUT / "external_predictions.npz"
SHAP_NPZ = STAGE10_OUT / "shap_values_bio_full.npz"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save(fig, name: str):
    """Save a figure as PNG only. No captions, no pdf/svg."""
    for fmt in SAVE_FORMATS:
        fig.savefig(FIG_DIR / f"{name}.{fmt}", format=fmt)
    plt.close(fig)
    logger.info("  saved %s (%s)", name, ", ".join(SAVE_FORMATS))


def _exists(p: Path, fig: str) -> bool:
    if Path(p).exists():
        return True
    logger.warning("SKIP %s: missing %s", fig, p)
    return False


def _guard(fn):
    try:
        fn()
    except Exception as e:  # never abort the whole render
        logger.warning("Figure %s failed: %s", fn.__name__, e)


def _load_enriched():
    df = pd.read_csv(ENRICHED_CSV)
    return df


def _third_model_key(z, tag):
    """Detect an optional TF-DFE OOF vector if the user dropped one in."""
    for k in (f"{tag}__tf_dfe", f"{tag}__tfdfe", f"{tag}__TF-DFE"):
        if k in z.files:
            return k
    return None


# ===========================================================================
# Figure 1 - Label evidence distribution (stacked bar)
# ===========================================================================
def fig1_label_evidence():
    fig_name = "fig01_label_evidence"
    if _exists(STAGE01_CSV, fig_name) is False:
        # fall back to metadata json counts
        if not _exists(STAGE01_META, fig_name):
            return
        meta = json.loads(Path(STAGE01_META).read_text())
        srcs = meta.get("label_sources", {})
        if not srcs:
            logger.warning("SKIP %s: no label_sources in metadata", fig_name)
            return
        s = pd.Series(srcs)
        # map source -> class by naming convention
        benign_like = s[s.index.str.contains("benign|gnomad|common", case=False)]
        path_like = s[~s.index.isin(benign_like.index)]
        cats = sorted(set(s.index))
        b = [benign_like.get(c, 0) for c in cats]
        p = [path_like.get(c, 0) for c in cats]
    else:
        df = pd.read_csv(
            STAGE01_CSV, usecols=lambda c: c in ("EVIDENCE_SOURCE", LABEL_COL)
        )
        if "EVIDENCE_SOURCE" not in df.columns:
            logger.warning("SKIP %s: no EVIDENCE_SOURCE column", fig_name)
            return
        tab = df.groupby(["EVIDENCE_SOURCE", LABEL_COL]).size().unstack(fill_value=0)
        cats = list(tab.index)
        b = tab.get(0, pd.Series(0, index=cats)).reindex(cats).values
        p = tab.get(1, pd.Series(0, index=cats)).reindex(cats).values

    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(cats))
    ax.bar(x, b, label="Benign", color=C_BENIGN)
    ax.bar(x, p, bottom=b, label="Pathogenic", color=C_PATH)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=35, ha="right")
    ax.set_ylabel("Number of variants")
    ax.set_title("Label evidence distribution")
    ax.legend()
    _save(fig, fig_name)


# ===========================================================================
# Figure 2 - Feature correlation heatmap (lower triangle)
# ===========================================================================
def fig2_correlation():
    fig_name = "fig02_feature_correlation"
    if not _exists(ENRICHED_CSV, fig_name):
        return
    df = _load_enriched()
    want = [
        "SASA",
        "RELATIVE_SASA",
        "PLDDT_SCORE",
        "esm_variant_score",
        "REVEL_score",
        "CADD_phred",
        "SIFT_score",
        "Polyphen2_HDIV_score",
        "GERP++_RS",
        "phyloP100way_vertebrate",
        LABEL_COL,
    ]
    cols = [c for c in want if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(cols) < 3:
        logger.warning("SKIP %s: <3 usable numeric columns", fig_name)
        return
    corr = df[cols].corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    cm = corr.copy()
    cm.values[mask] = np.nan

    fig, ax = plt.subplots(figsize=(7.2, 6))
    im = ax.imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            if not np.isnan(cm.values[i, j]):
                ax.text(
                    j,
                    i,
                    f"{cm.values[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    ax.set_title("Feature correlation (structural / ESM / predictors)")
    _save(fig, fig_name)
# ===========================================================================
# Figure 3 - Structural feature distributions (violin by class)
# ===========================================================================
def fig3_structural_violin():
    fig_name = "fig03_structural_distributions"
    if not _exists(ENRICHED_CSV, fig_name):
        return
    df = _load_enriched()
    feats = [c for c in ("SASA", "PLDDT_SCORE", "RELATIVE_SASA") if c in df.columns]
    feats = feats[:2] if len(feats) >= 2 else feats
    if not feats or LABEL_COL not in df.columns:
        logger.warning("SKIP %s: structural cols/label missing", fig_name)
        return

    fig, axes = plt.subplots(
        1, len(feats), figsize=(4.5 * len(feats), 4.2), squeeze=False
    )
    for ax, f in zip(axes[0], feats):
        d = df[[f, LABEL_COL]].replace(-1, np.nan).dropna()
        groups = [d[d[LABEL_COL] == 0][f].values, d[d[LABEL_COL] == 1][f].values]
        parts = ax.violinplot(groups, showmeans=True, showextrema=False)
        for pc, col in zip(parts["bodies"], (C_BENIGN, C_PATH)):
            pc.set_facecolor(col)
            pc.set_alpha(0.7)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Benign", "Pathogenic"])
        ax.set_ylabel(f)
        ax.set_title(f)
    fig.suptitle("Structural feature distributions by class", y=1.02)
    _save(fig, fig_name)


# ===========================================================================
# Figure 4 - UMAP / t-SNE of ESM-2 embeddings
# ===========================================================================
def fig4_embedding_projection():
    fig_name = "fig04_esm_embedding_projection"
    if not (_exists(EMB_NPY, fig_name) and _exists(ENRICHED_CSV, fig_name)):
        return
    E = np.load(EMB_NPY)
    df = _load_enriched()
    if len(E) != len(df):
        logger.warning("SKIP %s: emb/CSV length mismatch", fig_name)
        return
    keep = np.any(E != 0, axis=1)
    E, y = E[keep], df.loc[keep, LABEL_COL].to_numpy()
    # subsample for speed/legibility
    rng = np.random.RandomState(42)
    if len(E) > 8000:
        idx = rng.choice(len(E), 8000, replace=False)
        E, y = E[idx], y[idx]

    method = "UMAP"
    try:
        import umap
        proj = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42).fit_transform(E)
    except Exception:
        from sklearn.manifold import TSNE
        method = "t-SNE"
        proj = TSNE(
            n_components=2, init="pca", perplexity=30, random_state=42
        ).fit_transform(E)

    fig, ax = plt.subplots(figsize=(6, 5.4))
    for cls, col, lab in [(0, C_BENIGN, "Benign"), (1, C_PATH, "Pathogenic")]:
        m = y == cls
        ax.scatter(
            proj[m, 0], proj[m, 1], s=6, c=col, alpha=0.45, label=lab, edgecolors="none"
        )
    ax.set_xlabel(f"{method}-1")
    ax.set_ylabel(f"{method}-2")
    ax.set_title(f"ESM-2 embedding projection ({method})")
    ax.legend(markerscale=2)
    _save(fig, fig_name)


# ---------------------------------------------------------------------------
# OOF loader (internal, Stage 11 -> STAGE10_OUT/oof_predictions.npz)
# ---------------------------------------------------------------------------
def _load_oof(tag="bio_full"):
    if not _exists(OOF_NPZ, "ROC/PR"):
        return None
    z = np.load(OOF_NPZ, allow_pickle=True)
    ykey = f"{tag}__y"
    if ykey not in z.files:
        logger.warning("SKIP ROC/PR: %s not in OOF store", ykey)
        return None
    y = z[ykey]
    series = {}
    for name, disp in [("lightgbm", "LightGBM"), ("cross_attention", "CrossAttention")]:
        k = f"{tag}__{name}"
        if k in z.files:
            series[disp] = z[k]
    tf = _third_model_key(z, tag)
    if tf:
        series["TF-DFE"] = z[tf]
    return y, series


# ===========================================================================
# Figure 5 - ROC curves (internal OOF)
# ===========================================================================
def fig5_roc():
    fig_name = "fig05_roc_curves"
    loaded = _load_oof()
    if loaded is None:
        return
    from sklearn.metrics import roc_curve, roc_auc_score
    y, series = loaded
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for disp, prob in series.items():
        fpr, tpr, _ = roc_curve(y, prob)
        auc = roc_auc_score(y, prob)
        ax.plot(
            fpr,
            tpr,
            color=MODEL_COLORS.get(disp, None),
            label=f"{disp} (AUC={auc:.3f})",
        )
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC comparison (gene-level OOF)")
    ax.legend(loc="lower right")
    _save(fig, fig_name)


    # ===========================================================================
# Figure 6 - Precision-Recall (internal OOF)
# ===========================================================================
def fig6_pr():
    fig_name = "fig06_precision_recall"
    loaded = _load_oof()
    if loaded is None:
        return
    from sklearn.metrics import precision_recall_curve, average_precision_score
    y, series = loaded
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for disp, prob in series.items():
        prec, rec, _ = precision_recall_curve(y, prob)
        ap = average_precision_score(y, prob)
        ax.plot(
            rec, prec, color=MODEL_COLORS.get(disp, None), label=f"{disp} (AP={ap:.3f})"
        )
        # operating point closest to recall = 0.90
        ok = rec >= 0.90
        if ok.any():
            j = np.argmin(np.abs(rec - 0.90))
            ax.scatter(
                [rec[j]],
                [prec[j]],
                color=MODEL_COLORS.get(disp, None),
                s=40,
                zorder=5,
                edgecolors="black",
                linewidths=0.6,
            )
    ax.axvline(0.90, ls=":", color="gray", lw=1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall (operating point: recall $\\geq$ 0.90)")
    ax.legend(loc="lower left")
    _save(fig, fig_name)


# ---------------------------------------------------------------------------
# Confusion-matrix helper (shared by internal fig7 and external fig7b)
# ---------------------------------------------------------------------------
def _confusion_from_json(results_path, fig_name, suptitle):
    if not _exists(results_path, fig_name):
        return
    res = json.loads(Path(results_path).read_text())
    prim = res.get("primary", {})
    panels = []
    for key, disp in [("lightgbm", "LightGBM"), ("cross_attention", "CrossAttention")]:
        m = prim.get(key)
        if m and all(k in m for k in ("tp", "tn", "fp", "fn")):
            panels.append((disp, m))
    if "tf_dfe" in prim:
        panels.append(("TF-DFE", prim["tf_dfe"]))
    if not panels:
        logger.warning("SKIP %s: no confusion counts in %s", fig_name, results_path.name)
        return

    fig, axes = plt.subplots(
        1, len(panels), figsize=(4.2 * len(panels), 3.8), squeeze=False
    )
    for ax, (disp, m) in zip(axes[0], panels):
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:,}",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=11,
                    fontweight="bold",
                )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pred B", "Pred P"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["True B", "True P"])
        ax.set_title(f"{disp}\nMCC={m.get('mcc','?')}")
    fig.suptitle(suptitle, y=1.03)
    _save(fig, fig_name)


# ===========================================================================
# Figure 7 - Comparative confusion matrices (internal)
# ===========================================================================
def fig7_confusion():
    _confusion_from_json(
        RESULTS_JSON,
        "fig07_confusion_matrices",
        "Confusion matrices (gene-level OOF, recall-aware threshold)",
    )


# ===========================================================================
# Figure 8 - SHAP feature importance
# ===========================================================================
def fig8_shap():
    fig_name = "fig08_shap_importance"
    if not _exists(SHAP_NPZ, fig_name):
        return
    z = np.load(SHAP_NPZ, allow_pickle=True)
    sv, feats = z["shap"], list(z["features"])
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:20][::-1]
    names = [feats[i] for i in order]
    vals = mean_abs[order]

    fig, ax = plt.subplots(figsize=(6.4, 6))
    colors = [
        C_PATH if any(k in n for k in ("esm", "SASA", "PLDDT", "plddt")) else "#888888"
        for n in names
    ]
    ax.barh(range(len(names)), vals, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global SHAP feature importance (top 20)")
    _save(fig, fig_name)


# ---------------------------------------------------------------------------
# Type-1 circularity helper (shared by internal fig9 and external fig9b)
# ---------------------------------------------------------------------------
def _type1_from_json(results_path, fig_name, suptitle):
    if not _exists(results_path, fig_name):
        return
    res = json.loads(Path(results_path).read_text())
    prim, audit = res.get("primary", {}), res.get("leakage_audit", {})
    metrics = ["mcc", "auroc", "auprc"]
    models = [("lightgbm", "LightGBM"), ("cross_attention", "CrossAttention")]
    if "tf_dfe" in prim and "tf_dfe" in audit:
        models.append(("tf_dfe", "TF-DFE"))

    fig, axes = plt.subplots(
        1, len(metrics), figsize=(4.2 * len(metrics), 4.4), squeeze=False
    )
    for ax, met in zip(axes[0], metrics):
        for key, disp in models:
            if key in prim and key in audit and met in prim[key] and met in audit[key]:
                y0, y1 = prim[key][met], audit[key][met]
                col = MODEL_COLORS.get(disp, None)
                ax.plot([0, 1], [y0, y1], "-o", color=col, label=disp)
                ax.annotate(
                    f"{y0:.3f}",
                    (0, y0),
                    textcoords="offset points",
                    xytext=(-4, 4),
                    ha="right",
                    fontsize=8,
                )
                ax.annotate(
                    f"{y1:.3f}",
                    (1, y1),
                    textcoords="offset points",
                    xytext=(4, 4),
                    ha="left",
                    fontsize=8,
                )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["bio_full", "leakage_audit"])
        ax.set_title(met.upper())
        ax.set_ylabel(met.upper())
        ax.set_xlim(-0.3, 1.3)
    axes[0][-1].legend(loc="best")
    fig.suptitle(suptitle, y=1.03)
    _save(fig, fig_name)




# ---------------------------------------------------------------------------
# External prediction loader (Stage 12 -> STAGE12_OUT/external_predictions.npz)
# Mirrors _load_oof but reads external probability vectors.
# ---------------------------------------------------------------------------
def _load_ext_pred(tag="bio_full"):
    if not _exists(EXT_PRED_NPZ, "external ROC/PR"):
        return None
    z = np.load(EXT_PRED_NPZ, allow_pickle=True)
    ykey = f"{tag}__y"
    if ykey not in z.files:
        logger.warning("SKIP external ROC/PR: %s not in external pred store", ykey)
        return None
    y = z[ykey]
    series = {}
    for name, disp in [("lightgbm", "LightGBM"), ("cross_attention", "CrossAttention")]:
        k = f"{tag}__{name}"
        if k in z.files:
            series[disp] = z[k]
    tf = _third_model_key(z, tag)
    if tf:
        series["TF-DFE"] = z[tf]
    return y, series


# ===========================================================================
# Figure 7b - Confusion matrices (external validation)
# ===========================================================================
def fig7b_confusion_external():
    _confusion_from_json(
        EXT_JSON,
        "fig07b_confusion_matrices_external",
        "Confusion matrices (external validation, frozen internal threshold)",
    )


# ===========================================================================
# Figure 9 / 9b - Model comparison (LightGBM vs CrossAttention)
# Replaces the old Type-1 circularity panels. Grouped bars over key metrics,
# annotated with the McNemar winner + p-value.
# ===========================================================================
def _model_comparison_from_json(results_path, fig_name, title):
    if not _exists(results_path, fig_name):
        return
    res = json.loads(Path(results_path).read_text())
    prim = res.get("primary", {})
    models = [("lightgbm", "LightGBM"), ("cross_attention", "CrossAttention")]
    present = [(k, d) for k, d in models if isinstance(prim.get(k), dict)]
    if not present:
        logger.warning("SKIP %s: no model metrics in %s", fig_name, Path(results_path).name)
        return
    metrics = [("mcc", "MCC"), ("auroc", "AUROC"), ("auprc", "AUPRC"), ("f1", "F1")]
    x = np.arange(len(metrics))
    width = 0.8 / len(present)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for i, (k, disp) in enumerate(present):
        vals = [prim[k].get(m, np.nan) for m, _ in metrics]
        offset = (i - (len(present) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=disp,
                      color=MODEL_COLORS.get(disp, None))
        for b, v in zip(bars, vals):
            if v == v:  # skip NaN
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    mc = prim.get("mcnemar", {})
    if mc:
        ax.text(0.5, -0.16,
                f"McNemar: winner={mc.get('winner', '?')}, p={mc.get('p_value', '?')}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=8, color="#555555")
    ax.legend()
    _save(fig, fig_name)


def fig9_model_comparison():
    _model_comparison_from_json(
        RESULTS_JSON,
        "fig09_model_comparison",
        "Model comparison (internal, gene-level OOF)",
    )


def fig9b_model_comparison_external():
    _model_comparison_from_json(
        EXT_JSON,
        "fig09b_model_comparison_external",
        "Model comparison (external validation)",
    )


# ===========================================================================
# Figure 10 - ROC curves (external validation)
# ===========================================================================
def fig10_roc_external():
    fig_name = "fig10_roc_curves_external"
    loaded = _load_ext_pred()
    if loaded is None:
        return
    from sklearn.metrics import roc_curve, roc_auc_score
    y, series = loaded
    if len(np.unique(y)) < 2:
        logger.warning("SKIP %s: external set single-class", fig_name)
        return
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for disp, prob in series.items():
        fpr, tpr, _ = roc_curve(y, prob)
        auc = roc_auc_score(y, prob)
        ax.plot(
            fpr,
            tpr,
            color=MODEL_COLORS.get(disp, None),
            label=f"{disp} (AUC={auc:.3f})",
        )
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC comparison (external validation)")
    ax.legend(loc="lower right")
    _save(fig, fig_name)


# ===========================================================================
# Figure 11 - Precision-Recall (external validation)
# ===========================================================================
def fig11_pr_external():
    fig_name = "fig11_precision_recall_external"
    loaded = _load_ext_pred()
    if loaded is None:
        return
    from sklearn.metrics import precision_recall_curve, average_precision_score
    y, series = loaded
    if len(np.unique(y)) < 2:
        logger.warning("SKIP %s: external set single-class", fig_name)
        return
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for disp, prob in series.items():
        prec, rec, _ = precision_recall_curve(y, prob)
        ap = average_precision_score(y, prob)
        ax.plot(
            rec, prec, color=MODEL_COLORS.get(disp, None), label=f"{disp} (AP={ap:.3f})"
        )
        ok = rec >= 0.90
        if ok.any():
            j = np.argmin(np.abs(rec - 0.90))
            ax.scatter(
                [rec[j]],
                [prec[j]],
                color=MODEL_COLORS.get(disp, None),
                s=40,
                zorder=5,
                edgecolors="black",
                linewidths=0.6,
            )
    ax.axvline(0.90, ls=":", color="gray", lw=1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall (external validation, recall $\\geq$ 0.90)")
    ax.legend(loc="lower left")
    _save(fig, fig_name)


# ===========================================================================
# Figure 12 - Calibration / reliability (external validation)
# ===========================================================================
def fig12_calibration_external():
    fig_name = "fig12_calibration_external"
    loaded = _load_ext_pred()
    if loaded is None:
        return
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss
    y, series = loaded
    if len(np.unique(y)) < 2:
        logger.warning("SKIP %s: external set single-class", fig_name)
        return
    fig, ax = plt.subplots(figsize=(5.8, 5.6))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Perfect")
    for disp, prob in series.items():
        prob = np.clip(prob, 0.0, 1.0)
        frac_pos, mean_pred = calibration_curve(y, prob, n_bins=10, strategy="quantile")
        brier = brier_score_loss(y, prob)
        ax.plot(
            mean_pred,
            frac_pos,
            "-o",
            color=MODEL_COLORS.get(disp, None),
            markersize=4,
            label=f"{disp} (Brier={brier:.3f})",
        )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency (positives)")
    ax.set_title("Calibration / reliability (external validation)")
    ax.legend(loc="upper left")
    _save(fig, fig_name)


# ===========================================================================
# Main
# ===========================================================================
def main():
    logger.info("Rendering figures -> %s", FIG_DIR.resolve())
    for fn in (
        fig1_label_evidence,
        fig2_correlation,
        fig3_structural_violin,
        fig4_embedding_projection,
        fig5_roc,
        fig6_pr,
        fig7_confusion,
        fig7b_confusion_external,
        fig8_shap,
        fig9_model_comparison,             
        fig9b_model_comparison_external,   
        fig10_roc_external,
        fig11_pr_external,
        fig12_calibration_external,
    ):
        logger.info("Figure: %s", fn.__name__)
        _guard(fn)
    logger.info("Done. PNG figures in %s", FIG_DIR.resolve())


if __name__ == "__main__":
    main()