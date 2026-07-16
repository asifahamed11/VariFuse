import json
import logging

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import StratifiedGroupKFold

from config import DATA_DIR, STAGE09_OUT, STAGE12_OUT
import common as C

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stage12_external")

INT_CSV = STAGE09_OUT / "Final_Dataset_Balanced_with_ESM_Score.csv"
INT_NPY = STAGE09_OUT / "esm_residue_embeddings.npy"
EXT_CSV = DATA_DIR / "external" / "external_with_ESM_Score.csv"
EXT_NPY = DATA_DIR / "external" / "external_esm_embeddings.npy"

# Probability vectors for Stage 13 external ROC/PR/calibration figures
EXT_PRED_STORE = STAGE12_OUT / "external_predictions.npz"

STRICT_VARIANT_DEOVERLAP = True


def load_pair(csv_path, npy_path, name):
    logger.info("Loading %s: %s", name, csv_path)
    df = pd.read_csv(csv_path)
    E = np.load(npy_path)
    assert len(df) == len(E), f"{name} row mismatch: CSV={len(df)} NPY={len(E)}"
    valid = np.isfinite(df["esm_variant_score"].to_numpy()) & np.any(E != 0, axis=1)
    if (~valid).any():
        logger.warning(
            "%s: dropping %d rows with failed ESM extraction.",
            name,
            int((~valid).sum()),
        )
        df = df[valid].reset_index(drop=True)
        E = E[valid]
    return df, E


def variant_key(df):
    cols = [c for c in ("chr", "pos", "ref", "alt") if c in df.columns]
    if len(cols) < 4:
        return None
    return (
        df["chr"].astype(str) + ":" + df["pos"].astype(str) + ":" + df["ref"].astype(str) + ":" + df["alt"].astype(str)
    )


def deoverlap_external(int_df, ext_df, ext_E):
    train_genes = set(int_df[C.GENE_COL].dropna().unique())
    before = len(ext_df)
    keep = ~ext_df[C.GENE_COL].isin(train_genes)
    logger.info(
        "Gene-disjoint filter: removing %d external rows in %d shared genes.",
        int((~keep).sum()),
        len(set(ext_df[C.GENE_COL]) & train_genes),
    )
    if STRICT_VARIANT_DEOVERLAP:
        ik, ek = variant_key(int_df), variant_key(ext_df)
        if ik is not None and ek is not None:
            train_keys = set(ik)
            var_keep = ~ek.isin(train_keys)
            logger.info(
                "Strict variant de-overlap: removing %d additional exact matches.",
                int((~var_keep & keep).sum()),
            )
            keep = keep & var_keep
    ext_df2 = ext_df[keep].reset_index(drop=True)
    ext_E2 = ext_E[keep.to_numpy()]
    logger.info("External set: %d -> %d rows after de-overlap.", before, len(ext_df2))
    return ext_df2, ext_E2


def run_external(tag, int_df, int_E, ext_df, ext_E, pred_store):
    logger.info("=" * 66)
    logger.info("EXTERNAL VALIDATION | config: %s", tag)

    feats_int = C.select_features(int_df)
    feats = [f for f in feats_int if f in ext_df.columns]
    dropped_feats = set(feats_int) - set(feats)
    if dropped_feats:
        logger.warning(
            "Feature shrinkage! The following %d features are missing from external set: %s",
            len(dropped_feats),
            dropped_feats,
        )
    assert len(feats) == len(
        feats_int
    ), f"Schema mismatch: missing {len(dropped_feats)} features in external data."
    logger.info("Aligned feature set: %d features", len(feats))

    Xb_int = int_df[feats].to_numpy(dtype=np.float32)
    Xb_ext = ext_df[feats].to_numpy(dtype=np.float32)
    y_int = int_df[C.LABEL_COL].to_numpy(dtype=int)
    y_ext = ext_df[C.LABEL_COL].to_numpy(dtype=int)
    groups_int = int_df[C.GENE_COL].to_numpy()

    # Preprocessing fit on ALL internal, applied to external unchanged
    tb, te = C.fit_preprocessors(Xb_int, int_E)

    # Internal inner split for deep early stopping and threshold calculation
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=C.RANDOM_STATE)
    itr, ival = next(cv.split(Xb_int, y_int, groups_int))

    # (1) LightGBM
    clf = LGBMClassifier(**C.LGBM_PARAMS)
    clf.fit(
        Xb_int[itr],
        y_int[itr],
        eval_set=[(Xb_int[ival], y_int[ival])],
        eval_metric="auc",
        callbacks=[early_stopping(C.LGBM_EARLY_STOP, verbose=False), log_evaluation(0)],
    )
    p_lgbm_val = clf.predict_proba(Xb_int[ival])[:, 1]
    thr_lgbm = C.select_threshold(y_int[ival], p_lgbm_val)
    logger.info("Frozen internal threshold (LightGBM): %.3f", thr_lgbm)
    p_lgbm = clf.predict_proba(Xb_ext)[:, 1]

    # (2) Cross-Attention
    model = C.train_deep_model(
        tb(Xb_int[itr]),
        te(int_E[itr]),
        y_int[itr],
        tb(Xb_int[ival]),
        te(int_E[ival]),
        y_int[ival],
    )
    p_xattn_val = C.predict(model, tb(Xb_int[ival]), te(int_E[ival]))
    thr_xattn = C.select_threshold(y_int[ival], p_xattn_val)
    logger.info("Frozen internal threshold (Cross-Attention): %.3f", thr_xattn)
    p_xattn = C.predict(model, tb(Xb_ext), te(ext_E))
    del model
    if C.DEVICE == "cuda":
        torch.cuda.empty_cache()

    # ---- persist external probability vectors for Stage 13 figures ----
    C.save_oof_artifacts(
        pred_store, y_ext,
        {"lightgbm": p_lgbm, "cross_attention": p_xattn}, tag,
    )

    m_lgbm = C.evaluate(y_ext, p_lgbm, thr_lgbm)
    m_xattn = C.evaluate(y_ext, p_xattn, thr_xattn)
    mc = C.mcnemar_test(
        y_ext, (p_lgbm >= thr_lgbm).astype(int), (p_xattn >= thr_xattn).astype(int)
    )
    logger.info(
        "[%s|EXT] LGBM MCC=%.4f AUROC=%.4f | XATTN MCC=%.4f AUROC=%.4f | McNemar=%s",
        tag,
        m_lgbm["mcc"],
        m_lgbm["auroc"],
        m_xattn["mcc"],
        m_xattn["auroc"],
        mc["winner"],
    )
    return {
        "tag": tag,
        "frozen_threshold_lgbm": round(float(thr_lgbm), 4),
        "frozen_threshold_xattn": round(float(thr_xattn), 4),
        "n_features": len(feats),
        "n_external": int(len(y_ext)),
        "external_positives": int(y_ext.sum()),
        "lightgbm": m_lgbm,
        "cross_attention": m_xattn,
        "mcnemar": mc,
    }


def main():
    C.set_seeds()
    STAGE12_OUT.mkdir(parents=True, exist_ok=True)
    logger.info("Device: %s", C.DEVICE)

    int_df, int_E = load_pair(INT_CSV, INT_NPY, "internal")
    ext_df, ext_E = load_pair(EXT_CSV, EXT_NPY, "external")
    ext_df, ext_E = deoverlap_external(int_df, ext_df, ext_E)
    if len(ext_df) == 0:
        logger.error("No external rows left after de-overlap. Aborting.")
        return
    if len(np.unique(ext_df[C.LABEL_COL])) < 2:
        logger.error("External set has only one class after filtering. Aborting.")
        return

    # fresh prediction store so figures never read stale external probs
    if EXT_PRED_STORE.exists():
        EXT_PRED_STORE.unlink()

    results = {
        "external_source": str(EXT_CSV),
        "primary": run_external("bio_full", int_df, int_E, ext_df, ext_E, EXT_PRED_STORE),
    }

    rows = []
    for tag in ("primary",):
        r = results[tag]
        for name, key in [
            ("LightGBM", "lightgbm"),
            ("CrossAttention", "cross_attention"),
        ]:
            m = r[key]
            rows.append(
                {
                    "model": name,
                    "config": r["tag"],
                    "set": "external",
                    "n": r["n_external"],
                    "mcc": m["mcc"],
                    "auroc": m["auroc"],
                    "auprc": m["auprc"],
                    "recall": m["recall"],
                    "precision": m["precision"],
                    "f1": m["f1"],
                    "brier": m["brier"],
                    "threshold": m["threshold"],
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(STAGE12_OUT / "external_validation_table.csv", index=False)
    (STAGE12_OUT / "external_validation.json").write_text(json.dumps(results, indent=2))

    print()
    print("=" * 66)
    print("EXTERNAL VALIDATION (frozen internal threshold, gene-disjoint)")
    print("=" * 66)
    print(table.to_string(index=False))
    logger.info("Saved -> external_validation_table.csv, external_validation.json, external_predictions.npz")


if __name__ == "__main__":
    main()
