import json
import logging
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from config import STAGE09_OUT, STAGE10_OUT
import common as C

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("stage11_eval")

INPUT_CSV = STAGE09_OUT / "Final_Dataset_Balanced_with_ESM_Score.csv"
INPUT_NPY = STAGE09_OUT / "esm_residue_embeddings.npy"
N_SPLITS = 5


def run_config(df, E, y, groups, use_groups, tag, ablate, oof_store):
    logger.info("=" * 66)
    logger.info("CONFIG: %s", tag)
    feats = C.select_features(df, ablate)
    Xb_all = df[feats].to_numpy(dtype=np.float32)
    oof_lgbm = np.zeros(len(df))
    oof_xattn = np.zeros(len(df))
    last_clf = None

    if use_groups:
        outer = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=C.RANDOM_STATE)
        splits = outer.split(Xb_all, y, groups)
    else:
        outer = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=C.RANDOM_STATE)
        splits = outer.split(Xb_all, y)

    for fold, (tr, va) in enumerate(splits, 1):
        if use_groups:
            assert set(groups[tr]).isdisjoint(set(groups[va])), "Gene leakage!"
            inner = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=C.RANDOM_STATE)
            itr, ival = next(inner.split(Xb_all[tr], y[tr], groups[tr]))
        else:
            inner = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=C.RANDOM_STATE)
            itr, ival = next(inner.split(Xb_all[tr], y[tr]))
        tr_in, val_in = tr[itr], tr[ival]

        tb, te = C.fit_preprocessors(Xb_all[tr_in], E[tr_in])

        clf = LGBMClassifier(**C.LGBM_PARAMS)
        clf.fit(Xb_all[tr_in], y[tr_in], eval_set=[(Xb_all[val_in], y[val_in])],
                eval_metric="auc",
                callbacks=[early_stopping(C.LGBM_EARLY_STOP, verbose=False), log_evaluation(0)])
        oof_lgbm[va] = clf.predict_proba(Xb_all[va])[:, 1]
        last_clf = clf

        model = C.train_deep_model(tb(Xb_all[tr_in]), te(E[tr_in]), y[tr_in],
                                   tb(Xb_all[val_in]), te(E[val_in]), y[val_in])
        oof_xattn[va] = C.predict(model, tb(Xb_all[va]), te(E[va]))

        logger.info("Fold %d/%d | LGBM AUC=%.4f | XATTN AUC=%.4f", fold, N_SPLITS,
                    C.roc_auc_score(y[va], oof_lgbm[va]),
                    C.roc_auc_score(y[va], oof_xattn[va]))
        del model
        if C.DEVICE == "cuda":
            torch.cuda.empty_cache()

    # ---- persist artifacts for Stage 13 figures ----
    C.save_oof_artifacts(oof_store, y,
                         {"lightgbm": oof_lgbm, "cross_attention": oof_xattn}, tag)
    if last_clf is not None:
        C.compute_and_save_shap(last_clf, Xb_all, feats,
                                STAGE10_OUT / f"shap_values_{tag}.npz")

    thr_l, thr_x = C.select_threshold(y, oof_lgbm), C.select_threshold(y, oof_xattn)
    m_l, m_x = C.evaluate(y, oof_lgbm, thr_l), C.evaluate(y, oof_xattn, thr_x)
    mc = C.mcnemar_test(y, (oof_lgbm >= thr_l).astype(int), (oof_xattn >= thr_x).astype(int))
    logger.info("[%s] LGBM MCC=%.4f | XATTN MCC=%.4f | McNemar=%s",
                tag, m_l["mcc"], m_x["mcc"], mc["winner"])
    return {"tag": tag, "n_features": len(feats),
            "lightgbm": m_l, "cross_attention": m_x, "mcnemar": mc}


def run_evaluation():
    C.set_seeds()
    STAGE10_OUT.mkdir(parents=True, exist_ok=True)
    logger.info("Device: %s", C.DEVICE)
    df = pd.read_csv(INPUT_CSV)
    E = np.load(INPUT_NPY)
    assert len(df) == len(E), f"Row mismatch: CSV={len(df)} vs NPY={len(E)}"
    valid = np.isfinite(df["esm_variant_score"].to_numpy()) & np.any(E != 0, axis=1)
    if (~valid).any():
        logger.warning("Dropping %d rows with failed ESM extraction.", int((~valid).sum()))
        df = df[valid].reset_index(drop=True)
        E = E[valid]
    y = df[C.LABEL_COL].to_numpy(dtype=int)
    use_groups = C.GENE_COL in df.columns
    groups = df[C.GENE_COL].to_numpy() if use_groups else None
    logger.info("Split: %s", "StratifiedGroupKFold (gene-level)" if use_groups else "StratifiedKFold")

    oof_store = STAGE10_OUT / "oof_predictions.npz"
    if oof_store.exists():
        oof_store.unlink()  # fresh run

    results = {
        "primary": run_config(df, E, y, groups, use_groups, "bio_full", False, oof_store),
        "leakage_audit": run_config(df, E, y, groups, use_groups,
                                    "bio_minus_predictor_scores", True, oof_store),
    }
    results["type1_mcc_inflation"] = {
        "lightgbm": round(results["primary"]["lightgbm"]["mcc"]
                          - results["leakage_audit"]["lightgbm"]["mcc"], 4),
        "cross_attention": round(results["primary"]["cross_attention"]["mcc"]
                                 - results["leakage_audit"]["cross_attention"]["mcc"], 4),
    }

    rows = []
    for tag in ("primary", "leakage_audit"):
        r = results[tag]
        for name, key in [("LightGBM", "lightgbm"), ("CrossAttention", "cross_attention")]:
            m = r[key]
            rows.append({"model": name, "config": r["tag"], "mcc": m["mcc"],
                         "auroc": m["auroc"], "auprc": m["auprc"], "recall": m["recall"],
                         "precision": m["precision"], "f1": m["f1"], "brier": m["brier"],
                         "threshold": m["threshold"]})
    table = pd.DataFrame(rows)
    table.to_csv(STAGE10_OUT / "comparison_table.csv", index=False)
    (STAGE10_OUT / "results.json").write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 66)
    print(table.to_string(index=False))
    print("\nMcNemar (bio_full): ", results["primary"]["mcnemar"])
    print("McNemar (leakage_audit): ", results["leakage_audit"]["mcnemar"])
    print("Type-1 MCC inflation: ", results["type1_mcc_inflation"])
    logger.info("Saved -> comparison_table.csv, results.json, oof_predictions.npz, shap_values_*.npz")


if __name__ == "__main__":
    run_evaluation()
