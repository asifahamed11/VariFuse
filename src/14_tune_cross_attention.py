from __future__ import annotations
import argparse, json, logging, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from config import STAGE09_OUT, OUTPUT_DIR
import common as C

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("stage14_tune")

INPUT_CSV = STAGE09_OUT / "Final_Dataset_Balanced_with_ESM_Score.csv"
INPUT_NPY = STAGE09_OUT / "esm_residue_embeddings.npy"
TUNE_DIR  = OUTPUT_DIR / "14_tuning"
TUNE_DIR.mkdir(parents=True, exist_ok=True)
BEST_JSON = TUNE_DIR / "best_cross_attention_params.json"
TRIALS_CSV = TUNE_DIR / "trials_history.csv"

# ---- Tunable constants that live as globals inside `common` ---------------
TUNABLE = ["D_MODEL", "N_HEADS", "N_CROSS_BLOCKS", "N_FUSION_LAYERS", "N_ESM_SLOTS",
           "DROPOUT", "DEEP_LR", "DEEP_WD", "WARMUP_EPOCHS", "EMA_DECAY",
           "MIXUP_ALPHA", "LABEL_SMOOTH", "BATCH_SIZE", "N_ENSEMBLE",
           "DEEP_MAX_EPOCHS", "DEEP_PATIENCE"]


# ===========================================================================
# Determinism helpers (for reproducible train/test)
# ===========================================================================
def set_determinism(seed: int, strict: bool):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if strict:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True


def snapshot_constants():
    return {k: getattr(C, k) for k in TUNABLE}


def apply_params(params: dict):
    """Override the module-level constants that the model/train code reads."""
    for k, v in params.items():
        setattr(C, k, v)


# ===========================================================================
# Data
# ===========================================================================
def load_data():
    df = pd.read_csv(INPUT_CSV)
    E = np.load(INPUT_NPY)
    assert len(df) == len(E), f"Row mismatch CSV={len(df)} NPY={len(E)}"
    valid = np.isfinite(df["esm_variant_score"].to_numpy()) & np.any(E != 0, axis=1)
    if (~valid).any():
        df = df[valid].reset_index(drop=True); E = E[valid]
    y = df[C.LABEL_COL].to_numpy(dtype=int)
    use_groups = C.GENE_COL in df.columns
    groups = df[C.GENE_COL].to_numpy() if use_groups else None
    return df, E, y, groups, use_groups


# ===========================================================================
# One CV evaluation of the cross-attention model with the CURRENT C.* globals
# ===========================================================================
def evaluate_current(df, E, y, groups, use_groups, n_folds, seed):
    feats = C.select_features(df, ablate=False)          # bio_full feature set
    Xb_all = df[feats].to_numpy(dtype=np.float32)
    oof = np.zeros(len(df))

    if use_groups:
        outer = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(outer.split(Xb_all, y, groups))
    else:
        outer = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(outer.split(Xb_all, y))

    for fold, (tr, va) in enumerate(splits, 1):
        if use_groups:
            assert set(groups[tr]).isdisjoint(set(groups[va])), "Gene leakage!"
            inner = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            itr, ival = next(inner.split(Xb_all[tr], y[tr], groups[tr]))
        else:
            inner = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            itr, ival = next(inner.split(Xb_all[tr], y[tr]))
        tr_in, val_in = tr[itr], tr[ival]

        tb, te = C.fit_preprocessors(Xb_all[tr_in], E[tr_in])
        model = C.train_deep_model(tb(Xb_all[tr_in]), te(E[tr_in]), y[tr_in],
                                   tb(Xb_all[val_in]), te(E[val_in]), y[val_in])
        oof[va] = C.predict(model, tb(Xb_all[va]), te(E[va]))
        del model
        if C.DEVICE == "cuda":
            torch.cuda.empty_cache()
        logger.info("  fold %d/%d done", fold, n_folds)

    thr = C.select_threshold(y, oof)
    m = C.evaluate(y, oof, thr)
    return m


def composite_score(m: dict) -> float:
    return (0.50 * m["mcc"] + 0.25 * m["auroc"] + 0.15 * m["auprc"]
            - 0.10 * m["brier"])


def objective_value(m: dict, mode: str) -> float:
    if mode == "mcc":   return m["mcc"]
    if mode == "auroc": return m["auroc"]
    if mode == "auprc": return m["auprc"]
    return composite_score(m)


# ===========================================================================
# Search space
# ===========================================================================
def suggest_params(trial):
    """Optuna-style suggestion with the D_MODEL % N_HEADS == 0 constraint."""
    n_heads = trial.suggest_categorical("N_HEADS", [4, 8])
    d_units = trial.suggest_categorical("d_units", [16, 20, 24, 28, 32])  # per-head width
    d_model = n_heads * d_units
    return {
        "N_HEADS": n_heads,
        "D_MODEL": d_model,
        "N_CROSS_BLOCKS": trial.suggest_int("N_CROSS_BLOCKS", 2, 5),
        "N_FUSION_LAYERS": trial.suggest_int("N_FUSION_LAYERS", 1, 3),
        "N_ESM_SLOTS": trial.suggest_categorical("N_ESM_SLOTS", [8, 16, 24, 32]),
        "DROPOUT": trial.suggest_float("DROPOUT", 0.05, 0.30),
        "DEEP_LR": trial.suggest_float("DEEP_LR", 3e-4, 3e-3, log=True),
        "DEEP_WD": trial.suggest_float("DEEP_WD", 1e-5, 1e-3, log=True),
        "WARMUP_EPOCHS": trial.suggest_int("WARMUP_EPOCHS", 3, 12),
        "EMA_DECAY": trial.suggest_categorical("EMA_DECAY", [0.995, 0.999, 0.9995]),
        "MIXUP_ALPHA": trial.suggest_float("MIXUP_ALPHA", 0.0, 0.4),
        "LABEL_SMOOTH": trial.suggest_float("LABEL_SMOOTH", 0.0, 0.05),
        "BATCH_SIZE": trial.suggest_categorical("BATCH_SIZE", [256, 512, 1024]),
    }


# random fallback if optuna not installed
def suggest_params_random(rng):
    n_heads = rng.choice([4, 8])
    d_units = rng.choice([16, 20, 24, 28, 32])
    return {
        "N_HEADS": int(n_heads), "D_MODEL": int(n_heads * d_units),
        "N_CROSS_BLOCKS": int(rng.integers(2, 6)),
        "N_FUSION_LAYERS": int(rng.integers(1, 4)),
        "N_ESM_SLOTS": int(rng.choice([8, 16, 24, 32])),
        "DROPOUT": float(rng.uniform(0.05, 0.30)),
        "DEEP_LR": float(10 ** rng.uniform(np.log10(3e-4), np.log10(3e-3))),
        "DEEP_WD": float(10 ** rng.uniform(-5, -3)),
        "WARMUP_EPOCHS": int(rng.integers(3, 13)),
        "EMA_DECAY": float(rng.choice([0.995, 0.999, 0.9995])),
        "MIXUP_ALPHA": float(rng.uniform(0.0, 0.4)),
        "LABEL_SMOOTH": float(rng.uniform(0.0, 0.05)),
        "BATCH_SIZE": int(rng.choice([256, 512, 1024])),
    }


# ===========================================================================
# Search driver
# ===========================================================================
def run_search(args):
    set_determinism(args.seed, strict=False)   # search can use fast/nondeterministic
    df, E, y, groups, use_groups = load_data()
    base = snapshot_constants()
    history = []

    # search-phase overrides (fast)
    search_fixed = {"N_ENSEMBLE": args.search_ensemble,
                    "DEEP_MAX_EPOCHS": args.search_epochs,
                    "DEEP_PATIENCE": max(4, args.search_epochs // 5)}

    def eval_trial(params):
        apply_params(base)                 # reset to defaults first
        apply_params(search_fixed)
        apply_params(params)
        set_determinism(args.seed, strict=False)
        t0 = time.time()
        m = evaluate_current(df, E, y, groups, use_groups, args.search_folds, args.seed)
        rec = {**params, **{f"metric_{k}": v for k, v in m.items()
                            if k in ("mcc", "auroc", "auprc", "brier", "recall",
                                     "precision", "f1", "threshold")},
               "objective": objective_value(m, args.objective),
               "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        pd.DataFrame(history).to_csv(TRIALS_CSV, index=False)
        logger.info("trial obj=%.4f | MCC=%.4f AUROC=%.4f AUPRC=%.4f Brier=%.4f | %s",
                    rec["objective"], m["mcc"], m["auroc"], m["auprc"], m["brier"], params)
        return rec["objective"]

    best = None
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=args.seed))

        def _obj(trial):
            return eval_trial(suggest_params(trial))

        study.optimize(_obj, n_trials=args.trials)
        best_params = {k: v for k, v in study.best_params.items() if k != "d_units"}
        # rebuild D_MODEL from best n_heads/d_units
        best_params["N_HEADS"] = study.best_params["N_HEADS"]
        best_params["D_MODEL"] = study.best_params["N_HEADS"] * study.best_params["d_units"]
        best = best_params
        logger.info("Optuna best objective = %.4f", study.best_value)
    except ImportError:
        logger.warning("optuna not found -> random search fallback.")
        rng = np.random.default_rng(args.seed)
        best_obj = -1e9
        for i in range(args.trials):
            params = suggest_params_random(rng)
            obj = eval_trial(params)
            if obj > best_obj:
                best_obj, best = obj, params
        logger.info("Random-search best objective = %.4f", best_obj)

    # ---- FINAL full-fidelity re-evaluation of the best config -------------
    logger.info("=" * 66)
    logger.info("FINAL full-fidelity evaluation of best params")
    apply_params(base)
    final_fixed = {"N_ENSEMBLE": args.final_ensemble,
                   "DEEP_MAX_EPOCHS": args.final_epochs,
                   "DEEP_PATIENCE": args.final_patience}
    apply_params(final_fixed)
    apply_params(best)
    set_determinism(args.seed, strict=args.deterministic)
    m_final = evaluate_current(df, E, y, groups, use_groups, args.final_folds, args.seed)

    full_config = {**{k: getattr(C, k) for k in TUNABLE}}
    payload = {
        "search": {"trials": args.trials, "folds": args.search_folds,
                   "ensemble": args.search_ensemble, "epochs": args.search_epochs,
                   "objective": args.objective},
        "final_eval": {"folds": args.final_folds, "ensemble": args.final_ensemble,
                       "epochs": args.final_epochs, "seed": args.seed,
                       "deterministic": args.deterministic},
        "best_params": full_config,
        "final_metrics": m_final,
        "final_objective": objective_value(m_final, args.objective),
    }
    BEST_JSON.write_text(json.dumps(payload, indent=2))
    _print_summary(payload)


# ===========================================================================
# Reproduce mode: read best json, run the exact same eval, print metrics
# ===========================================================================
def run_reproduce(args):
    if not BEST_JSON.exists():
        logger.error("No saved best params at %s. Run a search first.", BEST_JSON)
        return
    payload = json.loads(BEST_JSON.read_text())
    df, E, y, groups, use_groups = load_data()

    apply_params(payload["best_params"])
    fe = payload["final_eval"]
    set_determinism(fe["seed"], strict=fe.get("deterministic", True))
    m = evaluate_current(df, E, y, groups, use_groups, fe["folds"], fe["seed"])

    logger.info("Reproduced metrics vs saved:")
    for k in ("mcc", "auroc", "auprc", "brier", "recall", "precision", "f1"):
        logger.info("  %-9s saved=%.4f  now=%.4f", k,
                    payload["final_metrics"][k], m[k])
    _print_summary({"best_params": payload["best_params"], "final_metrics": m,
                    "final_objective": objective_value(m, payload["search"]["objective"])})


def _print_summary(payload):
    print("\n" + "=" * 66)
    print("BEST CROSS-ATTENTION HYPERPARAMETERS")
    print("=" * 66)
    for k, v in payload["best_params"].items():
        print(f"  {k:18s} = {v}")
    print("-" * 66)
    m = payload["final_metrics"]
    print("PERFORMANCE (gene-level OOF, recall>=0.90 floor):")
    for k in ("mcc", "auroc", "auprc", "brier", "recall", "precision", "f1", "threshold"):
        print(f"  {k:10s} = {m[k]}")
    print(f"  {'objective':10s} = {round(payload['final_objective'], 4)}")
    print("=" * 66)
    print(f"Saved -> {BEST_JSON}")


# ===========================================================================
def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--objective", choices=["composite", "mcc", "auroc", "auprc"],
                   default="composite")
    p.add_argument("--seed", type=int, default=42)
    # search phase (fast)
    p.add_argument("--search-folds", type=int, default=3)
    p.add_argument("--search-ensemble", type=int, default=1)
    p.add_argument("--search-epochs", type=int, default=25)
    # final phase (full)
    p.add_argument("--final-folds", type=int, default=5)
    p.add_argument("--final-ensemble", type=int, default=3)
    p.add_argument("--final-epochs", type=int, default=60)
    p.add_argument("--final-patience", type=int, default=10)
    p.add_argument("--deterministic", action="store_true",
                   help="cudnn.deterministic for exact reproducibility (slower).")
    p.add_argument("--reproduce", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = build_args()
    if args.reproduce:
        run_reproduce(args)
    else:
        run_search(args)
