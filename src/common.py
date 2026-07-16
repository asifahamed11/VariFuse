from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("pipeline.common")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_STATE = 42
torch.backends.cudnn.benchmark = True

# ---- Shared schema constants ----------------------------------------------
LABEL_COL = "LABEL_PATHOGENIC"
GENE_COL = "genename"
ESM_DIM = 1280
RECALL_FLOOR = 0.90
FBETA_BETA = 2.0
ABSTAIN_MARGIN = 0.10

LEAKAGE_COLS = ["chr", "pos", "ref", "alt", "CONSENSUS_SCORE", "TIER",
                "DOMAIN_NAME", "SECONDARY_STRUCTURE", "IS_CLINVAR_BENIGN"]
# Kept for reference only. No longer dropped anywhere (bio_full uses these).
PREDICTOR_COLS = ["SIFT_score", "Polyphen2_HDIV_score", "CADD_phred", "REVEL_score"]
ID_COLS = ["aa_pos", "aa_ref", "aa_alt", "protein_sequence",
           "Ensembl_transcriptid", "HGVSp_snpEff", "HGVSc_snpEff"]

LGBM_PARAMS = dict(
    n_estimators=1000, learning_rate=0.02, num_leaves=30, max_depth=-1,
    min_child_samples=60, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_alpha=0.5, reg_lambda=2.0, n_jobs=-1, verbose=-1, random_state=RANDOM_STATE,
)
LGBM_EARLY_STOP = 100

D_MODEL = 256
N_HEADS = 8
N_CROSS_BLOCKS = 3
N_FUSION_LAYERS = 2
N_ESM_SLOTS = 16
DROPOUT = 0.23131003180165313
DEEP_LR = 0.0003577380577877598
DEEP_WD = 3.5786341385752017e-05
WARMUP_EPOCHS = 11
EMA_DECAY = 0.995
MIXUP_ALPHA = 0.22862321425429072
LABEL_SMOOTH = 0.00023378541799429914
BATCH_SIZE = 256
N_ENSEMBLE = 3
DEEP_MAX_EPOCHS = 60
DEEP_PATIENCE = 10
GRAD_CLIP = 1.0


def set_seeds(seed: int = RANDOM_STATE) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


# ===========================================================================
# Cross-Attention Fusion network
# ===========================================================================
class FeatureTokenizer(nn.Module):
    """FT-Transformer style: each scalar feature -> a d_model token."""

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_model))

    def forward(self, x):  # (B, Nf)
        return x.unsqueeze(-1) * self.weight + self.bias


class CrossAttentionBlock(nn.Module):
    """Bidirectional cross-attention: bio<->esm."""

    def __init__(self, d, h, mult, p):
        super().__init__()
        self.a2b = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.b2a = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.ln_a1, self.ln_b1, self.ln_a2, self.ln_b2 = [nn.LayerNorm(d) for _ in range(4)]
        self.ffn_a = nn.Sequential(nn.Linear(d, d*mult), nn.GELU(), nn.Dropout(p), nn.Linear(d*mult, d))
        self.ffn_b = nn.Sequential(nn.Linear(d, d*mult), nn.GELU(), nn.Dropout(p), nn.Linear(d*mult, d))

    def forward(self, a, b):
        a2, _ = self.a2b(a, b, b); a = self.ln_a1(a + a2); a = self.ln_a2(a + self.ffn_a(a))
        b2, _ = self.b2a(b, a, a); b = self.ln_b1(b + b2); b = self.ln_b2(b + self.ffn_b(b))
        return a, b


class CrossAttnFusionNet(nn.Module):
    """Feature tokens + ESM slot tokens -> cross-attention -> self-attention
    fusion over [CLS | bio | esm] -> (CLS + mean-pool) readout -> logit."""

    def __init__(self, n_features: int, esm_dim: int = ESM_DIM):
        super().__init__()
        d, h = D_MODEL, N_HEADS
        self.n_esm_slots = N_ESM_SLOTS
        self.tokenizer = FeatureTokenizer(n_features, d)
        self.esm_proj = nn.Sequential(
            nn.LayerNorm(esm_dim),
            nn.Linear(esm_dim, N_ESM_SLOTS * d),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.cross = nn.ModuleList(
            [CrossAttentionBlock(d, h, 2, DROPOUT) for _ in range(N_CROSS_BLOCKS)])
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=h, dim_feedforward=d*2,
                                         dropout=DROPOUT, activation="gelu",
                                         batch_first=True, norm_first=True)
        self.fusion = nn.TransformerEncoder(enc, num_layers=N_FUSION_LAYERS)
        # CLS token + mean-pooled sequence -> 2d input to the head
        self.head = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU(),
                                  nn.Dropout(DROPOUT), nn.Linear(d, 1))

    def forward(self, x_bio, x_esm):
        B = x_bio.size(0)
        bio = self.tokenizer(x_bio)
        esm = self.esm_proj(x_esm).view(B, self.n_esm_slots, -1)
        for blk in self.cross:
            bio, esm = blk(bio, esm)
        seq = torch.cat([self.cls.expand(B, -1, -1), bio, esm], dim=1)
        fused = self.fusion(seq)
        cls_out = fused[:, 0]
        mean_out = fused.mean(dim=1)
        z = torch.cat([cls_out, mean_out], dim=-1)
        return self.head(z).squeeze(-1)


# ===========================================================================
# EMA (exponential moving average of weights)
# ===========================================================================
class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()


# ===========================================================================
# Ensemble wrapper (logit-averaging). API-compatible with predict().
# ===========================================================================
class EnsembleModel(nn.Module):
    def __init__(self, members: list[nn.Module], T: float = 1.0):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.T = T

    def forward(self, x_bio, x_esm):
        logits = torch.stack([m(x_bio, x_esm) for m in self.members], dim=0)
        return logits.mean(dim=0)

    def eval(self):
        super().eval()
        for m in self.members:
            m.eval()
        return self


# ===========================================================================
# Training / prediction
# ===========================================================================
def _loader(Xb, Xe, y, bs, shuffle, workers=0):
    ds = TensorDataset(torch.from_numpy(Xb).float(),
                       torch.from_numpy(Xe).float(),
                       torch.from_numpy(y).float())
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, pin_memory=(DEVICE == "cuda"),
                      num_workers=workers, persistent_workers=(workers > 0))


@torch.no_grad()
def predict(model, Xb, Xe, bs: int = 2048) -> np.ndarray:
    model.eval()
    dummy = np.zeros(len(Xb), np.float32)
    T = getattr(model, 'T', 1.0)
    out = []
    with torch.amp.autocast('cuda', enabled=(DEVICE == "cuda")):
        for xb, xe, _ in _loader(Xb, Xe, dummy, bs, False):
            out.append(torch.sigmoid(model(xb.to(DEVICE), xe.to(DEVICE)) / T).cpu().numpy())
    return np.concatenate(out)


def _lr_lambda(epoch: int):
    """Linear warmup then cosine decay over DEEP_MAX_EPOCHS."""
    if epoch < WARMUP_EPOCHS:
        return float(epoch + 1) / float(max(1, WARMUP_EPOCHS))
    prog = (epoch - WARMUP_EPOCHS) / float(max(1, DEEP_MAX_EPOCHS - WARMUP_EPOCHS))
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))


def _train_single(Xb_tr, Xe_tr, y_tr, Xb_val, Xe_val, y_val, seed: int):
    """Train ONE fusion net with EMA + cosine LR + mixup + MCC early stop."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = CrossAttnFusionNet(Xb_tr.shape[1]).to(DEVICE)
    eval_model = CrossAttnFusionNet(Xb_tr.shape[1]).to(DEVICE)  # reused for EMA eval
    ema = EMA(model, EMA_DECAY)

    pos_weight = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
                              dtype=torch.float32, device=DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=DEEP_LR, weight_decay=DEEP_WD)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
    loader = _loader(Xb_tr, Xe_tr, y_tr, BATCH_SIZE, True, workers=2)
    rng = np.random.RandomState(seed)
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE == "cuda"))

    best_score, wait, best_state = -1.0, 0, None
    for _ in range(DEEP_MAX_EPOCHS):
        model.train()
        for xb, xe, yy in loader:
            xb, xe, yy = xb.to(DEVICE), xe.to(DEVICE), yy.to(DEVICE)
            # light label smoothing
            yy = yy * (1.0 - LABEL_SMOOTH) + 0.5 * LABEL_SMOOTH
            # input mixup (soft targets)
            if MIXUP_ALPHA > 0 and rng.rand() < 0.5 and xb.size(0) > 1:
                lam = float(rng.beta(MIXUP_ALPHA, MIXUP_ALPHA))
                perm = torch.randperm(xb.size(0), device=DEVICE)
                xb = lam * xb + (1.0 - lam) * xb[perm]
                xe = lam * xe + (1.0 - lam) * xe[perm]
                yy = lam * yy + (1.0 - lam) * yy[perm]
            opt.zero_grad()
            with torch.amp.autocast('cuda', enabled=(DEVICE == "cuda")):
                loss = crit(model(xb, xe), yy)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
            ema.update(model)
        sched.step()

        # early stopping evaluated on the EMA weights
        eval_model.load_state_dict(ema.shadow)
        p_val_raw = predict(eval_model, Xb_val, Xe_val)
        thr = select_threshold(y_val, p_val_raw)
        score = matthews_corrcoef(y_val, (p_val_raw >= thr).astype(int))
        if score > best_score + 1e-4:
            best_score, wait = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in ema.shadow.items()}
        else:
            wait += 1
            if wait >= DEEP_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    del eval_model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return model, best_score


def fit_temperature(model, Xb_val, Xe_val, y_val):
    logits = []
    model.eval()
    with torch.no_grad():
        dummy = np.zeros(len(Xb_val), np.float32)
        with torch.amp.autocast('cuda', enabled=(DEVICE == "cuda")):
            for xb, xe, _ in _loader(Xb_val, Xe_val, dummy, 2048, False):
                logits.append(model(xb.to(DEVICE), xe.to(DEVICE)).cpu())
    logits = torch.cat(logits)
    y = torch.from_numpy(y_val).float()
    T = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = nn.BCEWithLogitsLoss()(logits / T, y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.detach().clamp(min=1.0).item())


def train_deep_model(Xb_tr, Xe_tr, y_tr, Xb_val, Xe_val, y_val,
                     max_epochs: int = DEEP_MAX_EPOCHS, patience: int = DEEP_PATIENCE):
    """Train an ENSEMBLE of independently-seeded fusion nets and return a
    logit-averaging EnsembleModel."""
    members, early_stop_mccs = [], []
    for i in range(N_ENSEMBLE):
        net, auc = _train_single(Xb_tr, Xe_tr, y_tr, Xb_val, Xe_val, y_val,
                                 seed=RANDOM_STATE + 100 * (i + 1))
        members.append(net)
        early_stop_mccs.append(auc)
    ens = EnsembleModel(members).to(DEVICE)
    best_T = fit_temperature(ens, Xb_val, Xe_val, y_val)
    ens.T = best_T
    logger.info("Fitted temperature scaling T=%.3f", best_T)
    ens_auc = roc_auc_score(y_val, predict(ens, Xb_val, Xe_val))
    logger.info(" deep ensemble (%d members) inner-val AUROC=%.4f "
                "(members mean MCC=%.4f)", N_ENSEMBLE, ens_auc, float(np.mean(early_stop_mccs)))
    return ens


# ===========================================================================
# Preprocessing (fit on train, apply everywhere)
# ===========================================================================
def fit_preprocessors(Xb_fit, Xe_fit):
    """Return (transform_bio, transform_esm) closures fit on the given arrays."""
    imp_b = SimpleImputer(strategy="median").fit(Xb_fit)
    sc_b = StandardScaler().fit(imp_b.transform(Xb_fit))
    imp_e = SimpleImputer(strategy="median").fit(Xe_fit)
    sc_e = StandardScaler().fit(imp_e.transform(Xe_fit))
    tb = lambda X: sc_b.transform(imp_b.transform(X)).astype(np.float32)
    te = lambda X: sc_e.transform(imp_e.transform(X)).astype(np.float32)
    return tb, te


# ===========================================================================
# Feature selection  (bio_full only: predictor scores are ALWAYS kept)
# ===========================================================================
def select_features(df: pd.DataFrame) -> list[str]:
    drop = set(LEAKAGE_COLS) | {LABEL_COL, GENE_COL} | set(ID_COLS)
    drop |= {c for c in df.columns if "esm_emb" in c}
    drop |= {"dms_score"}  # DMS is a label source, never a feature
    feats = [c for c in df.columns
             if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    logger.info("Feature set (bio_full, predictor scores INCLUDED): %d features", len(feats))
    return feats


# ===========================================================================
# Decision policy
# ===========================================================================
def select_threshold(y_true, y_prob) -> float:
    """Max MCC subject to recall >= RECALL_FLOOR; fallback to max-F_beta."""
    grid = np.linspace(0.01, 0.99, 393)
    best_thr, best_mcc, feasible = 0.5, -1.0, False
    for thr in grid:
        pred = (y_prob >= thr).astype(int)
        if recall_score(y_true, pred, zero_division=0) < RECALL_FLOOR:
            continue
        feasible = True
        mcc = matthews_corrcoef(y_true, pred)
        if mcc > best_mcc:
            best_mcc, best_thr = mcc, thr
    if feasible:
        return best_thr
    best_thr, best_fb, b2 = 0.5, -1.0, FBETA_BETA ** 2
    for thr in grid:
        pred = (y_prob >= thr).astype(int)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        fb = ((1 + b2) * p * r / (b2 * p + r)) if (b2 * p + r) > 0 else 0.0
        if fb > best_fb:
            best_fb, best_thr = fb, thr
    logger.warning("Recall floor %.2f unreachable; max-F%.0f thr=%.3f",
                   RECALL_FLOOR, FBETA_BETA, best_thr)
    return best_thr


def evaluate(y_true, y_prob, threshold) -> dict:
    """Full metric suite with selective-abstention block."""
    pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    m = {"threshold": round(float(threshold), 4),
         "mcc": round(matthews_corrcoef(y_true, pred), 4),
         "auroc": round(roc_auc_score(y_true, y_prob), 4),
         "auprc": round(average_precision_score(y_true, y_prob), 4),
         "brier": round(brier_score_loss(y_true, y_prob), 6),
         "precision": round(precision_score(y_true, pred, zero_division=0), 4),
         "recall": round(recall_score(y_true, pred, zero_division=0), 4),
         "f1": round(f1_score(y_true, pred, zero_division=0), 4),
         "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}
    confident = np.abs(y_prob - threshold) >= ABSTAIN_MARGIN
    if confident.sum() > 0:
        pc, yc = pred[confident], y_true[confident]
        m["selective"] = {"coverage": round(float(confident.mean()), 4),
                          "mcc": round(matthews_corrcoef(yc, pc), 4)
                          if len(np.unique(yc)) > 1 else None,
                          "recall": round(recall_score(yc, pc, zero_division=0), 4),
                          "precision": round(precision_score(yc, pc, zero_division=0), 4)}
    return m


def mcnemar_test(y_true, p1, p2) -> dict:
    """McNemar on paired (thresholded) predictions. p1=lgbm, p2=xattn."""
    n01 = int(np.sum((p1 == y_true) & (p2 != y_true)))  # lgbm right, xattn wrong
    n10 = int(np.sum((p1 != y_true) & (p2 == y_true)))  # lgbm wrong, xattn right
    n = n01 + n10
    try:
        from statsmodels.stats.contingency_tables import mcnemar
        table = [[int(np.sum((p1 == y_true) & (p2 == y_true))), n01],
                 [n10, int(np.sum((p1 != y_true) & (p2 != y_true)))]]
        res = mcnemar(table, exact=(n < 25), correction=True)
        stat, pval = float(res.statistic), float(res.pvalue)
    except Exception:
        from scipy.stats import chi2
        stat = ((abs(n01 - n10) - 1) ** 2 / n) if n > 0 else 0.0
        pval = float(1 - chi2.cdf(stat, df=1)) if n > 0 else 1.0
    winner = ("cross_attention" if n10 > n01 else "lightgbm") if pval < 0.05 else "tie"
    return {"n01_lgbm_right_xattn_wrong": n01, "n10_lgbm_wrong_xattn_right": n10,
            "statistic": round(stat, 4), "p_value": round(pval, 6), "winner": winner}


# ============================================================================
# Artifact persistence for the figure module
# ============================================================================
def save_oof_artifacts(path, y_true, oof_by_model: dict, config_tag: str):
    """Append/save OOF probability vectors for ROC/PR/confusion figures."""
    import os
    data = {}
    if os.path.exists(path):
        with np.load(path, allow_pickle=True) as z:
            data = {k: z[k] for k in z.files}
    data[f"{config_tag}__y"] = np.asarray(y_true)
    for name, probs in oof_by_model.items():
        data[f"{config_tag}__{name}"] = np.asarray(probs)
    np.savez_compressed(path, **data)
    logger.info("Saved OOF artifacts -> %s (%d arrays)", path, len(data))


def compute_and_save_shap(model, X_background, feature_names, out_path,
                          max_samples: int = 2000):
    """Compute SHAP values for a fitted tree model and persist them."""
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed; skipping SHAP artifact.")
        return
    n = min(max_samples, X_background.shape[0])
    idx = np.random.RandomState(RANDOM_STATE).choice(
        X_background.shape[0], size=n, replace=False)
    Xs = X_background[idx]
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(Xs)
        if isinstance(sv, list):  # older API: [class0, class1]
            sv = sv[1]
        np.savez_compressed(
            out_path, shap=np.asarray(sv), X=np.asarray(Xs),
            features=np.asarray(feature_names, dtype=object))
        logger.info("Saved SHAP artifact -> %s", out_path)
    except Exception as e:
        logger.warning("SHAP computation failed (%s); skipping.", e)
