from __future__ import annotations

import json
import logging
import math
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, TensorDataset

from config import (
    ENABLE_LORA,
    ESM_EMBED_DIM,
    ESM_LAYER,
    ESM_LR,
    ESM_MODEL_NAME,
    ESM_UNFREEZE_AFTER,
    HEAD_LR,
    LORA_ALPHA,
    LORA_BATCH_SIZE,
    LORA_DROPOUT,
    LORA_GRAD_ACCUM_STEPS,
    LORA_GRADIENT_CHECKPOINTING,
    LORA_MAX_EPOCHS,
    LORA_PATIENCE,
    LORA_PRECISION,
    LORA_RANK,
    LORA_SEED,
    LORA_TARGET_MODULES,
    PROPOSED_ARCHITECTURE,
    RANDOM_STATE,
    REPRODUCIBLE,
    TUNING_BEST_JSON,
)
from schema import (
    GENE_COL as GENE_COL,
    LABEL_COL,
    ROW_ID_COL,
    select_model_features,
)
from gpu_runtime import cuda_inventory, data_parallel

logger = logging.getLogger("pipeline.common")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def gpu_runtime_summary() -> dict[str, Any]:
    """Return the validated single/dual-GPU execution contract."""
    return cuda_inventory(DEVICE)


def _parallel_model(model: nn.Module) -> nn.Module:
    """Create a transient DataParallel view without changing saved state keys."""
    return data_parallel(model, DEVICE)
ESM_DIM = ESM_EMBED_DIM
RECALL_FLOOR = 0.90
FBETA_BETA = 2.0
ABSTAIN_MARGIN = 0.10
CALIBRATION_EPSILON = 1e-6

LGBM_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.02,
    "num_leaves": 30,
    "max_depth": -1,
    "min_child_samples": 60,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "n_jobs": -1,
    "verbose": -1,
    "deterministic": True,
    "force_col_wise": True,
    "random_state": RANDOM_STATE,
}
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
MODALITY_DROPOUT = 0.25
RELIABILITY_RESIDUAL_SCALE = 2.0
# Learned positive expert weights and an optional anchor/consistency objective.
# Availability masks apply after the positive floor; absent experts keep zero
# weight. Disable both auxiliary loss weights for the plain-BCE ablation.
EVIDENTIAL_PRECISION_FLOOR = 0.05
EVIDENTIAL_GATE_TEMPERATURE = 1.0
# Zero reproduces the previous gate; one bounds the correction by observed
# auxiliary quality. Learned expert weights are not calibrated uncertainty.
EVIDENTIAL_RELIABILITY_POWER = 1.0
RCDI_ANCHOR_WEIGHT = 0.30
RCDI_CONSISTENCY_WEIGHT = 0.20
RCDI_CONSISTENCY_MASK_PROBABILITY = 0.50
N_ENSEMBLE = 3
DEEP_MAX_EPOCHS = 60
DEEP_PATIENCE = 10
GRAD_CLIP = 1.0
DEEP_EARLY_STOP_METRIC = os.environ.get(
    "DEEP_EARLY_STOP_METRIC", "auprc"
).strip().lower()
if DEEP_EARLY_STOP_METRIC not in {"auprc", "auroc"}:
    raise ValueError("DEEP_EARLY_STOP_METRIC must be 'auprc' or 'auroc'")

TUNABLE_CONSTANTS = (
    "D_MODEL",
    "N_HEADS",
    "N_CROSS_BLOCKS",
    "N_FUSION_LAYERS",
    "N_ESM_SLOTS",
    "DROPOUT",
    "DEEP_LR",
    "DEEP_WD",
    "WARMUP_EPOCHS",
    "EMA_DECAY",
    "MIXUP_ALPHA",
    "LABEL_SMOOTH",
    "BATCH_SIZE",
    "MODALITY_DROPOUT",
    "RELIABILITY_RESIDUAL_SCALE",
    "EVIDENTIAL_PRECISION_FLOOR",
    "EVIDENTIAL_GATE_TEMPERATURE",
    "EVIDENTIAL_RELIABILITY_POWER",
    "RCDI_ANCHOR_WEIGHT",
    "RCDI_CONSISTENCY_WEIGHT",
    "RCDI_CONSISTENCY_MASK_PROBABILITY",
    "N_ENSEMBLE",
    "DEEP_MAX_EPOCHS",
    "DEEP_PATIENCE",
)

# These columns are consumed only by the reliability gate.  They are not
# unrestricted predictive inputs to the residual branch.  Keeping that
# distinction explicit prevents annotation availability from becoming a
# shortcut while still allowing the model to abstain from unreliable
# auxiliary modalities.
# Both fusion models share feature binding, diagnostics and sequence fallback.
# RELIABILITY_ARCHITECTURE identifies the prespecified proposed model;
# is_reliability_family() selects schema/diagnostic behavior for either model.
RELIABILITY_RESIDUAL_ARCHITECTURE = "reliability_residual"
EVIDENTIAL_RESIDUAL_ARCHITECTURE = "evidential_residual"
RELIABILITY_FAMILY_ARCHITECTURES = (
    RELIABILITY_RESIDUAL_ARCHITECTURE,
    EVIDENTIAL_RESIDUAL_ARCHITECTURE,
)
if PROPOSED_ARCHITECTURE not in RELIABILITY_FAMILY_ARCHITECTURES:
    raise ValueError(
        "PROPOSED_ARCHITECTURE must name a reliability-family architecture, "
        f"got {PROPOSED_ARCHITECTURE!r}"
    )
RELIABILITY_ARCHITECTURE = PROPOSED_ARCHITECTURE
RELIABILITY_ANCHOR_FEATURE = "esm_variant_score"
RELIABILITY_REQUIRED_GATE_FEATURES = (
    "HAS_STRUCTURE",
    "LOW_CONFIDENCE_STRUCTURE",
)
RELIABILITY_CONSERVATION_MISSING_FEATURES = (
    "GERP++_RS__missing",
    "phyloP100way_vertebrate__missing",
    "phastCons100way_vertebrate__missing",
)
RELIABILITY_OPTIONAL_GATE_FEATURES = (
    "HAS_DOMAIN_ANNOTATION",
    "HAS_ACTIVE_SITE_ANNOTATION",
    "HAS_BINDING_SITE_ANNOTATION",
    "HAS_TRANSMEMBRANE_ANNOTATION",
    "PLDDT_SCORE",
    "LOCAL_MEAN_PLDDT_8A",
    "LOCAL_MIN_PLDDT_8A",
    "LOCAL_CONFIDENT_CONTACT_FRACTION_8A",
    "LOCAL_CONTACT_COUNT_8A",
    "LOCAL_CONTACT_COUNT_12A",
    "LOCAL_LONG_RANGE_CONTACT_COUNT_8A",
)
RELIABILITY_GATE_FEATURES = tuple(
    dict.fromkeys(
        (
            *RELIABILITY_REQUIRED_GATE_FEATURES,
            *RELIABILITY_CONSERVATION_MISSING_FEATURES,
            *RELIABILITY_OPTIONAL_GATE_FEATURES,
        )
    )
)
# Binary coverage/missingness fields are restricted to the gate.  Continuous
# confidence and local-geometry descriptors may also be encoded by the
# dedicated structural expert because they carry biological signal rather than
# merely identifying which annotation source produced a row.
RELIABILITY_GATE_ONLY_FEATURES = (
    *RELIABILITY_REQUIRED_GATE_FEATURES,
    *RELIABILITY_CONSERVATION_MISSING_FEATURES,
    "HAS_DOMAIN_ANNOTATION",
    "HAS_ACTIVE_SITE_ANNOTATION",
    "HAS_BINDING_SITE_ANNOTATION",
    "HAS_TRANSMEMBRANE_ANNOTATION",
)
RELIABILITY_STRUCTURE_SIGNAL_FEATURES = (
    "SASA",
    "RELATIVE_SASA",
    "PLDDT_SCORE",
    "IS_IN_DOMAIN",
    "DISTANCE_TO_ACTIVE_SITE",
    "IS_ACTIVE_SITE",
    "IS_BINDING_SITE",
    "IS_TRANSMEMBRANE",
    "LOCAL_CONTACT_COUNT_8A",
    "LOCAL_CONTACT_COUNT_12A",
    "LOCAL_LONG_RANGE_CONTACT_COUNT_8A",
    "LOCAL_MEAN_PLDDT_8A",
    "LOCAL_MIN_PLDDT_8A",
    "LOCAL_CONFIDENT_CONTACT_FRACTION_8A",
    "LOCAL_MEAN_DISTANCE_8A",
    "LOCAL_HYDROPHOBIC_FRACTION_8A",
    "LOCAL_CHARGED_FRACTION_8A",
)
RELIABILITY_EVOLUTION_SIGNAL_FEATURES = (
    "GERP++_RS",
    "phyloP100way_vertebrate",
    "phastCons100way_vertebrate",
)
RELIABILITY_DIAGNOSTIC_COMPONENTS = (
    "anchor_logit",
    "anchor_probability",
    "gate",
    "bounded_residual",
    "hard_availability",
    "structure_reliability",
    "conservation_reliability",
)
RELIABILITY_AVAILABILITY_STRATA = {
    0: "anchor_only",
    1: "structure_only",
    2: "conservation_only",
    3: "structure_and_conservation",
}


# CHANGELOG 2026-09 (novelty N1): single predicate for "does this architecture
# use the reliability feature schema and expose reliability diagnostics?".
# Before this change the same question was asked as
# ``architecture == RELIABILITY_ARCHITECTURE`` at every dispatch site, which
# conflated two different questions: (a) which feature schema / preprocessing
# contract applies, and (b) which single architecture is *reported* as the
# proposed model.  Only (a) generalises to the new family member, so (a) now
# calls this predicate while (b) keeps comparing against
# ``RELIABILITY_ARCHITECTURE``.  With the default configuration the family has
# exactly one selected member and both spellings agree, so behaviour is
# unchanged.
def is_reliability_family(architecture: str) -> bool:
    """Return ``True`` for architectures in the reliability-residual family.

    Family membership implies all of the following contracts:

    * the model is constructed with ``feature_names`` and binds features by
      name rather than by position, so it needs
      :func:`architecture_feature_names` / :func:`architecture_feature_mask`;
    * ``RELIABILITY_ANCHOR_FEATURE`` and ``RELIABILITY_GATE_FEATURES`` are
      passed through the preprocessor unscaled (see
      :func:`reliability_passthrough_indices`);
    * ``forward_components`` is implemented and returns the nine-element tuple
      consumed by :func:`predict_reliability_components`;
    * the model falls back exactly to the monotone sequence anchor when no
      auxiliary modality is available.
    """
    return architecture in RELIABILITY_FAMILY_ARCHITECTURES


def set_seeds(seed: int = RANDOM_STATE, deterministic: bool | None = None) -> None:
    """Seed all supported random generators."""
    if deterministic is None:
        deterministic = REPRODUCIBLE
    if deterministic:
        # CUDA reads this before the first cuBLAS operation.  The local runner
        # sets it before process start; setdefault also protects direct stage
        # invocations that call set_seeds before constructing a model.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    else:
        torch.use_deterministic_algorithms(False)


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def validate_binary_labels(
    values: Iterable[Any],
    name: str = LABEL_COL,
    *,
    require_both_classes: bool = True,
) -> np.ndarray:
    """Validate nonempty binary labels."""
    labels = np.asarray(values)
    if labels.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if labels.size == 0:
        raise ValueError(f"{name} is empty")
    if pd.isna(labels).any():
        raise ValueError(f"{name} contains missing values")
    if np.iscomplexobj(labels):
        raise ValueError(f"{name} must be real binary values")
    try:
        numeric = labels.astype(np.float64)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} must contain binary values") from error
    if not np.isfinite(numeric).all() or not np.isin(numeric, [0.0, 1.0]).all():
        raise ValueError(f"{name} must contain only 0 and 1")
    labels = numeric.astype(np.int64)
    observed = set(np.unique(labels).tolist())
    if require_both_classes and observed != {0, 1}:
        raise ValueError(f"{name} must contain both classes; found {observed}")
    return labels


def validate_groups(groups: Iterable[Any], y: np.ndarray, n_splits: int) -> np.ndarray:
    """Validate group-disjoint split feasibility."""
    group_values = np.asarray(groups, dtype=object)
    if len(group_values) != len(y):
        raise ValueError("Group and label lengths differ")
    missing = pd.isna(group_values) | (pd.Series(group_values).astype(str).str.strip() == "")
    if np.asarray(missing).any():
        raise ValueError("Split groups contain missing values")
    unique_groups = np.unique(group_values)
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"Need at least {n_splits} groups; found {len(unique_groups)}"
        )
    for label in (0, 1):
        count = len(np.unique(group_values[y == label]))
        if count < n_splits:
            raise ValueError(
                f"Class {label} spans {count} groups; need {n_splits}"
            )
    return group_values


def make_group_splits(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int = RANDOM_STATE,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create validated stratified group splits (gene or homology component)."""
    labels = validate_binary_labels(y)
    group_values = validate_groups(groups, labels, n_splits)
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    indices = np.arange(len(labels))
    splits = list(splitter.split(indices, labels, group_values))
    for fold, (train, test) in enumerate(splits, 1):
        if not set(group_values[train]).isdisjoint(set(group_values[test])):
            raise RuntimeError("Split-group leakage detected")
        # StratifiedGroupKFold is a constrained heuristic: global class/group
        # feasibility does not guarantee that every realised fold contains both
        # classes.  Undefined fold-level AUROC/AUPRC must be rejected before a
        # publication run starts rather than discovered after model fitting.
        validate_binary_labels(labels[train], f"group split {fold} train labels")
        validate_binary_labels(labels[test], f"group split {fold} test labels")
    return splits


def split_fit_stop_calibration(
    outer_train: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create group-disjoint fit, stop and calibration sets."""
    outer_train = np.asarray(outer_train, dtype=int)
    local_y = y[outer_train]
    local_groups = groups[outer_train]
    first = make_group_splits(local_y, local_groups, 3, seed)[0]
    fit_local, hold_local = first
    hold_global = outer_train[hold_local]
    hold_y = y[hold_global]
    hold_groups = groups[hold_global]
    second = make_group_splits(hold_y, hold_groups, 2, seed + 1)[0]
    stop_local, calibration_local = second
    fit = outer_train[fit_local]
    stop = hold_global[stop_local]
    calibration = hold_global[calibration_local]
    partitions = (fit, stop, calibration)
    group_sets = [set(groups[index]) for index in partitions]
    if group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2]:
        raise RuntimeError("Inner group leakage detected")
    if group_sets[1] & group_sets[2]:
        raise RuntimeError("Calibration group leakage detected")
    for index in partitions:
        validate_binary_labels(y[index])
    return partitions


def split_fit_stop_temperature_threshold(
    outer_train: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create four group-disjoint training partitions."""
    outer_train = np.asarray(outer_train, dtype=int)
    local_y = y[outer_train]
    local_groups = groups[outer_train]
    fit_local, hold_local = make_group_splits(
        local_y, local_groups, 3, seed
    )[0]
    hold_global = outer_train[hold_local]
    hold_y = y[hold_global]
    hold_groups = groups[hold_global]
    hold_splits = make_group_splits(hold_y, hold_groups, 3, seed + 1)
    stop = hold_global[hold_splits[0][1]]
    temperature = hold_global[hold_splits[1][1]]
    threshold = hold_global[hold_splits[2][1]]
    fit = outer_train[fit_local]
    partitions = (fit, stop, temperature, threshold)
    group_sets = [set(groups[index]) for index in partitions]
    for first_index in range(len(group_sets)):
        for second_index in range(first_index + 1, len(group_sets)):
            if group_sets[first_index] & group_sets[second_index]:
                raise RuntimeError("Inner group leakage detected")
    for index in partitions:
        validate_binary_labels(y[index])
    return partitions


class FeatureTokenizer(nn.Module):
    """Convert scalar features into learned tokens."""

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_model))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values.unsqueeze(-1) * self.weight + self.bias


class CrossAttentionBlock(nn.Module):
    """Apply bidirectional cross-attention."""

    def __init__(self, width: int, heads: int, multiplier: int, dropout: float):
        super().__init__()
        self.bio_to_esm = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.esm_to_bio = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.bio_norm_one = nn.LayerNorm(width)
        self.esm_norm_one = nn.LayerNorm(width)
        self.bio_norm_two = nn.LayerNorm(width)
        self.esm_norm_two = nn.LayerNorm(width)
        self.bio_feedforward = nn.Sequential(
            nn.Linear(width, width * multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * multiplier, width),
        )
        self.esm_feedforward = nn.Sequential(
            nn.Linear(width, width * multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * multiplier, width),
        )

    def forward(
        self, bio: torch.Tensor, esm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bio_update, _ = self.bio_to_esm(bio, esm, esm)
        bio = self.bio_norm_one(bio + bio_update)
        bio = self.bio_norm_two(bio + self.bio_feedforward(bio))
        esm_update, _ = self.esm_to_bio(esm, bio, bio)
        esm = self.esm_norm_one(esm + esm_update)
        esm = self.esm_norm_two(esm + self.esm_feedforward(esm))
        return bio, esm


class CrossAttnFusionNet(nn.Module):
    """Legacy pooled-vector attention retained only as an explicit comparator.

    The ESM input is one mutation-site vector, so the projected slots are not
    residue tokens.  Publication code must not describe this architecture as
    local residue-level cross-attention.
    """

    def __init__(self, n_features: int, esm_dim: int = ESM_DIM):
        super().__init__()
        if D_MODEL % N_HEADS:
            raise ValueError("D_MODEL must divide evenly across heads")
        self.d_model = D_MODEL
        self.n_esm_slots = N_ESM_SLOTS
        self.tokenizer = FeatureTokenizer(n_features, D_MODEL)
        self.esm_projection = nn.Sequential(
            nn.LayerNorm(esm_dim),
            nn.Linear(esm_dim, N_ESM_SLOTS * D_MODEL),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.cross_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(D_MODEL, N_HEADS, 2, DROPOUT)
                for _ in range(N_CROSS_BLOCKS)
            ]
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=D_MODEL * 2,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(
            encoder_layer, num_layers=N_FUSION_LAYERS
        )
        self.head = nn.Sequential(
            nn.LayerNorm(D_MODEL * 2),
            nn.Linear(D_MODEL * 2, D_MODEL),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, 1),
        )

    def forward(self, bio_values: torch.Tensor, esm_values: torch.Tensor) -> torch.Tensor:
        batch_size = bio_values.size(0)
        bio = self.tokenizer(bio_values)
        esm = self.esm_projection(esm_values).view(
            batch_size, self.n_esm_slots, self.d_model
        )
        for block in self.cross_blocks:
            bio, esm = block(bio, esm)
        tokens = torch.cat([self.cls.expand(batch_size, -1, -1), bio, esm], dim=1)
        fused = self.fusion(tokens)
        cls_output = fused[:, 0]
        mean_output = fused[:, 1:].mean(dim=1)
        return self.head(torch.cat([cls_output, mean_output], dim=-1)).squeeze(-1)


class ConcatenationMLP(nn.Module):
    """Provide a simple fusion baseline."""

    def __init__(self, n_features: int, esm_dim: int = ESM_DIM):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(n_features + esm_dim),
            nn.Linear(n_features + esm_dim, D_MODEL * 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL * 2, D_MODEL),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, 1),
        )

    def forward(self, bio_values: torch.Tensor, esm_values: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([bio_values, esm_values], dim=1)).squeeze(-1)


class GatedFusionNet(nn.Module):
    """Compact, parameter-efficient gated fusion for pooled modalities."""

    def __init__(self, n_features: int, esm_dim: int = ESM_DIM):
        super().__init__()
        width = min(D_MODEL, 192)
        self.bio_encoder = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, width),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )
        self.esm_encoder = nn.Sequential(
            nn.LayerNorm(esm_dim),
            nn.Linear(esm_dim, width),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )
        self.gate = nn.Sequential(
            nn.Linear(width * 2, width),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(width * 2),
            nn.Linear(width * 2, width),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(width, 1),
        )

    def forward(self, bio_values: torch.Tensor, esm_values: torch.Tensor) -> torch.Tensor:
        bio = self.bio_encoder(bio_values)
        esm = self.esm_encoder(esm_values)
        gate = self.gate(torch.cat([bio, esm], dim=1))
        fused = gate * bio + (1.0 - gate) * esm
        disagreement = torch.abs(bio - esm)
        return self.head(torch.cat([fused, disagreement], dim=1)).squeeze(-1)


def reliability_feature_names(
    frame: pd.DataFrame, base_feature_names: Iterable[str]
) -> list[str]:
    """Return the stable, architecture-specific reliability feature schema.

    Availability and missingness indicators are intentionally appended only
    for :class:`ReliabilityResidualNet`; they remain excluded from ordinary
    fusion baselines and tabular predictors.
    """
    base = list(dict.fromkeys(str(name) for name in base_feature_names))
    extras = [
        name
        for name in RELIABILITY_GATE_FEATURES
        if name not in base
        and name in frame.columns
        and pd.api.types.is_numeric_dtype(frame[name])
    ]
    return [*base, *extras]


def architecture_feature_names(
    architecture: str,
    frame: pd.DataFrame,
    base_feature_names: Iterable[str],
) -> list[str]:
    """Select an architecture's feature order without broadening baselines."""
    base = list(dict.fromkeys(str(name) for name in base_feature_names))
    # CHANGELOG 2026-09 (novelty N1): feature-schema dispatch is family-wide.
    # Both reliability-family members bind features by name and require the
    # gate columns to be present, so both need the widened name list.
    if is_reliability_family(architecture):
        return reliability_feature_names(frame, base)
    return base


def reliability_passthrough_indices(feature_names: Iterable[str]) -> list[int]:
    """Locate anchor and gate inputs that must retain their original scales."""
    names = [str(name) for name in feature_names]
    gate_names = {RELIABILITY_ANCHOR_FEATURE, *RELIABILITY_GATE_FEATURES}
    return [index for index, name in enumerate(names) if name in gate_names]


def reliability_architecture_protocol(
    feature_names: Iterable[str] | None = None,
    model_config: dict[str, Any] | None = None,
    architecture: str | None = None,
) -> dict[str, Any]:
    """Describe the actual architecture and configuration stored in an artifact."""
    names = [] if feature_names is None else [str(name) for name in feature_names]
    selected_config = model_config or {}
    available_gate = [name for name in names if name in RELIABILITY_GATE_FEATURES]
    selected_architecture = (
        RELIABILITY_ARCHITECTURE if architecture is None else str(architecture)
    )
    if not is_reliability_family(selected_architecture):
        raise ValueError(
            "reliability_architecture_protocol requires a reliability-family "
            f"architecture, got {selected_architecture!r}"
        )
    is_evidential = selected_architecture == EVIDENTIAL_RESIDUAL_ARCHITECTURE
    protocol = {
        "architecture": selected_architecture,
        "sequence_anchor": RELIABILITY_ANCHOR_FEATURE,
        "anchor_direction": "lower_esm_log_likelihood_ratio_is_more_pathogenic",
        "anchor_constraint": "positive_softplus_slope_on_negated_score",
        "fusion_rule": (
            "anchor_logit_plus_quality_and_evidence_gate_times_weighted_expert_residual"
            if is_evidential
            else "anchor_logit_plus_reliability_gate_times_bounded_residual"
        ),
        "safe_fallback": (
            "exact_sequence_anchor_when_no_reliable_structure_or_conservation"
        ),
        "availability_features_are_gate_only": True,
        "local_structure_encoder": (
            "dedicated_lightweight_structural_expert_zeroed_when_structure_unreliable"
        ),
        "evolution_encoder": (
            "dedicated_conservation_expert_zeroed_when_conservation_unavailable"
        ),
        "modality_dropout_training_only": True,
        "mixup_disabled": True,
        "required_gate_features": list(RELIABILITY_REQUIRED_GATE_FEATURES),
        "conservation_missing_features": list(
            RELIABILITY_CONSERVATION_MISSING_FEATURES
        ),
        "optional_gate_features": list(RELIABILITY_OPTIONAL_GATE_FEATURES),
        "available_gate_features": available_gate,
        "modality_dropout_probability": float(
            selected_config.get("MODALITY_DROPOUT", MODALITY_DROPOUT)
        ),
        "maximum_absolute_residual_logit": float(
            selected_config.get(
                "RELIABILITY_RESIDUAL_SCALE", RELIABILITY_RESIDUAL_SCALE
            )
        ),
    }
    if is_evidential:
        protocol.update(
            {
                "expert_names": list(EvidentialResidualNet.EXPERT_NAMES),
                "maskable_experts": ["structure", "evolution"],
                "expert_weighting": "normalized_positive_learned_weights_times_availability",
                "uncertainty_interpretation": "learned_fusion_weights_not_calibrated_uncertainty",
                "evidence_gate": "one_minus_exp_negative_total_precision_over_tau",
                "availability_masking_precedes_precision_floor": False,
                "absent_expert_weight_is_zero": True,
                "observed_quality_gate_power": float(
                    selected_config.get(
                        "EVIDENTIAL_RELIABILITY_POWER", EVIDENTIAL_RELIABILITY_POWER
                    )
                ),
                "precision_floor": float(
                    selected_config.get(
                        "EVIDENTIAL_PRECISION_FLOOR", EVIDENTIAL_PRECISION_FLOOR
                    )
                ),
                "gate_temperature": float(
                    selected_config.get(
                        "EVIDENTIAL_GATE_TEMPERATURE", EVIDENTIAL_GATE_TEMPERATURE
                    )
                ),
                "training_objective": (
                    "reliability_consistency_degradation_invariance"
                    if EvidentialResidualNet.uses_rcdi_objective and any(
                        float(selected_config.get(name, globals()[name])) > 0.0
                        for name in ("RCDI_ANCHOR_WEIGHT", "RCDI_CONSISTENCY_WEIGHT")
                    )
                    else "binary_cross_entropy"
                ),
                "rcdi_anchor_weight": float(
                    selected_config.get("RCDI_ANCHOR_WEIGHT", RCDI_ANCHOR_WEIGHT)
                ),
                "rcdi_consistency_weight": float(
                    selected_config.get(
                        "RCDI_CONSISTENCY_WEIGHT", RCDI_CONSISTENCY_WEIGHT
                    )
                ),
                "rcdi_consistency_mask_probability": float(
                    selected_config.get(
                        "RCDI_CONSISTENCY_MASK_PROBABILITY",
                        RCDI_CONSISTENCY_MASK_PROBABILITY,
                    )
                ),
            }
        )
    return protocol


# CHANGELOG 2026-09 (refactor supporting novelty N1): the feature binding, gate
# normalisation and natural-reliability estimation that used to live directly
# inside ``ReliabilityResidualNet`` were lifted into this shared base class so
# that the second family member (``EvidentialResidualNet``) can reuse them
# verbatim instead of duplicating roughly 130 lines.  Three properties were
# preserved deliberately, so that the original architecture is unaffected:
#
#   * every attribute keeps its original name, therefore ``state_dict`` keys --
#     and every checkpoint written by an earlier run -- remain valid;
#   * the order in which RNG-consuming submodules are *constructed* is
#     unchanged: the four branch encoders are built here and the subclass heads
#     immediately afterwards.  The two anchor parameters are initialised from
#     constants and consume no RNG, so moving them into the base class cannot
#     perturb the stream.  For a given seed ``ReliabilityResidualNet`` is
#     therefore initialised bit-identically to before this refactor;
#   * the runtime order of stochastic operations inside ``forward`` is unchanged
#     (two ``torch.rand`` modality-dropout draws, then the dropout layers of the
#     structure / evolution / context / ESM encoders, then the residual head,
#     then the gate network, which has no dropout).
#
# CHANGELOG 2026-09 (performance): feature-name -> column lookups are resolved
# once in ``__init__`` instead of on every forward pass.  ``_column`` previously
# called ``tuple.index`` -- a linear scan over ~100 names -- about eight times
# per forward, and ``_normalized_gate_values`` launched two to three tiny CUDA
# kernels per gate feature from inside a Python loop (14 gate features under the
# production schema).  Both are now O(1) dictionary lookups plus whole-tensor
# operations.  The arithmetic, including the column order of every mean
# reduction, is unchanged, so outputs are bit-identical.
class ReliabilityFamilyNet(nn.Module):
    """Shared feature binding and reliability estimation for the family.

    Concrete members differ only in how the per-modality evidence is *fused*
    into the bounded residual; they share

    * the by-name feature binding and the schema validation below,
    * the four branch encoders (structure, evolution, biological context and
      pooled ESM embedding),
    * the monotone sequence anchor and its positivity-constrained slope,
    * the natural (data-derived) structure and conservation reliabilities,
    * training-only modality dropout,
    * and the exact-fallback guarantee: when neither trustworthy structure nor
      any conservation score is available, ``hard_availability`` is exactly zero
      and the emitted logit equals the anchor logit bit-for-bit.

    Subclasses must implement :meth:`_fuse` and must return the nine keys listed
    in :meth:`forward_components`.
    """

    disable_mixup = True
    # CHANGELOG 2026-09 (novelty N2): opt-in marker read by ``_train_single``.
    # ``False`` keeps the plain single-term BCE objective, which is what every
    # pre-existing architecture used, so their training runs are unchanged.
    uses_rcdi_objective = False

    def __init__(
        self,
        n_features: int,
        feature_names: Iterable[str],
        esm_dim: int = ESM_DIM,
    ):
        super().__init__()
        self.feature_names = tuple(str(name) for name in feature_names)
        if len(self.feature_names) != n_features:
            raise ValueError("Reliability feature names and feature count differ")
        if len(set(self.feature_names)) != n_features:
            raise ValueError("Reliability feature names must be unique")
        if RELIABILITY_ANCHOR_FEATURE not in self.feature_names:
            raise ValueError(
                "Reliability-family architectures require "
                f"{RELIABILITY_ANCHOR_FEATURE}"
            )
        missing_required = set(RELIABILITY_REQUIRED_GATE_FEATURES) - set(
            self.feature_names
        )
        if missing_required:
            raise ValueError(
                "Reliability architecture lacks required gate features: "
                f"{sorted(missing_required)}"
            )
        conservation_missing = [
            name
            for name in RELIABILITY_CONSERVATION_MISSING_FEATURES
            if name in self.feature_names
        ]
        if not conservation_missing:
            raise ValueError(
                "Reliability architecture requires at least one conservation "
                "missingness indicator"
            )

        # ``setdefault`` reproduces ``tuple.index`` semantics exactly (first
        # occurrence wins) in the event that a caller supplies a duplicated
        # name.  Upstream callers deduplicate, so the two agree in practice.
        feature_index: dict[str, int] = {}
        for position, name in enumerate(self.feature_names):
            feature_index.setdefault(name, position)
        self._feature_index = feature_index

        self.anchor_index = feature_index[RELIABILITY_ANCHOR_FEATURE]
        self.gate_names = tuple(
            name for name in RELIABILITY_GATE_FEATURES if name in feature_index
        )
        self.gate_indices = tuple(feature_index[name] for name in self.gate_names)
        excluded = {RELIABILITY_ANCHOR_FEATURE, *RELIABILITY_GATE_ONLY_FEATURES}
        predictive_names = tuple(
            name for name in self.feature_names if name not in excluded
        )
        self.structure_names = tuple(
            name
            for name in predictive_names
            if name in RELIABILITY_STRUCTURE_SIGNAL_FEATURES
        )
        self.evolution_names = tuple(
            name
            for name in predictive_names
            if name in RELIABILITY_EVOLUTION_SIGNAL_FEATURES
        )
        specialized = {*self.structure_names, *self.evolution_names}
        self.context_names = tuple(
            name for name in predictive_names if name not in specialized
        )
        self.structure_indices = tuple(
            feature_index[name] for name in self.structure_names
        )
        self.evolution_indices = tuple(
            feature_index[name] for name in self.evolution_names
        )
        self.context_indices = tuple(
            feature_index[name] for name in self.context_names
        )
        self.conservation_missing_names = tuple(conservation_missing)

        # --- precomputed column addresses for the reliability estimator ------
        # ``HAS_STRUCTURE`` and ``LOW_CONFIDENCE_STRUCTURE`` are guaranteed
        # present by the ``missing_required`` check above.
        self._structure_flag_index = feature_index["HAS_STRUCTURE"]
        self._low_confidence_index = feature_index["LOW_CONFIDENCE_STRUCTURE"]
        self._plddt_quality_indices = tuple(
            feature_index[name]
            for name in ("PLDDT_SCORE", "LOCAL_MEAN_PLDDT_8A", "LOCAL_MIN_PLDDT_8A")
            if name in feature_index
        )
        self._contact_quality_indices = tuple(
            feature_index[name]
            for name in ("LOCAL_CONFIDENT_CONTACT_FRACTION_8A",)
            if name in feature_index
        )
        self._conservation_missing_indices = tuple(
            feature_index[name] for name in self.conservation_missing_names
        )

        # --- precomputed gate normalisation constants ------------------------
        # Registered non-persistently so that ``Module.to(device)`` moves them
        # while ``state_dict`` keys stay exactly as they were before this
        # refactor (a persistent buffer would break strict checkpoint loading
        # and would be tracked pointlessly by ``EMA``).
        plddt_gate_names = {"PLDDT_SCORE", "LOCAL_MEAN_PLDDT_8A", "LOCAL_MIN_PLDDT_8A"}
        count_gate_names = {
            "LOCAL_CONTACT_COUNT_8A",
            "LOCAL_CONTACT_COUNT_12A",
            "LOCAL_LONG_RANGE_CONTACT_COUNT_8A",
        }
        self.register_buffer(
            "_gate_divisor",
            torch.tensor(
                [
                    100.0 if name in plddt_gate_names else 1.0
                    for name in self.gate_names
                ],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_gate_is_count",
            torch.tensor(
                [name in count_gate_names for name in self.gate_names],
                dtype=torch.bool,
            ),
            persistent=False,
        )

        width = min(D_MODEL, 128)
        self.width = width
        self.modality_dropout = float(MODALITY_DROPOUT)
        self.residual_scale = float(RELIABILITY_RESIDUAL_SCALE)
        branch_width = max(24, width // 2)
        self.branch_width = branch_width

        def branch(input_width: int) -> nn.Sequential:
            actual_width = max(1, input_width)
            return nn.Sequential(
                nn.LayerNorm(actual_width),
                nn.Linear(actual_width, branch_width),
                nn.GELU(),
                nn.Dropout(DROPOUT),
            )

        self.structure_encoder = branch(len(self.structure_indices))
        self.evolution_encoder = branch(len(self.evolution_indices))
        self.context_encoder = branch(len(self.context_indices))
        self.esm_encoder = nn.Sequential(
            nn.LayerNorm(esm_dim),
            nn.Linear(esm_dim, branch_width),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )
        # inverse-softplus(1) initializes an interpretable unit monotonic slope.
        # Both parameters are constant-initialised and consume no RNG.
        self.anchor_raw_slope = nn.Parameter(torch.tensor(0.5413248546))
        self.anchor_intercept = nn.Parameter(torch.zeros(()))

    # ------------------------------------------------------------------ utils
    def _column(self, values: torch.Tensor, name: str) -> torch.Tensor:
        return values[:, self._feature_index[name]]

    @staticmethod
    def _select(values: torch.Tensor, indices: tuple[int, ...]) -> torch.Tensor:
        return (
            values[:, indices]
            if indices
            else values.new_zeros((len(values), 1))
        )

    def _normalized_gate_values(self, bio_values: torch.Tensor) -> torch.Tensor:
        """Map raw gate columns onto a common ``[0, 1]`` scale.

        Vectorised equivalent of the previous per-feature Python loop:
        pLDDT-like columns are divided by 100, raw contact counts are
        ``log1p``-compressed by ``log(65)`` (their divisor is 1.0, so ``scaled``
        equals the raw value for those columns), and everything else is passed
        through.  All columns are finally clamped to ``[0, 1]``, exactly as
        before.
        """
        scaled = bio_values[:, self.gate_indices] / self._gate_divisor
        compressed = torch.log1p(scaled.clamp_min(0.0)) / math.log(65.0)
        return torch.where(self._gate_is_count, compressed, scaled).clamp(0.0, 1.0)

    def _natural_reliability(
        self, bio_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        has_structure = bio_values[:, self._structure_flag_index].clamp(0.0, 1.0)
        low_confidence = bio_values[:, self._low_confidence_index].clamp(0.0, 1.0)
        # ``torch.cat`` of the two blocks reproduces the column order of the
        # previous ``torch.stack`` of individual columns (the three pLDDT
        # descriptors first, then the confident-contact fraction), so the mean
        # is taken over the same values in the same order.
        quality_blocks: list[torch.Tensor] = []
        if self._plddt_quality_indices:
            quality_blocks.append(
                (bio_values[:, self._plddt_quality_indices] / 100.0).clamp(0.0, 1.0)
            )
        if self._contact_quality_indices:
            quality_blocks.append(
                bio_values[:, self._contact_quality_indices].clamp(0.0, 1.0)
            )
        quality = (
            torch.cat(quality_blocks, dim=1).mean(dim=1)
            if quality_blocks
            else torch.ones_like(has_structure)
        )
        structure_reliability = has_structure * (1.0 - low_confidence) * quality

        missing_values = bio_values[:, self._conservation_missing_indices].clamp(
            0.0, 1.0
        )
        conservation_reliability = (1.0 - missing_values.mean(dim=1)).clamp(0.0, 1.0)
        hard_availability = torch.maximum(
            (structure_reliability > 0.0).to(bio_values.dtype),
            (conservation_reliability > 0.0).to(bio_values.dtype),
        )
        return structure_reliability, conservation_reliability, hard_availability

    def _anchor_logit(self, bio_values: torch.Tensor) -> torch.Tensor:
        anchor_score = bio_values[:, self.anchor_index]
        anchor_slope = torch.nn.functional.softplus(self.anchor_raw_slope)
        return self.anchor_intercept - anchor_slope * anchor_score

    def _encoded_modalities(
        self, bio_values: torch.Tensor, esm_values: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Run the shared encoders once and return their outputs.

        The consistency objective reuses these encodings for a degraded view,
        avoiding a second pass through the projection of frozen ESM features.
        """
        anchor_logit = self._anchor_logit(bio_values)
        structure_reliability, conservation_reliability, hard_availability = (
            self._natural_reliability(bio_values)
        )
        structure_values = self._select(bio_values, self.structure_indices)
        evolution_values = self._select(bio_values, self.evolution_indices)
        context_values = self._select(bio_values, self.context_indices)
        if self.training and self.modality_dropout > 0.0:
            structure_keep = (
                torch.rand(len(bio_values), device=bio_values.device)
                >= self.modality_dropout
            ).to(bio_values.dtype)
            evolution_keep = (
                torch.rand(len(bio_values), device=bio_values.device)
                >= self.modality_dropout
            ).to(bio_values.dtype)
            structure_values = structure_values * structure_keep[:, None]
            evolution_values = evolution_values * evolution_keep[:, None]
            structure_reliability = structure_reliability * structure_keep
            conservation_reliability = conservation_reliability * evolution_keep
            hard_availability = torch.maximum(
                (structure_reliability > 0.0).to(bio_values.dtype),
                (conservation_reliability > 0.0).to(bio_values.dtype),
            )

        structure = self.structure_encoder(structure_values)
        structure = structure * structure_reliability[:, None]
        evolution = self.evolution_encoder(evolution_values)
        evolution = evolution * conservation_reliability[:, None]
        context = self.context_encoder(context_values)
        esm = self.esm_encoder(esm_values)
        return {
            "anchor_logit": anchor_logit,
            "structure": structure,
            "evolution": evolution,
            "context": context,
            "esm": esm,
            "structure_reliability": structure_reliability,
            "conservation_reliability": conservation_reliability,
            "hard_availability": hard_availability,
        }

    def _degraded_views(
        self,
        encoded: dict[str, torch.Tensor],
        bio_values: torch.Tensor,
        structure_keep: torch.Tensor | None,
        evolution_keep: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Apply an optional availability mask to the encoded modalities.

        Returns ``(structure, evolution, structure_reliability,
        conservation_reliability, hard_availability)``.  With no mask the
        encoded tensors are returned unchanged -- no additional operations are
        issued -- so the undegraded path is bit-identical to the pre-refactor
        implementation.
        """
        if structure_keep is None and evolution_keep is None:
            return (
                encoded["structure"],
                encoded["evolution"],
                encoded["structure_reliability"],
                encoded["conservation_reliability"],
                encoded["hard_availability"],
            )
        structure = encoded["structure"]
        evolution = encoded["evolution"]
        structure_reliability = encoded["structure_reliability"]
        conservation_reliability = encoded["conservation_reliability"]
        if structure_keep is not None:
            structure = structure * structure_keep[:, None]
            structure_reliability = structure_reliability * structure_keep
        if evolution_keep is not None:
            evolution = evolution * evolution_keep[:, None]
            conservation_reliability = conservation_reliability * evolution_keep
        hard_availability = torch.maximum(
            (structure_reliability > 0.0).to(bio_values.dtype),
            (conservation_reliability > 0.0).to(bio_values.dtype),
        )
        return (
            structure,
            evolution,
            structure_reliability,
            conservation_reliability,
            hard_availability,
        )

    def _fuse(
        self,
        encoded: dict[str, torch.Tensor],
        bio_values: torch.Tensor,
        structure_keep: torch.Tensor | None = None,
        evolution_keep: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward_components(
        self, bio_values: torch.Tensor, esm_values: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return the diagnostic decomposition of the prediction.

        Every family member returns at least ``logits``, ``anchor_logit``,
        ``residual``, ``gate``, ``hard_availability``,
        ``structure_reliability``, ``conservation_reliability``,
        ``structure_embedding`` and ``evolution_embedding``.  Members may add
        further keys; consumers ignore unknown ones.
        """
        encoded = self._encoded_modalities(bio_values, esm_values)
        return self._fuse(encoded, bio_values)

    def rcdi_components(
        self, bio_values: torch.Tensor, esm_values: torch.Tensor, mask_probability: float
    ) -> dict[str, torch.Tensor]:
        """Return the full view plus a degraded view for the RCDI objective.

        The views share branch encodings. Only removal of present evidence
        contributes to the loss; removal of fully trusted evidence has zero
        weight. Independent head dropout also contributes to the logit gap,
        so this is a robustness regularizer, not a causal attribution.
        """
        encoded = self._encoded_modalities(bio_values, esm_values)
        full = self._fuse(encoded, bio_values)
        probability = float(mask_probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("Consistency mask probability must be in [0, 1]")
        if probability <= 0.0:
            zeros = torch.zeros_like(full["logits"])
            return {
                **full,
                "degraded_logits": full["logits"],
                "consistency_weight": zeros,
            }
        rows = len(bio_values)
        structure_keep = (
            torch.rand(rows, device=bio_values.device) >= probability
        ).to(bio_values.dtype)
        evolution_keep = (
            torch.rand(rows, device=bio_values.device) >= probability
        ).to(bio_values.dtype)
        degraded = self._fuse(
            encoded,
            bio_values,
            structure_keep=structure_keep,
            evolution_keep=evolution_keep,
        )
        structure_removed = (1.0 - structure_keep) * (
            encoded["structure_reliability"] > 0.0
        ).to(bio_values.dtype)
        evolution_removed = (1.0 - evolution_keep) * (
            encoded["conservation_reliability"] > 0.0
        ).to(bio_values.dtype)
        removed_reliability = torch.maximum(
            encoded["structure_reliability"] * structure_removed,
            encoded["conservation_reliability"] * evolution_removed,
        ).clamp(0.0, 1.0)
        removed_any = torch.maximum(structure_removed, evolution_removed)
        return {
            **full,
            "degraded_logits": degraded["logits"],
            "consistency_weight": removed_any * (1.0 - removed_reliability),
        }

    def forward(
        self, bio_values: torch.Tensor, esm_values: torch.Tensor
    ) -> torch.Tensor:
        return self.forward_components(bio_values, esm_values)["logits"]


class ReliabilityResidualNet(ReliabilityFamilyNet):
    """Reliability-conditioned residual fusion with an exact sequence fallback.

    The sequence anchor is a monotonic transform of the ESM variant score.
    Auxiliary tabular and pooled-embedding evidence can only provide a bounded
    residual correction through a gate whose hard availability component is
    zero when both trustworthy structure and conservation are unavailable.
    Availability indicators never enter the residual predictor itself.
    """

    # CHANGELOG 2026-09 (novelty N2): the original architecture keeps the plain
    # BCE objective so that it remains an exact reproduction of the published
    # baseline and a clean ablation of the evidential fusion rule.  Set this to
    # ``True`` (``common.ReliabilityResidualNet.uses_rcdi_objective = True``) to
    # run the "RCDI on the original architecture" ablation cell.
    uses_rcdi_objective = False

    def __init__(
        self,
        n_features: int,
        feature_names: Iterable[str],
        esm_dim: int = ESM_DIM,
    ):
        super().__init__(n_features, feature_names, esm_dim=esm_dim)
        width = self.width
        branch_width = self.branch_width
        self.residual_head = nn.Sequential(
            nn.LayerNorm(branch_width * 5),
            nn.Linear(branch_width * 5, width),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(width, 1),
            nn.Tanh(),
        )
        self.gate_network = nn.Sequential(
            nn.Linear(len(self.gate_indices), max(8, min(32, width // 4))),
            nn.GELU(),
            nn.Linear(max(8, min(32, width // 4)), 1),
            nn.Sigmoid(),
        )

    # CHANGELOG 2026-09 (refactor): the body below is the second half of the
    # original ``forward_components``, verbatim except that the encoder outputs
    # now arrive through ``encoded`` instead of being computed inline, and the
    # optional keep-masks (used only by the RCDI objective) are applied first.
    # With both masks ``None`` -- which is the case for every call made through
    # ``forward``/``forward_components`` -- the computation is identical.
    def _fuse(
        self,
        encoded: dict[str, torch.Tensor],
        bio_values: torch.Tensor,
        structure_keep: torch.Tensor | None = None,
        evolution_keep: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        (
            structure,
            evolution,
            structure_reliability,
            conservation_reliability,
            hard_availability,
        ) = self._degraded_views(encoded, bio_values, structure_keep, evolution_keep)
        anchor_logit = encoded["anchor_logit"]
        modality_disagreement = torch.abs(structure - evolution)
        residual = self.residual_head(
            torch.cat(
                [
                    structure,
                    evolution,
                    encoded["context"],
                    encoded["esm"],
                    modality_disagreement,
                ],
                dim=1,
            )
        ).squeeze(-1) * self.residual_scale
        learned_gate = self.gate_network(
            self._normalized_gate_values(bio_values)
        ).squeeze(-1)
        evidence_quality = torch.maximum(
            structure_reliability, conservation_reliability
        ).clamp(0.0, 1.0)
        gate = learned_gate * evidence_quality * hard_availability
        logits = anchor_logit + gate * residual
        return {
            "logits": logits,
            "anchor_logit": anchor_logit,
            "residual": residual,
            "gate": gate,
            "hard_availability": hard_availability,
            "structure_reliability": structure_reliability,
            "conservation_reliability": conservation_reliability,
            "structure_embedding": structure,
            "evolution_embedding": evolution,
        }


class EvidentialResidualNet(ReliabilityFamilyNet):
    """Availability-weighted bounded expert residual over the sequence anchor.

    The legacy name and precision field names are retained for checkpoints.
    These are learned positive fusion weights, without a probabilistic
    inverse-variance or epistemic-uncertainty interpretation. Such claims need
    separate validation (https://proceedings.mlr.press/v235/juergens24a.html).
    Quality attenuation is an ablatable hypothesis, not established novelty.
    """

    # Both auxiliary losses can be disabled for a separately trained ablation.
    uses_rcdi_objective = True

    # Expert order is fixed and persisted in the diagnostics contract.  Only the
    # first two are maskable by availability; biological context and the pooled
    # ESM embedding are always present by construction (context features are
    # imputed upstream and the ESM embedding is a hard requirement of the run).
    EXPERT_NAMES = ("structure", "evolution", "context", "esm")

    def __init__(
        self,
        n_features: int,
        feature_names: Iterable[str],
        esm_dim: int = ESM_DIM,
    ):
        super().__init__(n_features, feature_names, esm_dim=esm_dim)
        width = self.width
        branch_width = self.branch_width
        self.n_experts = len(self.EXPERT_NAMES)
        self.precision_floor = float(EVIDENTIAL_PRECISION_FLOOR)
        self.gate_temperature = float(EVIDENTIAL_GATE_TEMPERATURE)
        self.reliability_power = float(EVIDENTIAL_RELIABILITY_POWER)
        if not math.isfinite(self.precision_floor) or self.precision_floor < 0.0:
            raise ValueError("Expert weight floor must be finite and nonnegative")
        if not math.isfinite(self.gate_temperature) or self.gate_temperature <= 0.0:
            raise ValueError("Evidence gate temperature must be finite and positive")
        if not math.isfinite(self.reliability_power) or self.reliability_power < 0.0:
            raise ValueError("Reliability gate power must be finite and nonnegative")

        # Each small head emits an opinion and an unconstrained weight parameter
        # from its modality embedding and a shared disagreement summary.
        head_width = max(16, width // 2)
        self.expert_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(branch_width * 2),
                    nn.Linear(branch_width * 2, head_width),
                    nn.GELU(),
                    nn.Dropout(DROPOUT),
                    nn.Linear(head_width, 2),
                )
                for _ in self.EXPERT_NAMES
            ]
        )
        # Shared initialization of biases; head outputs still vary by expert.
        self.expert_precision_bias = nn.Parameter(torch.zeros(self.n_experts))

    def _fuse(
        self,
        encoded: dict[str, torch.Tensor],
        bio_values: torch.Tensor,
        structure_keep: torch.Tensor | None = None,
        evolution_keep: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        (
            structure,
            evolution,
            structure_reliability,
            conservation_reliability,
            hard_availability,
        ) = self._degraded_views(encoded, bio_values, structure_keep, evolution_keep)
        anchor_logit = encoded["anchor_logit"]
        context = encoded["context"]
        esm = encoded["esm"]

        # Shared disagreement summary, appended to every expert input so that a
        # single expert can moderate its own confidence when the modalities
        # conflict.  Shape (rows, branch_width).
        disagreement = torch.abs(structure - evolution)
        expert_inputs = (structure, evolution, context, esm)
        raw_opinions: list[torch.Tensor] = []
        raw_log_precisions: list[torch.Tensor] = []
        for index, head in enumerate(self.expert_heads):
            # (rows, 2) -> two (rows,) columns.
            output = head(torch.cat([expert_inputs[index], disagreement], dim=1))
            raw_opinions.append(output[:, 0])
            raw_log_precisions.append(output[:, 1])
        # (rows, n_experts)
        opinions = torch.tanh(torch.stack(raw_opinions, dim=1))
        raw_log_precision = (
            torch.stack(raw_log_precisions, dim=1) + self.expert_precision_bias
        )

        # Multiply AFTER adding the floor so absent experts retain zero weight.
        # Accumulate weights in FP32 even when expert heads run under autocast.
        ones = torch.ones_like(structure_reliability)
        availability = torch.stack(
            [structure_reliability, conservation_reliability, ones, ones], dim=1
        ).clamp(0.0, 1.0)
        precision = (
            torch.nn.functional.softplus(raw_log_precision.float()) + self.precision_floor
        ) * availability
        total_precision = precision.sum(dim=1)

        # Normalize learned weights over present experts; these are not
        # inverse-variance estimates. Guard underflow when the floor is zero.
        weights = precision / total_precision.clamp_min(CALIBRATION_EPSILON)[:, None]
        pooled_opinion = torch.tanh((weights * opinions).sum(dim=1))

        # Stable saturation of the total learned weight.
        evidence_gate = -torch.expm1(-total_precision / self.gate_temperature)
        # The hard availability factor is retained so that a row with no
        # trustworthy auxiliary modality falls back to the anchor even if the
        # always-present context/ESM experts report high precision.  This keeps
        # the safe-fallback contract identical to the original architecture and
        # keeps the two directly comparable.
        quality = torch.maximum(structure_reliability, conservation_reliability)
        quality_gate = (
            quality.pow(self.reliability_power)
            if self.reliability_power > 0.0
            else torch.ones_like(quality)
        )
        gate = evidence_gate * quality_gate * hard_availability
        residual = pooled_opinion * self.residual_scale
        logits = anchor_logit + gate * residual
        return {
            "logits": logits,
            "anchor_logit": anchor_logit,
            "residual": residual,
            "gate": gate,
            "hard_availability": hard_availability,
            "structure_reliability": structure_reliability,
            "conservation_reliability": conservation_reliability,
            "structure_embedding": structure,
            "evolution_embedding": evolution,
            # Extra diagnostics, ignored by consumers that do not know them.
            "expert_weights": weights,
            "expert_residuals": opinions,
            "total_precision": total_precision,
        }


class EMA:
    """Track an exponential moving average."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for key, value in model.state_dict().items():
            if value.dtype.is_floating_point:
                self.shadow[key].mul_(self.decay).add_(
                    value.detach(), alpha=1.0 - self.decay
                )
            else:
                self.shadow[key] = value.detach().clone()


class EnsembleModel(nn.Module):
    """Average member logits and apply an affine held-out calibration map."""

    def __init__(
        self,
        members: list[nn.Module],
        temperature: float = 1.0,
        calibration_slope: float | None = None,
        calibration_intercept: float = 0.0,
    ):
        super().__init__()
        self.members = nn.ModuleList(members)
        # T is retained so pre-publication bundles remain loadable.
        self.T = float(temperature)
        self.calibration_slope = (
            float(calibration_slope)
            if calibration_slope is not None
            else 1.0 / max(float(temperature), CALIBRATION_EPSILON)
        )
        self.calibration_intercept = float(calibration_intercept)

    def forward(self, bio_values: torch.Tensor, esm_values: torch.Tensor) -> torch.Tensor:
        logits = torch.stack(
            [member(bio_values, esm_values) for member in self.members], dim=0
        )
        return logits.mean(dim=0)


def _model_factory(
    architecture: str,
    n_features: int,
    feature_names: Iterable[str] | None = None,
) -> nn.Module:
    if architecture == "cross_attention":
        return CrossAttnFusionNet(n_features)
    if architecture == "concatenation":
        return ConcatenationMLP(n_features)
    if architecture == "gated_fusion":
        return GatedFusionNet(n_features)
    # CHANGELOG 2026-09 (novelty N1): construction dispatch is family-wide and
    # keyed on the architecture *name* rather than on RELIABILITY_ARCHITECTURE,
    # so that both members can be trained in the same run (the original one as a
    # mandatory ablation of the evidential fusion rule).
    if architecture in RELIABILITY_FAMILY_ARCHITECTURES:
        if feature_names is None:
            raise ValueError("Reliability architecture requires ordered feature names")
        if architecture == EVIDENTIAL_RESIDUAL_ARCHITECTURE:
            return EvidentialResidualNet(n_features, feature_names)
        return ReliabilityResidualNet(n_features, feature_names)
    raise ValueError(f"Unknown architecture: {architecture}")


def _loader(
    bio_values: np.ndarray,
    esm_values: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int = RANDOM_STATE,
    workers: int = 0,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(bio_values, dtype=np.float32)),
        torch.from_numpy(np.asarray(esm_values, dtype=np.float32)),
        torch.from_numpy(np.asarray(labels, dtype=np.float32)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=DEVICE == "cuda",
        num_workers=workers,
        persistent_workers=workers > 0,
        worker_init_fn=_seed_worker if workers else None,
        generator=generator,
    )


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    bio_values: np.ndarray,
    esm_values: np.ndarray,
    batch_size: int = 2048,
    *,
    loader: DataLoader | None = None,
    num_rows: int | None = None,
) -> np.ndarray:
    """Return uncalibrated logits safely.

    CHANGELOG 2026-09 (optimization): ``loader`` / ``num_rows`` let a caller
    hand in a prebuilt, evaluator-owned DataLoader so a fixed inference set
    (e.g. the early-stop validation set inside ``_train_single``) is wrapped in
    a TensorDataset / DataLoader only once instead of once per epoch.  When
    ``loader`` is supplied the ``bio_values`` / ``esm_values`` / ``batch_size``
    arguments are ignored; ``num_rows`` must then give the row count so the
    output can be preallocated instead of concatenated.
    """
    if loader is None:
        if len(bio_values) != len(esm_values):
            raise ValueError("Bio and ESM row counts differ")
        if num_rows is not None and num_rows != len(bio_values):
            raise ValueError("num_rows differs from the inference input length")
        num_rows = len(bio_values)
    elif num_rows is None:
        raise ValueError("A prebuilt inference loader requires num_rows")
    if num_rows < 0 or batch_size < 1:
        raise ValueError("Inference row count must be nonnegative and batch size positive")
    model.eval()
    inference_model = _parallel_model(model)
    inference_model.eval()
    if loader is None:
        dummy = np.zeros(len(bio_values), dtype=np.float32)
        loader = _loader(
            bio_values, esm_values, dummy, batch_size, False
        )
    result = np.empty(num_rows, dtype=np.float32)
    written = 0
    for bio_batch, esm_batch, _ in loader:
        with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
            logits = inference_model(
                bio_batch.to(DEVICE, non_blocking=DEVICE == "cuda"),
                esm_batch.to(DEVICE, non_blocking=DEVICE == "cuda"),
            )
        batch_logits = logits.float().cpu().numpy()
        if batch_logits.shape != (len(bio_batch),):
            raise ValueError("Inference must produce one logit per input row")
        end = written + len(batch_logits)
        if end > num_rows:
            raise ValueError("Inference loader yielded more rows than declared")
        result[written:end] = batch_logits
        written = end
    if written != num_rows:
        raise ValueError(f"Inference loader yielded {written} logits; expected {num_rows}")
    if not np.isfinite(result).all():
        raise ValueError("Inference logits contain nonfinite values")
    del inference_model
    return result


def predict(
    model: nn.Module,
    bio_values: np.ndarray,
    esm_values: np.ndarray,
    batch_size: int = 2048,
    *,
    loader: DataLoader | None = None,
    num_rows: int | None = None,
) -> np.ndarray:
    """Return calibrated probabilities safely.

    CHANGELOG 2026-09 (optimization): ``loader`` / ``num_rows`` are forwarded to
    ``predict_logits`` so a fixed validation set can reuse one prebuilt
    DataLoader across epochs (see ``_train_single``).
    """
    logits = predict_logits(
        model,
        bio_values,
        esm_values,
        batch_size,
        loader=loader,
        num_rows=num_rows,
    )
    if logits.size == 0:
        return logits
    if hasattr(model, "calibration_slope"):
        slope = float(model.calibration_slope)
        intercept = float(getattr(model, "calibration_intercept", 0.0))
        calibrated_logits = slope * logits + intercept
    else:
        temperature = max(float(getattr(model, "T", 1.0)), CALIBRATION_EPSILON)
        calibrated_logits = logits / temperature
    return torch.sigmoid(torch.from_numpy(calibrated_logits)).numpy()


class _ReliabilityComponentForward(nn.Module):
    """Expose reliability diagnostics through ``forward`` for DataParallel."""

    # CHANGELOG 2026-09 (novelty N1): the annotation was widened from
    # ``ReliabilityResidualNet`` to the shared base class.  Only the nine common
    # ``forward_components`` keys are read here, so both members work unchanged.
    def __init__(self, model: nn.Module, members: list[ReliabilityFamilyNet]):
        super().__init__()
        self.members = nn.ModuleList(members)
        if hasattr(model, "calibration_slope"):
            self.calibration_slope = float(model.calibration_slope)
            self.calibration_intercept = float(
                getattr(model, "calibration_intercept", 0.0)
            )
        else:
            temperature = max(
                float(getattr(model, "T", 1.0)), CALIBRATION_EPSILON
            )
            self.calibration_slope = 1.0 / temperature
            self.calibration_intercept = 0.0

    def forward(
        self, bio_values: torch.Tensor, esm_values: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        member_components = [
            member.forward_components(bio_values, esm_values)
            for member in self.members
        ]
        final_logit = torch.stack(
            [component["logits"] for component in member_components], dim=0
        ).mean(dim=0)
        anchor_logit = torch.stack(
            [component["anchor_logit"] for component in member_components], dim=0
        ).mean(dim=0)
        gate = torch.stack(
            [component["gate"] for component in member_components], dim=0
        ).mean(dim=0)
        bounded_residual = torch.stack(
            [component["residual"] for component in member_components], dim=0
        ).mean(dim=0)

        invariant_components: dict[str, torch.Tensor] = {}
        for name in (
            "hard_availability",
            "structure_reliability",
            "conservation_reliability",
        ):
            reference = member_components[0][name]
            if any(
                not torch.allclose(
                    reference, component[name], atol=1e-6, rtol=0.0
                )
                for component in member_components[1:]
            ):
                raise RuntimeError(f"Reliability ensemble members disagree on {name}")
            invariant_components[name] = reference

        final_calibrated_logit = (
            self.calibration_slope * final_logit.float()
            + self.calibration_intercept
        )
        anchor_calibrated_logit = (
            self.calibration_slope * anchor_logit.float()
            + self.calibration_intercept
        )
        return {
            "anchor_logit": anchor_logit,
            "anchor_probability": torch.sigmoid(anchor_calibrated_logit),
            "gate": gate,
            "bounded_residual": bounded_residual,
            **invariant_components,
            "model_probability": torch.sigmoid(final_calibrated_logit),
        }


@torch.no_grad()
def predict_reliability_components(
    model: nn.Module,
    bio_values: np.ndarray,
    esm_values: np.ndarray,
    batch_size: int = 2048,
) -> dict[str, np.ndarray]:
    """Extract deterministic deployment diagnostics from a reliability model.

    For an ensemble, logits and component values are averaged over the same
    members used by :func:`predict`.  ``anchor_probability`` applies the
    ensemble's frozen affine-logit calibration to the mean anchor logit, so it
    is exactly the deployed probability whenever the reliability gate is zero.
    The helper never consumes labels and never changes model parameters.
    """
    bio_array = np.asarray(bio_values, dtype=np.float32)
    esm_array = np.asarray(esm_values, dtype=np.float32)
    if len(bio_array) != len(esm_array):
        raise ValueError("Bio and ESM row counts differ")
    # CHANGELOG 2026-09 (novelty N1): the accepted type was widened from
    # ``ReliabilityResidualNet`` to ``ReliabilityFamilyNet``.  The base class is
    # what actually defines the diagnostic contract this function relies on (the
    # nine ``forward_components`` keys), and it is abstract -- ``_fuse`` raises
    # ``NotImplementedError`` -- so the check is no weaker than before: any
    # instantiable subclass necessarily implements the contract.  A mixed
    # ensemble would be rejected downstream by the feature-contract equality
    # check, and members are always built by a single ``_model_factory`` call
    # per fold, so a mixed ensemble cannot arise in practice.
    members: list[ReliabilityFamilyNet]
    if isinstance(model, EnsembleModel):
        members = list(model.members)  # type: ignore[assignment]
    elif isinstance(model, ReliabilityFamilyNet):
        members = [model]
    else:
        raise TypeError("Reliability diagnostics require a reliability model")
    if not members or any(
        not isinstance(member, ReliabilityFamilyNet) for member in members
    ):
        raise TypeError(
            "Reliability diagnostics require only reliability-family members"
        )
    expected_features = tuple(members[0].feature_names)
    if bio_array.ndim != 2 or bio_array.shape[1] != len(expected_features):
        raise ValueError("Reliability diagnostic feature matrix has invalid shape")
    if esm_array.ndim != 2:
        raise ValueError("Reliability diagnostic ESM matrix has invalid shape")
    if any(tuple(member.feature_names) != expected_features for member in members):
        raise ValueError("Reliability ensemble member feature contracts differ")
    if len(bio_array) == 0:
        return {
            **{
                name: np.empty(0, dtype=np.float32)
                for name in RELIABILITY_DIAGNOSTIC_COMPONENTS
            },
            "model_probability": np.empty(0, dtype=np.float32),
        }

    model.eval()
    component_model = _ReliabilityComponentForward(model, members).to(DEVICE)
    inference_model = _parallel_model(component_model)
    inference_model.eval()
    dummy = np.zeros(len(bio_array), dtype=np.float32)
    collected = {
        name: []
        for name in (*RELIABILITY_DIAGNOSTIC_COMPONENTS, "model_probability")
    }
    for bio_batch, esm_batch, _ in _loader(
        bio_array, esm_array, dummy, batch_size, False
    ):
        bio_tensor = bio_batch.to(DEVICE)
        esm_tensor = esm_batch.to(DEVICE)
        with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
            batch_components = inference_model(bio_tensor, esm_tensor)
        for name, values in batch_components.items():
            collected[name].append(values.float().cpu().numpy())

    output = {
        name: np.concatenate(parts).astype(np.float32, copy=False)
        for name, parts in collected.items()
    }
    for name, values in output.items():
        if values.shape != (len(bio_array),) or not np.isfinite(values).all():
            raise ValueError(f"Reliability diagnostic {name} is invalid")
    for name in (
        "anchor_probability",
        "model_probability",
        "gate",
        "hard_availability",
        "structure_reliability",
        "conservation_reliability",
    ):
        if ((output[name] < -1e-6) | (output[name] > 1.0 + 1e-6)).any():
            raise ValueError(f"Reliability diagnostic {name} is outside [0, 1]")
    del inference_model, component_model
    return output


def reliability_availability_codes(
    components: dict[str, np.ndarray],
    epsilon: float = 1e-7,
) -> np.ndarray:
    """Assign prespecified, label-independent auxiliary-modality strata."""
    structure = np.asarray(components["structure_reliability"], dtype=float)
    conservation = np.asarray(
        components["conservation_reliability"], dtype=float
    )
    if structure.ndim != 1 or conservation.shape != structure.shape:
        raise ValueError("Reliability stratum components are misaligned")
    if not np.isfinite(structure).all() or not np.isfinite(conservation).all():
        raise ValueError("Reliability stratum components contain nonfinite values")
    structure_available = structure > epsilon
    conservation_available = conservation > epsilon
    return (
        structure_available.astype(np.int8)
        + 2 * conservation_available.astype(np.int8)
    )


def summarize_reliability_diagnostics(
    components: dict[str, np.ndarray],
    labels: np.ndarray,
    model_probabilities: np.ndarray,
    thresholds: float | np.ndarray,
    model_decisions: np.ndarray,
    anchor_decisions: np.ndarray | None = None,
) -> dict[str, Any]:
    """Summarize post-selection reliability behavior without forming label strata."""
    missing = set(RELIABILITY_DIAGNOSTIC_COMPONENTS) - set(components)
    if missing:
        raise KeyError(f"Reliability diagnostic components are missing: {sorted(missing)}")
    labels = np.asarray(labels, dtype=int)
    model_probabilities = np.asarray(model_probabilities, dtype=float)
    model_decisions = np.asarray(model_decisions, dtype=np.int8)
    n_rows = len(labels)
    if n_rows == 0 or len(model_probabilities) != n_rows or len(model_decisions) != n_rows:
        raise ValueError("Reliability diagnostic outcome arrays are empty or misaligned")
    normalized = {
        name: np.asarray(components[name], dtype=float)
        for name in RELIABILITY_DIAGNOSTIC_COMPONENTS
    }
    if any(values.shape != (n_rows,) for values in normalized.values()):
        raise ValueError("Reliability diagnostic component arrays are misaligned")
    if any(not np.isfinite(values).all() for values in normalized.values()):
        raise ValueError("Reliability diagnostic component arrays are nonfinite")
    threshold_values = np.asarray(thresholds, dtype=float)
    if threshold_values.ndim == 0:
        threshold_values = np.repeat(float(threshold_values), n_rows)
    if threshold_values.shape != (n_rows,) or not np.isfinite(threshold_values).all():
        raise ValueError("Reliability diagnostic thresholds are invalid")
    if anchor_decisions is None:
        anchor_decisions = (
            normalized["anchor_probability"] >= threshold_values
        ).astype(np.int8)
    else:
        anchor_decisions = np.asarray(anchor_decisions, dtype=np.int8)
    if anchor_decisions.shape != (n_rows,):
        raise ValueError("Reliability anchor decisions are misaligned")

    codes = reliability_availability_codes(normalized)
    hard_fallback = normalized["hard_availability"] <= 1e-7
    exact_gate_fallback = normalized["gate"] <= 1e-7
    if np.any(hard_fallback & ~exact_gate_fallback):
        raise RuntimeError("Hard-unavailable rows do not use the exact anchor gate")
    fallback_difference = np.abs(
        model_probabilities[hard_fallback]
        - normalized["anchor_probability"][hard_fallback]
    )
    maximum_fallback_difference = (
        float(fallback_difference.max()) if fallback_difference.size else 0.0
    )
    if maximum_fallback_difference > 5e-6:
        raise RuntimeError("Hard-unavailable predictions differ from the anchor")

    metric_names = (
        "mcc",
        "auroc",
        "auprc",
        "brier",
        "precision",
        "recall",
        "f1",
        "specificity",
    )

    def compact_metrics(
        mask: np.ndarray, probabilities: np.ndarray, decisions: np.ndarray
    ) -> dict[str, Any] | None:
        if not mask.any():
            return None
        evaluated = evaluate(
            labels[mask],
            probabilities[mask],
            threshold_values[mask],
            predictions=decisions[mask],
        )
        return {name: evaluated.get(name) for name in metric_names}

    strata: dict[str, Any] = {}
    for code, name in RELIABILITY_AVAILABILITY_STRATA.items():
        mask = codes == code
        proposed = compact_metrics(mask, model_probabilities, model_decisions)
        anchor = compact_metrics(
            mask, normalized["anchor_probability"], anchor_decisions
        )
        delta = None
        if proposed is not None and anchor is not None:
            delta = {
                metric: (
                    None
                    if proposed[metric] is None or anchor[metric] is None
                    else round(float(proposed[metric]) - float(anchor[metric]), 6)
                )
                for metric in ("mcc", "auroc", "auprc", "brier")
            }
        strata[name] = {
            "code": code,
            "definition": {
                "structure_reliability_positive": bool(code & 1),
                "conservation_reliability_positive": bool(code & 2),
            },
            "n": int(mask.sum()),
            "coverage": round(float(mask.mean()), 6),
            "positives": int(labels[mask].sum()),
            "prevalence": (
                round(float(labels[mask].mean()), 6) if mask.any() else None
            ),
            "mean_gate": (
                round(float(normalized["gate"][mask].mean()), 6)
                if mask.any()
                else None
            ),
            "exact_fallback_rate": (
                round(float(exact_gate_fallback[mask].mean()), 6)
                if mask.any()
                else None
            ),
            "proposed_model": proposed,
            "anchor_ablation_same_frozen_threshold_vote": anchor,
            "proposed_minus_anchor": delta,
        }

    gate = normalized["gate"]
    residual = normalized["bounded_residual"]
    return {
        "status": "descriptive_post_selection_not_primary",
        "model": RELIABILITY_ARCHITECTURE,
        "n": n_rows,
        "stratification": {
            "uses_labels": False,
            "prespecified_from": (
                "positive natural structure/conservation reliability before "
                "outcome evaluation"
            ),
            "code_mapping": {
                str(code): name
                for code, name in RELIABILITY_AVAILABILITY_STRATA.items()
            },
        },
        "selection_or_refitting": {
            "used_for_model_selection": False,
            "used_for_threshold_selection": False,
            "external_labels_used_for_refitting": False,
            "inference_claim_allowed": False,
        },
        "component_semantics": {
            "anchor_logit": "mean_uncalibrated_sequence_anchor_logit",
            "anchor_probability": (
                "same_frozen_affine_logit_calibration_as_deployed_model"
            ),
            "gate": "mean_learned_gate_after_natural_evidence_quality",
            "bounded_residual": "mean_pre_gate_bounded_residual_logit",
            "hard_availability": (
                "label_independent_any_trustworthy_structure_or_conservation"
            ),
            "structure_reliability": "label_independent_natural_structure_quality",
            "conservation_reliability": (
                "label_independent_observed_conservation_fraction"
            ),
            "interpretation_limit": (
                "mechanistic_model_diagnostics_not_causal_feature_attributions"
            ),
        },
        "gate_behavior": {
            "mean": round(float(gate.mean()), 6),
            "median": round(float(np.median(gate)), 6),
            "p05": round(float(np.quantile(gate, 0.05)), 6),
            "p95": round(float(np.quantile(gate, 0.95)), 6),
            "active_rate": round(float((gate > 1e-7).mean()), 6),
            "exact_fallback_rate": round(float(exact_gate_fallback.mean()), 6),
            "hard_unavailable_rate": round(float(hard_fallback.mean()), 6),
            "maximum_anchor_identity_error_on_hard_fallback": round(
                maximum_fallback_difference, 8
            ),
        },
        "bounded_residual_behavior": {
            "mean": round(float(residual.mean()), 6),
            "mean_absolute": round(float(np.abs(residual).mean()), 6),
            "maximum_absolute_observed": round(float(np.abs(residual).max()), 6),
            "p05": round(float(np.quantile(residual, 0.05)), 6),
            "p95": round(float(np.quantile(residual, 0.95)), 6),
        },
        "strata": strata,
        "metric_interpretation": {
            "all_metrics": "descriptive_post_selection_without_uncertainty_claims",
            "anchor_ablation": (
                "same frozen deployment calibration and fold thresholds; no label-"
                "based refit"
            ),
            "brier_delta_orientation": "proposed_minus_anchor_lower_is_better",
            "other_delta_orientation": "proposed_minus_anchor_higher_is_better",
        },
    }


def reliability_diagnostic_rows(
    summary: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Flatten reliability strata for a compact publication supplement table."""
    rows: list[dict[str, Any]] = []
    shared = {
        **(context or {}),
        "diagnostic_status": summary.get("status"),
        "strata_use_labels": bool(
            summary.get("stratification", {}).get("uses_labels", True)
        ),
        "used_for_model_selection": bool(
            summary.get("selection_or_refitting", {}).get(
                "used_for_model_selection", True
            )
        ),
        "used_for_threshold_selection": bool(
            summary.get("selection_or_refitting", {}).get(
                "used_for_threshold_selection", True
            )
        ),
        "inferential_claim_allowed": bool(
            summary.get("selection_or_refitting", {}).get(
                "inference_claim_allowed", True
            )
        ),
    }
    for stratum, record in summary.get("strata", {}).items():
        proposed = record.get("proposed_model") or {}
        anchor = record.get("anchor_ablation_same_frozen_threshold_vote") or {}
        delta = record.get("proposed_minus_anchor") or {}
        rows.append(
            {
                **shared,
                "availability_stratum": stratum,
                "availability_code": record.get("code"),
                "n": record.get("n"),
                "coverage": record.get("coverage"),
                "positives": record.get("positives"),
                "prevalence": record.get("prevalence"),
                "mean_gate": record.get("mean_gate"),
                "exact_fallback_rate": record.get("exact_fallback_rate"),
                **{
                    f"proposed_{metric}": proposed.get(metric)
                    for metric in ("mcc", "auroc", "auprc", "brier")
                },
                **{
                    f"anchor_{metric}": anchor.get(metric)
                    for metric in ("mcc", "auroc", "auprc", "brier")
                },
                **{
                    f"proposed_minus_anchor_{metric}": delta.get(metric)
                    for metric in ("mcc", "auroc", "auprc", "brier")
                },
            }
        )
    return rows


def _lr_multiplier(epoch: int, max_epochs: int, warmup_epochs: int) -> float:
    if epoch < warmup_epochs:
        return float(epoch + 1) / float(max(1, warmup_epochs))
    progress = (epoch - warmup_epochs) / float(
        max(1, max_epochs - warmup_epochs)
    )
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _safe_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float | None:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, probabilities))


def _safe_auprc(y_true: np.ndarray, probabilities: np.ndarray) -> float | None:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, probabilities))


# Auxiliary BCE supervises the fallback anchor. Consistency pulls a degraded
# view toward the detached full view, weighted by removed-evidence unreliability.
def _rcdi_loss(
    model: nn.Module,
    training_model: nn.Module,
    bio_batch: torch.Tensor,
    esm_batch: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    """Return the RCDI training loss for one batch.

    ``training_model`` may be a ``DataParallel`` wrapper, so the components are
    obtained by calling it (its ``forward`` returns the component dict for
    reliability-family members, which ``DataParallel`` gathers key-wise).
    Loss coefficients come from the active fold's temporary configuration.
    """
    components = training_model(bio_batch, esm_batch)
    loss = criterion(components["logits"], targets)
    anchor_weight = float(RCDI_ANCHOR_WEIGHT)
    if anchor_weight > 0.0:
        loss = loss + anchor_weight * criterion(components["anchor_logit"], targets)
    consistency_weight = float(RCDI_CONSISTENCY_WEIGHT)
    if consistency_weight > 0.0:
        weight = components["consistency_weight"]
        # ``detach`` on the target: the penalty must pull the degraded view
        # towards the full view, not the other way round.
        gap = components["degraded_logits"].float() - components["logits"].detach().float()
        weight = weight.float()
        denominator = weight.sum().clamp_min(1.0)
        loss = loss + consistency_weight * (weight * gap.pow(2)).sum() / denominator
    return loss


class _RcdiForward(nn.Module):
    """Adapt ``rcdi_components`` to a plain ``forward`` for DataParallel.

    ``nn.DataParallel`` replicates the wrapped module and gathers dictionary
    outputs key by key along dim 0, so exposing the component dict through
    ``forward`` keeps the objective compatible with the existing
    ``SafeDataParallel`` path.  The mask probability is captured at construction
    time because ``DataParallel`` only forwards tensor-friendly arguments
    reliably.
    """

    def __init__(self, model: ReliabilityFamilyNet, mask_probability: float):
        super().__init__()
        self.model = model
        self.mask_probability = float(mask_probability)

    def forward(
        self, bio_values: torch.Tensor, esm_values: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return self.model.rcdi_components(
            bio_values, esm_values, self.mask_probability
        )


def _train_single(
    bio_train: np.ndarray,
    esm_train: np.ndarray,
    y_train: np.ndarray,
    bio_stop: np.ndarray,
    esm_stop: np.ndarray,
    y_stop: np.ndarray,
    seed: int,
    architecture: str,
    max_epochs: int,
    patience: int,
    feature_names: Iterable[str] | None = None,
) -> tuple[nn.Module, float, int]:
    set_seeds(seed)
    model = _model_factory(
        architecture, bio_train.shape[1], feature_names=feature_names
    ).to(DEVICE)
    evaluation_model = _model_factory(
        architecture, bio_train.shape[1], feature_names=feature_names
    ).to(DEVICE)
    moving_average = EMA(model, EMA_DECAY)
    # CHANGELOG 2026-09 (novelty N2): opt in to the RCDI objective only for
    # architectures that declare support for it.  Every pre-existing
    # architecture leaves ``uses_rcdi_objective`` at its ``False`` default (it is
    # read with ``getattr`` so that the non-family architectures, which do not
    # define the attribute at all, are unaffected), so their loss, RNG
    # consumption and results are unchanged.
    use_rcdi = bool(getattr(model, "uses_rcdi_objective", False)) and (
        float(RCDI_ANCHOR_WEIGHT) > 0.0 or float(RCDI_CONSISTENCY_WEIGHT) > 0.0
    )
    # CHANGELOG 2026-09 (novelty N2): ``_parallel_model`` was moved a few lines
    # down so that it can wrap the RCDI adapter when the objective is active.
    # ``_parallel_model`` never consumes randomness (it either returns ``model``
    # unchanged or wraps it in ``SafeDataParallel``) and ``_RcdiForward`` holds no
    # parameters of its own, so the RNG stream -- and therefore every existing
    # architecture's initialisation and training trajectory -- is unchanged by
    # the reordering.  ``model`` remains the module that owns the parameters, so
    # ``optimizer``, ``clip_grad_norm_``, ``EMA`` and ``load_state_dict`` below
    # are all untouched.
    training_model = _parallel_model(
        _RcdiForward(
            model,
            RCDI_CONSISTENCY_MASK_PROBABILITY if RCDI_CONSISTENCY_WEIGHT > 0.0 else 0.0,
        )
        if use_rcdi
        else model
    )
    positive_weight = torch.tensor(
        [(y_train == 0).sum() / max((y_train == 1).sum(), 1)],
        dtype=torch.float32,
        device=DEVICE,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=DEEP_LR, weight_decay=DEEP_WD
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: _lr_multiplier(epoch, max_epochs, WARMUP_EPOCHS),
    )
    train_loader = _loader(
        bio_train,
        esm_train,
        y_train,
        BATCH_SIZE,
        True,
        seed=seed,
        # These tensors are already in RAM. Windows spawn workers replicate
        # the large training dataset for every nested trial without disk I/O
        # to overlap; pinned-memory transfer is sufficient here.
        workers=0,
    )
    # CHANGELOG 2026-09 (optimization): the early-stop validation set is wrapped
    # in a TensorDataset / DataLoader once, before the epoch loop, instead of
    # being rebuilt on every epoch inside predict_logits.  The loader is
    # read-only (no shuffle, plain 1-D targets) and predict_logits only calls
    # .eval() on the model, so it is safe to iterate the same loader every
    # epoch; this removes a per-epoch tensor copy, a torch.Generator reseed and,
    # on GPU, a DataLoader worker-pool ramp-up each time.
    stop_loader = _loader(
        bio_stop,
        esm_stop,
        y_stop,
        BATCH_SIZE,
        False,
        seed=seed,
        workers=0,
    )
    random_state = np.random.RandomState(seed)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
    best_score = -math.inf
    best_epoch = 0
    wait = 0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(max_epochs):
        training_model.train()
        for bio_batch, esm_batch, targets in train_loader:
            bio_batch = bio_batch.to(DEVICE, non_blocking=DEVICE == "cuda")
            esm_batch = esm_batch.to(DEVICE, non_blocking=DEVICE == "cuda")
            targets = targets.to(DEVICE, non_blocking=DEVICE == "cuda")
            targets = targets * (1.0 - LABEL_SMOOTH) + 0.5 * LABEL_SMOOTH
            if (
                MIXUP_ALPHA > 0
                and not bool(getattr(model, "disable_mixup", False))
                and random_state.rand() < 0.5
                and len(targets) > 1
            ):
                weight = float(random_state.beta(MIXUP_ALPHA, MIXUP_ALPHA))
                permutation = torch.randperm(len(targets), device=DEVICE)
                bio_batch = weight * bio_batch + (1.0 - weight) * bio_batch[permutation]
                esm_batch = weight * esm_batch + (1.0 - weight) * esm_batch[permutation]
                targets = weight * targets + (1.0 - weight) * targets[permutation]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
                # CHANGELOG 2026-09 (novelty N2): branch on the objective.  The
                # ``else`` arm is the original single line, unchanged.
                if use_rcdi:
                    loss = _rcdi_loss(
                        model,
                        training_model,
                        bio_batch,
                        esm_batch,
                        targets,
                        criterion,
                    )
                else:
                    loss = criterion(training_model(bio_batch, esm_batch), targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            moving_average.update(model)
        scheduler.step()
        evaluation_model.load_state_dict(moving_average.shadow)
        stop_probabilities = predict(
            evaluation_model,
            bio_stop,
            esm_stop,
            loader=stop_loader,
            num_rows=len(bio_stop),
        )
        score = (
            _safe_auprc(y_stop, stop_probabilities)
            if DEEP_EARLY_STOP_METRIC == "auprc"
            else _safe_auc(y_stop, stop_probabilities)
        )
        score = -math.inf if score is None else score
        if score > best_score + 1e-4:
            best_score = score
            best_epoch = epoch + 1
            wait = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in moving_average.shadow.items()
            }
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is None:
        raise RuntimeError("Deep training produced no checkpoint")
    model.load_state_dict(best_state)
    del training_model, evaluation_model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return model, best_score, best_epoch


def fit_temperature_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit a positive calibration temperature."""
    logits = np.asarray(logits, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    if len(logits) == 0 or len(np.unique(labels)) < 2:
        logger.warning("Temperature fitting skipped for insufficient labels")
        return 1.0
    logits_tensor = torch.from_numpy(logits)
    labels_tensor = torch.from_numpy(labels)
    log_temperature = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.05, max_iter=100, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(1e-3, 1e3)
        loss = nn.BCEWithLogitsLoss()(logits_tensor / temperature, labels_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature.detach()).clamp(1e-3, 1e3).item())


@dataclass
class LogitCalibrator:
    """Affine Platt map fitted directly on uncalibrated model logits."""

    slope: float
    intercept: float

    def predict_from_logits(self, logits: np.ndarray) -> np.ndarray:
        values = self.slope * np.asarray(logits, dtype=np.float64) + self.intercept
        values = np.clip(values, -40.0, 40.0)
        return (1.0 / (1.0 + np.exp(-values))).astype(np.float64)


def fit_logit_calibrator(logits: np.ndarray, labels: np.ndarray) -> LogitCalibrator:
    """Fit slope and intercept on a dedicated, group-disjoint partition."""
    labels = validate_binary_labels(labels)
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 1 or len(logits) != len(labels):
        raise ValueError("Calibration logits and labels are misaligned")
    if not np.isfinite(logits).all():
        raise ValueError("Calibration logits contain nonfinite values")
    model = LogisticRegression(
        random_state=RANDOM_STATE,
        solver="lbfgs",
        C=1.0,
        max_iter=2000,
    )
    model.fit(logits.reshape(-1, 1), labels)
    return LogitCalibrator(
        slope=float(model.coef_[0, 0]),
        intercept=float(model.intercept_[0]),
    )


def fit_regularized_logistic(
    train: np.ndarray,
    stop: np.ndarray,
    labels_train: np.ndarray,
    labels_stop: np.ndarray,
    seed: int = RANDOM_STATE,
    primary_metric: str = "auprc",
) -> LogisticRegression:
    """Tune a compact logistic baseline inside one outer training fold."""
    validate_binary_labels(labels_train)
    validate_binary_labels(labels_stop)
    if primary_metric not in {"auprc", "auroc"}:
        raise ValueError("Logistic primary_metric must be auprc or auroc")
    best_model: LogisticRegression | None = None
    best_score = -math.inf
    for regularization in (0.01, 0.1, 1.0, 10.0):
        model = LogisticRegression(
            C=regularization,
            class_weight="balanced",
            max_iter=3000,
            random_state=seed,
            solver="lbfgs",
        )
        model.fit(train, labels_train)
        probabilities = model.predict_proba(stop)[:, 1]
        score = (
            _safe_auprc(labels_stop, probabilities)
            if primary_metric == "auprc"
            else _safe_auc(labels_stop, probabilities)
        )
        score = -math.inf if score is None else score
        if score > best_score:
            best_score = score
            best_model = model
    if best_model is None:
        raise RuntimeError("Logistic baseline fitting failed")
    return best_model


def fit_direction_preserving_calibrator(
    labels: np.ndarray, scores: np.ndarray
) -> LogitCalibrator:
    """Calibrate a prespecified higher-is-riskier score without allowing a flip."""
    labels = validate_binary_labels(labels)
    scores = np.asarray(scores, dtype=np.float64)
    calibrator = fit_logit_calibrator(scores, labels)
    if calibrator.slope > 0:
        return calibrator
    prevalence = float(labels.mean())
    intercept = float(np.log(prevalence / max(1.0 - prevalence, 1e-12)))
    logger.warning("Calibration slope was nonpositive; retaining score direction")
    return LogitCalibrator(slope=CALIBRATION_EPSILON, intercept=intercept)


def train_deep_model(
    bio_train: np.ndarray,
    esm_train: np.ndarray,
    y_train: np.ndarray,
    bio_stop: np.ndarray,
    esm_stop: np.ndarray,
    y_stop: np.ndarray,
    bio_calibration: np.ndarray | None = None,
    esm_calibration: np.ndarray | None = None,
    y_calibration: np.ndarray | None = None,
    max_epochs: int | None = None,
    patience: int | None = None,
    architecture: str = "cross_attention",
    seeds: Iterable[int] | None = None,
    feature_names: Iterable[str] | None = None,
) -> EnsembleModel:
    """Train and affine-calibrate an independently seeded ensemble."""
    y_train = validate_binary_labels(y_train)
    y_stop = validate_binary_labels(y_stop)
    max_epochs = DEEP_MAX_EPOCHS if max_epochs is None else max_epochs
    patience = DEEP_PATIENCE if patience is None else patience
    if max_epochs < 1 or patience < 1:
        raise ValueError("Epochs and patience must be positive")
    if seeds is None:
        seeds = [RANDOM_STATE + 100 * (index + 1) for index in range(N_ENSEMBLE)]
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Ensemble seeds must be nonempty and distinct")
    calibration_parts = (bio_calibration, esm_calibration, y_calibration)
    if any(value is not None for value in calibration_parts) and any(
        value is None for value in calibration_parts
    ):
        raise ValueError("Supply all calibration arrays or omit all of them")
    if y_calibration is not None:
        y_calibration = validate_binary_labels(y_calibration)
    partitions = [
        ("train", bio_train, esm_train, y_train),
        ("stop", bio_stop, esm_stop, y_stop),
    ]
    if y_calibration is not None:
        partitions.append(("calibration", bio_calibration, esm_calibration, y_calibration))
    bio_width = None
    for name, bio, esm_values, labels in partitions:
        if np.ndim(bio) != 2 or np.ndim(esm_values) != 2:
            raise ValueError(f"{name} inputs must be matrices")
        if len(bio) != len(labels) or len(esm_values) != len(labels):
            raise ValueError(f"{name} features and labels are misaligned")
        bio_width = bio.shape[1] if bio_width is None else bio_width
        if bio.shape[1] != bio_width or bio_width < 1 or esm_values.shape[1] != ESM_DIM:
            raise ValueError(f"{name} feature widths differ from the model schema")
    if feature_names is not None:
        feature_names = tuple(str(name) for name in feature_names)
        if len(feature_names) != bio_width or len(set(feature_names)) != bio_width:
            raise ValueError("Feature names must uniquely identify every bio column")
    members: list[nn.Module] = []
    scores: list[float] = []
    epochs: list[int] = []
    for seed in seeds:
        member, score, best_epoch = _train_single(
            bio_train,
            esm_train,
            y_train,
            bio_stop,
            esm_stop,
            y_stop,
            int(seed),
            architecture,
            max_epochs,
            patience,
            feature_names,
        )
        members.append(member)
        scores.append(score)
        epochs.append(best_epoch)
    ensemble = EnsembleModel(members).to(DEVICE)
    if bio_calibration is None or esm_calibration is None or y_calibration is None:
        logger.warning("Calibration set omitted; identity calibration retained")
    else:
        validate_binary_labels(y_calibration)
        calibration_logits = predict_logits(
            ensemble, bio_calibration, esm_calibration
        )
        calibrator = fit_direction_preserving_calibrator(
            y_calibration, calibration_logits
        )
        ensemble.calibration_slope = calibrator.slope
        ensemble.calibration_intercept = calibrator.intercept
        ensemble.T = 1.0 / max(calibrator.slope, CALIBRATION_EPSILON)
    logger.info(
        "%s ensemble members=%d stop_%s=%.4f calibration=(%.4f, %.4f) epochs=%s",
        architecture,
        len(members),
        DEEP_EARLY_STOP_METRIC.upper(),
        float(np.mean(scores)),
        ensemble.calibration_slope,
        ensemble.calibration_intercept,
        epochs,
    )
    return ensemble


@dataclass
class ArrayPreprocessor:
    """Store train-fitted imputation and scaling."""

    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, copy: bool = True) -> "ArrayPreprocessor":
        array = np.asarray(values, dtype=np.float32)
        if copy or not array.flags.writeable:
            array = array.copy()
        if array.ndim != 2 or array.shape[0] == 0:
            raise ValueError("Preprocessor requires a nonempty matrix")
        array[~np.isfinite(array)] = np.nan
        # Entirely absent training features have the declared zero fallback.
        # np.errstate cannot suppress nanmedian's Python All-NaN warning;
        # materialise that fallback first, without copying the feature matrix.
        array[:, np.isnan(array).all(axis=0)] = 0.0
        median = np.nanmedian(array, axis=0)
        median = np.where(np.isfinite(median), median, 0.0).astype(np.float32)
        np.copyto(array, np.broadcast_to(median, array.shape), where=np.isnan(array))
        mean = array.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = array.std(axis=0, dtype=np.float64).astype(np.float32)
        scale = np.where(scale > 0, scale, 1.0).astype(np.float32)
        return cls(median=median, mean=mean, scale=scale)

    @classmethod
    def fit_with_passthrough(
        cls,
        values: np.ndarray,
        passthrough_indices: Iterable[int],
        copy: bool = True,
    ) -> "ArrayPreprocessor":
        """Fit scaling while retaining raw, imputed values for selected columns."""
        fitted = cls.fit(values, copy=copy)
        indices = np.asarray(list(passthrough_indices), dtype=np.int64)
        if indices.size:
            if (indices < 0).any() or (indices >= len(fitted.median)).any():
                raise IndexError("Preprocessor passthrough index is out of range")
            fitted.mean[indices] = 0.0
            fitted.scale[indices] = 1.0
        return fitted

    def transform(self, values: np.ndarray, copy: bool = True) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if copy or not array.flags.writeable:
            array = array.copy()
        if array.ndim != 2 or array.shape[1] != len(self.median):
            raise ValueError("Preprocessor feature count mismatch")
        array[~np.isfinite(array)] = np.nan
        np.copyto(array, np.broadcast_to(self.median, array.shape), where=np.isnan(array))
        array -= self.mean
        array /= self.scale
        return array


@dataclass
class PreprocessorBundle:
    """Store modality-specific preprocessing."""

    bio: ArrayPreprocessor
    esm: ArrayPreprocessor

    def transform_bio(self, values: np.ndarray, copy: bool = True) -> np.ndarray:
        return self.bio.transform(values, copy=copy)

    def transform_esm(self, values: np.ndarray, copy: bool = True) -> np.ndarray:
        return self.esm.transform(values, copy=copy)


def fit_preprocessors(
    bio_fit: np.ndarray,
    esm_fit: np.ndarray,
    bio_passthrough_indices: Iterable[int] = (),
) -> PreprocessorBundle:
    """Fit preprocessing on training rows only."""
    return PreprocessorBundle(
        bio=ArrayPreprocessor.fit_with_passthrough(
            bio_fit, bio_passthrough_indices, copy=False
        ),
        esm=ArrayPreprocessor.fit(esm_fit, copy=False),
    )


def nonconstant_feature_mask(
    values: np.ndarray, tolerance: float = 1e-12
) -> np.ndarray:
    """Identify features with train-fold variation after finite-value handling."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError("Feature mask requires a nonempty two-dimensional matrix")
    finite = np.isfinite(array)
    observed = finite.any(axis=0)
    minima = np.where(finite, array, np.inf).min(axis=0)
    maxima = np.where(finite, array, -np.inf).max(axis=0)
    return observed & ((maxima - minima) > tolerance)


def architecture_feature_mask(
    values: np.ndarray,
    feature_names: Iterable[str],
    architecture: str,
) -> np.ndarray:
    """Drop fold constants while preserving the reliability model contract."""
    names = [str(name) for name in feature_names]
    if np.asarray(values).shape[1] != len(names):
        raise ValueError("Architecture feature names and matrix width differ")
    mask = nonconstant_feature_mask(values)
    # CHANGELOG 2026-09 (novelty N1): feature-schema dispatch is family-wide.
    # Both members bind the anchor and the required gate columns by name, so
    # both need those columns force-kept even when they are fold-constant.
    if not is_reliability_family(architecture):
        return mask
    required = {
        RELIABILITY_ANCHOR_FEATURE,
        *RELIABILITY_REQUIRED_GATE_FEATURES,
    }
    missing = required - set(names)
    if missing:
        raise ValueError(
            f"Reliability architecture is missing required features: {sorted(missing)}"
        )
    if not set(names) & set(RELIABILITY_CONSERVATION_MISSING_FEATURES):
        raise ValueError(
            "Reliability architecture has no conservation missingness indicator"
        )
    for index, name in enumerate(names):
        if name == RELIABILITY_ANCHOR_FEATURE or name in RELIABILITY_GATE_FEATURES:
            observed = np.isfinite(np.asarray(values)[:, index]).any()
            mandatory = (
                name == RELIABILITY_ANCHOR_FEATURE
                or name in RELIABILITY_REQUIRED_GATE_FEATURES
                or name in RELIABILITY_CONSERVATION_MISSING_FEATURES
            )
            if mandatory and not observed:
                raise ValueError(f"Reliability feature {name} is entirely missing")
            if observed:
                mask[index] = True
    return mask


def select_features(df: pd.DataFrame) -> list[str]:
    """Select only explicitly approved numeric features."""
    return select_model_features(df)


def _threshold_confusion_counts(
    labels: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute all >= threshold counts with one sort, including tied scores."""
    order = np.argsort(probabilities, kind="stable")
    sorted_scores = probabilities[order]
    positive_prefix = np.concatenate([[0], np.cumsum(labels[order], dtype=np.int64)])
    below = np.searchsorted(sorted_scores, thresholds, side="left")
    fn = positive_prefix[below].astype(np.float64)
    tn = below.astype(np.float64) - fn
    tp = float(positive_prefix[-1]) - fn
    fp = float(len(labels) - positive_prefix[-1]) - tn
    return tn, fp, fn, tp


def _threshold_mcc(
    tn: np.ndarray, fp: np.ndarray, fn: np.ndarray, tp: np.ndarray
) -> np.ndarray:
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.divide(
        tp * tn - fp * fn,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0.0,
    )


def _validated_probability_vector(values: np.ndarray, n_rows: int) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.shape != (n_rows,) or not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite and aligned one-dimensional values")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("Probabilities must be in [0, 1]")
    return probabilities


def select_threshold(y_true: np.ndarray, y_probability: np.ndarray) -> float:
    """Choose a recall-aware threshold on calibration data."""
    y_true = validate_binary_labels(y_true, require_both_classes=False)
    y_probability = _validated_probability_vector(y_probability, len(y_true))
    if len(np.unique(y_true)) < 2:
        logger.warning("Single-class calibration; using threshold 0.5")
        return 0.5
    grid = np.unique(np.concatenate([np.linspace(0.01, 0.99, 393), y_probability]))
    tn, fp, fn, tp = _threshold_confusion_counts(y_true, y_probability, grid)
    recall = tp / (tp + fn)
    feasible = recall >= RECALL_FLOOR
    if feasible.any():
        candidates = np.flatnonzero(feasible)
        best = candidates[np.argmax(_threshold_mcc(tn, fp, fn, tp)[feasible])]
        # Ascending grid and first argmax preserve the original lowest-tie rule.
        return float(grid[best])
    beta_squared = FBETA_BETA**2
    denominator = (1.0 + beta_squared) * tp + beta_squared * fn + fp
    fbeta = np.divide(
        (1.0 + beta_squared) * tp,
        denominator,
        out=np.zeros_like(tp),
        where=denominator > 0.0,
    )
    logger.warning("Recall floor unreachable; using F-beta threshold")
    return float(grid[np.argmax(fbeta)])


def select_operating_points(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    sensitivity_targets: Iterable[float] = (0.80, 0.90, 0.95),
) -> dict[str, float]:
    """Lock several interpretable operating points on calibration data only."""
    labels = validate_binary_labels(y_true, require_both_classes=False)
    probabilities = _validated_probability_vector(y_probability, len(labels))
    if len(np.unique(labels)) < 2:
        return {"default": 0.5}
    thresholds = np.unique(np.concatenate([[0.0, 1.0], probabilities]))
    tn, fp, fn, tp = _threshold_confusion_counts(labels, probabilities, thresholds)
    output = {"max_mcc": float(thresholds[np.argmax(_threshold_mcc(tn, fp, fn, tp))])}
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    for target in sensitivity_targets:
        if not 0.0 < float(target) < 1.0:
            raise ValueError("Sensitivity targets must be between zero and one")
        feasible = sensitivity + 1e-12 >= float(target)
        candidates = np.flatnonzero(feasible)
        key = f"sensitivity_{int(round(100 * target))}"
        if candidates.size:
            best_specificity = specificity[candidates].max()
            # The old max((specificity, threshold)) chose the highest tied cut.
            best = candidates[specificity[candidates] == best_specificity][-1]
            output[key] = float(thresholds[best])
        else:
            output[key] = 0.0
    return output


def _adaptive_ece(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    """Compute equal-frequency expected calibration error."""
    if len(labels) == 0:
        return float("nan")
    order = np.argsort(probabilities, kind="stable")
    chunks = np.array_split(order, min(bins, len(order)))
    error = 0.0
    for indices in chunks:
        if len(indices) == 0:
            continue
        error += (len(indices) / len(labels)) * abs(
            float(probabilities[indices].mean()) - float(labels[indices].mean())
        )
    return float(error)


def _calibration_diagnostics(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, float | None]:
    """Estimate descriptive calibration slope/intercept on evaluation rows."""
    prevalence = float(labels.mean())
    reference_brier = prevalence * (1.0 - prevalence)
    output: dict[str, float | None] = {
        "prevalence": round(prevalence, 6),
        "prevalence_brier": round(reference_brier, 6),
        "brier_skill": None,
        "log_loss": round(
            float(log_loss(labels, probabilities, labels=[0, 1])), 6
        ),
        "adaptive_ece": round(_adaptive_ece(labels, probabilities), 6),
        "calibration_slope": None,
        "calibration_intercept": None,
    }
    observed_brier = float(brier_score_loss(labels, probabilities))
    if reference_brier > 0:
        output["brier_skill"] = round(1.0 - observed_brier / reference_brier, 6)
    if len(np.unique(labels)) < 2:
        return output
    clipped = np.clip(probabilities, CALIBRATION_EPSILON, 1.0 - CALIBRATION_EPSILON)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    try:
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
        model.fit(logits, labels)
        output["calibration_slope"] = round(float(model.coef_[0, 0]), 6)
        output["calibration_intercept"] = round(float(model.intercept_[0]), 6)
    except (ValueError, FloatingPointError):
        pass
    return output


def _risk_coverage_summary(
    labels: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    threshold_values: np.ndarray,
    decision_confidence: np.ndarray | None = None,
) -> dict[str, dict[str, float | None]]:
    """Report selective performance at prespecified coverage levels."""
    if decision_confidence is None:
        clipped = np.clip(
            probabilities, CALIBRATION_EPSILON, 1.0 - CALIBRATION_EPSILON
        )
        probability_logits = np.log(clipped / (1.0 - clipped))
        threshold_array = np.broadcast_to(threshold_values, probabilities.shape)
        threshold_array = np.clip(
            threshold_array, CALIBRATION_EPSILON, 1.0 - CALIBRATION_EPSILON
        )
        threshold_logits = np.log(threshold_array / (1.0 - threshold_array))
        confidence = np.abs(probability_logits - threshold_logits)
    else:
        confidence = np.asarray(decision_confidence, dtype=float)
        if confidence.ndim != 1 or len(confidence) != len(labels):
            raise ValueError("Decision confidence is misaligned")
        if not np.isfinite(confidence).all():
            raise ValueError("Decision confidence contains nonfinite values")
    order = np.argsort(-confidence, kind="stable")
    summary: dict[str, dict[str, float | None]] = {}
    for coverage in (0.50, 0.80, 0.90, 1.00):
        size = max(1, int(math.ceil(coverage * len(labels))))
        indices = order[:size]
        subset_labels = labels[indices]
        subset_predictions = predictions[indices]
        errors = subset_predictions != subset_labels
        summary[f"coverage_{int(100 * coverage)}"] = {
            "coverage": round(float(size / len(labels)), 4),
            "risk": round(float(errors.mean()), 6),
            "mcc": (
                round(float(matthews_corrcoef(subset_labels, subset_predictions)), 4)
                if len(np.unique(subset_labels)) > 1
                else None
            ),
        }
    return summary


def evaluate(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    threshold: float | np.ndarray,
    predictions: np.ndarray | None = None,
    decision_confidence: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate probabilities and frozen decisions safely."""
    y_true = validate_binary_labels(y_true, require_both_classes=False)
    y_probability = _validated_probability_vector(y_probability, len(y_true))
    threshold_values = np.asarray(threshold, dtype=float)
    if (
        threshold_values.shape not in {(), (len(y_true),)}
        or not np.isfinite(threshold_values).all()
        or ((threshold_values < 0.0) | (threshold_values > 1.0)).any()
    ):
        raise ValueError("Evaluation thresholds must be finite scalar or aligned probabilities")
    if predictions is None:
        if threshold_values.ndim == 0:
            predictions = (y_probability >= float(threshold_values)).astype(int)
        else:
            if len(threshold_values) != len(y_true):
                raise ValueError("Threshold vector length mismatch")
            predictions = (y_probability >= threshold_values).astype(int)
    else:
        predictions = validate_binary_labels(
            predictions, "predictions", require_both_classes=False
        )
    if len(predictions) != len(y_true):
        raise ValueError("Prediction length mismatch")
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    auroc = _safe_auc(y_true, y_probability)
    auprc = _safe_auprc(y_true, y_probability)
    threshold_summary = float(np.mean(threshold_values))
    mcc = (
        float(matthews_corrcoef(y_true, predictions))
        if len(np.unique(y_true)) == 2
        else None
    )
    metrics: dict[str, Any] = {
        "threshold": round(threshold_summary, 4),
        "mcc": None if mcc is None else round(mcc, 4),
        "auroc": None if auroc is None else round(auroc, 4),
        "auprc": None if auprc is None else round(auprc, 4),
        "brier": round(float(brier_score_loss(y_true, y_probability)), 6),
        "precision": round(
            float(precision_score(y_true, predictions, zero_division=0)), 4
        ),
        "recall": round(
            float(recall_score(y_true, predictions, zero_division=0)), 4
        ),
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "specificity": round(float(tn / max(tn + fp, 1)), 4),
        "npv": round(float(tn / max(tn + fn, 1)), 4),
    }
    metrics["calibration"] = _calibration_diagnostics(y_true, y_probability)
    risk_coverage = _risk_coverage_summary(
        y_true,
        y_probability,
        predictions,
        threshold_values,
        decision_confidence=decision_confidence,
    )
    metrics["risk_coverage"] = risk_coverage
    metrics["risk_coverage_confidence_source"] = (
        "provided_classifier_decision_confidence"
        if decision_confidence is not None
        else "logit_distance_from_frozen_threshold"
    )
    # Backward-compatible summary; unlike the former fixed 0.10 margin this is
    # invariant to whether the operating threshold is 0.01 or 0.50.
    metrics["selective"] = risk_coverage["coverage_80"]
    return metrics


@dataclass
class ProbabilityCalibrator:
    """Apply Platt calibration to probabilities."""

    coefficient: float
    intercept: float

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
        logits = np.log(clipped / (1.0 - clipped))
        calibrated_logits = np.clip(self.coefficient * logits + self.intercept, -40.0, 40.0)
        calibrated = 1.0 / (1.0 + np.exp(-calibrated_logits))
        return calibrated.astype(np.float64)


def fit_probability_calibrator(
    labels: np.ndarray, probabilities: np.ndarray
) -> ProbabilityCalibrator:
    """Fit Platt scaling on dedicated data."""
    validate_binary_labels(labels)
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    model = LogisticRegression(random_state=RANDOM_STATE, solver="lbfgs")
    model.fit(logits, labels)
    coefficient = float(model.coef_[0, 0])
    intercept = float(model.intercept_[0])
    if coefficient <= 0:
        prevalence = float(labels.mean())
        coefficient = CALIBRATION_EPSILON
        intercept = float(np.log(prevalence / max(1.0 - prevalence, 1e-12)))
        logger.warning(
            "Probability calibration slope was nonpositive; retaining score direction"
        )
    return ProbabilityCalibrator(
        coefficient=coefficient,
        intercept=intercept,
    )


def fold_class_weight(labels: np.ndarray) -> float:
    """Return a fold-specific positive weight."""
    labels = np.asarray(labels, dtype=int)
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("Class weight requires both classes")
    return negatives / positives


def group_bootstrap_intervals(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    groups: np.ndarray,
    iterations: int = 1000,
    seed: int = RANDOM_STATE,
) -> dict[str, list[float] | None]:
    """Estimate group-bootstrap confidence intervals."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    predictions = np.asarray(predictions)
    groups = np.asarray(groups, dtype=object)
    if not (len(y_true) == len(probabilities) == len(predictions) == len(groups)):
        raise ValueError("Bootstrap inputs are misaligned")
    codes, unique_groups = pd.factorize(groups, sort=False)
    order = np.argsort(codes, kind="stable")
    boundaries = np.cumsum(np.bincount(codes, minlength=len(unique_groups)))
    group_indices = dict(zip(unique_groups, np.split(order, boundaries[:-1])))
    random_state = np.random.RandomState(seed)
    values = {name: [] for name in ("mcc", "auroc", "auprc", "brier")}
    for _ in range(iterations):
        sampled = random_state.choice(unique_groups, len(unique_groups), replace=True)
        indices = np.concatenate([group_indices[group] for group in sampled])
        labels = y_true[indices]
        probs = probabilities[indices]
        preds = predictions[indices]
        values["brier"].append(float(brier_score_loss(labels, probs)))
        auc = _safe_auc(labels, probs)
        auprc = _safe_auprc(labels, probs)
        if len(np.unique(labels)) > 1:
            values["mcc"].append(float(matthews_corrcoef(labels, preds)))
        if auc is not None:
            values["auroc"].append(auc)
        if auprc is not None:
            values["auprc"].append(auprc)
    intervals: dict[str, list[float] | None] = {}
    for name, samples in values.items():
        intervals[name] = (
            [round(float(value), 4) for value in np.percentile(samples, [2.5, 97.5])]
            if samples
            else None
        )
    return intervals


def _paired_metric_differences(
    labels: np.ndarray,
    first_probability: np.ndarray,
    second_probability: np.ndarray,
    first_prediction: np.ndarray,
    second_prediction: np.ndarray,
) -> dict[str, float]:
    """Compute paired second-minus-first endpoint differences."""
    return {
        "mcc": float(matthews_corrcoef(labels, second_prediction))
        - float(matthews_corrcoef(labels, first_prediction)),
        "auroc": float(roc_auc_score(labels, second_probability))
        - float(roc_auc_score(labels, first_probability)),
        "auprc": float(average_precision_score(labels, second_probability))
        - float(average_precision_score(labels, first_probability)),
    }


def paired_group_randomization_test(
    y_true: np.ndarray,
    first_probability: np.ndarray,
    second_probability: np.ndarray,
    first_prediction: np.ndarray,
    second_prediction: np.ndarray,
    groups: np.ndarray,
    iterations: int = 1000,
    seed: int = RANDOM_STATE,
) -> dict[str, dict[str, Any]]:
    """Test paired model differences by swapping complete split groups.

    Under the null, the two models are exchangeable within every independent
    gene/homology group.  A Monte Carlo group-level swap therefore supplies a
    valid null distribution without treating correlated variants as rows that
    may be permuted independently.
    """
    labels = validate_binary_labels(y_true)
    first_probability = np.asarray(first_probability, dtype=np.float64)
    second_probability = np.asarray(second_probability, dtype=np.float64)
    first_prediction = np.asarray(first_prediction, dtype=np.int8)
    second_prediction = np.asarray(second_prediction, dtype=np.int8)
    groups = np.asarray(groups, dtype=object)
    arrays = (
        first_probability,
        second_probability,
        first_prediction,
        second_prediction,
        groups,
    )
    if any(len(values) != len(labels) for values in arrays):
        raise ValueError("Paired randomization inputs are misaligned")
    if iterations < 1:
        raise ValueError("Randomization iterations must be positive")
    if not np.isfinite(first_probability).all() or not np.isfinite(
        second_probability
    ).all():
        raise ValueError("Paired randomization probabilities contain nonfinite values")
    if not set(np.unique(first_prediction)).issubset({0, 1}) or not set(
        np.unique(second_prediction)
    ).issubset({0, 1}):
        raise ValueError("Paired randomization predictions must be binary")
    if pd.isna(groups).any():
        raise ValueError("Paired randomization groups contain missing values")
    group_codes, unique_groups = pd.factorize(groups, sort=False)
    if len(unique_groups) < 2:
        raise ValueError("Paired randomization requires at least two groups")

    observed = _paired_metric_differences(
        labels,
        first_probability,
        second_probability,
        first_prediction,
        second_prediction,
    )
    exceedances = {name: 0 for name in observed}
    random_state = np.random.RandomState(seed)
    for _ in range(iterations):
        group_swaps = random_state.randint(0, 2, size=len(unique_groups)).astype(bool)
        row_swaps = group_swaps[group_codes]
        permuted_first_probability = np.where(
            row_swaps, second_probability, first_probability
        )
        permuted_second_probability = np.where(
            row_swaps, first_probability, second_probability
        )
        permuted_first_prediction = np.where(
            row_swaps, second_prediction, first_prediction
        )
        permuted_second_prediction = np.where(
            row_swaps, first_prediction, second_prediction
        )
        permuted = _paired_metric_differences(
            labels,
            permuted_first_probability,
            permuted_second_probability,
            permuted_first_prediction,
            permuted_second_prediction,
        )
        for name, value in permuted.items():
            if abs(value) + 1e-15 >= abs(observed[name]):
                exceedances[name] += 1
    return {
        name: {
            "observed_difference_second_minus_first": round(float(value), 6),
            "two_sided_probability": round(
                float((exceedances[name] + 1.0) / (iterations + 1.0)), 6
            ),
            "randomization_iterations": int(iterations),
            "inference_method": "paired_split_group_randomization",
            "exchangeability_unit": "split_group",
            "inference_scope": "conditional_on_fitted_model_predictions",
            "training_procedure_uncertainty_included": False,
        }
        for name, value in observed.items()
    }


def clustered_model_comparison(
    y_true: np.ndarray,
    first_probability: np.ndarray,
    second_probability: np.ndarray,
    first_prediction: np.ndarray,
    second_prediction: np.ndarray,
    groups: np.ndarray,
    iterations: int = 1000,
    seed: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Compare models with descriptive CIs and valid paired randomization."""
    randomization = paired_group_randomization_test(
        y_true,
        first_probability,
        second_probability,
        first_prediction,
        second_prediction,
        groups,
        iterations=iterations,
        seed=seed + 1,
    )
    codes, unique_groups = pd.factorize(groups, sort=False)
    order = np.argsort(codes, kind="stable")
    boundaries = np.cumsum(np.bincount(codes, minlength=len(unique_groups)))
    group_indices = dict(zip(unique_groups, np.split(order, boundaries[:-1])))
    random_state = np.random.RandomState(seed)
    differences = {name: [] for name in ("mcc", "auroc", "auprc")}
    for _ in range(iterations):
        sampled = random_state.choice(unique_groups, len(unique_groups), replace=True)
        indices = np.concatenate([group_indices[group] for group in sampled])
        labels = y_true[indices]
        if len(np.unique(labels)) > 1:
            differences["mcc"].append(
                float(matthews_corrcoef(labels, second_prediction[indices]))
                - float(matthews_corrcoef(labels, first_prediction[indices]))
            )
            differences["auroc"].append(
                float(roc_auc_score(labels, second_probability[indices]))
                - float(roc_auc_score(labels, first_probability[indices]))
            )
            differences["auprc"].append(
                float(average_precision_score(labels, second_probability[indices]))
                - float(average_precision_score(labels, first_probability[indices]))
            )
    output: dict[str, Any] = {}
    for metric, samples in differences.items():
        array = np.asarray(samples, dtype=float)
        if array.size == 0:
            output[metric] = None
            continue
        inference = randomization[metric]
        output[metric] = {
            "mean_difference_second_minus_first": round(
                float(inference["observed_difference_second_minus_first"]), 4
            ),
            "bootstrap_mean_difference_second_minus_first": round(
                float(array.mean()), 4
            ),
            "ci95": [
                round(float(value), 4) for value in np.percentile(array, [2.5, 97.5])
            ],
            "ci_method": "descriptive_split_group_bootstrap_percentile",
            "two_sided_probability": inference["two_sided_probability"],
            "randomization_iterations": inference["randomization_iterations"],
            "inference_method": inference["inference_method"],
            "exchangeability_unit": inference["exchangeability_unit"],
        }
    return output


def save_oof_artifacts(
    path: Path,
    y_true: np.ndarray,
    probabilities: dict[str, np.ndarray],
    config_tag: str,
    groups: np.ndarray | None = None,
    thresholds: dict[str, np.ndarray] | None = None,
    row_ids: np.ndarray | None = None,
    fold_ids: np.ndarray | None = None,
    extra_arrays: dict[str, np.ndarray] | None = None,
) -> None:
    """Save aligned prediction artifacts atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, np.ndarray] = {}
    if path.exists():
        with np.load(path, allow_pickle=True) as stored:
            data = {key: stored[key] for key in stored.files}
    prefix = f"{config_tag}__"
    data = {key: value for key, value in data.items() if not key.startswith(prefix)}
    data[f"{config_tag}__y"] = np.asarray(y_true)
    for name, values in probabilities.items():
        data[f"{config_tag}__{name}"] = np.asarray(values)
    if groups is not None:
        data[f"{config_tag}__groups"] = np.asarray(groups, dtype=object)
    if thresholds:
        for name, values in thresholds.items():
            data[f"{config_tag}__{name}__thresholds"] = np.asarray(values)
    if row_ids is not None:
        data[f"{config_tag}__row_ids"] = np.asarray(row_ids, dtype=object)
    if fold_ids is not None:
        data[f"{config_tag}__fold_ids"] = np.asarray(fold_ids, dtype=int)
    for name, values in (extra_arrays or {}).items():
        clean_name = str(name).strip()
        array = np.asarray(values)
        if not clean_name or clean_name in {
            "y",
            "groups",
            "row_ids",
            "fold_ids",
        }:
            raise ValueError(f"Invalid extra OOF artifact name: {name!r}")
        if array.ndim == 0 or len(array) != len(y_true):
            raise ValueError(f"Extra OOF artifact {clean_name} is misaligned")
        data[f"{config_tag}__{clean_name}"] = array
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **data)
    temporary.replace(path)
    logger.info("Saved prediction artifacts to %s", path)


def compute_shap_values(
    model: Any,
    values: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Compute held-out tree SHAP values."""
    try:
        import shap
    except ImportError as error:
        logger.warning("SHAP unavailable: %s", type(error).__name__)
        return None
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(values)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3:
            shap_values = shap_values[..., -1]
        if shap_values.shape != values.shape:
            raise ValueError("SHAP output shape differs from feature input")
        return (
            shap_values,
            np.asarray(values),
            np.asarray(feature_names, dtype=object),
        )
    except (ValueError, TypeError, RuntimeError) as error:
        logger.warning("SHAP failed: %s: %s", type(error).__name__, error)
        return None


def save_shap_artifact(
    path: Path,
    shap_values: list[np.ndarray],
    feature_values: list[np.ndarray],
    feature_names: list[str],
    fold_ids: list[np.ndarray],
) -> None:
    """Save fold-aggregated SHAP values."""
    if not shap_values:
        logger.warning("No SHAP values were generated")
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            shap=np.concatenate(shap_values),
            X=np.concatenate(feature_values),
            features=np.asarray(feature_names, dtype=object),
            fold_ids=np.concatenate(fold_ids),
        )
    temporary.replace(path)


def save_deep_bundle(
    path: Path,
    model: EnsembleModel,
    preprocessors: PreprocessorBundle,
    feature_names: list[str],
    threshold: float,
    architecture: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist inference state and feature order."""
    payload = {
        "architecture": architecture,
        "feature_names": feature_names,
        "threshold": float(threshold),
        "temperature": float(model.T),
        "calibration_slope": float(model.calibration_slope),
        "calibration_intercept": float(model.calibration_intercept),
        "calibration_method": "affine_logit_platt",
        "model_states": [member.state_dict() for member in model.members],
        "preprocessors": preprocessors,
        "model_config": {
            name: globals()[name] for name in TUNABLE_CONSTANTS
        },
        "metadata": {
            **(metadata or {}),
            # CHANGELOG 2026-09 (novelty N1): persist the protocol for either
            # family member and document the architecture that was actually
            # trained.  Under the default configuration this is the same
            # payload as before.
            **(
                {
                    "reliability_protocol": reliability_architecture_protocol(
                        feature_names, architecture=architecture
                    )
                }
                if is_reliability_family(architecture)
                else {}
            ),
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _torch_load(path: Path) -> dict[str, Any]:
    """Load trusted local PyTorch artifacts."""
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def load_deep_bundle(
    path: Path,
) -> tuple[EnsembleModel, PreprocessorBundle, list[str], float, dict[str, Any]]:
    """Restore a persisted deep ensemble."""
    payload = _torch_load(Path(path))
    architecture = str(payload["architecture"])
    feature_names = [str(value) for value in payload["feature_names"]]
    stored_config = dict(payload.get("model_config", {}))
    if architecture == EVIDENTIAL_RESIDUAL_ARCHITECTURE:
        # Older weights were trained without quality attenuation. Preserve
        # their inference rule instead of silently changing a saved predictor.
        stored_config.setdefault("EVIDENTIAL_RELIABILITY_POWER", 0.0)
    previous_config = {
        name: globals()[name]
        for name in stored_config
        if name in TUNABLE_CONSTANTS
    }
    members: list[nn.Module] = []
    try:
        for name, value in stored_config.items():
            if name in TUNABLE_CONSTANTS:
                globals()[name] = value
        for state in payload["model_states"]:
            member = _model_factory(
                architecture,
                len(feature_names),
                feature_names=feature_names,
            ).to(DEVICE)
            member.load_state_dict(state)
            members.append(member)
    finally:
        for name, value in previous_config.items():
            globals()[name] = value
    model = EnsembleModel(
        members,
        float(payload.get("temperature", 1.0)),
        (
            float(payload["calibration_slope"])
            if "calibration_slope" in payload
            else None
        ),
        float(payload.get("calibration_intercept", 0.0)),
    ).to(DEVICE)
    preprocessors = payload["preprocessors"]
    if not isinstance(preprocessors, PreprocessorBundle):
        raise TypeError("Deep bundle contains invalid preprocessing state")
    return (
        model,
        preprocessors,
        feature_names,
        float(payload["threshold"]),
        dict(payload.get("metadata", {})),
    )


def load_tuned_parameters(path: Path = TUNING_BEST_JSON) -> dict[str, Any]:
    """Load validated tuning parameters when present."""
    path = Path(path)
    if not path.exists():
        logger.info("No tuning artifact found; using configured defaults")
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    parameters = payload.get("production_params", payload.get("best_params", {}))
    accepted: dict[str, Any] = {}
    for name, value in parameters.items():
        if name in TUNABLE_CONSTANTS:
            globals()[name] = value
            accepted[name] = value
    if accepted and D_MODEL % N_HEADS:
        raise ValueError("Tuned D_MODEL is incompatible with N_HEADS")
    logger.info("Loaded %d tuned parameters", len(accepted))
    return accepted


def read_tuned_parameter_sets(
    path: Path = TUNING_BEST_JSON,
) -> dict[str, dict[str, Any]]:
    """Read architecture-scoped parameters without mutating module globals."""
    path = Path(path)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload.get("production_params_by_architecture")
    if candidates is None:
        architectures = payload.get("architectures")
        if isinstance(architectures, dict):
            candidates = {
                name: details.get("production_params", details.get("best_params", {}))
                for name, details in architectures.items()
                if isinstance(details, dict)
            }
    if not isinstance(candidates, dict):
        legacy = payload.get("production_params", payload.get("best_params", {}))
        candidates = {"cross_attention": legacy}
    output: dict[str, dict[str, Any]] = {}
    for architecture, parameters in candidates.items():
        if not isinstance(parameters, dict):
            continue
        accepted = {
            name: value
            for name, value in parameters.items()
            if name in TUNABLE_CONSTANTS
        }
        if accepted:
            heads = int(accepted.get("N_HEADS", N_HEADS))
            width = int(accepted.get("D_MODEL", D_MODEL))
            if width % heads:
                raise ValueError(
                    f"Tuned {architecture} D_MODEL is incompatible with N_HEADS"
                )
        output[str(architecture)] = accepted
    return output


@contextmanager
def temporary_model_config(parameters: dict[str, Any] | None):
    """Apply one architecture's configuration for training or restoration."""
    selected = {
        name: value
        for name, value in (parameters or {}).items()
        if name in TUNABLE_CONSTANTS
    }
    previous = {name: globals()[name] for name in selected}
    try:
        for name, value in selected.items():
            globals()[name] = value
        if D_MODEL % N_HEADS:
            raise ValueError("D_MODEL must divide evenly across N_HEADS")
        yield
    finally:
        for name, value in previous.items():
            globals()[name] = value


class LoRALinear(nn.Module):
    """Apply a trainable low-rank update."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = LORA_RANK,
        alpha: float = LORA_ALPHA,
        dropout: float = LORA_DROPOUT,
    ):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        factory = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.adapter_a = nn.Linear(base.in_features, rank, bias=False, **factory)
        self.adapter_b = nn.Linear(rank, base.out_features, bias=False, **factory)
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.adapter_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.adapter_b.weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.base(values) + self.adapter_b(
            self.adapter_a(self.dropout(values))
        ) * self.scale


def inject_lora(
    module: nn.Module,
    targets: tuple[str, ...] = LORA_TARGET_MODULES,
) -> int:
    """Replace matching linear layers recursively."""
    replacements = 0
    for child_name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and child_name in targets:
            setattr(module, child_name, LoRALinear(child))
            replacements += 1
        else:
            replacements += inject_lora(child, targets)
    if replacements and hasattr(module, "enable_torch_version"):
        # fair-esm's functional attention shortcut reads raw projection weights
        # and bypasses module.forward(), which would omit the adapters.
        module.enable_torch_version = False
    return replacements


class ESMVariantClassifier(nn.Module):
    """Classify mutation windows with fold-local LoRA."""

    def __init__(self, backbone: nn.Module, alphabet: Any, layer: int = ESM_LAYER):
        super().__init__()
        if not getattr(alphabet, "prepend_bos", False):
            raise ValueError("LoRA residue indexing requires an ESM alphabet with BOS")
        self.backbone = backbone
        self.alphabet = alphabet
        self.layer = layer
        width = int(getattr(backbone, "embed_dim", ESM_DIM))
        self.head = nn.Sequential(
            nn.LayerNorm(width * 2),
            nn.Linear(width * 2, width // 2),
            nn.GELU(),
            nn.Dropout(LORA_DROPOUT),
            nn.Linear(width // 2, 1),
        )
        self.T = 1.0

    def forward(
        self,
        tokens: torch.Tensor,
        residue_positions: torch.Tensor,
        alternate_tokens: torch.Tensor,
    ) -> torch.Tensor:
        output = self.backbone(tokens, repr_layers=[self.layer])
        representations = output["representations"][self.layer]
        batch = torch.arange(len(tokens), device=tokens.device)
        residue = representations[batch, residue_positions]
        alternate = self.backbone.embed_tokens(alternate_tokens)
        reference = self.backbone.embed_tokens(tokens[batch, residue_positions])
        mutation_delta = alternate - reference
        return self.head(torch.cat([residue, mutation_delta], dim=1)).squeeze(-1)


def _lora_dataframe_batches(
    df: pd.DataFrame,
    alphabet: Any,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    indices = np.arange(len(df))
    if shuffle:
        np.random.RandomState(seed).shuffle(indices)
    converter = alphabet.get_batch_converter()
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        rows = df.iloc[selected]
        data = [
            (str(row[ROW_ID_COL]), str(row["mutation_window"]))
            for _, row in rows.iterrows()
        ]
        _, _, tokens = converter(data)
        positions = torch.tensor(
            rows["window_aa_pos"].to_numpy(dtype=int), dtype=torch.long
        )
        alternate = torch.tensor(
            [alphabet.get_idx(str(value)) for value in rows["aa_alt"]],
            dtype=torch.long,
        )
        labels = torch.tensor(
            rows[LABEL_COL].to_numpy(dtype=np.float32), dtype=torch.float32
        )
        yield tokens, positions, alternate, labels


@torch.no_grad()
def predict_lora(model: ESMVariantClassifier, df: pd.DataFrame) -> np.ndarray:
    """Predict with a fold-local LoRA model."""
    if df.empty:
        return np.empty(0, dtype=np.float32)
    model.eval()
    inference_model = _parallel_model(model)
    inference_model.eval()
    outputs: list[np.ndarray] = []
    for tokens, positions, alternate, _ in _lora_dataframe_batches(
        df, model.alphabet, LORA_BATCH_SIZE, False, LORA_SEED
    ):
        tokens = tokens.to(DEVICE)
        positions = positions.to(DEVICE)
        alternate = alternate.to(DEVICE)
        with torch.amp.autocast(
            "cuda",
            enabled=DEVICE == "cuda" and LORA_PRECISION in {"fp16", "bf16"},
            dtype=torch.bfloat16 if LORA_PRECISION == "bf16" else torch.float16,
        ):
            logits = inference_model(tokens, positions, alternate)
        outputs.append(
            torch.sigmoid(logits.float() / max(model.T, 1e-6)).cpu().numpy()
        )
    result = np.concatenate(outputs)
    del inference_model
    return result


def train_lora_model(
    train_df: pd.DataFrame,
    stop_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    seed: int = LORA_SEED,
) -> ESMVariantClassifier:
    """Train a leakage-free fold-local LoRA model."""
    if not ENABLE_LORA:
        raise RuntimeError("LoRA training is disabled")
    required = {
        ROW_ID_COL,
        "mutation_window",
        "window_aa_pos",
        "aa_alt",
        LABEL_COL,
    }
    for name, frame in (
        ("train", train_df),
        ("stop", stop_df),
        ("calibration", calibration_df),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(f"{name} LoRA data misses {sorted(missing)}")
        validate_binary_labels(frame[LABEL_COL].to_numpy(), f"{name} labels")
    try:
        import esm
    except ImportError as error:
        raise RuntimeError("fair-esm is required for LoRA") from error
    set_seeds(seed)
    backbone, alphabet = esm.pretrained.load_model_and_alphabet(ESM_MODEL_NAME)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    replacements = inject_lora(backbone)
    if replacements == 0:
        raise RuntimeError("No matching ESM layers accepted LoRA")
    if LORA_GRADIENT_CHECKPOINTING and hasattr(
        backbone, "gradient_checkpointing_enable"
    ):
        backbone.gradient_checkpointing_enable()
    model = ESMVariantClassifier(backbone, alphabet).to(DEVICE)
    training_model = _parallel_model(model)
    adapter_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("head.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": adapter_parameters, "lr": ESM_LR},
            {"params": model.head.parameters(), "lr": HEAD_LR},
        ],
        weight_decay=DEEP_WD,
    )
    train_labels = train_df[LABEL_COL].to_numpy(dtype=int)
    positive_weight = torch.tensor(
        [fold_class_weight(train_labels)], dtype=torch.float32, device=DEVICE
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=DEVICE == "cuda" and LORA_PRECISION == "fp16"
    )
    best_score = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0
    for epoch in range(LORA_MAX_EPOCHS):
        if ESM_UNFREEZE_AFTER > 0 and epoch + 1 == ESM_UNFREEZE_AFTER:
            layers = getattr(model.backbone, "layers", None)
            if layers:
                newly_trainable = []
                for parameter in layers[-1].parameters():
                    if not parameter.requires_grad:
                        parameter.requires_grad = True
                        newly_trainable.append(parameter)
                if newly_trainable:
                    optimizer.add_param_group(
                        {"params": newly_trainable, "lr": ESM_LR * 0.1}
                    )
        training_model.train()
        optimizer.zero_grad(set_to_none=True)
        pending_examples = 0
        for batch_index, (tokens, positions, alternate, labels) in enumerate(
            _lora_dataframe_batches(
                train_df, alphabet, LORA_BATCH_SIZE, True, seed + epoch
            ),
            1,
        ):
            tokens = tokens.to(DEVICE)
            positions = positions.to(DEVICE)
            alternate = alternate.to(DEVICE)
            labels = labels.to(DEVICE)
            with torch.amp.autocast(
                "cuda",
                enabled=DEVICE == "cuda" and LORA_PRECISION in {"fp16", "bf16"},
                dtype=(
                    torch.bfloat16 if LORA_PRECISION == "bf16" else torch.float16
                ),
            ):
                loss = criterion(training_model(tokens, positions, alternate), labels)
                # Accumulate a sum, then normalize by the actual group size.
                # This also handles a short final microbatch or accumulation group.
                loss = loss * len(labels)
            scaler.scale(loss).backward()
            pending_examples += len(labels)
            if batch_index % LORA_GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(pending_examples)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                pending_examples = 0
        if pending_examples:
            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(pending_examples)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        stop_probabilities = predict_lora(model, stop_df)
        score = _safe_auc(
            stop_df[LABEL_COL].to_numpy(dtype=int), stop_probabilities
        )
        score = -math.inf if score is None else score
        if score > best_score + 1e-4:
            best_score = score
            wait = 0
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
        else:
            wait += 1
            if wait >= LORA_PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("LoRA training produced no checkpoint")
    model.load_state_dict(best_state)
    del training_model
    calibration_probabilities = predict_lora(model, calibration_df)
    clipped = np.clip(calibration_probabilities, 1e-6, 1.0 - 1e-6)
    calibration_logits = np.log(clipped / (1.0 - clipped))
    model.T = fit_temperature_from_logits(
        calibration_logits,
        calibration_df[LABEL_COL].to_numpy(dtype=int),
    )
    return model


def load_lora_bundle(path: Path) -> tuple[ESMVariantClassifier, float]:
    """Restore a fold-local LoRA classifier."""
    if not ENABLE_LORA:
        raise RuntimeError("LoRA inference is disabled")
    try:
        import esm
    except ImportError as error:
        raise RuntimeError("fair-esm is required for LoRA") from error
    payload = _torch_load(Path(path))
    stored_model = payload.get("esm_model", ESM_MODEL_NAME)
    if stored_model != ESM_MODEL_NAME:
        raise ValueError("LoRA checkpoint uses a different ESM model")
    if int(payload.get("esm_layer", ESM_LAYER)) != ESM_LAYER:
        raise ValueError("LoRA checkpoint uses a different ESM layer")
    stored_targets = tuple(payload.get("target_modules", LORA_TARGET_MODULES))
    if stored_targets != LORA_TARGET_MODULES:
        raise ValueError("LoRA checkpoint uses different target modules")
    if int(payload.get("rank", LORA_RANK)) != LORA_RANK:
        raise ValueError("LoRA checkpoint uses a different rank")
    if not math.isclose(float(payload.get("alpha", LORA_ALPHA)), LORA_ALPHA):
        raise ValueError("LoRA checkpoint uses a different alpha")
    if not math.isclose(
        float(payload.get("dropout", LORA_DROPOUT)), LORA_DROPOUT
    ):
        raise ValueError("LoRA checkpoint uses a different dropout")
    backbone, alphabet = esm.pretrained.load_model_and_alphabet(ESM_MODEL_NAME)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    replacements = inject_lora(backbone)
    if replacements == 0:
        raise RuntimeError("No matching ESM layers accepted LoRA")
    model = ESMVariantClassifier(backbone, alphabet).to(DEVICE)
    incompatible = model.load_state_dict(payload["state"], strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(
            f"Unexpected LoRA state keys: {incompatible.unexpected_keys}"
        )
    required_keys = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    declared_trainable = payload.get("trainable_parameter_names", [])
    if not isinstance(declared_trainable, list) or any(
        not isinstance(name, str) for name in declared_trainable
    ):
        raise ValueError("Invalid LoRA trainable-parameter manifest")
    required_keys.update(declared_trainable)
    missing_declared = required_keys - set(payload["state"])
    if missing_declared:
        raise ValueError(f"Missing LoRA adapter/head state: {sorted(missing_declared)}")
    model.T = float(payload["temperature"])
    threshold = float(payload["threshold"])
    if not math.isfinite(model.T) or model.T <= 0.0:
        raise ValueError("LoRA temperature must be finite and positive")
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("LoRA threshold must be a probability")
    model.eval()
    return model, threshold
