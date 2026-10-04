"""Publication-safe robustness and uncertainty reporting utilities.

The helpers in this module deliberately separate three ideas that are often
conflated in variant-effect papers:

* deterministic stress testing is a sensitivity analysis, not missing-data
  imputation or evidence of clinical utility;
* selective-risk summaries are descriptive unless their selection rule was
  fixed independently of the evaluation outcomes; and
* split conformal guarantees require an explicit, sufficiently large
  calibration sample that is exchangeable with the evaluation sample.

Stage 12 uses the first two capabilities and records why a conformal claim is
not available for temporally or assay-shifted external cohorts.  The conformal
implementation is exposed for a future genuinely exchangeable calibration
cohort and fails closed when that contract is not met.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    matthews_corrcoef,
    roc_auc_score,
)

STRESS_PROTOCOL_VERSION = "prespecified_group_masking_v1"
STRESS_LEVELS = (0.0, 0.25, 0.50, 0.75, 1.0)

CONSERVATION_COLUMNS = (
    "GERP++_RS",
    "phyloP100way_vertebrate",
    "phastCons100way_vertebrate",
)
STRUCTURE_COLUMNS = (
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
STRESS_SCENARIOS: Mapping[str, tuple[str, ...]] = {
    "structure": STRUCTURE_COLUMNS,
    "conservation": CONSERVATION_COLUMNS,
    "structure_and_conservation": (*STRUCTURE_COLUMNS, *CONSERVATION_COLUMNS),
}

# These are reporting safeguards, not a retrospective power calculation.  A
# cohort below them can still be shown, but cannot support the corresponding
# inferential or clinical-calibration language in generated artifacts.
MIN_INFERENCE_N = 100
MIN_INFERENCE_PER_CLASS = 20
MIN_INFERENCE_GROUPS = 20
MIN_CALIBRATION_N = 200
MIN_CALIBRATION_PER_CLASS = 50
MIN_CALIBRATION_GROUPS = 30
MIN_CONFORMAL_CALIBRATION_N = 200
MIN_CONFORMAL_PER_CLASS = 50


def _as_clean_groups(groups: Iterable[Any]) -> np.ndarray:
    values = np.asarray(list(groups), dtype=object)
    if values.ndim != 1:
        raise ValueError("Stress-test groups must be one-dimensional")
    if pd.isna(values).any():
        raise ValueError("Stress-test groups contain missing values")
    cleaned = np.asarray([str(value).strip() for value in values], dtype=object)
    if np.any(cleaned == ""):
        raise ValueError("Stress-test groups contain blank values")
    return cleaned


def _stable_digest(seed: int, namespace: str, value: str) -> str:
    payload = f"{int(seed)}\0{namespace}\0{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ordered_identifier_sha256(values: Iterable[Any]) -> str:
    """Hash an ordered identifier sequence without ambiguous concatenation."""
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


@dataclass(frozen=True)
class GroupMaskPlan:
    """A deterministic, nested group-level masking plan."""

    row_mask: np.ndarray
    target_fraction: float
    selected_group_count: int
    total_group_count: int
    selected_groups: tuple[str, ...]
    grouping_unit: str
    namespace: str

    def audit(self) -> dict[str, Any]:
        row_count = int(len(self.row_mask))
        return {
            "target_fraction": float(self.target_fraction),
            "grouping_unit": self.grouping_unit,
            "selection_policy": "sha256_ordered_complete_groups_without_labels",
            "namespace": self.namespace,
            "selected_groups": int(self.selected_group_count),
            "total_groups": int(self.total_group_count),
            "realized_group_fraction": round(
                self.selected_group_count / max(self.total_group_count, 1), 6
            ),
            "masked_rows": int(self.row_mask.sum()),
            "total_rows": row_count,
            "realized_row_fraction": round(
                float(self.row_mask.mean()) if row_count else 0.0, 6
            ),
            "selected_group_ids_sha256": ordered_identifier_sha256(
                self.selected_groups
            ),
        }


def deterministic_group_mask(
    groups: Iterable[Any],
    fraction: float,
    *,
    seed: int,
    namespace: str,
    grouping_unit: str,
) -> GroupMaskPlan:
    """Select complete groups using a label-independent stable hash order.

    A single order is used for every masking level in a namespace, so masks at
    0/25/50/75/100% are nested.  The target applies to groups; the realized row
    fraction is reported separately and is never silently treated as exact.
    """
    if not 0.0 <= float(fraction) <= 1.0:
        raise ValueError("Stress-test fraction must be in [0, 1]")
    cleaned = _as_clean_groups(groups)
    unique = sorted(set(cleaned), key=lambda value: (_stable_digest(seed, namespace, value), value))
    target = int(math.floor(float(fraction) * len(unique) + 0.5))
    target = min(max(target, 0), len(unique))
    selected = tuple(unique[:target])
    row_mask = np.isin(cleaned, np.asarray(selected, dtype=object))
    return GroupMaskPlan(
        row_mask=row_mask.astype(bool, copy=False),
        target_fraction=float(fraction),
        selected_group_count=target,
        total_group_count=len(unique),
        selected_groups=selected,
        grouping_unit=str(grouping_unit),
        namespace=str(namespace),
    )


def source_modality_coverage(
    frame: pd.DataFrame, columns: Iterable[str]
) -> dict[str, Any]:
    """Measure source-valued coverage before train-fitted median imputation."""
    available = [column for column in columns if column in frame]
    if not available:
        present = np.zeros(len(frame), dtype=bool)
        finite_cells = 0
    else:
        numeric = frame[available].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        finite = np.isfinite(numeric)
        present = finite.any(axis=1)
        finite_cells = int(finite.sum())
    return {
        "declared_columns": [str(column) for column in columns],
        "available_columns": available,
        "rows_with_any_source_value": int(present.sum()),
        "row_coverage": round(float(present.mean()) if len(frame) else 0.0, 6),
        "finite_source_cells": finite_cells,
    }


def apply_modality_mask(
    frame: pd.DataFrame,
    row_mask: np.ndarray,
    columns: Iterable[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replace selected source values with NaN and update audit indicators.

    NaN is intentional: each frozen fold then applies its train-fitted median.
    No statistics are estimated from the external evaluation cohort.
    """
    mask = np.asarray(row_mask, dtype=bool)
    if mask.ndim != 1 or len(mask) != len(frame):
        raise ValueError("Modality mask and frame are misaligned")
    requested = tuple(dict.fromkeys(str(column) for column in columns))
    available = [column for column in requested if column in frame]
    output = frame.copy()
    finite_before = 0
    for column in available:
        # Normalize numeric storage to floating point before introducing NaN;
        # this avoids pandas' incompatible-dtype assignment path for int8 flags.
        output[column] = pd.to_numeric(output[column], errors="coerce").astype(float)
        numeric = output.loc[mask, column]
        finite_before += int(np.isfinite(numeric.to_numpy(dtype=float)).sum())
        output.loc[mask, column] = np.nan
        indicator = f"{column}__missing"
        if indicator in output:
            output.loc[mask, indicator] = 1

    structure_requested = bool(set(requested) & set(STRUCTURE_COLUMNS))
    updated_indicators: list[str] = []
    if structure_requested:
        for indicator in ("HAS_STRUCTURE", "STRUCTURE_FILE_AVAILABLE"):
            if indicator in output:
                output.loc[mask, indicator] = 0
                updated_indicators.append(indicator)
        if "LOW_CONFIDENCE_STRUCTURE" in output:
            # Absence is not low structural confidence; keep the two states
            # distinct for the audit-only availability control.
            output.loc[mask, "LOW_CONFIDENCE_STRUCTURE"] = 0
            updated_indicators.append("LOW_CONFIDENCE_STRUCTURE")
    return output, {
        "masked_columns": available,
        "unavailable_declared_columns": sorted(set(requested) - set(available)),
        "masked_rows": int(mask.sum()),
        "finite_source_cells_removed": finite_before,
        "availability_indicators_updated": updated_indicators,
        "imputation_policy": "frozen_training_fold_medians_only",
        "external_statistics_fitted": False,
    }


def clinical_evidence_assessment(
    labels: Iterable[int], groups: Iterable[Any]
) -> dict[str, Any]:
    """Return a fail-closed reporting policy for an external clinical cohort."""
    y = np.asarray(list(labels), dtype=int)
    clean_groups = _as_clean_groups(groups)
    if y.ndim != 1 or len(y) != len(clean_groups) or not np.isin(y, [0, 1]).all():
        raise ValueError("Clinical evidence labels/groups are invalid")
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    group_count = int(len(set(clean_groups)))
    inference_failures = []
    for condition, description in (
        (len(y) < MIN_INFERENCE_N, f"n<{MIN_INFERENCE_N}"),
        (positives < MIN_INFERENCE_PER_CLASS, f"positives<{MIN_INFERENCE_PER_CLASS}"),
        (negatives < MIN_INFERENCE_PER_CLASS, f"negatives<{MIN_INFERENCE_PER_CLASS}"),
        (group_count < MIN_INFERENCE_GROUPS, f"groups<{MIN_INFERENCE_GROUPS}"),
    ):
        if condition:
            inference_failures.append(description)
    calibration_failures = []
    for condition, description in (
        (len(y) < MIN_CALIBRATION_N, f"n<{MIN_CALIBRATION_N}"),
        (positives < MIN_CALIBRATION_PER_CLASS, f"positives<{MIN_CALIBRATION_PER_CLASS}"),
        (negatives < MIN_CALIBRATION_PER_CLASS, f"negatives<{MIN_CALIBRATION_PER_CLASS}"),
        (group_count < MIN_CALIBRATION_GROUPS, f"groups<{MIN_CALIBRATION_GROUPS}"),
    ):
        if condition:
            calibration_failures.append(description)
    inference_allowed = not inference_failures
    calibration_allowed = not calibration_failures
    return {
        "status": "adequately_sized_for_prespecified_reporting" if inference_allowed else "underpowered",
        "reporting_mode": "descriptive_with_group_inference" if inference_allowed else "descriptive_only",
        "n": int(len(y)),
        "positives": positives,
        "negatives": negatives,
        "groups": group_count,
        "inferential_model_comparison_allowed": inference_allowed,
        "clinical_calibration_claim_allowed": calibration_allowed,
        "clinical_utility_claim_allowed": False,
        "inference_guard_failures": inference_failures,
        "calibration_guard_failures": calibration_failures,
        "thresholds_are_prespecified_reporting_safeguards_not_power_analysis": True,
        "thresholds": {
            "model_comparison": {
                "minimum_n": MIN_INFERENCE_N,
                "minimum_per_class": MIN_INFERENCE_PER_CLASS,
                "minimum_groups": MIN_INFERENCE_GROUPS,
            },
            "external_calibration_interpretation": {
                "minimum_n": MIN_CALIBRATION_N,
                "minimum_per_class": MIN_CALIBRATION_PER_CLASS,
                "minimum_groups": MIN_CALIBRATION_GROUPS,
            },
        },
        "allowed_claims": [
            "descriptive_discrimination",
            "descriptive_calibration",
            "descriptive_selective_risk",
        ],
        "prohibited_claims": [
            "clinical_utility",
            *([] if inference_allowed else ["inferential_model_superiority"]),
            *([] if calibration_allowed else ["reliable_clinical_calibration"]),
        ],
    }


def conformal_availability_report(
    *,
    calibration_labels: Iterable[int] | None,
    exchangeability_justification: str | None,
    evaluation_shift: str,
) -> dict[str, Any]:
    """State whether a split-conformal claim can be attempted.

    This helper intentionally does not infer exchangeability from similar file
    names or shared features.  A caller must supply a substantive justification.
    """
    if calibration_labels is None:
        return {
            "status": "not_estimated",
            "reason": "no_distinct_labeled_calibration_sample",
            "coverage_guarantee_claimed": False,
            "evaluation_shift": evaluation_shift,
            "exchangeability_caveat": (
                "Finite-sample split-conformal coverage requires calibration and "
                "evaluation examples to be exchangeable; temporal, gene, homology, "
                "or assay shift can invalidate that guarantee."
            ),
        }
    labels = np.asarray(list(calibration_labels), dtype=int)
    if labels.ndim != 1 or not np.isin(labels, [0, 1]).all():
        raise ValueError("Conformal calibration labels must be binary")
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    failures = []
    if len(labels) < MIN_CONFORMAL_CALIBRATION_N:
        failures.append(f"calibration_n<{MIN_CONFORMAL_CALIBRATION_N}")
    if positives < MIN_CONFORMAL_PER_CLASS:
        failures.append(f"calibration_positives<{MIN_CONFORMAL_PER_CLASS}")
    if negatives < MIN_CONFORMAL_PER_CLASS:
        failures.append(f"calibration_negatives<{MIN_CONFORMAL_PER_CLASS}")
    if not exchangeability_justification:
        failures.append("exchangeability_not_justified")
    return {
        "status": "eligible" if not failures else "not_estimated",
        "reason": None if not failures else "guard_failed",
        "guard_failures": failures,
        "calibration_n": int(len(labels)),
        "calibration_positives": positives,
        "calibration_negatives": negatives,
        "coverage_guarantee_claimed": False,
        "evaluation_shift": evaluation_shift,
        "exchangeability_justification": exchangeability_justification,
        "exchangeability_caveat": (
            "Eligibility is not proof of exchangeability; the scientific sampling "
            "argument must be defended for the target population."
        ),
    }


def mondrian_split_conformal_binary(
    calibration_labels: Iterable[int],
    calibration_probabilities: Iterable[float],
    evaluation_probabilities: Iterable[float],
    *,
    alpha: float = 0.1,
    exchangeability_justification: str | None,
) -> dict[str, Any]:
    """Construct class-conditional binary split-conformal prediction sets.

    A result is returned only after the same sample-size and explicit
    exchangeability guards used by :func:`conformal_availability_report` pass.
    The calibration and evaluation arrays must be supplied separately.
    """
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("Conformal alpha must be in (0, 1)")
    labels = np.asarray(list(calibration_labels), dtype=int)
    calibration = np.asarray(list(calibration_probabilities), dtype=float)
    evaluation = np.asarray(list(evaluation_probabilities), dtype=float)
    if len(labels) != len(calibration) or not len(evaluation):
        raise ValueError("Conformal arrays are empty or misaligned")
    if (
        not np.isfinite(calibration).all()
        or not np.isfinite(evaluation).all()
        or ((calibration < 0.0) | (calibration > 1.0)).any()
        or ((evaluation < 0.0) | (evaluation > 1.0)).any()
    ):
        raise ValueError("Conformal probabilities must be finite values in [0, 1]")
    availability = conformal_availability_report(
        calibration_labels=labels,
        exchangeability_justification=exchangeability_justification,
        evaluation_shift="caller_asserted_exchangeable_target_sample",
    )
    if availability["status"] != "eligible":
        return availability

    thresholds: dict[int, float] = {}
    membership: dict[int, np.ndarray] = {}
    for label in (0, 1):
        selected = labels == label
        true_class_probability = (
            calibration[selected] if label == 1 else 1.0 - calibration[selected]
        )
        scores = 1.0 - true_class_probability
        n_class = len(scores)
        rank = min(n_class, int(math.ceil((n_class + 1) * (1.0 - alpha))))
        threshold = float(np.partition(scores, rank - 1)[rank - 1])
        thresholds[label] = threshold
        eval_class_probability = evaluation if label == 1 else 1.0 - evaluation
        membership[label] = (1.0 - eval_class_probability) <= threshold
    set_size = membership[0].astype(np.int8) + membership[1].astype(np.int8)
    return {
        **availability,
        "status": "estimated",
        "alpha": float(alpha),
        "method": "class_conditional_mondrian_split_conformal",
        "class_nonconformity_quantiles": {
            str(label): thresholds[label] for label in (0, 1)
        },
        "include_class_0": membership[0],
        "include_class_1": membership[1],
        "set_size": set_size,
        "empty_set_rate": round(float(np.mean(set_size == 0)), 6),
        "singleton_rate": round(float(np.mean(set_size == 1)), 6),
        "coverage_guarantee_claimed": True,
        "guarantee_scope": (
            "marginal within each class under the caller's exchangeability assertion"
        ),
    }


def safe_fallback(
    candidate_values: Iterable[float],
    anchor_values: Iterable[float],
    masked_rows: Iterable[bool],
) -> np.ndarray:
    """Use the sequence anchor exactly on rows whose modality was removed."""
    candidate = np.asarray(list(candidate_values))
    anchor = np.asarray(list(anchor_values))
    mask = np.asarray(list(masked_rows), dtype=bool)
    if candidate.shape != anchor.shape or candidate.ndim != 1 or len(mask) != len(candidate):
        raise ValueError("Safe-fallback arrays are misaligned")
    return np.where(mask, anchor, candidate)


def descriptive_stress_metrics(
    labels: Iterable[int],
    probabilities: Iterable[float],
    decisions: Iterable[int],
    groups: Iterable[Any],
    *,
    dms_functional_scores: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Compute prespecified descriptive endpoints without p-values or CIs."""
    y = np.asarray(list(labels), dtype=int)
    probability = np.asarray(list(probabilities), dtype=float)
    decision = np.asarray(list(decisions), dtype=int)
    clean_groups = _as_clean_groups(groups)
    if not (
        len(y) == len(probability) == len(decision) == len(clean_groups)
        and np.isin(y, [0, 1]).all()
        and np.isin(decision, [0, 1]).all()
        and np.isfinite(probability).all()
    ):
        raise ValueError("Stress metric inputs are invalid")
    both_classes = len(np.unique(y)) == 2
    output: dict[str, Any] = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "groups": int(len(set(clean_groups))),
        "auroc": round(float(roc_auc_score(y, probability)), 4) if both_classes else None,
        "auprc": round(float(average_precision_score(y, probability)), 4) if both_classes else None,
        "mcc": round(float(matthews_corrcoef(y, decision)), 4) if both_classes else None,
        "brier": round(float(brier_score_loss(y, probability)), 6),
        "reporting_scope": "descriptive_sensitivity_analysis",
        "confidence_interval": None,
        "hypothesis_test": None,
    }
    functional = (
        np.asarray(list(dms_functional_scores), dtype=float)
        if dms_functional_scores is not None
        else None
    )
    if functional is not None and len(functional) != len(y):
        raise ValueError("DMS functional scores are misaligned")
    group_aurocs: list[float] = []
    group_correlations: list[float] = []
    for group in sorted(set(clean_groups)):
        selected = clean_groups == group
        if len(np.unique(y[selected])) == 2:
            group_aurocs.append(float(roc_auc_score(y[selected], probability[selected])))
        if functional is not None:
            finite = selected & np.isfinite(functional)
            if finite.sum() >= 3:
                correlation = spearmanr(functional[finite], 1.0 - probability[finite]).statistic
                if np.isfinite(correlation):
                    group_correlations.append(float(correlation))
    output["group_macro_auroc"] = (
        round(float(np.mean(group_aurocs)), 4) if group_aurocs else None
    )
    output["group_macro_functional_spearman"] = (
        round(float(np.mean(group_correlations)), 4)
        if group_correlations
        else None
    )
    return output
