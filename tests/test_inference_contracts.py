from __future__ import annotations

import numpy as np

import common


stage12 = __import__("12_external_validation")


def test_group_model_inference_uses_reproducible_cluster_randomization() -> None:
    labels = np.tile(np.array([0, 1], dtype=int), 6)
    groups = np.repeat(np.array([f"G{index}" for index in range(6)]), 2)
    first_probability = np.tile(np.array([0.4, 0.6]), 6)
    second_probability = np.tile(np.array([0.1, 0.9]), 6)
    first_prediction = (first_probability >= 0.5).astype(int)
    second_prediction = (second_probability >= 0.5).astype(int)

    first = common.paired_group_randomization_test(
        labels,
        first_probability,
        second_probability,
        first_prediction,
        second_prediction,
        groups,
        iterations=127,
        seed=19,
    )
    second = common.paired_group_randomization_test(
        labels,
        first_probability,
        second_probability,
        first_prediction,
        second_prediction,
        groups,
        iterations=127,
        seed=19,
    )

    assert first == second
    for result in first.values():
        assert result["inference_method"] == "paired_split_group_randomization"
        assert result["exchangeability_unit"] == "split_group"
        assert 0.0 < result["two_sided_probability"] <= 1.0


def test_dms_paired_inference_uses_assay_sign_flip_not_bootstrap_tail() -> None:
    assays = {
        f"A{index}": {
            "models": {
                "baseline": {
                    "functional_spearman": 0.1 + index * 0.01,
                    "auroc": 0.55 + index * 0.01,
                },
                "candidate": {
                    "functional_spearman": 0.3 + index * 0.01,
                    "auroc": 0.75 + index * 0.01,
                },
            }
        }
        for index in range(8)
    }

    result = stage12._assay_paired_comparisons(
        assays,
        ["baseline", "candidate"],
        iterations=127,
        seed=23,
    )["candidate_minus_baseline"]

    for metric in ("functional_spearman", "auroc"):
        inference = result["metrics"][metric]
        assert inference["inference_method"] == "paired_assay_sign_flip_randomization"
        assert inference["exchangeability_unit"] == "assay"
        assert inference["ci_method"] == (
            "descriptive_paired_assay_bootstrap_percentile"
        )
        assert 0.0 < inference["two_sided_probability"] <= 1.0


def test_dms_inference_requires_at_least_two_paired_assays() -> None:
    assays = {
        "A": {
            "models": {
                "baseline": {"functional_spearman": 0.1, "auroc": 0.6},
                "candidate": {"functional_spearman": 0.2, "auroc": 0.7},
            }
        }
    }
    result = stage12._assay_paired_comparisons(
        assays, ["baseline", "candidate"], iterations=31, seed=5
    )["candidate_minus_baseline"]
    for metric in ("functional_spearman", "auroc"):
        inference = result["metrics"][metric]
        assert inference["paired_assays"] == 1
        assert inference["two_sided_probability"] is None
        assert inference["inference_status"] == "insufficient_paired_assays"
