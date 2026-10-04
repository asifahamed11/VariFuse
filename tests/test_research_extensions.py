from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

import common
from research_evidence import (
    align_rows,
    attach_public_scores,
    paired_group_bootstrap,
    proteingym_aggregate,
)
from research_models import AblationResidualNet, research_factory


FEATURES = [
    'esm_variant_score', 'GERP++_RS', 'SASA', 'LOCAL_CONTACT_COUNT_8A',
    'LOCAL_MEAN_PLDDT_8A', 'LOCAL_MIN_PLDDT_8A',
    'LOCAL_CONFIDENT_CONTACT_FRACTION_8A', 'HAS_STRUCTURE',
    'LOW_CONFIDENCE_STRUCTURE', 'GERP++_RS__missing',
]


def _bio() -> torch.Tensor:
    values = torch.zeros((2, len(FEATURES)), dtype=torch.float32)
    values[:, FEATURES.index('esm_variant_score')] = torch.tensor([-1.0, 1.0])
    values[:, FEATURES.index('GERP++_RS__missing')] = 1.0
    return values


def test_row_alignment_is_exact_and_rejects_ambiguous_keys() -> None:
    frame = pd.DataFrame({'row_id': ['b', 'a'], 'value': [2, 1]})
    assert align_rows(frame, ['a', 'b']).value.tolist() == [1, 2]
    with pytest.raises(ValueError, match='uniquely'):
        align_rows(pd.concat([frame, frame.iloc[[0]]]), ['a', 'b'])
    with pytest.raises(ValueError, match='Unmapped'):
        align_rows(frame, ['missing'])


def test_paired_group_bootstrap_preserves_pairing() -> None:
    y = np.array([0, 1, 0, 1, 0, 1])
    score = np.array([.1, .9, .2, .8, .3, .7])
    result = paired_group_bootstrap(y, score, score, ['a', 'a', 'b', 'b', 'c', 'c'],
                                    iterations=40)
    for metric in ['auroc', 'auprc']:
        assert result['metrics'][metric]['difference'] == 0
        assert result['metrics'][metric]['ci95'] == [0.0, 0.0]


def test_proteingym_hierarchy_does_not_equal_weight_assays() -> None:
    assays = pd.DataFrame({
        'model': ['m'] * 3, 'DMS_id': ['a1', 'a2', 'b1'],
        'spearman': [0.0, 0.0, 1.0],
    })
    metadata = pd.DataFrame({
        'DMS_id': ['a1', 'a2', 'b1'], 'UniProt_ID': ['P1', 'P1', 'P2'],
        'coarse_selection_type': ['Activity', 'Activity', 'Binding'],
    })
    result = proteingym_aggregate(assays, metadata, 'spearman')['m']
    assert result['assay_macro'] == pytest.approx(1 / 3)
    assert result['score'] == pytest.approx(0.5)
    with pytest.raises(ValueError, match='uniquely'):
        proteingym_aggregate(assays, pd.concat([metadata, metadata.iloc[[0]]]), 'spearman')


def test_public_score_join_requires_unique_keys_and_declared_direction(tmp_path) -> None:
    frame = pd.DataFrame({'ASSAY_ID': ['x'], 'aa_ref': ['A'], 'aa_pos': [2], 'aa_alt': ['V']})
    path = tmp_path / 'scores.csv'
    pd.DataFrame({'DMS_id': ['x'], 'mutant': ['A2V'], 'score': [3.0]}).to_csv(path, index=False)
    assert attach_public_scores(frame, path, direction='higher_is_functional')[0] == -3
    with pytest.raises(ValueError, match='orientation'):
        attach_public_scores(frame, path, direction='unknown')


@pytest.mark.parametrize('variant', ['reliability_control', 'fixed_quality_gate', 'hard_switch',
                                     'no_modality_dropout', 'unbounded_residual'])
def test_every_residual_control_preserves_exact_missing_modality_fallback(variant) -> None:
    with common.temporary_model_config({'D_MODEL': 32, 'DROPOUT': 0.0}):
        model = AblationResidualNet(len(FEATURES), FEATURES, variant=variant, esm_dim=8).eval()
    components = model.forward_components(_bio(), torch.randn(2, 8))
    assert torch.equal(components['gate'], torch.zeros(2))
    assert torch.equal(components['logits'], components['anchor_logit'])


def test_ablation_changes_are_component_scoped() -> None:
    with common.temporary_model_config({'D_MODEL': 32, 'DROPOUT': 0.0, 'MODALITY_DROPOUT': 0.4}):
        fixed = AblationResidualNet(len(FEATURES), FEATURES, variant='fixed_quality_gate', esm_dim=8)
        no_dropout = AblationResidualNet(len(FEATURES), FEATURES, variant='no_modality_dropout', esm_dim=8)
        unbounded = AblationResidualNet(len(FEATURES), FEATURES, variant='unbounded_residual', esm_dim=8)
    assert isinstance(fixed.gate_network, torch.nn.Identity)
    assert no_dropout.modality_dropout == 0
    assert isinstance(unbounded.residual_head[-1], torch.nn.Identity)


def test_research_factory_is_restored_even_on_error() -> None:
    original = common._model_factory
    with pytest.raises(RuntimeError):
        with research_factory('hard_switch'):
            assert common._model_factory is not original
            raise RuntimeError('test')
    assert common._model_factory is original
