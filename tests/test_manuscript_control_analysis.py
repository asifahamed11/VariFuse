"""Guards for frozen clinical scoring and its evaluation unit."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    'manuscript_analysis', Path(__file__).parents[1] / 'tools/analyze_manuscript_controls.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_identity_alignment_reorders_and_checks_labels():
    frame = pd.DataFrame({'row_id': ['b', 'a'], 'LABEL_PATHOGENIC': [0, 1]})
    assert module.checked_alignment(frame, ['a', 'b'], [1, 0]).row_id.tolist() == ['a', 'b']
    with pytest.raises(ValueError, match='labels'):
        module.checked_alignment(frame, ['a', 'b'], [0, 1])
    with pytest.raises(ValueError, match='unique'):
        module.checked_alignment(frame, ['a', 'a'], [1, 1])


def test_majority_vote_uses_each_frozen_threshold_not_mean_threshold():
    scores = np.array([[.4], [.4], [.4], [.95], [.95]])
    mean, votes = module.aggregate_folds(scores, [.3, .3, .3, .99, .99])
    assert mean[0] == pytest.approx(.62)
    assert votes.tolist() == [1]
    assert (mean >= np.mean([.3, .3, .3, .99, .99])).tolist() == [True]
    # The reverse case distinguishes the majority policy from a mean cutoff.
    mean, votes = module.aggregate_folds(scores, [.5, .5, .5, .7, .7])
    assert votes.tolist() == [0]
    assert (mean >= np.mean([.5, .5, .5, .7, .7])).tolist() == [True]


def test_invalid_or_incomplete_fold_predictions_fail_closed():
    with pytest.raises(ValueError, match='five'):
        module.aggregate_folds(np.ones((4, 2)) * .5, np.ones(4) * .5)
    with pytest.raises(ValueError, match='Nonfinite'):
        module.aggregate_folds(np.full((5, 2), np.nan), np.ones(5) * .5)
    with pytest.raises(ValueError, match='probabilities'):
        module.aggregate_folds(np.ones((5, 2)) * 1.1, np.ones(5) * .5)
