"""Isolated ablation models; legacy production architectures are unchanged."""
from __future__ import annotations

from contextlib import contextmanager
import numpy as np
import torch
from torch import nn

import common as C


NEURAL_MODELS = ('reliability_control', 'fixed_quality_gate', 'hard_switch',
                 'no_modality_dropout', 'unbounded_residual', 'same_input_mlp')
BASELINE_MODELS = ('same_input_logistic', 'same_input_lightgbm')
ALL_MODELS = (*NEURAL_MODELS, *BASELINE_MODELS)


def predict_bundle(bundle, frame, embeddings):
    """Score saved controls using their own frozen feature/preprocessing schema."""
    if embeddings.shape != (len(frame), bundle['esm_dim']):
        raise ValueError('Prediction embedding shape mismatch')
    names = bundle['feature_names']
    if len(set(names)) != len(names) or set(names) - set(frame):
        raise ValueError('Prediction feature schema mismatch')
    bio = frame[names].to_numpy(np.float32)
    model, prep = bundle['model'], bundle['preprocessors']
    kind = bundle['model_type']
    if kind in BASELINE_MODELS:
        values = np.column_stack([bio, np.asarray(embeddings, dtype=np.float32)])
        if kind == 'same_input_logistic':
            scores = bundle['calibrator'].predict_from_logits(model.decision_function(prep.transform(values)))
        else:
            scores = bundle['calibrator'].predict(model.booster_.predict(values))
    elif kind in NEURAL_MODELS:
        model = model.to(C.DEVICE)
        try:
            scores = C.predict(model, prep.transform_bio(bio), prep.transform_esm(embeddings))
        finally:
            model.cpu()
    else:
        raise ValueError(f'Unknown control model: {kind}')
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite saved-model predictions')
    return scores


class AblationResidualNet(C.ReliabilityResidualNet):
    """One-factor changes to the existing model, trained independently.

    fixed_quality_gate removes only the learned gate; hard_switch also removes
    continuous evidence-quality attenuation. Neither removes exact fallback.
    """
    def __init__(self, n_features, feature_names, *, variant, esm_dim=C.ESM_DIM):
        super().__init__(n_features, feature_names, esm_dim=esm_dim)
        if variant not in NEURAL_MODELS or variant == 'same_input_mlp':
            raise ValueError(f'Invalid reliability ablation: {variant}')
        self.variant = variant
        if variant in ['fixed_quality_gate', 'hard_switch']:
            # Keep construction order/RNG identical to control, then remove the
            # unused learned gate so reported trainable parameters are truthful.
            self.gate_network = nn.Identity()
        if variant == 'no_modality_dropout':
            self.modality_dropout = 0.0
        if variant == 'unbounded_residual':
            self.residual_head[-1] = nn.Identity()

    def _fuse(self, encoded, bio_values, structure_keep=None, evolution_keep=None):
        if self.variant not in ['fixed_quality_gate', 'hard_switch']:
            return super()._fuse(encoded, bio_values, structure_keep, evolution_keep)
        structure, evolution, sr, cr, available = self._degraded_views(
            encoded, bio_values, structure_keep, evolution_keep)
        residual = self.residual_head(torch.cat([
            structure, evolution, encoded['context'], encoded['esm'],
            torch.abs(structure-evolution)], dim=1)).squeeze(-1) * self.residual_scale
        gate = available
        if self.variant == 'fixed_quality_gate':
            gate = gate * torch.maximum(sr, cr).clamp(0, 1)
        return {'logits': encoded['anchor_logit'] + gate * residual,
                'anchor_logit': encoded['anchor_logit'], 'residual': residual,
                'gate': gate, 'hard_availability': available,
                'structure_reliability': sr, 'conservation_reliability': cr,
                'structure_embedding': structure, 'evolution_embedding': evolution}


@contextmanager
def research_factory(variant: str):
    """Scoped hook for the existing trainer in a single-process research job."""
    if variant not in NEURAL_MODELS:
        raise ValueError(variant)
    original = C._model_factory

    def factory(architecture, n_features, feature_names=None):
        if architecture != 'research_extension':
            return original(architecture, n_features, feature_names)
        if variant == 'same_input_mlp':
            return C.ConcatenationMLP(n_features)
        return AblationResidualNet(n_features, feature_names, variant=variant)

    C._model_factory = factory
    try:
        yield
    finally:
        C._model_factory = original
