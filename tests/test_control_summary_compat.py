"""Legacy serialization must not weaken frozen prediction identity checks."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "summary_compat", Path(__file__).resolve().parents[1] / "tools/summarize_research_controls.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def checkpoint(tmp_path, dtype):
    values = {"indices": np.array([1, 0]), "probabilities": np.array([.8, .2]),
              "decisions": np.array([1, 0]), "y": np.array([1, 0]),
              "row_ids": np.array(["b", "a"], dtype=dtype)}
    (tmp_path / "model.pkl").write_bytes(b"not-loaded-model")
    np.savez_compressed(tmp_path / "predictions.npz", **values)
    record = {"fingerprint": "frozen", "model": "control", "fold": 1,
              "engineering_only": False, "artifacts": {
                  name: summary.sha256(tmp_path / name)
                  for name in ["model.pkl", "predictions.npz"]}}
    (tmp_path / "result.json").write_text(json.dumps(record))
    return values


@pytest.mark.parametrize("dtype", [object, str])
def test_checksum_verified_ids_export_without_pickle(tmp_path, dtype):
    original = checkpoint(tmp_path, dtype)
    loaded = summary.read_checkpoint(tmp_path, "frozen", "control", 1)
    frame = pd.DataFrame({"row_id": ["a", "b"], summary.C.LABEL_COL: [0, 1]})
    values = summary.assemble_oof([loaded], frame, [{"outer_validation": {"indices": [1, 0]}}])
    assert values["row_ids"].dtype.kind == "U"
    assert np.array_equal(values["probabilities"], original["probabilities"][[1, 0]])
    np.savez_compressed(tmp_path / "oof.npz", **values)
    with np.load(tmp_path / "oof.npz", allow_pickle=False) as saved:
        assert saved["row_ids"].tolist() == ["a", "b"]


def test_changed_checkpoint_is_rejected_before_loading(tmp_path):
    checkpoint(tmp_path, object)
    with (tmp_path / "predictions.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        summary.read_checkpoint(tmp_path, "frozen", "control", 1)


def test_protocol_and_row_identity_mismatches_are_rejected(tmp_path):
    checkpoint(tmp_path, object)
    with pytest.raises(ValueError, match="protocol mismatch"):
        summary.read_checkpoint(tmp_path, "different", "control", 1)
    loaded = summary.read_checkpoint(tmp_path, "frozen", "control", 1)
    frame = pd.DataFrame({"row_id": ["wrong", "b"], summary.C.LABEL_COL: [0, 1]})
    with pytest.raises(ValueError, match="row identity mismatch"):
        summary.assemble_oof([loaded], frame, [{"outer_validation": {"indices": [1, 0]}}])


def test_frozen_split_and_label_mismatches_are_rejected(tmp_path):
    checkpoint(tmp_path, str)
    loaded = summary.read_checkpoint(tmp_path, "frozen", "control", 1)
    frame = pd.DataFrame({"row_id": ["a", "b"], summary.C.LABEL_COL: [1, 0]})
    with pytest.raises(ValueError, match="validation indices differ"):
        summary.assemble_oof([loaded], frame, [{"outer_validation": {"indices": [0, 1]}}])
    with pytest.raises(ValueError, match="label identity mismatch"):
        summary.assemble_oof([loaded], frame, [{"outer_validation": {"indices": [1, 0]}}])
