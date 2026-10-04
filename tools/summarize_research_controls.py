"""Summarize frozen controls, including legacy pandas object-string row IDs.

The training runner is deliberately unchanged: its source hash is part of every
completed fold's locked protocol. Original fold artifacts are never rewritten.
"""
# ruff: noqa: E402
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import common as C
import run_research_controls as runner
from research_evidence import dump_json, extension_dir, rank_metrics, sha256
from research_models import ALL_MODELS

FIELDS = {"indices", "probabilities", "decisions", "y", "row_ids"}


def read_checkpoint(folder: Path, fingerprint: str, model: str, fold: int) -> dict:
    record = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    if record["fingerprint"] != fingerprint:
        raise ValueError("Checkpoint protocol mismatch")
    if record["model"] != model or record["fold"] != fold or record["engineering_only"]:
        raise ValueError("Checkpoint is not the declared full-data model/fold")
    if set(record["artifacts"]) != {"model.pkl", "predictions.npz"}:
        raise ValueError("Unexpected checkpoint artifact inventory")
    for name, checksum in record["artifacts"].items():
        if sha256(folder / name) != checksum:
            raise ValueError(f"Checkpoint artifact changed: {folder / name}")
    path = folder / "predictions.npz"
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != FIELDS:
            raise ValueError("Unexpected prediction fields")
        values = {key: saved[key] for key in FIELDS - {"row_ids"}}
        try:
            ids = saved["row_ids"]
        except ValueError as error:
            if "Object arrays cannot be loaded" not in str(error):
                raise
            # Only row_ids from the checksum-verified, locally trained checkpoint
            # use legacy pickle. Numerical fields must remain non-object arrays.
            with np.load(path, allow_pickle=True) as legacy:
                ids = legacy["row_ids"]
    if ids.ndim != 1 or not all(isinstance(value, str) for value in ids):
        raise ValueError("Checkpoint row IDs must be a vector of strings")
    values["row_ids"] = np.asarray(ids, dtype=str)
    size = len(ids)
    if any(value.shape != (size,) for value in values.values()):
        raise ValueError("Checkpoint prediction vectors are misaligned")
    if values["indices"].dtype.kind not in "iu":
        raise ValueError("Checkpoint indices must be integers")
    if not np.isfinite(values["probabilities"]).all():
        raise ValueError("Nonfinite checkpoint probabilities")
    if not np.isin(values["y"], [0, 1]).all() or not np.isin(values["decisions"], [0, 1]).all():
        raise ValueError("Checkpoint labels/decisions must be binary")
    return values


def assemble_oof(pieces: list[dict], frame, folds: list[dict]) -> dict:
    if len(pieces) != len(folds):
        raise ValueError("Missing OOF folds")
    for piece, fold in zip(pieces, folds):
        if not np.array_equal(piece["indices"], fold["outer_validation"]["indices"]):
            raise ValueError("Checkpoint validation indices differ from frozen split")
    indices = np.concatenate([piece["indices"] for piece in pieces])
    if len(indices) != len(frame) or set(indices) != set(range(len(frame))):
        raise ValueError("Incomplete/duplicated OOF coverage")
    order = np.argsort(indices)
    values = {key: np.concatenate([piece[key] for piece in pieces])[order] for key in FIELDS}
    if not np.array_equal(values["row_ids"], frame.row_id.astype(str).to_numpy()):
        raise ValueError("OOF row identity mismatch")
    if not np.array_equal(values["y"], frame[C.LABEL_COL].to_numpy()):
        raise ValueError("OOF label identity mismatch")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/15_research_extensions")
    args = parser.parse_args()
    output = extension_dir(args.source_root, args.output_root)
    frame, _embeddings, plan, original, names = runner.read_inputs(args.source_root)
    protocol = runner.protocol(args.source_root, output, frame, plan, original, names)
    assembled, records, input_hashes = {}, {}, {}
    for model in ALL_MODELS:
        pieces = []
        for fold in plan["folds"]:
            folder = output / "controls" / model / f'fold_{fold["outer_fold"]}'
            if not (folder / "result.json").is_file():
                raise ValueError(f"Training incomplete: {model}, fold {fold['outer_fold']}")
            pieces.append(read_checkpoint(folder, protocol["fingerprint"], model, fold["outer_fold"]))
            for filename in ["result.json", "model.pkl", "predictions.npz"]:
                path = folder / filename
                input_hashes[str(path.relative_to(output))] = sha256(path)
        assembled[model] = assemble_oof(pieces, frame, plan["folds"])
        records[model] = {"status": "complete", "metrics": rank_metrics(
            assembled[model]["y"], assembled[model]["probabilities"])}
    repair = output / "summary_repair"
    repair.mkdir(exist_ok=True)
    for name in ["controls_status.json", "protocol.json"]:
        backup = repair / ("before_" + name)
        if (output / name).exists() and not backup.exists():
            shutil.copy2(output / name, backup)
    output_hashes = {}
    for model, values in assembled.items():
        path = output / "controls" / f"{model}_oof.npz"
        temp = path.with_suffix(".npz.tmp")
        with temp.open("wb") as stream:
            np.savez_compressed(stream, **values)
        with np.load(temp, allow_pickle=False) as saved:
            if any(not np.array_equal(saved[key], values[key]) for key in FIELDS):
                raise RuntimeError("OOF export roundtrip mismatch")
        temp.replace(path)
        output_hashes[str(path.relative_to(output))] = sha256(path)
    dump_json(output / "controls_status.json", {
        "role": "exploratory_component_and_same_input_controls", "models": records,
        "all_complete": True})
    if any(sha256(output / name) != checksum for name, checksum in input_hashes.items()):
        raise RuntimeError("A frozen checkpoint changed while summarizing")
    dump_json(repair / "summary_provenance.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "Legacy object-string row_ids could not be read with allow_pickle=False",
        "training_repeated": False, "original_fold_artifacts_modified": False,
        "protocol_fingerprint": protocol["fingerprint"], "script_sha256": sha256(Path(__file__)),
        "input_sha256": input_hashes, "output_sha256": output_hashes,
        "exported_row_id_dtype": "unicode", "all_models": list(ALL_MODELS),
        "folds_per_model": len(plan["folds"]), "rows_per_model": len(frame)})
    print(f"Complete: {len(records)} models, {sum(len(plan['folds']) for _ in records)} folds; all_complete=true.")


if __name__ == "__main__":
    main()
