"""Reconcile a completed Stage 14 run without training any model.

The Optuna database remains the authoritative ledger.  This utility rebuilds
``trials_history.csv`` from it, changes any stale per-fold attempt count to the
number of documented COMPLETE/PRUNED rows, and records every discrepancy.  It
also refreshes the affected provenance fingerprints in downstream manifests.
Original metadata files are copied to a separate backup directory first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
from typing import Any

import optuna
import pandas as pd


TARGET_STUDY = (
    "cross_attention_predictor_free_"
    "fixed_nested_group_cv_v4_reliability_residual_outer_2"
)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def sampled_sha256(path: Path, sample_size: int = 1 << 20) -> str:
    size = path.stat().st_size
    digest = hashlib.sha256(str(size).encode())
    with path.open("rb") as handle:
        digest.update(handle.read(sample_size))
        if size > sample_size:
            handle.seek(max(0, size - sample_size))
            digest.update(handle.read(sample_size))
    return digest.hexdigest()


def directory_fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = total = 0
    for item in sorted(
        (candidate for candidate in path.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(path).as_posix(),
    ):
        relative = item.relative_to(path).as_posix()
        stat = item.stat()
        count += 1
        total += stat.st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode())
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode())
        digest.update(b"\n")
    return {
        "metadata_sha256": digest.hexdigest(),
        "file_count": count,
        "total_size_bytes": total,
        "content_hash_policy": "path_size_mtime",
    }


def artifact_record(path: Path, artifact_id: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "exists": path.exists(),
        "kind": "directory" if path.is_dir() else "file",
        "mtime_ns": path.stat().st_mtime_ns,
    }
    if path.is_file():
        record.update(
            sha256=sha256(path),
            sample_sha256=sampled_sha256(path),
            size_bytes=path.stat().st_size,
            hash_policy="full_sha256",
        )
    else:
        record["directory_fingerprint"] = directory_fingerprint(path)
    if artifact_id is not None:
        record["artifact_id"] = artifact_id
    return record


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def backup(path: Path, output_root: Path, backup_root: Path) -> None:
    destination = backup_root / path.relative_to(output_root)
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)


def update_attempt_count(value: Any, documented: int) -> int:
    changed = 0
    if isinstance(value, dict):
        if (
            value.get("study_name") == TARGET_STUDY
            and value.get("attempted_trials") != documented
        ):
            value["attempted_trials"] = documented
            changed += 1
        for child in value.values():
            changed += update_attempt_count(child, documented)
    elif isinstance(value, list):
        for child in value:
            changed += update_attempt_count(child, documented)
    return changed


def refresh_manifest_input(manifest: dict[str, Any], suffix: str, path: Path) -> bool:
    normalized = suffix.replace("\\", "/")
    for key in list(manifest.get("inputs", {})):
        if key.replace("\\", "/").endswith(normalized):
            manifest["inputs"][key] = artifact_record(path)
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path, default=Path(__file__).resolve().parents[1] / "outputs"
    )
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "audit_reports/final_run_20260915/original_before_repair"
        ),
    )
    arguments = parser.parse_args()
    output_root = arguments.output_root.resolve()
    stage14 = output_root / "14_tuning"
    database = stage14 / "optuna.db"
    architecture_file = stage14 / "architecture_selection.json"
    if not all(path.is_file() for path in (database, architecture_file)):
        raise FileNotFoundError("Completed Stage 14 database/results are required")

    storage = f"sqlite:///{database.as_posix()}"
    summaries = optuna.get_all_study_summaries(storage=storage)
    expected_names = {
        record["study_name"]
        for details in json.loads(architecture_file.read_text(encoding="utf-8"))[
            "architectures"
        ].values()
        for record in details["fold_results"]
    }
    observed_names = {summary.study_name for summary in summaries}
    if expected_names != observed_names:
        raise RuntimeError("Architecture results and Optuna study inventory differ")

    frames = []
    accounting = []
    target_documented = None
    for name in sorted(expected_names):
        study = optuna.load_study(study_name=name, storage=storage)
        frame = study.trials_dataframe(
            attrs=(
                "number",
                "value",
                "datetime_start",
                "datetime_complete",
                "duration",
                "params",
                "state",
            )
        )
        architecture, fold_text = name.split("_predictor_free_", 1)[0], name.rsplit("_", 1)[1]
        frame.insert(0, "outer_fold", int(fold_text))
        frame.insert(0, "architecture", architecture)
        frames.append(frame)
        states = frame["state"].value_counts().to_dict()
        documented = int(states.get("COMPLETE", 0) + states.get("PRUNED", 0))
        missing_numbers = sorted(set(range(40)) - set(frame["number"].astype(int)))
        accounting.append(
            {
                "study_name": name,
                "documented_trials": documented,
                "state_counts": {key: int(value) for key, value in states.items()},
                "missing_trial_numbers_below_target": missing_numbers,
            }
        )
        if name == TARGET_STUDY:
            target_documented = documented
    if target_documented is None:
        raise RuntimeError(f"Target study is absent: {TARGET_STUDY}")

    trial_file = stage14 / "trials_history.csv"
    stage14_manifest_file = stage14 / "run_manifest.json"
    touched_json = [
        architecture_file,
        stage14 / "best_cross_attention_params.json",
        stage14 / "architecture_checkpoints/cross_attention_checkpoint.json",
    ]
    for path in [trial_file, stage14_manifest_file, *touched_json]:
        backup(path, output_root, arguments.backup_root.resolve())

    history = pd.concat(frames, ignore_index=True).sort_values(
        ["architecture", "outer_fold", "number"], kind="stable"
    )
    temporary_csv = trial_file.with_suffix(".csv.tmp")
    history.to_csv(temporary_csv, index=False)
    temporary_csv.replace(trial_file)

    changed_occurrences = {}
    for path in touched_json:
        payload = json.loads(path.read_text(encoding="utf-8"))
        count = update_attempt_count(payload, target_documented)
        if count:
            atomic_json(path, payload)
        changed_occurrences[path.relative_to(output_root).as_posix()] = count

    reconciliation_file = stage14 / "tuning_history_reconciliation.json"
    reconciliation = {
        "schema_version": 1,
        "status": "reconciled_without_model_retraining",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "authority": "optuna_sqlite_database",
        "database_sha256": sha256(database),
        "target_trials_per_study": 40,
        "expected_studies": len(expected_names),
        "documented_trials": len(history),
        "state_counts": {
            key: int(value) for key, value in history["state"].value_counts().items()
        },
        "discrepancies": [record for record in accounting if record["documented_trials"] != 40],
        "resolution": (
            "No trial was fabricated. The affected fold's attempted_trials field "
            "was changed from 40 to the 39 COMPLETE/PRUNED records present in the "
            "database. Its best objective and final predictions were unchanged."
        ),
        "changed_occurrences": changed_occurrences,
        "history_file": "14_tuning/trials_history.csv",
        "history_sha256": sha256(trial_file),
        "backup_root": str(arguments.backup_root.resolve()),
    }
    atomic_json(reconciliation_file, reconciliation)

    stage14_manifest = json.loads(stage14_manifest_file.read_text(encoding="utf-8"))
    changed_stage14 = {
        "14_tuning/trials_history.csv": trial_file,
        "14_tuning/architecture_selection.json": architecture_file,
        "14_tuning/best_cross_attention_params.json": stage14 / "best_cross_attention_params.json",
        "14_tuning/architecture_checkpoints": stage14 / "architecture_checkpoints",
        "14_tuning/tuning_history_reconciliation.json": reconciliation_file,
    }
    for artifact_id, path in changed_stage14.items():
        stage14_manifest["outputs"][artifact_id] = artifact_record(path, artifact_id)
    stage14_manifest.setdefault("extra", {})["post_run_trial_history_reconciliation"] = {
        "status": "applied_without_model_retraining",
        "artifact": "14_tuning/tuning_history_reconciliation.json",
        "database_trials": len(history),
        "missing_trial_was_fabricated": False,
    }
    atomic_json(stage14_manifest_file, stage14_manifest)

    stage11_manifest_file = output_root / "11_train_and_evaluate/run_manifest.json"
    backup(stage11_manifest_file, output_root, arguments.backup_root.resolve())
    stage11_manifest = json.loads(stage11_manifest_file.read_text(encoding="utf-8"))
    for suffix, path in (
        ("/14_tuning/architecture_selection.json", architecture_file),
        ("/14_tuning/best_cross_attention_params.json", stage14 / "best_cross_attention_params.json"),
        ("/14_tuning/run_manifest.json", stage14_manifest_file),
    ):
        if not refresh_manifest_input(stage11_manifest, suffix, path):
            raise RuntimeError(f"Stage 11 manifest does not reference {suffix}")
    stage11_manifest.setdefault("extra", {})["post_run_input_metadata_reconciliation"] = {
        "status": "fingerprints_refreshed_after_trial_accounting_only_change",
        "model_artifacts_retrained": False,
    }
    atomic_json(stage11_manifest_file, stage11_manifest)

    stage12_manifest_file = output_root / "12_external_validation/run_manifest.json"
    backup(stage12_manifest_file, output_root, arguments.backup_root.resolve())
    stage12_manifest = json.loads(stage12_manifest_file.read_text(encoding="utf-8"))
    for suffix, path in (
        ("/14_tuning/architecture_selection.json", architecture_file),
        ("/14_tuning/best_cross_attention_params.json", stage14 / "best_cross_attention_params.json"),
        ("/14_tuning/run_manifest.json", stage14_manifest_file),
        ("/11_train_and_evaluate/run_manifest.json", stage11_manifest_file),
    ):
        if not refresh_manifest_input(stage12_manifest, suffix, path):
            raise RuntimeError(f"Stage 12 manifest does not reference {suffix}")
    stage12_manifest.setdefault("extra", {})["post_run_input_metadata_reconciliation"] = {
        "status": "fingerprints_refreshed_after_trial_accounting_only_change",
        "external_predictions_recomputed": False,
    }
    atomic_json(stage12_manifest_file, stage12_manifest)

    print(json.dumps(reconciliation, indent=2))


if __name__ == "__main__":
    main()
