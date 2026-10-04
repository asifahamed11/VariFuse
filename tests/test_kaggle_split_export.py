from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


P = load("split_prepare_test", ROOT / "kaggle/prepare_kaggle.py")
B = load("split_build_test", ROOT / "tools/build_kaggle_split_bundle.py")


def package(tmp_path, kind, *, pair="same", relative=None, corrupt=False):
    relative = relative or f"{kind}/file.txt"
    payload = kind.encode()
    manifest = {
        "format_version": 1,
        "kind": kind,
        "pair_id": pair,
        "files": [
            {
                "path": relative,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    path = tmp_path / f"{kind}.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"VariFuse_2/VARIFUSE_{kind.upper()}_MANIFEST.json", json.dumps(manifest))
        archive.writestr(f"VariFuse_2/{relative}", b"oops" if corrupt else payload)
    return P.Package(path, kind)


def test_split_packages_import_and_resume_without_overwriting_results(tmp_path):
    code, data = package(tmp_path, "code"), package(tmp_path, "data")
    root = tmp_path / "work"
    try:
        first = P.prepare(code, data, root)
        assert first["copied_files"] == 2
        (root / "training_result.json").write_text("keep")
        second = P.prepare(code, data, root)
        assert second["copied_files"] == 0
        assert (root / "training_result.json").read_text() == "keep"
        (root / "data/file.txt").write_text("changed")
        with pytest.raises(ValueError, match="Existing file differs"):
            P.prepare(code, data, root)
    finally:
        code.close()
        data.close()


def test_mixed_export_pairs_rejected_before_copy(tmp_path):
    code, data = package(tmp_path, "code"), package(tmp_path, "data", pair="other")
    try:
        with pytest.raises(ValueError, match="different exports"):
            P.prepare(code, data, tmp_path / "work")
        assert not (tmp_path / "work").exists()
    finally:
        code.close()
        data.close()


@pytest.mark.parametrize("path", ["../escape", "/absolute", "C:/escape", "a\\escape", "."])
def test_unsafe_package_paths_rejected(path):
    with pytest.raises(ValueError, match="Unsafe"):
        P.relative_path(path)


def test_corrupted_zip_does_not_create_final_file(tmp_path):
    code, data = package(tmp_path, "code"), package(tmp_path, "data", corrupt=True)
    try:
        with pytest.raises(ValueError, match="checksum failed"):
            P.prepare(code, data, tmp_path / "work")
        assert not (tmp_path / "work/data/file.txt").exists()
    finally:
        code.close()
        data.close()


def test_exported_stage10_main_records_real_validation(tmp_path, monkeypatch):
    original = (ROOT / "src/10_extract_esm_features.py").read_bytes()
    exported = tmp_path / "exported_stage10.py"
    exported.write_bytes(B.patched_stage10(original))
    stage = load("exported_stage10_test", exported)
    calls = []
    monkeypatch.setattr(stage, "DEVICE", "cpu")
    monkeypatch.setattr(stage, "ensure_directories", lambda *args: None)
    monkeypatch.setattr(stage, "_selected_tasks", lambda selection: ["internal"])
    monkeypatch.setattr(
        stage, "_validate_task_upstream", lambda task: calls.append(("preflight", task))
    )
    monkeypatch.setattr(stage, "_hardware_summary", lambda: {})
    monkeypatch.setattr(stage, "_load_esm_model", lambda esm: (None, None))
    monkeypatch.setitem(sys.modules, "esm", SimpleNamespace())
    monkeypatch.setattr(
        stage,
        "extract_task",
        lambda *args, **kwargs: calls.append(("extract", kwargs["validate_upstream"])),
    )
    monkeypatch.setattr(sys, "argv", [str(exported), "--dataset", "internal"])
    stage.main()
    assert calls == [("preflight", "internal"), ("extract", True)]
    assert (ROOT / "src/10_extract_esm_features.py").read_bytes() == original


def test_notebook_has_required_dependency_order_and_valid_python():
    notebook = json.loads(B.notebook())
    stage_cells = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            compile(source, "notebook_cell", "exec")
            if source.startswith("run_stage("):
                stage_cells.append(source.strip())
    assert stage_cells == [
        'run_stage("stage10", "--dataset", "internal")',
        'run_stage("stage10", "--dataset", "clinvar")',
        'run_stage("stage10", "--dataset", "dms")',
        'run_stage("stage14")',
        'run_stage("stage11")',
        'run_stage("stage12")',
        'run_stage("stage13")',
    ]
