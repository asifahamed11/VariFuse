from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


run_pipeline = importlib.import_module("run_pipeline")


def test_pipeline_refuses_nonempty_figure_root_for_clean_stage01(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "new_outputs"
    figures = tmp_path / "old_figures"
    output.mkdir()
    figures.mkdir()
    (figures / "old_result.pdf").write_bytes(b"legacy")
    monkeypatch.setenv("VARIANT_OUTPUT_DIR", str(output))
    monkeypatch.setenv("VARIANT_FIGURE_DIR", str(figures))
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--stages", "01"])

    with pytest.raises(RuntimeError, match="non-empty artifact roots"):
        run_pipeline.main()

    assert (figures / "old_result.pdf").read_bytes() == b"legacy"


def test_pipeline_explicit_override_reaches_direct_stage13_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "outputs"
    figures = tmp_path / "figures"
    output.mkdir()
    figures.mkdir()
    (figures / "old_result.pdf").write_bytes(b"legacy")
    monkeypatch.setenv("VARIANT_OUTPUT_DIR", str(output))
    monkeypatch.setenv("VARIANT_FIGURE_DIR", str(figures))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pipeline.py",
            "--stages",
            "13",
            "--allow-existing-output",
        ],
    )
    captured: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_pipeline.subprocess, "run", fake_run)

    run_pipeline.main()

    environment = captured["env"]
    assert isinstance(environment, dict)
    assert environment["ALLOW_EXISTING_FIGURE_DIR"] == "1"


def test_pipeline_forwards_prespecified_confirmation_seeds_only_to_stage14(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pipeline.py",
            "--stages",
            "14",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--figure-dir",
            str(tmp_path / "figures"),
            "--confirmation-split-seeds",
            "1701",
            "2903",
            "4159",
        ],
    )
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_pipeline.subprocess, "run", fake_run)
    run_pipeline.main()

    assert len(commands) == 1
    assert commands[0][-4:] == [
        "--confirmation-split-seeds",
        "1701",
        "2903",
        "4159",
    ]


def test_local_publication_auto_binds_bundled_mmseqs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "mmseqs.exe"
    executable.write_bytes(b"fixture")
    monkeypatch.delenv("MMSEQS_EXECUTABLE", raising=False)
    monkeypatch.setattr(run_pipeline, "LOCAL_MMSEQS_CANDIDATES", (executable,))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pipeline.py",
            "--local-publication",
            "--run-id",
            "test",
            "--stages",
            "08b",
        ],
    )
    captured: dict[str, object] = {}

    def fake_run(*_: object, **kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_pipeline.subprocess, "run", fake_run)

    run_pipeline.main()

    environment = captured["env"]
    assert isinstance(environment, dict)
    assert environment["MMSEQS_EXECUTABLE"] == str(executable.resolve())
    assert environment["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert environment["VARIANT_TRAIN_CUTOFF_DATE"] == "2024-06-30"
    assert environment["CLINVAR_REQUIRE_SCV_EVIDENCE"] == "1"
    assert environment["CLINVAR_SCV_MULTIPLE_SUBMITTER_MINIMUM"] == "2"
    assert environment["REQUIRE_TRANSCRIPT_MAPPING"] == "1"
