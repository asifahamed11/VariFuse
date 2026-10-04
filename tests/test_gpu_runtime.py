from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import gpu_runtime as G


def _dual_t4(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(G.USE_ALL_GPUS_ENV, "1")
    monkeypatch.setenv(G.REQUIRED_GPU_COUNT_ENV, "2")
    monkeypatch.setenv(G.MAX_GPU_COUNT_ENV, "2")
    monkeypatch.setenv(G.GPU_NAME_CONTAINS_ENV, "T4")
    monkeypatch.setattr(G.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        G.torch.cuda,
        "get_device_name",
        lambda index: f"Tesla T4 worker {index}",
    )


def test_dual_t4_contract_selects_both_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _dual_t4(monkeypatch)
    assert G.selected_cuda_device_ids("cuda") == [0, 1]


def test_dual_t4_contract_rejects_silent_single_gpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _dual_t4(monkeypatch)
    monkeypatch.setattr(G.torch.cuda, "device_count", lambda: 1)
    with pytest.raises(RuntimeError, match="requires 2 CUDA GPUs"):
        G.selected_cuda_device_ids("cuda")


def test_dual_t4_contract_rejects_wrong_accelerator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _dual_t4(monkeypatch)
    monkeypatch.setattr(
        G.torch.cuda,
        "get_device_name",
        lambda index: "NVIDIA L4" if index else "Tesla T4",
    )
    with pytest.raises(RuntimeError, match="must contain 't4'"):
        G.selected_cuda_device_ids("cuda")


def test_data_parallel_is_transient_and_uses_both_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _dual_t4(monkeypatch)
    captured: dict[str, object] = {}

    class FakeDataParallel(nn.Module):
        def __init__(
            self,
            module: nn.Module,
            device_ids: list[int],
            output_device: int,
        ) -> None:
            super().__init__()
            captured.update(
                module=module,
                device_ids=device_ids,
                output_device=output_device,
            )
            self.module = module

        def forward(self, values):  # type: ignore[no-untyped-def]
            return self.module(values)

    monkeypatch.setattr(G, "SafeDataParallel", FakeDataParallel)
    model = nn.Linear(3, 1)
    wrapped = G.data_parallel(model, "cuda")
    assert isinstance(wrapped, FakeDataParallel)
    assert captured == {
        "module": model,
        "device_ids": [0, 1],
        "output_device": 0,
    }
    assert wrapped.module is model


def test_safe_data_parallel_runs_singleton_on_primary_without_replica_error() -> None:
    class TokensRequired(nn.Module):
        def forward(self, tokens: torch.Tensor, repr_layers: list[int]):
            return tokens + len(repr_layers)

    parallel = object.__new__(G.SafeDataParallel)
    nn.Module.__init__(parallel)
    parallel.module = TokensRequired()
    parallel.device_ids = [0, 1]
    tokens = torch.ones((1, 4))
    observed = parallel(tokens, repr_layers=[33])
    torch.testing.assert_close(observed, torch.full((1, 4), 2.0))


def test_kaggle_runner_accepts_current_v4_reliability_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "kaggle" / "kaggle_runner.py"
    spec = importlib.util.spec_from_file_location("kaggle_runner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.PROTOCOL == "fixed_nested_group_cv_v4_reliability_residual"
    # CHANGELOG 2026-09 (novelty): the reliability family grew to two members;
    # this gate must track the same tuple as src/common.
    assert module.REQUIRED_ARCHITECTURES == {
        "concatenation",
        "gated_fusion",
        "cross_attention",
        "reliability_residual",
        "evidential_residual",
    }
    required = module.EXPECTED_OUTPUTS["stage14"]
    assert any(path.name == "best_reliability_residual_params.json" for path in required)


def test_kaggle_runner_binds_exact_stage09_external_contract(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "kaggle" / "kaggle_runner.py"
    spec = importlib.util.spec_from_file_location("kaggle_runner_contract_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "stage": "09_prepare_external_esm_dataset",
                "external_contract": {
                    "require_clinvar": True,
                    "require_dms": True,
                    "strict_clinvar_post_cutoff_last_evaluated": True,
                    "require_clinvar_scv_evidence": True,
                    "clinvar_scv_min_matching": 1,
                    "clinvar_scv_min_unique_submitters": 1,
                    "clinvar_scv_multiple_submitter_minimum": 2,
                    "dms_sampling_policy": "all",
                    "dms_max_rows_per_assay": 500,
                    "allow_sequence_only_dms": True,
                },
            }
        ),
        encoding="utf-8",
    )

    assert module._external_contract_environment(manifest_path) == {
        "REQUIRE_EXTERNAL_CLINVAR": "1",
        "REQUIRE_EXTERNAL_DMS": "1",
        "CLINVAR_REQUIRE_POST_CUTOFF_EVALUATION": "1",
        "CLINVAR_REQUIRE_SCV_EVIDENCE": "1",
        "CLINVAR_SCV_MIN_MATCHING": "1",
        "CLINVAR_SCV_MIN_UNIQUE_SUBMITTERS": "1",
        "CLINVAR_SCV_MULTIPLE_SUBMITTER_MINIMUM": "2",
        "DMS_SAMPLING_POLICY": "all",
        "DMS_MAX_ROWS_PER_ASSAY": "500",
        "ALLOW_SEQUENCE_ONLY_DMS": "1",
    }


def test_kaggle_runner_passes_stage09_contract_to_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "kaggle" / "kaggle_runner.py"
    spec = importlib.util.spec_from_file_location("kaggle_runner_process_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    captured: dict[str, object] = {}

    def fake_run(command, *, cwd, env, check):  # type: ignore[no-untyped-def]
        captured.update(command=command, cwd=cwd, env=env, check=check)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        module,
        "_external_contract_environment",
        lambda: {"DMS_SAMPLING_POLICY": "all"},
    )
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    arguments = SimpleNamespace(
        command="stage10",
        dataset="clinvar",
        scoring_mode="masked-marginal",
    )

    assert module.run_stage(arguments) == 0
    environment = captured["env"]
    assert isinstance(environment, dict)
    assert environment["DMS_SAMPLING_POLICY"] == "all"
