from __future__ import annotations

import argparse
import heapq
import hashlib
import logging
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from config import (
    ESM_EMBED_DIM,
    ESM_DEVICE,
    ESM_INTERNAL_MAX_ROWS,
    ESM_LAYER,
    ESM_MAX_BATCH_ATTENTION,
    ESM_MAX_BATCH_PROTEINS,
    ESM_MAX_BATCH_TOKENS,
    ESM_CACHE_COMPRESS,
    ESM_MASK_BATCH_ATTENTION,
    ESM_MAX_MASKS_PER_BATCH,
    ESM_MODEL_NAME,
    ESM_SCORING_MODE,
    ESM_SAMPLE_MIN_PER_CLASS,
    ESM_USE_FP16,
    ESM_WINDOW_SIZE,
    ENABLE_LORA,
    HOMOLOGY_CLUSTER_MODE,
    HOMOLOGY_COVERAGE_MODE,
    HOMOLOGY_MIN_COVERAGE,
    HOMOLOGY_MIN_SEQUENCE_IDENTITY,
    REQUIRE_HOMOLOGY_GROUPS,
    STAGE08_OUT,
    STAGE09_OUT,
    STAGE10_OUT,
    ensure_directories,
    validate_upstream_manifest,
    write_run_manifest,
)
from schema import GENE_COL, LABEL_COL, ROW_ID_COL, STANDARD_AA
from gpu_runtime import cuda_inventory, data_parallel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stage10_esm")

if ESM_DEVICE == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("ESM_DEVICE=cuda was requested but CUDA is unavailable")
DEVICE = (
    "cuda"
    if ESM_DEVICE == "cuda" or (ESM_DEVICE == "auto" and torch.cuda.is_available())
    else "cpu"
)


def _hardware_summary() -> dict[str, Any]:
    """Record the exact validated extraction devices."""
    summary: dict[str, Any] = {
        "requested_device": ESM_DEVICE,
        "resolved_device": DEVICE,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if DEVICE == "cuda":
        inventory = cuda_inventory(DEVICE)
        devices = inventory["devices"]
        primary = devices[0]
        summary.update(
            {
                **inventory,
                # Retain the original primary-device fields for manifest
                # compatibility while recording every selected GPU above.
                "gpu_name": primary["name"],
                "gpu_total_bytes": primary["total_bytes"],
                "gpu_free_bytes_before_load": primary["free_bytes"],
                "cuda_runtime": torch.version.cuda,
            }
        )
    return summary


@dataclass
class ExtractionTask:
    name: str
    input_table: Path
    output_table: Path
    embedding_file: Path
    status_file: Path
    manifest_file: Path
    cache_dir: Path
    sequence_table: Path | None = None


@dataclass
class ContextItem:
    sequence: str
    start: int
    end: int
    variants: list[tuple[int, int, str, str]]

    @property
    def context_hash(self) -> str:
        payload = (
            f"{self.sequence}|{self.start}|{self.end}|{ESM_MODEL_NAME}|{ESM_LAYER}|"
            f"fp16={int(DEVICE == 'cuda' and ESM_USE_FP16)}"
        )
        return hashlib.sha256(payload.encode()).hexdigest()


def _task_definitions() -> dict[str, ExtractionTask]:
    homology_ready = STAGE08_OUT / "internal_esm_ready_homology.parquet"
    return {
        "internal": ExtractionTask(
            "internal",
            homology_ready,
            STAGE10_OUT / "internal_with_esm.parquet",
            STAGE10_OUT / "internal_esm_embeddings.npy",
            STAGE10_OUT / "internal_esm_extraction.parquet",
            STAGE10_OUT / "internal_esm_manifest.json",
            STAGE10_OUT / "cache" / "internal",
            STAGE08_OUT / "internal_sequences.parquet",
        ),
        "clinvar": ExtractionTask(
            "clinvar",
            STAGE09_OUT / "clinvar_esm_ready.csv",
            STAGE10_OUT / "clinvar_with_esm.parquet",
            STAGE10_OUT / "clinvar_esm_embeddings.npy",
            STAGE10_OUT / "clinvar_esm_extraction.parquet",
            STAGE10_OUT / "clinvar_esm_manifest.json",
            STAGE10_OUT / "cache" / "clinvar",
        ),
        "dms": ExtractionTask(
            "dms",
            STAGE09_OUT / "dms_esm_ready.csv",
            STAGE10_OUT / "dms_with_esm.parquet",
            STAGE10_OUT / "dms_esm_embeddings.npy",
            STAGE10_OUT / "dms_esm_extraction.parquet",
            STAGE10_OUT / "dms_esm_manifest.json",
            STAGE10_OUT / "cache" / "dms",
            STAGE09_OUT / "dms_sequences.parquet",
        ),
    }


def _task_upstream_manifests(task: ExtractionTask) -> list[Path]:
    """Resolve manifests beside the exact task inputs, not import-time roots."""
    if task.name == "internal":
        sequence_parent = (
            task.sequence_table.parent
            if task.sequence_table is not None
            else task.input_table.parent
        )
        return [
            task.input_table.parent / "homology_run_manifest.json",
            sequence_parent / "run_manifest.json",
        ]
    return [task.input_table.parent / "run_manifest.json"]


def _validate_task_upstream(task: ExtractionTask) -> None:
    """Authenticate the exact prepared table before loading the ESM checkpoint."""
    if task.name == "internal":
        if not REQUIRE_HOMOLOGY_GROUPS:
            raise RuntimeError(
                "Publication Stage 10 requires Stage 08b homology groups; "
                "REQUIRE_HOMOLOGY_GROUPS cannot be disabled"
            )
        homology_manifest_path, stage08_manifest_path = _task_upstream_manifests(task)
        homology_manifest = validate_upstream_manifest(
            homology_manifest_path,
            "08b_build_homology_groups",
            [task.input_table],
        )
        validate_upstream_manifest(
            stage08_manifest_path,
            "08_prepare_esm_dataset",
            [task.sequence_table] if task.sequence_table is not None else [],
        )
        homology_contract = homology_manifest.get("extra", {})
        expected = {
            "min_sequence_identity": HOMOLOGY_MIN_SEQUENCE_IDENTITY,
            "minimum_coverage": HOMOLOGY_MIN_COVERAGE,
            "coverage_mode": HOMOLOGY_COVERAGE_MODE,
            "cluster_mode": HOMOLOGY_CLUSTER_MODE,
            "member_universe_validation": "exact",
        }
        mismatches = {
            key: {"expected": value, "observed": homology_contract.get(key)}
            for key, value in expected.items()
            if homology_contract.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"Stage 08b homology policy differs from Stage 10: {mismatches}"
            )
        columns = set(pq.ParquetFile(task.input_table).schema_arrow.names)
        missing = {"split_group", "homology_cluster"} - columns
        if missing:
            raise KeyError(
                f"Validated Stage 08b table misses homology fields {sorted(missing)}"
            )
    else:
        (stage09_manifest_path,) = _task_upstream_manifests(task)
        validate_upstream_manifest(
            stage09_manifest_path,
            "09_prepare_external_esm_dataset",
            [
                task.input_table,
                *([task.sequence_table] if task.sequence_table is not None else []),
            ],
        )


def _context_items(df: pd.DataFrame, statuses: pd.DataFrame) -> list[ContextItem]:
    grouped: dict[tuple[str, int, int], list[tuple[int, int, str, str]]] = {}
    standard = set(STANDARD_AA)
    extraction_status = ["pending"] * len(df)
    error_type = [""] * len(df)
    window_start = np.full(len(df), np.nan, dtype=np.float32)
    window_end = np.full(len(df), np.nan, dtype=np.float32)
    input_length = np.full(len(df), np.nan, dtype=np.float32)
    mutation_windows = [""] * len(df)
    local_positions = np.full(len(df), -1, dtype=np.int32)
    for row in df.itertuples():
        row_index = int(row.Index)
        full_sequence = str(row.protein_sequence)
        position = int(row.aa_pos)
        reference = str(row.aa_ref).upper()
        alternate = str(row.aa_alt).upper()
        if reference not in standard or alternate not in standard or reference == alternate:
            extraction_status[row_index] = "invalid_substitution"
            error_type[row_index] = "validation"
            continue
        if len(full_sequence) <= ESM_WINDOW_SIZE:
            sequence = full_sequence
            start = 1
            end = len(full_sequence)
            local_position = position
        else:
            start_zero = max(0, position - 1 - ESM_WINDOW_SIZE // 2)
            end_zero = min(len(full_sequence), start_zero + ESM_WINDOW_SIZE)
            if end_zero == len(full_sequence):
                start_zero = max(0, end_zero - ESM_WINDOW_SIZE)
            sequence = full_sequence[start_zero:end_zero]
            start = start_zero + 1
            end = end_zero
            local_position = position - start_zero
        if len(sequence) > ESM_WINDOW_SIZE:
            extraction_status[row_index] = "window_too_long"
            error_type[row_index] = "validation"
            continue
        if not 1 <= local_position <= len(sequence):
            extraction_status[row_index] = "position_out_of_bounds"
            error_type[row_index] = "validation"
            continue
        if sequence[local_position - 1] != reference:
            extraction_status[row_index] = "reference_mismatch"
            error_type[row_index] = "validation"
            continue
        key = (sequence, start, end)
        grouped.setdefault(key, []).append(
            (row_index, local_position, reference, alternate)
        )
        window_start[row_index] = start
        window_end[row_index] = end
        input_length[row_index] = len(sequence)
        mutation_windows[row_index] = sequence
        local_positions[row_index] = local_position
    statuses["extraction_status"] = extraction_status
    statuses["error_type"] = error_type
    statuses["window_start"] = window_start
    statuses["window_end"] = window_end
    statuses["input_length"] = input_length
    df["mutation_window"] = mutation_windows
    df["window_aa_pos"] = local_positions
    df["window_start"] = window_start
    df["window_end"] = window_end
    return [
        ContextItem(sequence, start, end, variants)
        for (sequence, start, end), variants in grouped.items()
    ]


def _batch_cost(items: list[ContextItem]) -> tuple[int, int]:
    if not items:
        return 0, 0
    padded = max(len(item.sequence) + 2 for item in items)
    return padded * len(items), padded * padded * len(items)


def _make_batches(items: list[ContextItem]) -> list[list[ContextItem]]:
    ordered = sorted(items, key=lambda item: len(item.sequence))
    batches: list[list[ContextItem]] = []
    current: list[ContextItem] = []
    for item in ordered:
        proposed = [*current, item]
        tokens, attention = _batch_cost(proposed)
        exceeds = (
            len(proposed) > ESM_MAX_BATCH_PROTEINS
            or tokens > ESM_MAX_BATCH_TOKENS
            or attention > ESM_MAX_BATCH_ATTENTION
        )
        if current and exceeds:
            batches.append(current)
            current = [item]
        else:
            current = proposed
    if current:
        batches.append(current)
    return batches


def _cache_path(task: ExtractionTask, item: ContextItem, scoring_mode: str) -> Path:
    # CHANGELOG 2026-09 (performance / correctness of caching): the cache key
    # used to be derived from the short-lived row indices (``row:position:
    # reference:alternate``).  Those indices are assigned by whatever input
    # frame reached Stage 10, so any upstream re-sampling or re-numbering made
    # every identical residue re-extract.  The key now depends only on the
    # residue content (position, reference, alternate), which is stable across
    # differently-numbered inputs, so an existing cache is reused whenever the
    # protein context, residue, scoring mode and model are unchanged.
    variant_signature = "|".join(
        f"{position}:{reference}:{alternate}"
        for _, position, reference, alternate in item.variants
    )
    key = hashlib.sha256(
        f"{item.context_hash}|{variant_signature}|{scoring_mode}".encode()
    ).hexdigest()
    return task.cache_dir / f"{key}.npz"


def _legacy_cache_path(
    task: ExtractionTask, item: ContextItem, scoring_mode: str
) -> Path:
    """Path of the pre-2026-09 row-index-based cache, if an old cache exists."""
    variant_signature = "|".join(
        f"{row}:{position}:{reference}:{alternate}"
        for row, position, reference, alternate in item.variants
    )
    key = hashlib.sha256(
        f"{item.context_hash}|{variant_signature}|{scoring_mode}".encode()
    ).hexdigest()
    return task.cache_dir / f"{key}.npz"


def _load_cache(
    task: ExtractionTask,
    item: ContextItem,
    scoring_mode: str,
    embeddings: np.ndarray,
    wt_scores: np.ndarray,
    masked_scores: np.ndarray,
    statuses: pd.DataFrame,
) -> bool:
    # CHANGELOG 2026-09 (performance): ``_cache_path`` is index-independent now,
    # but caches written by earlier runs still live under the legacy
    # row-based name.  Resolve the new path first; on a miss look up the legacy
    # path, validate it (it stores the same payload), reuse it and re-save it
    # under the new name so only one migration pass ever happens per key.  The
    # legacy file is removed only after the re-save succeeds, so a crash mid-way
    # leaves the original cache intact and the next run retries, idempotently.
    path = _cache_path(task, item, scoring_mode)
    if not path.exists():
        legacy_path = _legacy_cache_path(task, item, scoring_mode)
        if not legacy_path.exists():
            return False
        if not _load_cache_from(
            legacy_path, item, scoring_mode, embeddings, wt_scores,
            masked_scores, statuses,
        ):
            return False
        try:
            _save_cache(
                task, item, scoring_mode, embeddings, wt_scores, masked_scores
            )
        except (OSError, ValueError) as error:
            logger.warning(
                "Legacy cache reuse fell back to per-item save for %s",
                item.context_hash[:16],
            )
            logger.debug("Legacy re-save failed: %s", error)
        else:
            # A failed or skipped replacement must never delete the usable
            # legacy entry. Publication artifacts are independent of caches.
            if path.is_file():
                try:
                    legacy_path.unlink()
                except OSError:
                    pass
        return True
    return _load_cache_from(
        path, item, scoring_mode, embeddings, wt_scores, masked_scores, statuses
    )


def _load_cache_from(
    path: Path,
    item: ContextItem,
    scoring_mode: str,
    embeddings: np.ndarray,
    wt_scores: np.ndarray,
    masked_scores: np.ndarray,
    statuses: pd.DataFrame,
) -> bool:
    try:
        with np.load(path, allow_pickle=False) as stored:
            if (
                str(stored["model"].item()) != ESM_MODEL_NAME
                or int(stored["layer"].item()) != ESM_LAYER
                or str(stored["scoring_mode"].item()) != scoring_mode
                or bool(stored["fp16"].item())
                != bool(DEVICE == "cuda" and ESM_USE_FP16)
            ):
                return False
            rows = stored["rows"].astype(int)
            expected = np.asarray([variant[0] for variant in item.variants], dtype=int)
            if "variants" in stored and "context_hash" in stored:
                variants = np.asarray(
                    [variant[1:] for variant in item.variants], dtype=str
                )
                if (
                    str(stored["context_hash"].item()) != item.context_hash
                    or not np.array_equal(stored["variants"], variants)
                ):
                    return False
            elif not np.array_equal(rows, expected):
                # Old payloads cannot authenticate a mapping to new row IDs.
                return False
            cached_embeddings = stored["embeddings"]
            cached_wt = stored["wt_scores"]
            cached_masked = stored["masked_scores"]
            if (
                cached_embeddings.shape != (len(expected), ESM_EMBED_DIM)
                or cached_wt.shape != (len(expected),)
                or cached_masked.shape != (len(expected),)
                or not np.isfinite(cached_embeddings).all()
                or (
                    scoring_mode in {"wt-marginal", "both"}
                    and not np.isfinite(cached_wt).all()
                )
                or (
                    scoring_mode in {"masked-marginal", "both"}
                    and not np.isfinite(cached_masked).all()
                )
            ):
                return False
            embeddings[expected] = cached_embeddings
            wt_scores[expected] = cached_wt
            masked_scores[expected] = cached_masked
            statuses.loc[expected, "extraction_status"] = "cached"
            statuses.loc[expected, "error_type"] = ""
            return True
    except (OSError, ValueError, KeyError, EOFError, TypeError, BadZipFile) as error:
        logger.warning("Ignoring cache %s after %s", path.name, type(error).__name__)
        return False


def _save_cache(
    task: ExtractionTask,
    item: ContextItem,
    scoring_mode: str,
    embeddings: np.ndarray,
    wt_scores: np.ndarray,
    masked_scores: np.ndarray,
) -> None:
    path = _cache_path(task, item, scoring_mode)
    rows = np.asarray([variant[0] for variant in item.variants], dtype=int)
    if not np.isfinite(embeddings[rows]).all():
        return
    if scoring_mode in {"wt-marginal", "both"} and not np.isfinite(wt_scores[rows]).all():
        return
    if scoring_mode in {"masked-marginal", "both"} and not np.isfinite(masked_scores[rows]).all():
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        save = np.savez_compressed if ESM_CACHE_COMPRESS else np.savez
        save(
            handle,
            rows=rows,
            variants=np.asarray([variant[1:] for variant in item.variants], dtype=str),
            context_hash=np.asarray(item.context_hash),
            embeddings=embeddings[rows],
            wt_scores=wt_scores[rows],
            masked_scores=masked_scores[rows],
            model=np.asarray(ESM_MODEL_NAME),
            layer=np.asarray(ESM_LAYER),
            scoring_mode=np.asarray(scoring_mode),
            fp16=np.asarray(DEVICE == "cuda" and ESM_USE_FP16),
        )
    temporary.replace(path)


def _forward_batch(
    items: list[ContextItem],
    model: Any,
    batch_converter: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = [(item.context_hash[:16], item.sequence) for item in items]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(DEVICE)
    with torch.inference_mode(), torch.amp.autocast(
        "cuda", enabled=DEVICE == "cuda" and ESM_USE_FP16
    ):
        output = model(tokens, repr_layers=[ESM_LAYER])
    representations = (
        output["representations"][ESM_LAYER].detach().float().cpu()
    )
    log_probabilities = (
        torch.log_softmax(output["logits"].float(), dim=-1).detach().cpu()
    )
    del output, tokens
    return representations, log_probabilities


def _mask_batch_capacity(token_length: int) -> int:
    """Bound masked rows by row, token and quadratic attention budgets."""
    masks_hard = (
        ESM_MAX_MASKS_PER_BATCH
        if ESM_MAX_MASKS_PER_BATCH > 0
        else ESM_MAX_BATCH_PROTEINS
    )
    if token_length < 1:
        raise ValueError("Masked token length must be positive")
    token_budget = ESM_MAX_BATCH_TOKENS // token_length
    attention_limit = (
        ESM_MASK_BATCH_ATTENTION
        if ESM_MASK_BATCH_ATTENTION > 0
        else ESM_MAX_BATCH_ATTENTION
    )
    attention_budget = attention_limit // (token_length * token_length)
    if min(token_budget, attention_budget) < 1:
        raise ValueError("A masked context exceeds the configured token/attention budget")
    return min(masks_hard, token_budget, attention_budget)


def _masked_log_probabilities(
    item: ContextItem,
    positions: list[int],
    model: Any,
    batch_converter: Any,
    mask_index: int,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    _, _, base_tokens = batch_converter([(item.context_hash[:16], item.sequence)])
    base_tokens = base_tokens.to(DEVICE)
    results: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    token_length = int(base_tokens.shape[1])
    masks_per_batch = _mask_batch_capacity(token_length)
    start = 0
    while start < len(positions):
        attempt = masks_per_batch
        while True:
            position_batch = positions[start : start + attempt]
            masked = base_tokens.expand(len(position_batch), -1).clone()
            row_indices = torch.arange(len(position_batch), device=masked.device)
            masked[
                row_indices,
                torch.as_tensor(position_batch, device=masked.device),
            ] = mask_index
            try:
                with torch.inference_mode(), torch.amp.autocast(
                    "cuda", enabled=DEVICE == "cuda" and ESM_USE_FP16
                ):
                    output = model(masked, repr_layers=[ESM_LAYER])
            except RuntimeError as error:
                # CHANGELOG 2026-09 (performance / robustness): on an OOM the
                # batch is halved and retried from the SAME start, so a single
                # long window that would not fit together still gets scored mask
                # by mask instead of aborting the whole item.  The retry only
                # ever shrinks the batch, so with the default budget of 1 mask
                # this never triggers.
                if "out of memory" in str(error).lower() and attempt > 1:
                    error.__traceback__ = None
                    del masked, row_indices
                    if DEVICE == "cuda":
                        try:
                            torch.cuda.empty_cache()
                        except RuntimeError:
                            logger.warning(
                                "CUDA cache cleanup failed after %s",
                                type(error).__name__,
                            )
                    attempt = max(1, attempt // 2)
                    masks_per_batch = attempt
                    continue
                raise
            for batch_index, position in enumerate(position_batch):
                results[position] = (
                    torch.log_softmax(
                        output["logits"][batch_index, position].float(), dim=-1
                    )
                    .detach()
                    .cpu()
                    .clone(),
                    output["representations"][ESM_LAYER][batch_index, position]
                    .detach()
                    .float()
                    .cpu()
                    .clone(),
                )
            del output, masked
            start += attempt
            break
    del base_tokens
    return results


def _masked_log_probabilities_for_items(
    items: list[ContextItem],
    model: Any,
    batch_converter: Any,
    mask_index: int,
) -> dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]]:
    """Batch masked positions across contexts so both T4s receive work."""
    if len(items) == 1:
        item = items[0]
        positions = sorted({variant[1] for variant in item.variants})
        return {
            item.context_hash: _masked_log_probabilities(
                item, positions, model, batch_converter, mask_index
            )
        }

    jobs = [
        (item, position)
        for item in items
        for position in sorted({variant[1] for variant in item.variants})
    ]
    results: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {
        item.context_hash: {} for item in items
    }
    batches: list[list[tuple[ContextItem, int]]] = []
    current: list[tuple[ContextItem, int]] = []
    # CHANGELOG 2026-09 (performance): mirror _mask_batch_capacity's fallback
    # so this multi-context batch packing honours the mask-specific budget when
    # set, and reproduces the pre-existing ESM_MAX_BATCH_* arithmetic when both
    # values are 0 (the default is therefore a no-op).
    masks_limit = (
        ESM_MAX_MASKS_PER_BATCH
        if ESM_MAX_MASKS_PER_BATCH > 0
        else ESM_MAX_BATCH_PROTEINS
    )
    attention_limit = (
        ESM_MASK_BATCH_ATTENTION
        if ESM_MASK_BATCH_ATTENTION > 0
        else ESM_MAX_BATCH_ATTENTION
    )
    for job in jobs:
        _mask_batch_capacity(len(job[0].sequence) + 2)
        proposed = [*current, job]
        padded = max(len(candidate.sequence) + 2 for candidate, _ in proposed)
        if current and (
            len(proposed) > masks_limit
            or padded * len(proposed) > ESM_MAX_BATCH_TOKENS
            or padded * padded * len(proposed) > attention_limit
        ):
            batches.append(current)
            current = [job]
        else:
            current = proposed
    if current:
        batches.append(current)

    for batch in batches:
        data = [
            (f"{item.context_hash[:12]}-{position}", item.sequence)
            for item, position in batch
        ]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(DEVICE)
        row_indices = torch.arange(len(batch), device=tokens.device)
        positions = torch.as_tensor(
            [position for _, position in batch],
            dtype=torch.long,
            device=tokens.device,
        )
        tokens[row_indices, positions] = mask_index
        with torch.inference_mode(), torch.amp.autocast(
            "cuda", enabled=DEVICE == "cuda" and ESM_USE_FP16
        ):
            output = model(tokens, repr_layers=[ESM_LAYER])
        for batch_index, (item, position) in enumerate(batch):
            results[item.context_hash][position] = (
                torch.log_softmax(
                    output["logits"][batch_index, position].float(), dim=-1
                )
                .detach()
                .cpu()
                .clone(),
                output["representations"][ESM_LAYER][batch_index, position]
                .detach()
                .float()
                .cpu()
                .clone(),
            )
        del output, tokens
    return results


def _fill_item(
    item: ContextItem,
    representation: torch.Tensor | None,
    wt_log_probability: torch.Tensor | None,
    model: Any,
    alphabet: Any,
    batch_converter: Any,
    mask_index: int,
    scoring_mode: str,
    embeddings: np.ndarray,
    wt_scores: np.ndarray,
    masked_scores: np.ndarray,
    statuses: pd.DataFrame,
    masked_by_position: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> None:
    needs_masked = scoring_mode in {"masked-marginal", "both"}
    if needs_masked and masked_by_position is None:
        positions = sorted({variant[1] for variant in item.variants})
        masked_by_position = _masked_log_probabilities(
            item, positions, model, batch_converter, mask_index
        )
    masked_by_position = masked_by_position or {}
    statuses_local = statuses  # local binding; rows written below in one pass
    success_rows: list[int] = []
    failure_rows: list[int] = []
    failure_types: list[str] = []
    for row, position, reference, alternate in item.variants:
        try:
            reference_index = alphabet.get_idx(reference)
            alternate_index = alphabet.get_idx(alternate)
            if needs_masked:
                embedding_value = masked_by_position[position][1]
            elif representation is not None:
                embedding_value = representation[position]
            else:
                raise RuntimeError("Unmasked representation was not computed")
            embeddings[row] = embedding_value.detach().float().cpu().numpy()
            if wt_log_probability is not None:
                wt_values = wt_log_probability[position]
                wt_scores[row] = float(
                    wt_values[alternate_index] - wt_values[reference_index]
                )
            if needs_masked:
                masked_values = masked_by_position[position][0]
                masked_scores[row] = float(
                    masked_values[alternate_index] - masked_values[reference_index]
                )
            success_rows.append(row)
        except (IndexError, KeyError, ValueError, RuntimeError) as error:
            failure_rows.append(row)
            failure_types.append(type(error).__name__)
    # CHANGELOG 2026-09 (performance): write the statuses in two vectorised
    # assignments for the whole item instead of one .loc per variant, which is
    # far cheaper on a DataFrame with many variants.
    if success_rows:
        statuses_local.loc[
            success_rows, ["extraction_status", "error_type"]
        ] = ["success", ""]
    if failure_rows:
        statuses_local.loc[failure_rows, "extraction_status"] = "variant_failed"
        statuses_local.loc[failure_rows, "error_type"] = failure_types


def _process_batch(
    items: list[ContextItem],
    model: Any,
    alphabet: Any,
    batch_converter: Any,
    mask_index: int,
    scoring_mode: str,
    embeddings: np.ndarray,
    wt_scores: np.ndarray,
    masked_scores: np.ndarray,
    statuses: pd.DataFrame,
) -> None:
    try:
        masked_by_item = (
            _masked_log_probabilities_for_items(
                items, model, batch_converter, mask_index
            )
            if scoring_mode in {"masked-marginal", "both"}
            else {}
        )
        # The masked-marginal embedding and score both come from the masked
        # pass. Running an unmasked pass as well would double the publication
        # default's compute solely for an unrelated auxiliary score.
        if scoring_mode == "masked-marginal":
            for item in items:
                _fill_item(
                    item,
                    None,
                    None,
                    model,
                    alphabet,
                    batch_converter,
                    mask_index,
                    scoring_mode,
                    embeddings,
                    wt_scores,
                    masked_scores,
                    statuses,
                    masked_by_item[item.context_hash],
                )
            return
        representations, log_probabilities = _forward_batch(
            items, model, batch_converter
        )
        for index, item in enumerate(items):
            _fill_item(
                item,
                representations[index],
                log_probabilities[index],
                model,
                alphabet,
                batch_converter,
                mask_index,
                scoring_mode,
                embeddings,
                wt_scores,
                masked_scores,
                statuses,
                masked_by_item.get(item.context_hash),
            )
        del representations, log_probabilities
    except RuntimeError as error:
        is_out_of_memory = "out of memory" in str(error).lower()
        # Release the failed forward's frames before retrying smaller batches.
        error.__traceback__ = None
        if DEVICE == "cuda":
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                logger.warning("CUDA cache cleanup failed after %s", type(error).__name__)
        if len(items) > 1:
            midpoint = len(items) // 2
            _process_batch(
                items[:midpoint],
                model,
                alphabet,
                batch_converter,
                mask_index,
                scoring_mode,
                embeddings,
                wt_scores,
                masked_scores,
                statuses,
            )
            _process_batch(
                items[midpoint:],
                model,
                alphabet,
                batch_converter,
                mask_index,
                scoring_mode,
                embeddings,
                wt_scores,
                masked_scores,
                statuses,
            )
            return
        status = "out_of_memory" if is_out_of_memory else "runtime_error"
        for item in items:
            rows = [variant[0] for variant in item.variants]
            statuses.loc[rows, ["extraction_status", "error_type"]] = [
                status,
                type(error).__name__,
            ]


def _write_array(path: Path, values: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.save(handle, values)


def _priority(row_id: str) -> int:
    return int.from_bytes(hashlib.sha256(row_id.encode()).digest()[:8], "big")


def _select_internal_rows(path: Path, max_rows: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Select a deterministic class-stratified, gene-diverse modelling subset."""
    parquet = pq.ParquetFile(path)
    total_rows = parquet.metadata.num_rows
    if max_rows <= 0 or total_rows <= max_rows:
        return np.arange(total_rows, dtype=np.int64), {
            "source_rows": total_rows,
            "selected_rows": total_rows,
            "sampling": "all_rows",
        }
    counts: Counter[int] = Counter()
    gene_representatives: dict[tuple[int, str], tuple[int, int]] = {}
    offset = 0
    for batch in parquet.iter_batches(
        batch_size=65_536, columns=[ROW_ID_COL, LABEL_COL, GENE_COL]
    ):
        metadata = batch.to_pandas()
        for local, row in enumerate(metadata.itertuples(index=False)):
            row_id = str(getattr(row, ROW_ID_COL))
            label = int(getattr(row, LABEL_COL))
            gene = str(getattr(row, GENE_COL))
            if label not in {0, 1}:
                raise ValueError(f"Internal sampling found invalid label {label}")
            index = offset + local
            priority = _priority(row_id)
            counts[label] += 1
            key = (label, gene)
            previous = gene_representatives.get(key)
            if previous is None or (priority, index) < previous:
                gene_representatives[key] = (priority, index)
        offset += len(metadata)
    if set(counts) != {0, 1}:
        raise ValueError(f"Internal data must contain both classes: {dict(counts)}")
    minimum = min(ESM_SAMPLE_MIN_PER_CLASS, max_rows // 4)
    positive_target = int(round(max_rows * counts[1] / total_rows))
    positive_target = max(positive_target, min(minimum, counts[1]))
    positive_target = min(positive_target, counts[1])
    benign_target = max_rows - positive_target
    if benign_target > counts[0]:
        benign_target = counts[0]
        positive_target = min(counts[1], max_rows - benign_target)
    targets = {0: benign_target, 1: positive_target}
    selected: set[int] = set()
    selected_by_class: Counter[int] = Counter()
    for label in (0, 1):
        representatives = sorted(
            value
            for (candidate_label, _), value in gene_representatives.items()
            if candidate_label == label
        )
        for _, index in representatives[: targets[label]]:
            selected.add(index)
            selected_by_class[label] += 1
    capacities = {
        label: targets[label] - selected_by_class[label] for label in (0, 1)
    }
    heaps: dict[int, list[tuple[int, int]]] = {0: [], 1: []}
    offset = 0
    for batch in parquet.iter_batches(
        batch_size=65_536, columns=[ROW_ID_COL, LABEL_COL]
    ):
        metadata = batch.to_pandas()
        for local, row in enumerate(metadata.itertuples(index=False)):
            index = offset + local
            if index in selected:
                continue
            label = int(getattr(row, LABEL_COL))
            capacity = capacities[label]
            if capacity <= 0:
                continue
            priority = _priority(str(getattr(row, ROW_ID_COL)))
            candidate = (-priority, -index)
            heap = heaps[label]
            if len(heap) < capacity:
                heapq.heappush(heap, candidate)
            elif candidate > heap[0]:
                heapq.heapreplace(heap, candidate)
        offset += len(metadata)
    for label, heap in heaps.items():
        selected.update(-index for _, index in heap)
        selected_by_class[label] += len(heap)
    indices = np.asarray(sorted(selected), dtype=np.int64)
    return indices, {
        "source_rows": total_rows,
        "selected_rows": len(indices),
        "sampling": "deterministic_gene_diverse_class_stratified",
        "source_class_counts": dict(counts),
        "selected_class_counts": dict(selected_by_class),
        "max_rows": max_rows,
    }


def _read_selected_parquet(path: Path, indices: np.ndarray) -> pd.DataFrame:
    parquet = pq.ParquetFile(path)
    tables: list[pa.Table] = []
    offset = 0
    left = 0
    for batch in parquet.iter_batches(batch_size=65_536):
        stop = offset + batch.num_rows
        right = int(np.searchsorted(indices, stop, side="left"))
        if right > left:
            local = indices[left:right] - offset
            tables.append(pa.Table.from_batches([batch]).take(pa.array(local)))
        left = right
        offset = stop
        if left == len(indices):
            break
    if not tables:
        raise RuntimeError("No selected Parquet rows were read")
    return pa.concat_tables(tables, promote_options="default").to_pandas().reset_index(
        drop=True
    )


def _read_task_frame(task: ExtractionTask) -> tuple[pd.DataFrame, dict[str, Any]]:
    if task.input_table.suffix.lower() == ".parquet":
        if task.name == "internal":
            indices, sampling = _select_internal_rows(
                task.input_table, ESM_INTERNAL_MAX_ROWS
            )
            frame = _read_selected_parquet(task.input_table, indices)
        else:
            frame = pd.read_parquet(task.input_table).reset_index(drop=True)
            sampling = {
                "source_rows": len(frame),
                "selected_rows": len(frame),
                "sampling": "all_rows",
            }
    else:
        frame = pd.read_csv(
            task.input_table,
            usecols=(
                (lambda column: column not in {"protein_sequence", "mutation_window"})
                if task.sequence_table is not None
                else None
            ),
            low_memory=False,
        ).reset_index(drop=True)
        sampling = {
            "source_rows": len(frame),
            "selected_rows": len(frame),
            "sampling": "all_rows",
        }
    if task.sequence_table is not None:
        if not task.sequence_table.exists():
            raise FileNotFoundError(task.sequence_table)
        sequences = pd.read_parquet(
            task.sequence_table, columns=["sequence_hash", "protein_sequence"]
        ).drop_duplicates("sequence_hash")
        sequence_map = dict(zip(sequences["sequence_hash"], sequences["protein_sequence"]))
        frame["protein_sequence"] = frame["sequence_hash"].map(sequence_map)
        if frame["protein_sequence"].isna().any():
            raise ValueError("Selected internal rows have missing sequence-table entries")
    return frame, sampling


def extract_task(
    task: ExtractionTask,
    model: Any,
    alphabet: Any,
    scoring_mode: str,
    *,
    validate_upstream: bool = True,
) -> None:
    """Extract one aligned ESM dataset."""
    if not task.input_table.exists():
        raise FileNotFoundError(task.input_table)
    if validate_upstream:
        _validate_task_upstream(task)
    task.cache_dir.mkdir(parents=True, exist_ok=True)
    df, sampling = _read_task_frame(task)
    # CHANGELOG 2026-09 (correctness): ensure a clean 0..n-1 index before the
    # row-index-based embeddings / scores arrays are assigned.  A non-default
    # index from a prior upstream parquet write would otherwise leave the
    # __setitem__ positions misaligned for the row-major np arrays.
    df = df.reset_index(drop=True)
    required = {
        ROW_ID_COL,
        "protein_sequence",
        "aa_pos",
        "aa_ref",
        "aa_alt",
        "sequence_hash",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{task.name} input misses {sorted(missing)}")
    if task.name == "internal":
        missing_homology = {"split_group", "homology_cluster"} - set(df.columns)
        if missing_homology:
            raise KeyError(
                f"Internal ESM input misses {sorted(missing_homology)}"
            )
    if df[ROW_ID_COL].duplicated().any():
        raise ValueError(f"{task.name} contains duplicate row identifiers")
    row_count = len(df)
    embedding_temporary = task.embedding_file.with_suffix(
        task.embedding_file.suffix + ".tmp"
    )
    embedding_temporary.unlink(missing_ok=True)
    embeddings = np.lib.format.open_memmap(
        embedding_temporary,
        mode="w+",
        dtype=np.float32,
        shape=(row_count, ESM_EMBED_DIM),
    )
    embeddings[:] = np.nan
    wt_scores = np.full(row_count, np.nan, dtype=np.float32)
    masked_scores = np.full(row_count, np.nan, dtype=np.float32)
    statuses = pd.DataFrame(
        {
            ROW_ID_COL: df[ROW_ID_COL].astype(str),
            "sequence_hash": df["sequence_hash"].astype(str),
            "model_name": ESM_MODEL_NAME,
            "layer": ESM_LAYER,
            "scoring_mode": scoring_mode,
            "window_start": np.nan,
            "window_end": np.nan,
            "input_length": np.nan,
            "extraction_status": "pending",
            "error_type": "",
        }
    )
    items = _context_items(df, statuses)
    pending: list[ContextItem] = []
    for item in items:
        if not _load_cache(
            task,
            item,
            scoring_mode,
            embeddings,
            wt_scores,
            masked_scores,
            statuses,
        ):
            pending.append(item)
    batch_converter = alphabet.get_batch_converter()
    for batch_number, batch in enumerate(_make_batches(pending), 1):
        _process_batch(
            batch,
            model,
            alphabet,
            batch_converter,
            alphabet.mask_idx,
            scoring_mode,
            embeddings,
            wt_scores,
            masked_scores,
            statuses,
        )
        for item in batch:
            _save_cache(
                task,
                item,
                scoring_mode,
                embeddings,
                wt_scores,
                masked_scores,
            )
        if batch_number % 10 == 0:
            logger.info("%s processed %d batches", task.name, batch_number)
    if scoring_mode == "wt-marginal":
        primary_scores = wt_scores
    else:
        primary_scores = masked_scores
    success = np.isfinite(embeddings).all(axis=1) & np.isfinite(primary_scores)
    statuses["success"] = success.astype(np.int8)
    statuses.loc[
        statuses["extraction_status"].eq("pending"),
        ["extraction_status", "error_type"],
    ] = ["not_processed", "internal"]
    statuses.loc[
        ~success & statuses["extraction_status"].isin(["success", "cached"]),
        ["extraction_status", "error_type"],
    ] = ["score_failed", "nonfinite"]
    if not success.any():
        embeddings.flush()
        del embeddings
        embedding_temporary.unlink(missing_ok=True)
        raise RuntimeError(f"{task.name} produced no complete ESM features")
    failure_report = STAGE10_OUT / f"{task.name}_esm_failures.csv"
    if not success.all():
        failure_temporary = failure_report.with_suffix(
            failure_report.suffix + ".tmp"
        )
        statuses.loc[~success].to_csv(failure_temporary, index=False)
        failure_temporary.replace(failure_report)
        embeddings.flush()
        del embeddings
        embedding_temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"{task.name} has {int((~success).sum())} incomplete ESM rows; "
            f"inspect {failure_report} and rerun to resume cached successes"
        )
    failure_report.unlink(missing_ok=True)
    df["esm_variant_score"] = primary_scores
    if scoring_mode in {"wt-marginal", "both"}:
        df["esm_variant_score_wt_marginal"] = wt_scores
    if scoring_mode in {"masked-marginal", "both"}:
        df["esm_variant_score_masked_marginal"] = masked_scores
    df["ESM_EXTRACTION_SUCCESS"] = success.astype(np.int8)
    df["esm_variant_score__missing"] = (~np.isfinite(primary_scores)).astype(np.int8)
    task.output_table.parent.mkdir(parents=True, exist_ok=True)
    table_temporary = task.output_table.with_suffix(task.output_table.suffix + ".tmp")
    status_temporary = task.status_file.with_suffix(task.status_file.suffix + ".tmp")
    for path in (table_temporary, status_temporary):
        path.unlink(missing_ok=True)
    try:
        compact_drop = ["protein_sequence"]
        if task.name != "internal" or not ENABLE_LORA:
            compact_drop.append("mutation_window")
        modelling_frame = df.drop(columns=compact_drop, errors="ignore")
        modelling_frame.to_parquet(
            table_temporary, index=False, compression="zstd"
        )
        statuses.to_parquet(status_temporary, index=False, compression="zstd")
        embeddings.flush()
        del embeddings
        table_temporary.replace(task.output_table)
        status_temporary.replace(task.status_file)
        embedding_temporary.replace(task.embedding_file)
    except BaseException:
        for path in (table_temporary, status_temporary, embedding_temporary):
            path.unlink(missing_ok=True)
        raise
    row_order_digest = hashlib.sha256()
    for row_id in df[ROW_ID_COL].astype(str):
        row_order_digest.update(row_id.encode())
        row_order_digest.update(b"\n")
    write_run_manifest(
        task.manifest_file,
        f"10_extract_esm_features:{task.name}",
        [
            task.input_table,
            *([task.sequence_table] if task.sequence_table else []),
            *_task_upstream_manifests(task),
            *sorted(
                (
                    Path(
                        os.environ.get(
                            "TORCH_HOME", str(STAGE10_OUT / "torch_cache")
                        )
                    )
                    / "hub"
                    / "checkpoints"
                ).glob(f"{ESM_MODEL_NAME}*.pt")
            ),
        ],
        {
            "rows": row_count,
            "successful_rows": int(success.sum()),
            "failed_rows": int((~success).sum()),
            "coverage": float(success.mean()),
            "model_name": ESM_MODEL_NAME,
            "layer": ESM_LAYER,
            "embedding_dimension": ESM_EMBED_DIM,
            "scoring_mode": scoring_mode,
            "numeric_precision": (
                "fp16_autocast" if DEVICE == "cuda" and ESM_USE_FP16 else "fp32"
            ),
            "window_size": ESM_WINDOW_SIZE,
            "row_order_sha256": row_order_digest.hexdigest(),
            "status_file": str(task.status_file),
            "output_table": str(task.output_table),
            "embedding_storage": "npy_memmap",
            "sampling": sampling,
            "hardware": _hardware_summary(),
            "validated_upstream_stage": (
                ("08b_build_homology_groups" if task.name == "internal"
                 else "09_prepare_external_esm_dataset")
                if validate_upstream else "explicitly_skipped_nonpublication"
            ),
        },
        outputs=[task.output_table, task.embedding_file, task.status_file],
    )
    logger.info(
        "%s extraction coverage %.2f%%",
        task.name,
        100.0 * success.mean(),
    )


def _selected_tasks(selection: str) -> list[ExtractionTask]:
    tasks = _task_definitions()
    if selection == "internal":
        names = ["internal"]
    elif selection == "clinvar":
        names = ["clinvar"]
    elif selection == "dms":
        names = ["dms"]
    elif selection == "external":
        names = ["clinvar", "dms"]
    else:
        names = ["internal", "clinvar", "dms"]
    return [tasks[name] for name in names]


def _load_esm_model(esm_module: Any) -> tuple[Any, Any]:
    """Load the model without constructing a transient full-precision GPU copy."""
    use_fp16 = DEVICE == "cuda" and ESM_USE_FP16
    previous_dtype = torch.get_default_dtype()
    try:
        if use_fp16:
            logger.info("Constructing %s directly in FP16", ESM_MODEL_NAME)
            torch.set_default_dtype(torch.float16)
        model, alphabet = esm_module.pretrained.load_model_and_alphabet(
            ESM_MODEL_NAME
        )
    finally:
        torch.set_default_dtype(previous_dtype)
    model.eval()
    if use_fp16:
        model = model.half()
    model = model.to(DEVICE)
    model = data_parallel(model, DEVICE)
    # CHANGELOG 2026-09 (correctness): the masked-position scoring indexes
    # ``logits[batch_index, position]`` and ``representations[batch_index,
    # position]`` under the assumption that token ``position`` corresponds to
    # residue ``position``.  That alignment only holds when the tokenizer
    # prepends a leading boundary (BOS) token so the residue index equals the
    # token index.  Guard against any future ESM variant whose alphabet drops
    # BOS: masking ``tokens[row, position]`` would then mutate the wrong token
    # and silently score a different residue.
    if not getattr(alphabet, "prepend_bos", False):
        raise RuntimeError(
            f"{ESM_MODEL_NAME} alphabet does not prepend BOS; the masked-marginal "
            "position indexing in Stage 10 would be misaligned"
        )
    return model, alphabet


def main() -> None:
    """Extract frozen ESM representations."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["internal", "clinvar", "dms", "external", "all", "both"],
        default="all",
    )
    parser.add_argument(
        "--scoring-mode",
        choices=["wt-marginal", "masked-marginal", "both"],
        default=ESM_SCORING_MODE,
    )
    arguments = parser.parse_args()
    selection = "all" if arguments.dataset == "both" else arguments.dataset
    ensure_directories(STAGE10_OUT)
    os.environ.setdefault("TORCH_HOME", str(STAGE10_OUT / "torch_cache"))
    tasks = _selected_tasks(selection)
    for task in tasks:
        _validate_task_upstream(task)
    try:
        import esm
    except ImportError as error:
        raise RuntimeError("fair-esm is required for Stage 10") from error
    logger.info("Loading %s on %s", ESM_MODEL_NAME, DEVICE)
    hardware = _hardware_summary()
    if DEVICE == "cuda":
        for device in hardware["devices"]:
            logger.info(
                "Selected cuda:%d: %s (%.2f GiB free / %.2f GiB total)",
                device["index"],
                device["name"],
                device["free_bytes"] / 2**30,
                device["total_bytes"] / 2**30,
            )
        if any(
            device["free_bytes"] < 3 * 2**30
            for device in hardware["devices"]
        ):
            raise RuntimeError(
                "Less than 3 GiB memory is free on a selected GPU before ESM "
                "loading. Restart the Kaggle session; continuing risks a native "
                "CUDA crash that Python cannot recover from."
            )
    model, alphabet = _load_esm_model(esm)
    for task in tasks:
        extract_task(
            task,
            model,
            alphabet,
            arguments.scoring_mode,
            validate_upstream=True,
        )


if __name__ == "__main__":
    main()
