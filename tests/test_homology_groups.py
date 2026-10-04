from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import pytest


stage08b = importlib.import_module("08b_build_homology_groups")


def test_connected_groups_join_genes_and_homology_clusters() -> None:
    rows = pd.DataFrame(
        {
            "sequence_hash": ["s1", "s2", "s3", "s4"],
            "genename": ["G1", "G2", "G2", "G3"],
        }
    )
    clusters = {"s1": "c1", "s2": "c1", "s3": "c2", "s4": "c3"}
    mapping = stage08b._connected_split_groups(rows, clusters)
    groups = dict(zip(mapping["sequence_hash"], mapping["split_group"]))
    assert groups["s1"] == groups["s2"] == groups["s3"]
    assert groups["s4"] != groups["s1"]


def test_cluster_reader_rejects_unreported_sequences(tmp_path: Path) -> None:
    cluster_file = tmp_path / "cluster.tsv"
    cluster_file.write_text("s1\ts1\ns1\ts2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="omits 1 current sequences"):
        stage08b._read_clusters(cluster_file, {"s1", "s2", "s3"})


def test_cluster_reader_requires_exact_current_member_universe(tmp_path: Path) -> None:
    cluster_file = tmp_path / "cluster.tsv"
    cluster_file.write_text("s1\ts1\ns1\ts2\ns3\ts3\n", encoding="utf-8")
    clusters = stage08b._read_clusters(cluster_file, {"s1", "s2", "s3"})
    assert clusters == {"s1": "s1", "s2": "s1", "s3": "s3"}

    cluster_file.write_text("s1\ts1\nforeign\tforeign\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside the current sequence universe"):
        stage08b._read_clusters(cluster_file, {"s1"})
