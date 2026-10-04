from __future__ import annotations

import importlib
from pathlib import Path

import pytest


stage04 = importlib.import_module("04_feature_engineering")


def _atom_line(
    serial: int,
    residue: str,
    position: int,
    x: float,
    bfactor: float,
) -> str:
    return (
        f"ATOM  {serial:5d}  CA  {residue:>3s} A{position:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{bfactor:6.2f}"
        "           C  \n"
    )


@pytest.mark.skipif(
    not stage04.BIOPYTHON_AVAILABLE, reason="Biopython is required for PDB parsing"
)
def test_local_structure_environment_is_rotation_invariant_descriptor(
    tmp_path: Path,
) -> None:
    structure = tmp_path / "AF-PTEST-F1-model_v4.pdb"
    structure.write_text(
        "".join(
            [
                _atom_line(1, "ALA", 1, 0.0, 95.0),
                _atom_line(2, "VAL", 2, 4.0, 90.0),
                _atom_line(3, "ASP", 20, 7.0, 80.0),
                _atom_line(4, "LYS", 30, 11.0, 60.0),
                "TER\nEND\n",
            ]
        ),
        encoding="ascii",
    )
    fragment = stage04.StructureFragment(structure, fragment=1, offset=0)

    accession, features, status, message = stage04._read_structure_features(
        "PTEST", fragment, {1}
    )

    assert accession == "PTEST"
    assert status == "ok", message
    environment = features[1]
    assert environment["local_contact_count_8a"] == 2.0
    assert environment["local_contact_count_12a"] == 3.0
    assert environment["local_long_range_contact_count_8a"] == 1.0
    assert environment["local_mean_plddt_8a"] == pytest.approx(85.0)
    assert environment["local_min_plddt_8a"] == pytest.approx(80.0)
    assert environment["local_confident_contact_fraction_8a"] == 1.0
    assert environment["local_mean_distance_8a"] == pytest.approx(5.5)
    assert environment["local_hydrophobic_fraction_8a"] == pytest.approx(0.5)
    assert environment["local_charged_fraction_8a"] == pytest.approx(0.5)


@pytest.mark.skipif(
    not stage04.BIOPYTHON_AVAILABLE, reason="Biopython is required for PDB parsing"
)
def test_structure_reference_mismatch_is_rejected_before_sasa(tmp_path, monkeypatch) -> None:
    structure = tmp_path / "AF-PTEST-F1-model_v4.pdb"
    structure.write_text(_atom_line(1, "ALA", 1, 0.0, 95.0) + "TER\nEND\n", encoding="ascii")

    def unexpected_sasa():
        pytest.fail("An incompatible structure must be rejected before SASA computation")

    monkeypatch.setattr(stage04, "ShrakeRupley", unexpected_sasa)
    fragment = stage04.StructureFragment(structure, fragment=1, offset=0)
    _, features, status, _ = stage04._read_structure_features("PTEST", fragment, {1}, "V")
    assert not features
    assert status == "reference_mismatch"
