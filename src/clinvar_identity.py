from __future__ import annotations

import pandas as pd


def normalize_variation_ids(values: pd.Series) -> pd.Series:
    """Return positive canonical ClinVar VariationIDs.

    VariationID is a stable variant identity; AlleleID is deliberately not a
    substitute. Leading zeroes are removed so copied text and integer exports
    compare identically, while missing, non-numeric and zero identifiers remain
    unavailable rather than becoming fabricated keys.
    """
    raw = values.astype("string").str.strip()
    valid = raw.str.fullmatch(r"[0-9]+", na=False)
    canonical = raw.where(valid).str.lstrip("0")
    canonical = canonical.mask(canonical.eq(""), "0")
    canonical = canonical.where(canonical.ne("0"), pd.NA)
    return canonical.astype("string")
