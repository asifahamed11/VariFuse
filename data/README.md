# Dataset setup for the current clinical pipeline

Raw databases, structures, embeddings and fitted weights are not tracked in Git.
Use [DATA_SOURCES.md](../DATA_SOURCES.md) for pinned releases, checksums, official
source links and transformations, and
[LOCAL_PUBLICATION_RUN.md](../LOCAL_PUBLICATION_RUN.md) for the execution order.
The source catalog is `tools/publication_data_catalog.json`.

The current primary task is high-confidence clinical missense classification.
Later clinical variants are exact-variant-disjoint from development. The main
clinical cohort contains overlapping genes; a smaller gene-disjoint subset is
reported separately and is not certified as homology-disjoint.

`VARIFUSE_DATA_DIR` can point at a local input directory. Stage-specific generated
artifacts follow the current configuration and run ID; do not combine artifacts
from different runs. The older `external/format_dms.py` is retained for history
and is not the documented current ProteinGym preparation entrypoint.

Upstream resources retain their own licenses and terms. Use official pinned
sources rather than assuming that a prior shared download reconstructs this run.
