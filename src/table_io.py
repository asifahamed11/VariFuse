from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def table_columns(path: Path) -> list[str]:
    """Return columns without loading table rows."""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pq.ParquetFile(path).schema_arrow.names
    return pd.read_csv(path, nrows=0).columns.tolist()


def iter_table(
    path: Path,
    chunksize: int,
    columns: Sequence[str] | None = None,
    max_rows: int | None = None,
) -> Iterator[pd.DataFrame]:
    """Stream CSV or Parquet data as pandas chunks."""
    path = Path(path)
    if path.suffix.lower() != ".parquet":
        yield from pd.read_csv(
            path,
            usecols=list(columns) if columns is not None else None,
            chunksize=chunksize,
            nrows=max_rows,
            low_memory=False,
        )
        return
    remaining = max_rows
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=chunksize,
        columns=list(columns) if columns is not None else None,
    ):
        frame = batch.to_pandas()
        if remaining is not None:
            if remaining <= 0:
                break
            frame = frame.head(remaining)
            remaining -= len(frame)
        if not frame.empty:
            yield frame.reset_index(drop=True)


class AtomicParquetWriter:
    """Write compressed Parquet chunks and atomically publish on success."""

    def __init__(self, path: Path, compression: str = "zstd"):
        self.path = Path(path)
        self.temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        self.compression = compression
        self.writer: pq.ParquetWriter | None = None
        self.schema: pa.Schema | None = None
        self.rows = 0

    def __enter__(self) -> "AtomicParquetWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary.unlink(missing_ok=True)
        return self

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        table = pa.Table.from_pandas(frame.reset_index(drop=True), preserve_index=False)
        if self.writer is None:
            self.schema = table.schema
            self.writer = pq.ParquetWriter(
                self.temporary,
                self.schema,
                compression=self.compression,
                use_dictionary=True,
            )
        else:
            table = table.cast(self.schema, safe=False)
        self.writer.write_table(table)
        self.rows += len(frame)

    def __exit__(self, error_type, error, traceback) -> bool:
        if self.writer is not None:
            self.writer.close()
        if error_type is None and self.rows > 0:
            self.temporary.replace(self.path)
        else:
            self.temporary.unlink(missing_ok=True)
        return False
