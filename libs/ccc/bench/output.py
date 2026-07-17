"""Incremental structured writers for benchmark records (JSON Lines / CSV).

Records are flushed as they are produced so a long sweep that is interrupted
still leaves a parseable file with every completed case.
"""

import csv
import json
import sys


class JSONLWriter:
    """One JSON object per line, flushed after every record."""

    def __init__(self, stream):
        self._stream = stream

    def write(self, record: dict) -> None:
        self._stream.write(json.dumps(record, default=str) + "\n")
        self._stream.flush()

    def close(self) -> None:
        if self._stream not in (sys.stdout, sys.stderr):
            self._stream.close()


class CSVWriter:
    """CSV with a header taken from the first record's keys."""

    def __init__(self, stream):
        self._stream = stream
        self._writer = None

    def write(self, record: dict) -> None:
        if self._writer is None:
            self._writer = csv.DictWriter(self._stream, fieldnames=list(record.keys()))
            self._writer.writeheader()
        # Flatten anything non-scalar to a JSON string so the row stays 1-D.
        row = {
            k: (json.dumps(v, default=str) if isinstance(v, (dict, list)) else v)
            for k, v in record.items()
        }
        self._writer.writerow(row)
        self._stream.flush()

    def close(self) -> None:
        if self._stream not in (sys.stdout, sys.stderr):
            self._stream.close()


def open_writer(output_path: str | None, fmt: str):
    """Return a writer for ``fmt`` ('jsonl'|'csv') writing to ``output_path``.

    ``output_path`` of None or ``-`` writes to stdout.
    """
    if output_path in (None, "-"):
        stream = sys.stdout
    else:
        # Kept open across incremental writes; closed via the writer's .close().
        stream = open(output_path, "w", newline="")  # noqa: SIM115

    if fmt == "csv":
        return CSVWriter(stream)
    if fmt == "jsonl":
        return JSONLWriter(stream)
    msg = f"Unknown output format: {fmt!r} (expected 'jsonl' or 'csv')"
    raise ValueError(msg)
