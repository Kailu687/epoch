"""Safe, bounded CSV reading. Content is parsed as data only; nothing is executed."""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

MAX_BYTES = 500 * 1024 * 1024
MAX_ROWS = 5_000_000
MIN_ROWS = 10
ENCODINGS = ("utf-8-sig", "utf-16", "cp1252", "latin-1")


class DatasetError(Exception):
    """Raised for files that cannot be imported; message is user-facing."""


@dataclass
class CsvInfo:
    path: str
    sha256: str
    size: int
    encoding: str
    delimiter: str


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sniff(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()[:65536]
    if b"\x00" in raw and not raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        raise DatasetError("File looks binary, not a text CSV.")
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        candidates = ("utf-16",)
    else:
        candidates = ("utf-8-sig", "cp1252", "latin-1")
    sample = None
    for enc in candidates:
        for cut in (0, 1, 2, 3):  # a 64 KiB cut can split a multibyte character
            try:
                sample = (raw[:len(raw) - cut] if cut else raw).decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if sample is not None:
            break
    if sample is None:
        raise DatasetError("Could not determine the text encoding.")
    try:
        delim = csv.Sniffer().sniff("\n".join(sample.splitlines()[:50]), delimiters=",;\t|").delimiter
    except csv.Error:
        delim = ","
    return enc, delim


def read_csv(path: str | Path, max_rows: int = MAX_ROWS) -> tuple[pd.DataFrame, CsvInfo]:
    p = Path(path)
    if not p.is_file():
        raise DatasetError(f"File not found: {p}")
    size = p.stat().st_size
    if size == 0:
        raise DatasetError("File is empty.")
    if size > MAX_BYTES:
        raise DatasetError(f"File is {size / 1e6:.0f} MB; the limit is {MAX_BYTES // 1_000_000} MB for this release.")
    enc, delim = sniff(p)
    try:
        df = pd.read_csv(p, sep=delim, encoding=enc, nrows=max_rows + 1, low_memory=False,
                         skipinitialspace=True, na_values=["", "NA", "N/A", "NaN", "nan", "null", "NULL", "?"])
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as e:
        raise DatasetError(f"Could not parse CSV: {e}") from e
    if len(df) > max_rows:
        raise DatasetError(f"More than {max_rows:,} rows; out-of-core processing is not implemented in this release.")
    if df.shape[1] < 2:
        raise DatasetError("Need at least two columns (features and a target). Check the delimiter.")
    if len(df) < MIN_ROWS:
        raise DatasetError(f"Only {len(df)} rows; at least {MIN_ROWS} are needed.")
    with open(p, newline="", encoding=enc) as f:  # pandas silently renames duplicates, so read the raw header
        header = [h.strip() for h in next(csv.reader(f, delimiter=delim))]
    if len(set(header)) != len(header):
        raise DatasetError(f"Duplicate column names: {sorted({h for h in header if header.count(h) > 1})}")
    df.columns = [str(c).strip() for c in df.columns]
    return df, CsvInfo(str(p), sha256_file(p), size, enc, delim)
