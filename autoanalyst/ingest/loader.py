import csv
import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from autoanalyst.config import Settings, get_settings

ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")
DELIMITERS = ",;\t|"
EXTRA_NA = ["?"]


class IngestError(Exception):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


@dataclass
class LoadedData:
    df: pd.DataFrame
    file_name: str
    data_hash: str
    encoding: str
    delimiter: str
    n_bytes: int
    warnings: list[str] = field(default_factory=list)


def data_hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _decode(raw: bytes) -> tuple[str, str]:
    for enc in ENCODINGS:
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise IngestError(["Could not decode file"])  # unreachable: latin-1 decodes any bytes


def _sniff_delimiter(text: str) -> str:
    sample = text[:65536]
    try:
        return csv.Sniffer().sniff(sample, delimiters=DELIMITERS).delimiter
    except csv.Error:
        return ","


def _header_warnings(text: str, delimiter: str) -> list[str]:
    first_line = next(csv.reader(io.StringIO(text), delimiter=delimiter), [])
    names = [n.strip() for n in first_line]
    warnings = []
    dupes = sorted({n for n in names if names.count(n) > 1 and n})
    if dupes:
        warnings.append(f"Duplicate column names were renamed: {', '.join(dupes)}")
    if any(not n for n in names):
        warnings.append("Some columns have no header name")
    return warnings


def load_csv(
    source: str | Path | bytes,
    file_name: str | None = None,
    settings: Settings | None = None,
) -> LoadedData:
    """Read and validate an uploaded CSV. Raises IngestError listing every problem found."""
    settings = settings or get_settings()
    if isinstance(source, (str, Path)):
        path = Path(source)
        file_name = file_name or path.name
        raw = path.read_bytes()
    else:
        raw = source
        file_name = file_name or "upload.csv"

    errors = []
    if not file_name.lower().endswith(".csv"):
        errors.append(f"Expected a .csv file, got '{file_name}'")
    size_mb = len(raw) / 1024**2
    if size_mb > settings.max_file_mb:
        errors.append(f"File is {size_mb:.1f} MB, limit is {settings.max_file_mb:g} MB")
    if not raw.strip():
        errors.append("File is empty")
    if errors:
        raise IngestError(errors)

    text, encoding = _decode(raw)
    delimiter = _sniff_delimiter(text)
    warnings = []
    if encoding != "utf-8-sig":
        warnings.append(f"File is not UTF-8; read as {encoding}")

    try:
        df = pd.read_csv(
            io.StringIO(text),
            sep=delimiter,
            na_values=EXTRA_NA,
            skipinitialspace=True,
            low_memory=False,
        )
    except pd.errors.EmptyDataError:
        raise IngestError(["File has no columns"])
    except pd.errors.ParserError as exc:
        raise IngestError([f"Malformed CSV: {exc}".strip()])

    df.columns = [str(c).strip() for c in df.columns]
    warnings += _header_warnings(text, delimiter)

    if len(df) > settings.max_rows:
        errors.append(f"File has {len(df):,} rows, limit is {settings.max_rows:,}")
    if len(df) < settings.min_rows:
        errors.append(f"File has {len(df)} rows, at least {settings.min_rows} are needed")
    if df.shape[1] < 2:
        errors.append("At least two columns are needed (features and a target)")
    if errors:
        raise IngestError(errors)

    empty_rows = int(df.isna().all(axis=1).sum())
    if empty_rows:
        df = df.loc[~df.isna().all(axis=1)].reset_index(drop=True)
        warnings.append(f"Removed {empty_rows} completely empty rows")

    return LoadedData(
        df=df,
        file_name=file_name,
        data_hash=data_hash(raw),
        encoding=encoding,
        delimiter=delimiter,
        n_bytes=len(raw),
        warnings=warnings,
    )
