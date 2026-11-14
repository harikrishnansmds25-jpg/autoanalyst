import pandas as pd
import pytest

from autoanalyst.config import Settings
from autoanalyst.ingest import IngestError, load_csv
from autoanalyst.ingest.loader import data_hash


def _csv(rows: int = 30, sep: str = ",") -> str:
    lines = [sep.join(["a", "b", "label"])]
    lines += [sep.join([str(i), f"x{i % 3}", str(i % 2)]) for i in range(rows)]
    return "\n".join(lines) + "\n"


def test_loads_valid_csv(settings):
    data = load_csv(_csv().encode(), "ok.csv", settings)
    assert data.df.shape == (30, 3)
    assert data.delimiter == ","
    assert data.encoding == "utf-8-sig"
    assert data.data_hash == data_hash(_csv().encode())


def test_detects_semicolon_delimiter(settings):
    data = load_csv(_csv(sep=";").encode(), "semi.csv", settings)
    assert data.delimiter == ";"
    assert list(data.df.columns) == ["a", "b", "label"]


def test_handles_bom_and_cp1252(settings):
    bom = load_csv(b"\xef\xbb\xbf" + _csv().encode(), "bom.csv", settings)
    assert bom.df.columns[0] == "a"
    text = _csv().replace("x0", "caf\xe9")
    legacy = load_csv(text.encode("cp1252"), "legacy.csv", settings)
    assert legacy.encoding == "cp1252"
    assert "café" in set(legacy.df["b"])
    assert any("not UTF-8" in w for w in legacy.warnings)


def test_question_mark_is_missing(settings):
    text = _csv().replace("x1", "?")
    data = load_csv(text.encode(), "q.csv", settings)
    assert data.df["b"].isna().sum() == 10


def test_rejects_wrong_extension(settings):
    with pytest.raises(IngestError, match="Expected a .csv"):
        load_csv(_csv().encode(), "data.xlsx", settings)


def test_rejects_empty_file(settings):
    with pytest.raises(IngestError, match="empty"):
        load_csv(b"   \n", "empty.csv", settings)


def test_rejects_large_file():
    small_limit = Settings(_env_file=None, max_file_mb=0.0001)
    with pytest.raises(IngestError, match="limit"):
        load_csv(_csv(200).encode(), "big.csv", small_limit)


def test_rejects_too_many_and_too_few_rows():
    with pytest.raises(IngestError, match="rows, limit"):
        load_csv(_csv(100).encode(), "many.csv", Settings(_env_file=None, max_rows=50))
    with pytest.raises(IngestError, match="at least 20"):
        load_csv(_csv(5).encode(), "few.csv", Settings(_env_file=None))


def test_rejects_single_column(settings):
    text = "only\n" + "\n".join(str(i) for i in range(30))
    with pytest.raises(IngestError, match="two columns"):
        load_csv(text.encode(), "one.csv", settings)


def test_rejects_malformed_rows(settings):
    text = _csv() + "1,2,3,4,5\n"
    with pytest.raises(IngestError, match="Malformed"):
        load_csv(text.encode(), "bad.csv", settings)


def test_warns_on_duplicate_headers_and_drops_empty_rows(settings):
    text = "a,a,label\n" + "\n".join(f"{i},{i},{i % 2}" for i in range(25)) + "\n,,\n"
    data = load_csv(text.encode(), "dup.csv", settings)
    assert any("Duplicate column names" in w for w in data.warnings)
    assert any("empty rows" in w for w in data.warnings)
    assert len(data.df) == 25
    assert data.df.index.equals(pd.RangeIndex(25))


def test_reads_from_path(tmp_path, settings):
    path = tmp_path / "file.csv"
    path.write_text(_csv())
    data = load_csv(path, settings=settings)
    assert data.file_name == "file.csv"
