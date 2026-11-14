import numpy as np
import pandas as pd
import pytest

from autoanalyst.ingest import LoadedData
from autoanalyst.profiling import build_profile, profile_column
from autoanalyst.schemas import Profile


def _kinds(profile):
    return {c.name: c.kind for c in profile.columns}


def test_detects_column_kinds(messy_profile):
    assert _kinds(messy_profile) == {
        "customer_id": "id",
        "age": "numeric",
        "income": "numeric_text",
        "signup_date": "datetime",
        "city": "categorical",
        "member": "categorical",
        "notes": "text",
        "constant": "constant",
        "churn": "numeric",
    }


def test_counts_missing_duplicates_outliers(messy_profile):
    age = messy_profile.column("age")
    assert age.n_missing == 20
    assert age.n_outliers_extreme == 2
    assert "extreme_outliers" in age.flags
    assert messy_profile.n_duplicates == 5
    income = messy_profile.column("income")
    assert income.n_unparseable == 4
    assert "currency_symbols" in income.flags
    assert income.stats is not None and 30000 < income.stats.median < 70000


def test_flags_mixed_dates_and_warns_about_random_split(messy_profile):
    assert "mixed_date_formats" in messy_profile.column("signup_date").flags
    assert any("random train/test split" in w for w in messy_profile.warnings)


def test_profile_is_compact_and_serialisable(messy_profile):
    assert len(messy_profile.sample_rows) == 5
    assert all(len(c.samples) <= 3 and len(c.top_values) <= 5 for c in messy_profile.columns)
    restored = Profile.model_validate_json(messy_profile.model_dump_json())
    assert restored == messy_profile
    assert len(messy_profile.model_dump_json()) < 20_000


def test_target_guess_and_validation(messy_loaded, settings):
    guessed = build_profile(messy_loaded, settings=settings)
    assert guessed.target == "churn" and guessed.target_guessed
    with pytest.raises(ValueError, match="not found"):
        build_profile(messy_loaded, target="nope", settings=settings)


def test_target_warnings(settings):
    df = pd.DataFrame({"x": np.arange(30.0), "y": [None] * 3 + list(range(27))})
    data = LoadedData(df=df, file_name="t.csv", data_hash="h", encoding="utf-8", delimiter=",", n_bytes=0)
    profile = build_profile(data, target="y", settings=settings)
    assert any("3 missing values" in w for w in profile.warnings)


def test_float_column_is_not_id(settings):
    col = profile_column(pd.Series(np.linspace(0, 1, 100), name="ratio"), settings)
    assert col.kind == "numeric"


def test_sequential_integers_are_id(settings):
    col = profile_column(pd.Series(np.arange(100, 200), name="row"), settings)
    assert col.kind == "id"


def test_unique_strings_are_id_without_numeric_stats(settings):
    col = profile_column(pd.Series([f"Person, Mr. {i}" for i in range(50)], name="Name"), settings)
    assert col.kind == "id" and col.stats is None


def test_month_names_are_not_dates(settings):
    col = profile_column(pd.Series(["May", "June", "July"] * 10, name="month"), settings)
    assert col.kind == "categorical"
