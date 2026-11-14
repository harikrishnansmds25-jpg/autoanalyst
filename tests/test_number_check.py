import pytest

from autoanalyst.report.number_check import (
    extract_numbers,
    fact_numbers,
    strip_unsupported,
    unsupported_numbers,
)

FACTS = {
    "dataset": {"n_rows": 20640, "data_hash": "9f3a77", "missing_frac": 0.1987},
    "model": {"cv": {"mean": 0.8103, "std": 0.0163}, "folds": 5, "corr": -0.9248},
    "warnings": ["'Age' has 177 missing values"],
    "flag": True,
}


def test_extract_numbers_formats():
    found = extract_numbers("Rows: 20,640, share 19.9%, r = -0.92, 5-fold, 12 percent.")
    assert [(m.value, m.decimals, m.percent) for m in found] == [
        (20640.0, 0, False), (19.9, 1, True), (-0.92, 2, False), (5.0, 0, False), (12.0, 0, True),
    ]


def test_extract_ignores_identifiers_and_ordinals():
    assert extract_numbers("column c3, feature_2, the 3rd class, x2y") == []


def test_fact_numbers_reads_text_and_skips_hash_and_bools():
    numbers = fact_numbers(FACTS)
    assert 177.0 in numbers
    assert 9.0 not in numbers and 3.0 not in numbers
    assert 1.0 not in numbers


@pytest.mark.parametrize("text", [
    "The file has 20,640 rows.",
    "Macro F1 was 0.810 with a standard deviation of 0.016.",
    "About 0.81 on average.",
    "19.9% of values are missing.",
    "Roughly 20% are missing.",
    "Latitude and longitude correlate at -0.92.",
    "A 5-fold cross-validation was used.",
    "'Age' has 177 missing values.",
])
def test_supported_numbers(text):
    assert unsupported_numbers(text, FACTS) == []


@pytest.mark.parametrize("text, bad", [
    ("The model reached 0.95 accuracy.", "0.95"),
    ("Macro F1 was 0.811.", "0.811"),
    ("The file has 20,641 rows.", "20,641"),
    ("About 25% are missing.", "25%"),
])
def test_invented_numbers_are_caught(text, bad):
    assert [m.text for m in unsupported_numbers(text, FACTS)] == [bad]


def test_strip_removes_only_bad_sentences():
    text = ("Macro F1 was 0.810. The model is 99.5% accurate. "
            "It used 5 folds. Results are encouraging.")
    clean, removed = strip_unsupported(text, FACTS)
    assert clean == "Macro F1 was 0.810. It used 5 folds. Results are encouraging."
    assert removed == ["The model is 99.5% accurate."]


def test_strip_accepts_pydantic_facts(messy_profile):
    clean, removed = strip_unsupported(f"There are {messy_profile.n_rows} rows. Also 31337 rows.", messy_profile)
    assert removed == ["Also 31337 rows."]
    assert clean == f"There are {messy_profile.n_rows} rows."
