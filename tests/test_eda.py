import numpy as np
import pandas as pd
import pytest

from autoanalyst.config import Settings
from autoanalyst.eda import run_eda
from autoanalyst.eda.analysis import correlation_ratio, cramers_v
from autoanalyst.training import TargetError, infer_task, prepare_target
from refdata import TARGETS, prepared

SETTINGS = Settings(_env_file=None)


def test_iris_eda():
    eda = run_eda(prepared("iris").df, "species", settings=SETTINGS)
    assert eda.target.task == "classification"
    assert sorted(eda.target.class_counts.values()) == [49, 50, 50]  # one duplicate removed
    assert ("petal length (cm)", "petal width (cm)") in {(a, b) for a, b, _ in eda.strong_pairs}
    assert eda.associations[0].feature.startswith("petal")
    assert len(eda.correlations) == 4


def test_titanic_eda():
    eda = run_eda(prepared("titanic").df, "Survived", "classification", SETTINGS)
    assert eda.associations[0].feature == "Sex"
    assert eda.target.minority_share == pytest.approx(342 / 891, abs=1e-3)
    assert {c.column for c in eda.categorical} == {"Sex", "Embarked"}
    assert any("Skewed" in w and "Fare" in w for w in eda.warnings)


def test_california_eda():
    eda = run_eda(prepared("california_housing").df, "MedHouseVal", settings=SETTINGS)
    assert eda.target.task == "regression"
    assert eda.target.stats is not None and eda.target.stats.max == pytest.approx(5.0)
    assert eda.associations[0].feature == "MedInc" and eda.associations[0].measure == "pearson"
    assert any("Latitude" in w and "Longitude" in w for w in eda.warnings)


def test_eda_is_json_serialisable():
    eda = run_eda(prepared("titanic").df, "Survived", "classification", SETTINGS)
    assert type(eda).model_validate_json(eda.model_dump_json()) == eda


def test_imbalance_and_small_data_warnings():
    df = pd.DataFrame({"x": np.arange(60.0), "y": ["a"] * 57 + ["b"] * 3})
    eda = run_eda(df, "y", settings=SETTINGS)
    assert any("Class imbalance" in w for w in eda.warnings)
    assert any("Only 60 rows" in w for w in eda.warnings)


def test_association_measures():
    groups = pd.Series(["a"] * 50 + ["b"] * 50)
    assert correlation_ratio(groups, pd.Series([0.0] * 50 + [1.0] * 50)) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    assert correlation_ratio(groups, pd.Series(rng.normal(size=100))) < 0.3
    assert cramers_v(groups, groups.map({"a": "x", "b": "y"})) == pytest.approx(1.0)
    assert cramers_v(groups, pd.Series(rng.choice(["x", "y"], 100))) < 0.3


@pytest.mark.parametrize("values, task", [
    ([0, 1, 1, 0], "classification"),
    (["cat", "dog", "cat"], "classification"),
    (np.linspace(0, 1, 50), "regression"),
    (list(range(100)), "regression"),
    ([f"${i},000" for i in range(50)], "regression"),
])
def test_infer_task(values, task):
    assert infer_task(pd.Series(values)) == task


def test_prepare_target():
    y, notes = prepare_target(pd.Series(["$1,000", "$2,500", "$3,000"], name="price"), "regression", 50)
    assert y.tolist() == [1000.0, 2500.0, 3000.0] and notes
    labels, _ = prepare_target(pd.Series([1.0, 0.0, 1.0]), "classification", 50)
    assert labels.dtype == int
    with pytest.raises(TargetError, match="single class"):
        prepare_target(pd.Series([1, 1, 1]), "classification", 50)
    with pytest.raises(TargetError, match="continuous"):
        prepare_target(pd.Series([0.5, 1.2, 3.3]), "classification", 50)
    with pytest.raises(TargetError, match="not numeric"):
        prepare_target(pd.Series(["a", "b", "c"]), "regression", 50)
    with pytest.raises(TargetError, match="classes"):
        prepare_target(pd.Series(range(60)), "classification", 50)


def test_all_reference_targets_infer_correctly():
    expected = {"iris": "classification", "titanic": "classification", "california_housing": "regression"}
    for name, target in TARGETS.items():
        assert infer_task(prepared(name).df[target]) == expected[name]
