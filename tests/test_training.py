import numpy as np
import pandas as pd
import pytest

from autoanalyst.config import Settings
from autoanalyst.models import models_for
from autoanalyst.schemas import PlanStep
from autoanalyst.training import TrainingError, train_and_evaluate
from refdata import prepared

FAST = Settings(_env_file=None, cv_folds=3, search_iter=3, n_jobs=1, importance_repeats=2)


def _train(name, task, model, max_rows=None, settings=FAST):
    result = prepared(name, max_rows)
    target = {"iris": "species", "titanic": "Survived", "california_housing": "MedHouseVal"}[name]
    return train_and_evaluate(result.df, target, task, model, result.steps_by_feature(), settings)


def _column_step(pipeline, column, step_suffix):
    for _, pipe, cols in pipeline.named_steps["pre"].transformers_:
        if cols == [column]:
            return next(est for name, est in pipe.steps if name.endswith(step_suffix))
    raise AssertionError(f"no transformer for {column}")


@pytest.mark.parametrize("model", list(models_for("classification")))
def test_iris_all_classifiers(model):
    out = _train("iris", "classification", model)
    r = out.result
    assert r.beats_baseline
    assert r.test["f1_macro"] > 0.85
    assert r.test_baseline["balanced_accuracy"] == pytest.approx(1 / 3, abs=0.01)
    assert np.array(r.confusion_matrix).shape == (3, 3)
    assert np.array(r.confusion_matrix).sum() == r.n_test == 30
    assert set(r.cv) == {"f1_macro", "accuracy", "balanced_accuracy", "roc_auc"}
    assert r.feature_importance[0].feature.startswith("petal")


@pytest.mark.parametrize("model", ["random_forest", "hist_gradient_boosting", "logistic_regression"])
def test_titanic(model):
    r = _train("titanic", "classification", model).result
    assert r.beats_baseline
    assert r.test["accuracy"] > 0.75 and r.test["roc_auc"] > 0.8
    assert "Sex" in [f.feature for f in r.feature_importance[:2]]
    assert r.class_labels == ["0", "1"]
    assert r.n_train + r.n_test == 891


@pytest.mark.parametrize("model", list(models_for("regression")))
def test_california_all_regressors(model):
    r = _train("california_housing", "regression", model, max_rows=3000).result
    assert r.beats_baseline
    assert r.test["r2"] > 0.5
    assert r.test["rmse"] < r.test_baseline["rmse"]
    assert set(r.cv) == {"rmse", "mae", "r2"}
    assert r.confusion_matrix == []


def test_imputer_learns_from_train_rows_only():
    rng = np.random.default_rng(3)
    x = rng.exponential(10, 400)
    x[rng.choice(400, 60, replace=False)] = np.nan
    df = pd.DataFrame({"x": x, "z": rng.normal(size=400)})
    df["y"] = 2 * np.nan_to_num(x, nan=5.0) + rng.normal(size=400)
    steps = {"x": [PlanStep(column="x", action="impute_median", reason="test")]}
    out = train_and_evaluate(df, "y", "regression", "ridge", steps, FAST)
    train_rows = df.drop(index=out.X_test.index)
    learned = _column_step(out.pipeline, "x", "impute_median").statistics_[0]
    assert learned == pytest.approx(train_rows["x"].median())
    assert learned != pytest.approx(df["x"].median())


def test_titanic_age_imputed_with_train_median():
    out = _train("titanic", "classification", "decision_tree")
    df = prepared("titanic").df
    learned = _column_step(out.pipeline, "Age", "impute_median").statistics_[0]
    assert learned == pytest.approx(df.drop(index=out.X_test.index)["Age"].median())


def test_scaling_follows_the_registry():
    linear = _train("iris", "classification", "logistic_regression").pipeline
    tree = _train("iris", "classification", "random_forest").pipeline
    assert "auto_scale" in str(linear.named_steps["pre"].transformers_)
    assert "auto_scale" not in str(tree.named_steps["pre"].transformers_)


def test_runs_are_deterministic():
    a = _train("titanic", "classification", "hist_gradient_boosting").result
    b = _train("titanic", "classification", "hist_gradient_boosting").result
    assert a.model_dump(exclude={"seconds"}) == b.model_dump(exclude={"seconds"})


@pytest.mark.parametrize("seed", range(4))
def test_noise_does_not_beat_baseline(seed):
    # The margin is one CV std, so it needs the default 5 folds; 3 folds underestimate the spread.
    settings = FAST.model_copy(update={"cv_folds": 5})
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(400, 5)), columns=list("abcde"))
    reg = train_and_evaluate(X.assign(y=rng.normal(size=400)), "y", "regression", "random_forest",
                             settings=settings).result
    assert not reg.beats_baseline
    assert any("does not clearly beat the baseline" in w for w in reg.warnings)
    clf = train_and_evaluate(X.assign(y=rng.integers(0, 2, 400)), "y", "classification",
                             "random_forest", settings=settings).result
    assert not clf.beats_baseline


def test_missing_target_rows_are_excluded():
    df = prepared("iris").df.copy()
    df.loc[df.index[:10], "species"] = np.nan
    r = train_and_evaluate(df, "species", "classification", "decision_tree", settings=FAST).result
    assert r.n_dropped_target == 10
    assert r.n_train + r.n_test == len(df) - 10
    assert any("10 rows without a target" in w for w in r.warnings)


def test_currency_target_is_parsed_for_regression():
    rng = np.random.default_rng(1)
    x = rng.normal(size=200)
    df = pd.DataFrame({"x": x, "price": [f"${v:,.2f}" for v in 1000 + 300 * x + rng.normal(0, 5, 200)]})
    r = train_and_evaluate(df, "price", "regression", "ridge", settings=FAST).result
    assert r.test["r2"] > 0.9
    assert any("parsed as numbers" in w for w in r.warnings)


def test_training_errors():
    df = pd.DataFrame({"x": range(30), "y": ["a"] * 29 + ["b"]})
    with pytest.raises(TrainingError, match="fewer than 2"):
        train_and_evaluate(df, "y", "classification", "decision_tree", settings=FAST)
    with pytest.raises(TrainingError, match="continuous"):
        train_and_evaluate(df.assign(y=np.linspace(0, 1, 30)), "y", "classification",
                           "decision_tree", settings=FAST)
    with pytest.raises(TrainingError, match="not found"):
        train_and_evaluate(df, "nope", "classification", "decision_tree", settings=FAST)
    with pytest.raises(KeyError):
        train_and_evaluate(df, "y", "classification", "ridge", settings=FAST)
