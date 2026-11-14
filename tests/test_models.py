import pytest
from sklearn.base import is_classifier, is_regressor
from sklearn.model_selection import ParameterGrid

from autoanalyst.models import DEFAULT_MODEL, REGISTRY, get_model, models_for


def test_registry_contents():
    assert {s.label for s in models_for("classification").values()} == {
        "Logistic Regression", "Decision Tree", "Random Forest", "HistGradientBoosting"
    }
    assert {s.label for s in models_for("regression").values()} == {
        "Ridge", "Decision Tree", "Random Forest", "HistGradientBoosting"
    }
    for task, key in DEFAULT_MODEL.items():
        assert key in REGISTRY[task]


def test_scaling_only_for_linear_models():
    scaled = {(s.task, s.key) for t in REGISTRY.values() for s in t.values() if s.needs_scaling}
    assert scaled == {("classification", "logistic_regression"), ("regression", "ridge")}


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_every_model_builds_and_accepts_its_search_space(task):
    for spec in models_for(task).values():
        model = spec.build(7)
        assert is_classifier(model) if task == "classification" else is_regressor(model)
        assert model.get_params()["random_state"] == 7
        for params in list(ParameterGrid(spec.search_space))[:5]:
            model.set_params(**params)


def test_unknown_model():
    with pytest.raises(KeyError, match="choose from"):
        get_model("regression", "logistic_regression")
