"""Model registry shared by the UI (choices) and the trainer (construction and search space)."""

from dataclasses import dataclass
from typing import Callable

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from autoanalyst.schemas import Task


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    task: Task
    build: Callable[[int], object]
    needs_scaling: bool
    search_space: dict[str, list]


_TREE_SPACE = {"max_depth": [3, 5, 8, 12, None], "min_samples_leaf": [1, 5, 10, 20]}
_FOREST_SPACE = {
    "n_estimators": [100, 200],
    "max_depth": [None, 8, 16],
    "min_samples_leaf": [1, 2, 5],
    "max_features": ["sqrt", 0.5, 1.0],
}
_HGB_SPACE = {
    "learning_rate": [0.03, 0.1, 0.3],
    "max_leaf_nodes": [15, 31, 63],
    "l2_regularization": [0.0, 0.1, 1.0],
}

REGISTRY: dict[Task, dict[str, ModelSpec]] = {
    "classification": {
        "logistic_regression": ModelSpec(
            "logistic_regression", "Logistic Regression", "classification",
            lambda seed: LogisticRegression(max_iter=5000, random_state=seed),
            True, {"C": [0.01, 0.1, 1.0, 10.0, 100.0]},
        ),
        "decision_tree": ModelSpec(
            "decision_tree", "Decision Tree", "classification",
            lambda seed: DecisionTreeClassifier(random_state=seed),
            False, _TREE_SPACE,
        ),
        "random_forest": ModelSpec(
            "random_forest", "Random Forest", "classification",
            lambda seed: RandomForestClassifier(random_state=seed),
            False, _FOREST_SPACE,
        ),
        "hist_gradient_boosting": ModelSpec(
            "hist_gradient_boosting", "HistGradientBoosting", "classification",
            lambda seed: HistGradientBoostingClassifier(random_state=seed),
            False, _HGB_SPACE,
        ),
    },
    "regression": {
        "ridge": ModelSpec(
            "ridge", "Ridge", "regression",
            lambda seed: Ridge(random_state=seed),
            True, {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        ),
        "decision_tree": ModelSpec(
            "decision_tree", "Decision Tree", "regression",
            lambda seed: DecisionTreeRegressor(random_state=seed),
            False, _TREE_SPACE,
        ),
        "random_forest": ModelSpec(
            "random_forest", "Random Forest", "regression",
            lambda seed: RandomForestRegressor(random_state=seed),
            False, _FOREST_SPACE,
        ),
        "hist_gradient_boosting": ModelSpec(
            "hist_gradient_boosting", "HistGradientBoosting", "regression",
            lambda seed: HistGradientBoostingRegressor(random_state=seed),
            False, _HGB_SPACE,
        ),
    },
}

DEFAULT_MODEL: dict[Task, str] = {
    "classification": "hist_gradient_boosting",
    "regression": "hist_gradient_boosting",
}


def models_for(task: Task) -> dict[str, ModelSpec]:
    return REGISTRY[task]


def get_model(task: Task, key: str) -> ModelSpec:
    try:
        return REGISTRY[task][key]
    except KeyError:
        raise KeyError(f"No model '{key}' for {task}; choose from {sorted(REGISTRY.get(task, {}))}")
