"""Leakage-safe training: split first, then fit all preprocessing inside the pipeline on train only."""

import time
from dataclasses import dataclass
from math import prod
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from autoanalyst.config import Settings, get_settings
from autoanalyst.models import get_model
from autoanalyst.schemas import PlanStep, Task
from autoanalyst.toolbox.fitted import build_preprocessor
from autoanalyst.training.target import TargetError, prepare_target

PRIMARY = {"classification": "f1_macro", "regression": "rmse"}
# Balanced accuracy is chance-corrected (a majority-class baseline scores 1/k),
# so it is a fairer yardstick against the dummy model than F1.
COMPARISON = {"classification": "balanced_accuracy", "regression": "rmse"}
LOWER_IS_BETTER = {"rmse", "mae"}
METRIC_NAMES = {
    "f1_macro": "macro F1",
    "balanced_accuracy": "balanced accuracy",
    "roc_auc": "ROC AUC",
    "accuracy": "accuracy",
    "rmse": "RMSE",
    "mae": "MAE",
    "r2": "R2",
}


class TrainingError(ValueError):
    pass


class Score(BaseModel):
    mean: float
    std: float


class FeatureImportance(BaseModel):
    feature: str
    importance: float
    std: float


class TrainResult(BaseModel):
    task: Task
    target: str
    model_key: str
    model_label: str
    primary_metric: str
    comparison_metric: str
    seed: int
    n_train: int
    n_test: int
    n_dropped_target: int
    cv_folds: int
    features: list[str]
    best_params: dict[str, Any]
    cv: dict[str, Score]
    cv_baseline: dict[str, Score]
    train_score: float
    test: dict[str, float]
    test_baseline: dict[str, float]
    beats_baseline: bool
    class_labels: list[str] = []
    confusion_matrix: list[list[int]] = []
    feature_importance: list[FeatureImportance]
    warnings: list[str]
    seconds: float


@dataclass
class TrainOutput:
    result: TrainResult
    pipeline: Pipeline
    X_test: pd.DataFrame
    y_test: pd.Series
    y_pred: np.ndarray
    y_proba: np.ndarray | None


def _scorers(task: Task, n_classes: int) -> dict[str, str]:
    if task == "classification":
        return {
            "f1_macro": "f1_macro",
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "roc_auc": "roc_auc" if n_classes == 2 else "roc_auc_ovr_weighted",
        }
    return {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}


def _sign(metric: str) -> float:
    return -1.0 if metric in LOWER_IS_BETTER else 1.0


def _better(a: float, b: float, metric: str, margin: float = 0.0) -> bool:
    return (a - b) * _sign(metric) > margin


def _cv_scores(cv_results: dict, index: int, metrics: list[str]) -> dict[str, Score]:
    return {
        m: Score(
            mean=round(float(_sign(m) * cv_results[f"mean_test_{m}"][index]), 4),
            std=round(float(cv_results[f"std_test_{m}"][index]), 4),
        )
        for m in metrics
    }


def _test_metrics(task: Task, y_true, y_pred, proba, classes, warnings: list[str]) -> dict[str, float]:
    if task == "regression":
        return {
            "rmse": round(float(root_mean_squared_error(y_true, y_pred)), 4),
            "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
            "r2": round(float(r2_score(y_true, y_pred)), 4),
        }
    out = {
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }
    try:
        if len(classes) == 2:
            out["roc_auc"] = roc_auc_score(y_true, proba[:, 1])
        else:
            out["roc_auc"] = roc_auc_score(y_true, proba, multi_class="ovr", average="weighted", labels=classes)
    except ValueError as exc:
        warnings.append(f"ROC AUC could not be computed on the test set: {exc}")
    return {k: round(float(v), 4) for k, v in out.items()}


def _make_cv(task: Task, y_train: pd.Series, folds: int, seed: int):
    if task == "regression":
        return KFold(n_splits=folds, shuffle=True, random_state=seed), folds
    smallest = int(y_train.value_counts().min())
    folds = min(folds, smallest)
    if folds < 2:
        raise TrainingError("A class has fewer than 2 training rows; cross-validation is not possible")
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed), folds


def train_and_evaluate(
    df: pd.DataFrame,
    target: str,
    task: Task,
    model_key: str,
    steps_by_feature: dict[str, list[PlanStep]] | None = None,
    settings: Settings | None = None,
) -> TrainOutput:
    """Train the chosen model on cleaned data. `steps_by_feature` are the plan's fitted steps."""
    settings = settings or get_settings()
    started = time.perf_counter()
    spec = get_model(task, model_key)
    seed = settings.seed
    warnings: list[str] = []

    if target not in df.columns:
        raise TrainingError(f"Target '{target}' not found")
    known = df[target].notna()
    n_dropped = int((~known).sum())
    if n_dropped:
        warnings.append(f"{n_dropped} rows without a target value were excluded")
    try:
        y, notes = prepare_target(df.loc[known, target], task, settings.max_classes)
    except TargetError as exc:
        raise TrainingError(str(exc)) from exc
    warnings += notes
    X = df.loc[y.index].drop(columns=[target])
    if X.shape[1] == 0:
        raise TrainingError("No feature columns left")

    stratify = y if task == "classification" else None
    if stratify is not None and y.value_counts().min() < 2:
        raise TrainingError("A class has fewer than 2 rows; it cannot be split into train and test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=settings.test_size, random_state=seed, stratify=stratify
    )
    cv, folds = _make_cv(task, y_train, settings.cv_folds, seed)
    if folds < settings.cv_folds:
        warnings.append(f"Cross-validation reduced to {folds} folds because a class is small")

    preprocessor = build_preprocessor(X_train, steps_by_feature or {}, scale_numeric=spec.needs_scaling)
    pipeline = Pipeline([("pre", preprocessor), ("model", spec.build(seed))])
    classes = sorted(y.unique().tolist())
    scorers = _scorers(task, len(classes))
    primary, comparison = PRIMARY[task], COMPARISON[task]

    space = {f"model__{k}": v for k, v in spec.search_space.items()}
    search = RandomizedSearchCV(
        pipeline, space,
        n_iter=min(settings.search_iter, prod(len(v) for v in space.values())),
        scoring=scorers, refit=primary, cv=cv, random_state=seed,
        n_jobs=settings.n_jobs, return_train_score=True, error_score="raise",
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    cv_scores = _cv_scores(search.cv_results_, search.best_index_, list(scorers))

    dummy = DummyClassifier(strategy="prior") if task == "classification" else DummyRegressor()
    baseline = Pipeline([("pre", clone(preprocessor)), ("model", dummy)])
    base_cv = cross_validate(baseline, X_train, y_train, cv=cv, scoring=scorers)
    cv_baseline = {
        m: Score(mean=round(float(_sign(m) * base_cv[f"test_{m}"].mean()), 4),
                 std=round(float(base_cv[f"test_{m}"].std()), 4))
        for m in scorers
    }
    baseline.fit(X_train, y_train)

    y_pred = best.predict(X_test)
    proba = best.predict_proba(X_test) if task == "classification" else None
    test = _test_metrics(task, y_test, y_pred, proba, classes, warnings)
    base_proba = baseline.predict_proba(X_test) if task == "classification" else None
    test_baseline = _test_metrics(task, y_test, baseline.predict(X_test), base_proba, classes, [])

    model_cv = cv_scores[comparison]
    beats = (
        _better(model_cv.mean, cv_baseline[comparison].mean, comparison, margin=model_cv.std)
        and _better(test[comparison], test_baseline[comparison], comparison)
    )
    if not beats:
        warnings.append(
            f"{spec.label} does not clearly beat the baseline on {comparison} "
            f"(CV {model_cv.mean} vs {cv_baseline[comparison].mean}; "
            f"test {test[comparison]} vs {test_baseline[comparison]})"
        )

    gap_metric = "f1_macro" if task == "classification" else "r2"
    train_score = float(search.cv_results_[f"mean_train_{gap_metric}"][search.best_index_])
    if train_score - cv_scores[gap_metric].mean > settings.overfit_gap:
        warnings.append(
            f"Possible overfitting: train {gap_metric} {train_score:.3f} vs CV {cv_scores[gap_metric].mean:.3f}"
        )
    if len(y_test) < 50:
        warnings.append(f"Test set has only {len(y_test)} rows; test metrics are noisy")

    importance = permutation_importance(
        best, X_test, y_test, scoring=scorers[primary], n_repeats=settings.importance_repeats,
        random_state=seed, n_jobs=1,
    )
    features = [
        FeatureImportance(feature=str(c), importance=round(float(m), 4), std=round(float(s), 4))
        for c, m, s in zip(X_test.columns, importance.importances_mean, importance.importances_std)
    ]
    features.sort(key=lambda f: f.importance, reverse=True)

    cm = confusion_matrix(y_test, y_pred, labels=classes).tolist() if task == "classification" else []
    result = TrainResult(
        task=task, target=target, model_key=spec.key, model_label=spec.label,
        primary_metric=primary, comparison_metric=comparison, seed=seed,
        n_train=len(y_train), n_test=len(y_test), n_dropped_target=n_dropped, cv_folds=folds,
        features=[str(c) for c in X.columns],
        best_params={k.removeprefix("model__"): v for k, v in search.best_params_.items()},
        cv=cv_scores, cv_baseline=cv_baseline, train_score=round(train_score, 4),
        test=test, test_baseline=test_baseline, beats_baseline=beats,
        class_labels=[str(c) for c in classes] if task == "classification" else [],
        confusion_matrix=cm, feature_importance=features, warnings=warnings,
        seconds=round(time.perf_counter() - started, 2),
    )
    return TrainOutput(result, best, X_test, y_test, y_pred, proba)
