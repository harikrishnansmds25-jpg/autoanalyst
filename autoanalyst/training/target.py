import numpy as np
import pandas as pd

from autoanalyst.schemas import Task
from autoanalyst.toolbox.cleaning import is_numeric, parse_number_like

MAX_CLASS_LEVELS_FOR_NUMERIC = 20


class TargetError(ValueError):
    pass


def _integral(values: pd.Series) -> bool:
    return bool(np.all(np.mod(values.astype(float), 1) == 0))


def infer_task(y: pd.Series) -> Task:
    """Suggest a task type; the user confirms it at the model-selection checkpoint."""
    known = y.dropna()
    if is_numeric(known):
        few_levels = known.nunique() <= MAX_CLASS_LEVELS_FOR_NUMERIC
        return "classification" if few_levels and _integral(known) else "regression"
    if known.nunique() > MAX_CLASS_LEVELS_FOR_NUMERIC and parse_number_like(known).notna().mean() >= 0.9:
        return "regression"
    return "classification"


def prepare_target(y: pd.Series, task: Task, max_classes: int) -> tuple[pd.Series, list[str]]:
    """Validate a target with no missing values for the confirmed task. Returns (y, notes)."""
    notes = []
    if task == "regression":
        if not is_numeric(y):
            parsed = parse_number_like(y)
            if parsed.notna().mean() < 0.9:
                raise TargetError(f"Target '{y.name}' is not numeric; choose classification")
            dropped = int(parsed.isna().sum())
            notes.append(f"Target '{y.name}' was stored as text and parsed as numbers")
            if dropped:
                notes.append(f"{dropped} rows with an unparseable target were dropped")
            y = parsed.dropna()
        if y.nunique() <= 2:
            notes.append(f"Target '{y.name}' has only {y.nunique()} values; classification may fit better")
        return y.astype(float), notes

    n_classes = y.nunique()
    if n_classes < 2:
        raise TargetError(f"Target '{y.name}' has a single class")
    if n_classes > max_classes:
        raise TargetError(f"Target '{y.name}' has {n_classes} classes (limit {max_classes}); choose regression")
    if is_numeric(y) and not _integral(y):
        raise TargetError(f"Target '{y.name}' has continuous values; choose regression")
    if is_numeric(y):
        y = y.astype(int)
    else:
        y = y.astype(str)
    return y, notes
