"""Static PNG charts for the report. Greyscale by design: the report is printed black and white."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from autoanalyst.eda.analysis import EdaResult  # noqa: E402
from autoanalyst.schemas import Profile  # noqa: E402
from autoanalyst.training.trainer import METRIC_NAMES, TrainOutput  # noqa: E402

INK = "#222222"
MARK = "#4d4d4d"
LIGHT = "#b3b3b3"
GRID = "#e6e6e6"
DPI = 150
MAX_BARS = 15
MAX_HISTS = 9

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": "#808080",
    "axes.linewidth": 0.6,
    "axes.labelcolor": INK,
    "axes.titlesize": 10,
    "axes.titleweight": "normal",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


@dataclass
class Chart:
    key: str
    section: str
    caption: str
    path: Path


def _save(fig, out_dir: Path, key: str) -> Path:
    path = out_dir / f"{key}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def _hbar(ax, labels: list[str], values: list[float], errors: list[float] | None = None) -> None:
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, height=0.7, color=MARK, xerr=errors,
            error_kw={"ecolor": INK, "elinewidth": 0.8, "capsize": 2})
    ax.set_yticks(y, labels)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.axvline(0, color="#808080", linewidth=0.6)


def missing_values(profile: Profile, out_dir: Path) -> Chart | None:
    cols = sorted((c for c in profile.columns if c.n_missing), key=lambda c: c.missing_frac, reverse=True)
    if not cols:
        return None
    cols = cols[:MAX_BARS]
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(cols) + 1))
    _hbar(ax, [c.name for c in cols], [100 * c.missing_frac for c in cols])
    ax.set_xlabel("Missing values (% of rows)")
    ax.set_xlim(0, 100)
    return Chart("missing_values", "data_quality", "Share of missing values per column, before cleaning.",
                 _save(fig, out_dir, "missing_values"))


def target_distribution(df: pd.DataFrame, eda: EdaResult, out_dir: Path) -> Chart:
    target = eda.target.name
    fig, ax = plt.subplots(figsize=(6, 3))
    if eda.target.task == "classification":
        counts = eda.target.class_counts
        labels = list(counts)[:MAX_BARS]
        ax.bar(labels, [counts[k] for k in labels], width=0.6, color=MARK)
        ax.set_ylabel("Rows")
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        caption = f"Number of rows per class of '{target}'."
    else:
        ax.hist(df[target].dropna().astype(float), bins=40, color=MARK, edgecolor="white", linewidth=0.5)
        ax.set_xlabel(target)
        ax.set_ylabel("Rows")
        caption = f"Distribution of the target '{target}'."
    return Chart("target_distribution", "eda", caption, _save(fig, out_dir, "target_distribution"))


def numeric_distributions(df: pd.DataFrame, eda: EdaResult, out_dir: Path) -> Chart | None:
    numeric = {n.column for n in eda.numeric}
    ranked = [a.feature for a in eda.associations if a.feature in numeric][:MAX_HISTS]
    if not ranked:
        return None
    ncols = 3
    nrows = int(np.ceil(len(ranked) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7, 2.0 * nrows), squeeze=False)
    for ax, col in zip(axes.flat, ranked):
        ax.hist(df[col].dropna().astype(float), bins=30, color=MARK, edgecolor="white", linewidth=0.4)
        ax.set_title(col, fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(ranked):]:
        ax.set_visible(False)
    return Chart("numeric_distributions", "eda",
                 "Distributions of the numeric features most associated with the target.",
                 _save(fig, out_dir, "numeric_distributions"))


def correlation_heatmap(eda: EdaResult, out_dir: Path) -> Chart | None:
    cols = list(eda.correlations)
    if len(cols) < 2:
        return None
    matrix = np.array([[eda.correlations[a][b] for b in cols] for a in cols])
    size = max(3.5, 0.5 * len(cols) + 1.5)
    fig, ax = plt.subplots(figsize=(size + 1, size))
    image = ax.imshow(np.abs(matrix), cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    ax.spines[:].set_visible(False)
    if len(cols) <= 15:
        for i in range(len(cols)):
            for j in range(len(cols)):
                v = matrix[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.6 else INK)
    bar = fig.colorbar(image, ax=ax, fraction=0.04)
    bar.set_label("|r|")
    bar.outline.set_visible(False)
    return Chart("correlation_heatmap", "eda",
                 "Pearson correlations between numeric features. Shading shows strength; "
                 "the printed value keeps the sign.",
                 _save(fig, out_dir, "correlation_heatmap"))


def feature_importance(out: TrainOutput, out_dir: Path) -> Chart:
    items = out.result.feature_importance[:MAX_BARS]
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(items) + 1))
    _hbar(ax, [f.feature for f in items], [f.importance for f in items], [f.std for f in items])
    ax.set_xlabel(f"Drop in {METRIC_NAMES[out.result.primary_metric]} when the feature is shuffled")
    return Chart("feature_importance", "importance",
                 "Permutation importance on the test set. Error bars show one standard deviation "
                 "across repeats.", _save(fig, out_dir, "feature_importance"))


def model_vs_baseline(out: TrainOutput, out_dir: Path) -> Chart:
    r = out.result
    metric = r.comparison_metric
    groups = ["Cross-validation", "Test set"]
    model = [r.cv[metric].mean, r.test[metric]]
    base = [r.cv_baseline[metric].mean, r.test_baseline[metric]]
    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(5, 3))
    bars_m = ax.bar(x - 0.18, model, width=0.34, color=INK, label=r.model_label)
    ax.errorbar(x[0] - 0.18, model[0], yerr=r.cv[metric].std, color="#808080", elinewidth=0.8, capsize=3)
    bars_b = ax.bar(x + 0.18, base, width=0.34, color=LIGHT, label="Baseline (dummy)")
    for bars in (bars_m, bars_b):
        ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=7, color=INK)
    ax.set_xticks(x, groups)
    ax.set_ylabel(METRIC_NAMES[metric])
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.margins(y=0.15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    return Chart("model_vs_baseline", "model",
                 f"{r.model_label} against the dummy baseline on {METRIC_NAMES[metric]}. "
                 "The error bar is one cross-validation standard deviation.",
                 _save(fig, out_dir, "model_vs_baseline"))


def confusion(out: TrainOutput, out_dir: Path) -> Chart:
    r = out.result
    cm = np.array(r.confusion_matrix)
    share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    n = len(r.class_labels)
    fig, ax = plt.subplots(figsize=(max(3.5, 0.6 * n + 2), max(3, 0.6 * n + 1.5)))
    ax.imshow(share, cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks(range(n), r.class_labels, rotation=45 if n > 4 else 0)
    ax.set_yticks(range(n), r.class_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.spines[:].set_visible(False)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if share[i, j] > 0.6 else INK)
    return Chart("confusion_matrix", "model",
                 "Confusion matrix on the test set (counts; shading is the share of each actual class).",
                 _save(fig, out_dir, "confusion_matrix"))


def regression_fit(out: TrainOutput, out_dir: Path) -> list[Chart]:
    actual = np.asarray(out.y_test, dtype=float)
    pred = np.asarray(out.y_pred, dtype=float)
    lo, hi = min(actual.min(), pred.min()), max(actual.max(), pred.max())

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.scatter(actual, pred, s=8, color=MARK, alpha=0.4, linewidths=0)
    ax.plot([lo, hi], [lo, hi], color=INK, linewidth=0.8)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    first = Chart("predicted_vs_actual", "model",
                  "Predicted against actual values on the test set; the line marks perfect predictions.",
                  _save(fig, out_dir, "predicted_vs_actual"))

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.scatter(pred, actual - pred, s=8, color=MARK, alpha=0.4, linewidths=0)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (actual - predicted)")
    second = Chart("residuals", "model", "Residuals on the test set against predicted values.",
                   _save(fig, out_dir, "residuals"))
    return [first, second]


def make_charts(df: pd.DataFrame, profile: Profile, eda: EdaResult, out: TrainOutput,
                out_dir: str | Path) -> list[Chart]:
    """Render every chart for a run. `df` is the cleaned data used for EDA."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    charts = [
        missing_values(profile, out_dir),
        target_distribution(df, eda, out_dir),
        numeric_distributions(df, eda, out_dir),
        correlation_heatmap(eda, out_dir),
        model_vs_baseline(out, out_dir),
    ]
    if out.result.task == "classification":
        charts.append(confusion(out, out_dir))
    else:
        charts += regression_fit(out, out_dir)
    charts.append(feature_importance(out, out_dir))
    return [c for c in charts if c is not None]
