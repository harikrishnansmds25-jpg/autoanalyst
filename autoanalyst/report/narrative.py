from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel, Field

from autoanalyst.report.facts import Facts
from autoanalyst.training.trainer import METRIC_NAMES

SECTIONS = ("overview", "data_quality", "eda", "model", "conclusion")
MEASURE_NAMES = {"pearson": "Pearson r", "eta": "eta", "cramers_v": "Cramer's V"}


class Narrative(BaseModel):
    overview: str = Field(min_length=1)
    data_quality: str = Field(min_length=1)
    eda: str = Field(min_length=1)
    model: str = Field(min_length=1)
    conclusion: str = Field(min_length=1)


def _env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["num"] = lambda v: f"{int(v):,}"
    env.filters["pct"] = lambda v: f"{100 * v:.1f}%"
    env.filters["metric"] = lambda k: METRIC_NAMES.get(k, k)
    env.filters["measure"] = lambda k: MEASURE_NAMES.get(k, k)
    env.filters["plural"] = lambda n, word: f"{n} {word}" + ("" if n == 1 else "s")
    return env


def template_narrative(facts: Facts) -> Narrative:
    """Deterministic narrative built only from facts. Used when no LLM writer is available."""
    template = _env().get_template("narrative.j2")
    context = template.new_context({
        "facts": facts,
        "d": facts.dataset,
        "m": facts.model,
        "s": facts.summary,
        "primary": facts.model.primary_metric,
        "comp": facts.model.comparison_metric,
    })
    parts = {name: " ".join("".join(template.blocks[name](context)).split()) for name in SECTIONS}
    return Narrative(**parts)
