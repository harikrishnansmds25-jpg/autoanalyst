from dataclasses import dataclass, field

from autoanalyst.agents.prompts import WRITER_SYSTEM
from autoanalyst.config import Settings, get_settings
from autoanalyst.llm.client import LLMClient, LLMError
from autoanalyst.report.facts import Facts
from autoanalyst.report.narrative import SECTIONS, Narrative, template_narrative
from autoanalyst.report.number_check import unsupported_numbers
from autoanalyst.schemas import DecisionEntry


@dataclass
class WriterResult:
    narrative: Narrative
    source: str
    attempts: int = 0
    rejected: list[dict[str, list[str]]] = field(default_factory=list)
    log: list[DecisionEntry] = field(default_factory=list)


def facts_for_prompt(facts: Facts) -> Facts:
    """Drop parts that cost tokens but carry nothing to write about."""
    eda = facts.eda.model_copy(update={"correlations": {}})
    run = {k: v for k, v in facts.run.items() if k not in ("libraries", "llm_calls")}
    return facts.model_copy(update={"eda": eda, "run": run})


def writer_prompt(facts: Facts, rejected: dict[str, list[str]] | None = None) -> str:
    prompt = f"<facts>\n{facts_for_prompt(facts).model_dump_json(indent=1)}\n</facts>"
    if rejected:
        listed = "; ".join(f"{section}: {', '.join(nums)}" for section, nums in rejected.items())
        prompt += (
            f"\n\nYour previous draft used numbers that are not in facts.json ({listed}). "
            "Rewrite every section using only numbers that appear in facts.json."
        )
    return prompt


def _unsupported(narrative: Narrative, facts: Facts) -> dict[str, list[str]]:
    found = {}
    for section in SECTIONS:
        bad = unsupported_numbers(getattr(narrative, section), facts)
        if bad:
            found[section] = [m.text for m in bad]
    return found


def write_narrative(facts: Facts, client: LLMClient | None, settings: Settings | None = None) -> WriterResult:
    """LLM narrative checked against facts; regenerated when it invents numbers.
    Falls back to the template narrative if the LLM is unavailable."""
    settings = settings or get_settings()
    if client is None:
        return WriterResult(template_narrative(facts), "template", log=[
            DecisionEntry(stage="report", actor="system", decision="Template narrative used",
                          reason="No LLM configured")])
    result = WriterResult(narrative=template_narrative(facts), source="template")
    rejected = None
    for attempt in range(1, settings.writer_max_attempts + 1):
        try:
            draft = client.structured(WRITER_SYSTEM, writer_prompt(facts, rejected), Narrative, agent="writer")
        except LLMError as exc:
            result.log.append(DecisionEntry(stage="report", actor="system", decision="Template narrative used",
                                            reason=f"Writer unavailable: {str(exc)[:200]}"))
            return result
        result.attempts = attempt
        result.narrative, result.source = draft, "writer"
        rejected = _unsupported(draft, facts)
        if not rejected:
            result.log.append(DecisionEntry(
                stage="report", actor="writer", decision="Narrative accepted",
                reason=f"All numbers found in facts.json (attempt {attempt})"))
            return result
        result.rejected.append(rejected)
    result.log.append(DecisionEntry(
        stage="report", actor="system", decision="Sentences with unverified numbers removed",
        reason=f"Writer still used unverified numbers after {settings.writer_max_attempts} attempts"))
    return result
