"""Planner-Critic loop before execution, and the fix/retry loop after validation.

Validators always have the final say: a plan with validator errors is never approved,
whatever the Critic answers.
"""

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from autoanalyst.agents.cache import PlanCache, cache_key
from autoanalyst.agents.critic import review_plan, review_results
from autoanalyst.agents.fallback import fallback_plan
from autoanalyst.agents.messages import CriticOutput
from autoanalyst.agents.planner import propose_plan
from autoanalyst.config import Settings, get_settings
from autoanalyst.llm.client import LLMClient, LLMError
from autoanalyst.schemas import CheckResult, DecisionEntry, Plan, Profile, ValidationReport
from autoanalyst.toolbox.executor import ExecutionError, ExecutionResult, execute
from autoanalyst.validators import check_plan, validate_execution


@dataclass
class Round:
    number: int
    plan: Plan
    checks: list[CheckResult]
    review: CriticOutput | None = None
    notes: str = ""


@dataclass
class DebateResult:
    plan: Plan
    status: Literal["approved", "escalated", "fallback", "cached"]
    rounds: list[Round] = field(default_factory=list)
    open_issues: list[str] = field(default_factory=list)
    log: list[DecisionEntry] = field(default_factory=list)


def _errors(checks: list[CheckResult]) -> list[CheckResult]:
    return [c for c in checks if not c.passed and c.severity == "error"]


def _feedback(checks: list[CheckResult], review: CriticOutput | None) -> list[str]:
    items = [f"Validator: {c.message}" for c in _errors(checks)]
    items += [f"Validator warning: {c.message}" for c in checks if not c.passed and c.severity == "warning"]
    if review is not None:
        items += [
            f"Critic ({i.severity}){' on ' + repr(i.column) if i.column else ''}: {i.problem} Fix: {i.suggestion}"
            for i in review.issues
        ]
    return items


def _fallback(profile: Profile, settings: Settings, reason: str, rounds: list[Round],
              log: list[DecisionEntry]) -> DebateResult:
    plan = fallback_plan(profile, settings)
    checks = check_plan(plan, profile, settings)
    log.append(DecisionEntry(stage="plan", actor="fallback", decision="Rule-based plan used", reason=reason))
    status = "fallback" if not _errors(checks) else "escalated"
    return DebateResult(plan, status, rounds, [c.message for c in _errors(checks)], log)


def plan_with_debate(
    profile: Profile,
    client: LLMClient | None,
    settings: Settings | None = None,
    use_critic: bool = True,
    cache: PlanCache | None = None,
    previous: Plan | None = None,
    feedback: list[str] | None = None,
) -> DebateResult:
    """Planner proposes, validators check, Critic reviews; at most `debate_max_rounds` rounds.
    Unresolved issues after the last round are escalated to the user. `previous` and `feedback`
    start the first round from a rejected plan."""
    settings = settings or get_settings()
    log: list[DecisionEntry] = []
    rounds: list[Round] = []
    key = cache_key(profile, settings, "critic" if use_critic else "no_critic")
    if cache is not None and not feedback and (cached := cache.get(key)) is not None:
        log.append(DecisionEntry(stage="plan", actor="system", decision="Cached approved plan reused",
                                 reason="Same data hash and configuration"))
        return DebateResult(cached, "cached", [], [], log)
    if client is None:
        return _fallback(profile, settings, "No LLM configured", rounds, log)

    plan, feedback = previous, list(feedback or [])
    for number in range(1, settings.debate_max_rounds + 1):
        try:
            plan, notes = propose_plan(client, profile, plan, feedback)
        except LLMError as exc:
            return _fallback(profile, settings, f"Planner unavailable: {str(exc)[:200]}", rounds, log)
        log.append(DecisionEntry(stage="plan", actor="planner", decision=f"Plan proposed (round {number})",
                                 reason=notes[:300]))
        checks = check_plan(plan, profile, settings)
        review = None
        if use_critic:
            try:
                review = review_plan(client, plan, profile, checks)
            except LLMError as exc:
                log.append(DecisionEntry(stage="plan", actor="system", decision="Critic unavailable",
                                         reason=f"Validators decide alone: {str(exc)[:200]}"))
        rounds.append(Round(number, plan, checks, review, notes))

        errors = _errors(checks)
        if review is not None:
            log.append(DecisionEntry(stage="plan", actor="critic", decision=review.verdict,
                                     reason=review.summary[:300]))
            if review.verdict == "approve" and errors:
                log.append(DecisionEntry(stage="plan", actor="validator", decision="Approval overruled",
                                         reason="; ".join(c.message for c in errors)[:300]))
        approved = not errors and (review is None or review.verdict == "approve")
        if approved:
            if cache is not None:
                cache.put(key, plan)
            return DebateResult(plan, "approved", rounds, [], log)
        feedback = _feedback(checks, review)

    log.append(DecisionEntry(stage="plan", actor="system", decision="Escalated to the user",
                             reason=f"No agreement after {settings.debate_max_rounds} rounds"))
    return DebateResult(plan, "escalated", rounds, feedback, log)


@dataclass
class FixRound:
    number: int
    plan: Plan
    report: ValidationReport
    review: CriticOutput | None = None


@dataclass
class ExecutionOutcome:
    plan: Plan
    status: Literal["passed", "stopped"]
    result: ExecutionResult | None
    report: ValidationReport
    rounds: list[FixRound] = field(default_factory=list)
    explanation: str = ""
    log: list[DecisionEntry] = field(default_factory=list)


def _run(df: pd.DataFrame, plan: Plan, settings: Settings) -> tuple[ExecutionResult | None, ValidationReport]:
    try:
        result = execute(df, plan)
    except ExecutionError as exc:
        return None, ValidationReport(checks=[CheckResult(name="execution", passed=False, message=str(exc))])
    return result, validate_execution(df, plan, result, settings)


def execute_with_review(
    df: pd.DataFrame,
    plan: Plan,
    profile: Profile,
    client: LLMClient | None,
    settings: Settings | None = None,
    use_critic: bool = True,
) -> ExecutionOutcome:
    """Execute and validate. On validator errors the Critic explains them and the Planner revises,
    at most `fix_max_rounds` times; then stop and explain. Critic remarks on a passing run are logged."""
    settings = settings or get_settings()
    log: list[DecisionEntry] = []
    rounds: list[FixRound] = []
    for number in range(1, settings.fix_max_rounds + 1):
        result, report = _run(df, plan, settings)
        review = None
        if client is not None and use_critic:
            try:
                review = review_results(client, plan, profile, report.checks)
                log.append(DecisionEntry(stage="validation", actor="critic", decision=review.verdict,
                                         reason=review.summary[:300]))
            except LLMError as exc:
                log.append(DecisionEntry(stage="validation", actor="system", decision="Critic unavailable",
                                         reason=str(exc)[:200]))
        rounds.append(FixRound(number, plan, report, review))
        log.append(DecisionEntry(
            stage="validation", actor="validator", decision="passed" if report.passed else "failed",
            reason="; ".join(c.message for c in report.errors)[:300] or f"{len(report.checks)} checks passed"))
        if report.passed:
            return ExecutionOutcome(plan, "passed", result, report, rounds, "", log)
        if client is None or number == settings.fix_max_rounds:
            break
        try:
            plan, notes = propose_plan(client, profile, plan, _feedback(report.checks, review))
        except LLMError as exc:
            log.append(DecisionEntry(stage="validation", actor="system", decision="Planner unavailable",
                                     reason=str(exc)[:200]))
            break
        log.append(DecisionEntry(stage="validation", actor="planner", decision=f"Plan revised (fix {number})",
                                 reason=notes[:300]))

    errors = "; ".join(c.message for c in report.errors)
    explanation = f"Stopped after {len(rounds)} validation round(s). Unresolved: {errors}"
    log.append(DecisionEntry(stage="validation", actor="system", decision="Stopped", reason=explanation[:300]))
    return ExecutionOutcome(plan, "stopped", result, report, rounds, explanation, log)
