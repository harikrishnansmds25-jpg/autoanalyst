import json

import pytest
from conftest import make_messy_frame

from autoanalyst.agents import fallback_plan
from autoanalyst.config import Settings
from autoanalyst.ingest import load_csv
from autoanalyst.profiling import build_profile
from autoanalyst.agents.cache import PlanCache
from autoanalyst.agents.debate import execute_with_review, plan_with_debate
from autoanalyst.agents.messages import CriticOutput, plan_to_output
from autoanalyst.agents.planner import planner_prompt, propose_plan
from autoanalyst.agents.writer import facts_for_prompt, write_narrative
from autoanalyst.eda import run_eda
from autoanalyst.llm import LLMClient, MockProvider
from autoanalyst.report import build_facts, checked_narrative, template_narrative
from autoanalyst.schemas import Plan, PlanStep
from autoanalyst.toolbox.executor import execute
from autoanalyst.training import train_and_evaluate
from autoanalyst.validators import validate_execution

REVISE = json.dumps({
    "verdict": "revise", "summary": "Problems found.",
    "issues": [{"column": "age", "problem": "Capping removes plausible ages.",
                "suggestion": "Drop the cap_outliers step for age.", "severity": "blocker"}],
})
APPROVE = '{"verdict": "approve", "issues": [], "summary": "Plan is sound."}'


def _client(**script):
    return LLMClient(MockProvider(script))


def _agents(client):
    return [c.agent for c in client.calls]


def _plan_json(plan: Plan) -> str:
    return plan_to_output(plan).model_dump_json()


@pytest.fixture
def good_plan(messy_profile, settings):
    return fallback_plan(messy_profile, settings)


@pytest.fixture
def unencoded_plan(good_plan):
    """City is imputed but never encoded: a validator error."""
    return good_plan.model_copy(update={"steps": [
        s for s in good_plan.steps if not (s.column == "city" and s.action == "encode_onehot")
    ]})


@pytest.fixture
def unimputed_plan(good_plan):
    """Passes nothing at runtime: age keeps its missing values."""
    return good_plan.model_copy(update={"steps": [
        s for s in good_plan.steps if not (s.column == "age" and s.action == "impute_median")
    ]})


def test_planner_prompt_shows_profile_not_data(messy_profile):
    prompt = planner_prompt(messy_profile)
    assert "<profile>" in prompt and "Target column: churn" in prompt
    assert messy_profile.data_hash not in prompt
    assert "customer note number 4 " in prompt  # sample rows only
    assert "customer note number 150" not in prompt


def test_planner_returns_validated_plan(messy_profile, good_plan):
    plan, notes = propose_plan(_client(), messy_profile)
    assert plan.source == "planner" and plan.target == "churn"
    assert [s.model_dump() for s in plan.steps] == [s.model_dump() for s in good_plan.steps]


def test_planner_output_outside_toolbox_is_rejected(messy_profile):
    bad = json.dumps({"steps": [{"column": "age", "action": "run_python", "reason": "custom code"}]})
    table_with_column = json.dumps({"steps": [{"column": "age", "action": "drop_duplicates", "reason": "dupes"}]})
    client = _client(planner=[bad, table_with_column])
    plan, _ = propose_plan(client, messy_profile)
    assert [c.ok for c in client.calls] == [False, False, True]
    assert "run_python" not in {s.action for s in plan.steps}


def test_critic_output_must_be_consistent():
    with pytest.raises(ValueError):
        CriticOutput.model_validate({"verdict": "revise", "issues": [], "summary": "no issues listed"})


def test_debate_approves_in_first_round(messy_profile, settings):
    client = _client()
    result = plan_with_debate(messy_profile, client, settings)
    assert result.status == "approved" and len(result.rounds) == 1
    assert _agents(client) == ["planner", "critic"]


def test_debate_revises_after_critic_objection(messy_profile, settings):
    client = _client(critic=[REVISE, APPROVE])
    result = plan_with_debate(messy_profile, client, settings)
    assert result.status == "approved" and len(result.rounds) == 2
    second_planner_prompt = [p for a, _, p in client.provider.prompts if a == "planner"][1]
    assert "Capping removes plausible ages." in second_planner_prompt
    assert "<previous_plan>" in second_planner_prompt
    assert [e.decision for e in result.log if e.actor == "critic"] == ["revise", "approve"]


def test_validators_overrule_critic_approval(messy_profile, settings, unencoded_plan):
    client = _client(planner=[_plan_json(unencoded_plan)], critic=[APPROVE, APPROVE])
    result = plan_with_debate(messy_profile, client, settings)
    assert result.status == "approved" and len(result.rounds) == 2
    assert any(e.decision == "Approval overruled" for e in result.log)
    planner_feedback = [p for a, _, p in client.provider.prompts if a == "planner"][1]
    assert "never encoded" in planner_feedback


def test_debate_escalates_after_max_rounds(messy_profile, settings):
    client = _client(critic=[REVISE] * 3)
    result = plan_with_debate(messy_profile, client, settings)
    assert result.status == "escalated" and len(result.rounds) == settings.debate_max_rounds
    assert _agents(client).count("planner") == 3
    assert any("plausible ages" in issue for issue in result.open_issues)


def test_variant_without_critic_uses_validators_only(messy_profile, settings, unencoded_plan):
    client = _client(planner=[_plan_json(unencoded_plan)])
    result = plan_with_debate(messy_profile, client, settings, use_critic=False)
    assert result.status == "approved" and len(result.rounds) == 2
    assert "critic" not in _agents(client)


def test_unavailable_planner_falls_back(messy_profile, settings):
    client = _client(planner=[ConnectionError("offline")] * 3)
    result = plan_with_debate(messy_profile, client, settings)
    assert result.status == "fallback" and result.plan.source == "fallback"
    assert plan_with_debate(messy_profile, None, settings).status == "fallback"


def test_unavailable_critic_leaves_validators_in_charge(messy_profile, settings):
    client = _client(critic=[TimeoutError("slow")] * 3)
    result = plan_with_debate(messy_profile, client, settings)
    assert result.status == "approved"
    assert any(e.decision == "Critic unavailable" for e in result.log)


def test_approved_plans_are_cached(messy_profile, settings, tmp_path):
    cache = PlanCache(tmp_path)
    first = plan_with_debate(messy_profile, _client(), settings, cache=cache)
    client = _client()
    second = plan_with_debate(messy_profile, client, settings, cache=cache)
    assert second.status == "cached" and client.calls == []
    assert second.plan == first.plan
    changed = settings.model_copy(update={"max_row_drop_frac": 0.2})
    assert plan_with_debate(messy_profile, _client(), changed, cache=cache).status == "approved"


def test_escalated_plans_are_not_cached(messy_profile, settings, tmp_path):
    cache = PlanCache(tmp_path)
    plan_with_debate(messy_profile, _client(critic=[REVISE] * 3), settings, cache=cache)
    assert list(tmp_path.iterdir()) == []


def test_execution_passes_first_time(messy_loaded, messy_profile, settings, good_plan):
    client = _client()
    outcome = execute_with_review(messy_loaded.df, good_plan, messy_profile, client, settings)
    assert outcome.status == "passed" and len(outcome.rounds) == 1
    assert outcome.result is not None and outcome.report.passed


def test_execution_failure_is_fixed_by_planner(messy_loaded, messy_profile, settings, unimputed_plan):
    critic_says = json.dumps({"verdict": "revise", "summary": "age still has nulls", "issues": [
        {"column": "age", "problem": "Missing values remain.", "suggestion": "Add impute_median.",
         "severity": "blocker"}]})
    client = _client(critic=[critic_says, APPROVE])
    outcome = execute_with_review(messy_loaded.df, unimputed_plan, messy_profile, client, settings)
    assert outcome.status == "passed" and len(outcome.rounds) == 2
    assert not outcome.rounds[0].report.passed
    assert "no_nulls" in {c.name for c in outcome.rounds[0].report.errors}
    fix_prompt = [p for a, _, p in client.provider.prompts if a == "planner"][0]
    assert "Missing values remain in: age" in fix_prompt


def test_execution_stops_and_explains(messy_loaded, messy_profile, settings, unimputed_plan):
    client = _client(planner=[_plan_json(unimputed_plan)] * 3)
    outcome = execute_with_review(messy_loaded.df, unimputed_plan, messy_profile, client, settings)
    assert outcome.status == "stopped" and len(outcome.rounds) == settings.fix_max_rounds
    assert "Missing values remain in: age" in outcome.explanation
    no_llm = execute_with_review(messy_loaded.df, unimputed_plan, messy_profile, None, settings)
    assert no_llm.status == "stopped" and len(no_llm.rounds) == 1


def test_execution_errors_become_check_failures(messy_loaded, messy_profile, settings, good_plan):
    broken = good_plan.model_copy(update={"steps": good_plan.steps + [
        PlanStep(column="age", action="parse_dates", reason="wrong")]})
    outcome = execute_with_review(messy_loaded.df, broken, messy_profile, None, settings)
    assert outcome.status == "stopped" and outcome.report.errors[0].name == "execution"


@pytest.fixture(scope="module")
def facts():
    fast = Settings(_env_file=None, cv_folds=3, search_iter=2, n_jobs=1, importance_repeats=2)
    data = load_csv(make_messy_frame().to_csv(index=False).encode(), "messy.csv", fast)
    profile = build_profile(data, target="churn", settings=fast)
    plan = fallback_plan(profile, fast)
    result = execute(data.df, plan)
    report = validate_execution(data.df, plan, result, fast)
    eda = run_eda(result.df, "churn", "classification", fast)
    out = train_and_evaluate(result.df, "churn", "classification", "decision_tree",
                             result.steps_by_feature(), fast)
    return build_facts(profile, plan, report, len(result.df), eda, out.result)


def _narrative_json(facts, **override):
    return template_narrative(facts).model_copy(update=override).model_dump_json()


def test_writer_accepts_grounded_narrative(facts, settings):
    client = _client()
    result = write_narrative(facts, client, settings)
    assert result.source == "writer" and result.attempts == 1 and result.rejected == []


def test_writer_regenerates_when_numbers_are_invented(facts, settings):
    client = _client(writer=[_narrative_json(facts, model="It reached 97.3% accuracy.")])
    result = write_narrative(facts, client, settings)
    assert result.attempts == 2 and result.rejected == [{"model": ["97.3%"]}]
    retry_prompt = [p for a, _, p in client.provider.prompts if a == "writer"][1]
    assert "97.3%" in retry_prompt and "not in facts.json" in retry_prompt


def test_writer_output_is_stripped_when_it_keeps_inventing(facts, settings):
    invented = _narrative_json(facts, model="It reached 97.3% accuracy. The model was trained.")
    client = _client(writer=[invented] * settings.writer_max_attempts)
    result = write_narrative(facts, client, settings)
    assert result.source == "writer" and len(result.rejected) == settings.writer_max_attempts
    clean, removed = checked_narrative(result.narrative, facts)
    assert removed == {"model": ["It reached 97.3% accuracy."]}
    assert clean.model == "The model was trained."


def test_writer_falls_back_to_template(facts, settings):
    assert write_narrative(facts, None, settings).source == "template"
    failing = _client(writer=[ConnectionError("offline")] * 3)
    result = write_narrative(facts, failing, settings)
    assert result.source == "template" and "Writer unavailable" in result.log[0].reason


def test_writer_prompt_drops_bulky_fields(facts):
    trimmed = facts_for_prompt(facts)
    assert trimmed.eda.correlations == {} and "libraries" not in trimmed.run
    assert facts.eda.correlations and "libraries" in facts.run
