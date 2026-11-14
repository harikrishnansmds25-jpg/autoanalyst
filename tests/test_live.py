"""Checks against a real model. Skipped unless AUTOANALYST_LIVE is set to 'ollama' or 'hosted':

    AUTOANALYST_LIVE=ollama pytest tests/test_live.py -s
"""

import os

import pytest

from autoanalyst.agents.debate import execute_with_review, plan_with_debate
from autoanalyst.agents.writer import write_narrative
from autoanalyst.config import Settings
from autoanalyst.eda import run_eda
from autoanalyst.ingest import load_csv
from autoanalyst.llm import make_client
from autoanalyst.profiling import build_profile
from autoanalyst.report import build_facts
from autoanalyst.training import train_and_evaluate
from autoanalyst.validators import check_plan
from refdata import DATA

PROVIDER = os.environ.get("AUTOANALYST_LIVE")
pytestmark = pytest.mark.skipif(PROVIDER not in ("ollama", "hosted"), reason="live LLM tests not enabled")


@pytest.fixture(scope="module")
def live():
    settings = Settings(llm_provider=PROVIDER, cv_folds=3, search_iter=2, n_jobs=1, importance_repeats=2)
    return settings, make_client(settings)


@pytest.mark.parametrize("name, target", [("titanic", "Survived"), ("messy", "churn")])
def test_live_plan_debate_and_execution(live, name, target, messy_loaded):
    settings, client = live
    data = messy_loaded if name == "messy" else load_csv(DATA / f"{name}.csv", settings=settings)
    profile = build_profile(data, target=target, settings=settings)
    debate = plan_with_debate(profile, client, settings)
    print(f"\n[{name}] debate: {debate.status}, rounds {len(debate.rounds)}")
    for r in debate.rounds:
        errors = [c.message for c in r.checks if not c.passed and c.severity == 'error']
        print(f"  round {r.number}: {len(r.plan.steps)} steps, validator errors {errors}")
        if r.review:
            print(f"  critic {r.review.verdict}: {[i.problem for i in r.review.issues]}")
    assert debate.status in ("approved", "escalated", "fallback")
    if debate.status == "approved":
        assert not [c for c in check_plan(debate.plan, profile, settings) if c.severity == "error"]
    outcome = execute_with_review(data.df, debate.plan, profile, client, settings)
    print(f"  execution: {outcome.status} after {len(outcome.rounds)} round(s) {outcome.explanation}")
    print(f"  llm: {client.stats()}")


def test_live_writer(live):
    settings, client = live
    data = load_csv(DATA / "titanic.csv", settings=settings)
    profile = build_profile(data, target="Survived", settings=settings)
    debate = plan_with_debate(profile, None, settings)
    outcome = execute_with_review(data.df, debate.plan, profile, None, settings)
    eda = run_eda(outcome.result.df, "Survived", "classification", settings)
    out = train_and_evaluate(outcome.result.df, "Survived", "classification", "decision_tree",
                             outcome.result.steps_by_feature(), settings)
    facts = build_facts(profile, debate.plan, outcome.report, len(outcome.result.df), eda, out.result)
    result = write_narrative(facts, client, settings)
    print(f"\nwriter: source {result.source}, attempts {result.attempts}, rejected {result.rejected}")
    for section, text in result.narrative.model_dump().items():
        print(f"  [{section}] {text}")
    assert result.source in ("writer", "template")
