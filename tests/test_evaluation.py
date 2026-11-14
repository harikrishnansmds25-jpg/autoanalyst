import json

import pandas as pd
import pytest

from autoanalyst.agents import fallback_plan
from autoanalyst.agents.messages import plan_to_output
from autoanalyst.config import Settings
from autoanalyst.ingest import load_csv
from autoanalyst.llm import LLMClient, MockProvider
from autoanalyst.profiling import build_profile
from autoanalyst.schemas import Plan, PlanStep
from evaluation.corrupt import DATASETS, load
from evaluation.gold import GOLD, score_plan
from evaluation.run_eval import main, run_plan
from evaluation.summarize import render

SETTINGS = Settings(_env_file=None, llm_provider="mock", cv_folds=3, search_iter=2, n_jobs=1,
                    importance_repeats=2)


def _profile(name):
    raw, target = load(name)
    return build_profile(load_csv(raw, f"{name}.csv", SETTINGS), target, SETTINGS)


def test_every_dataset_has_a_gold_plan():
    assert set(DATASETS) == set(GOLD)
    for name, gold in GOLD.items():
        columns = {c.name for c in _profile(name).columns}
        assert set(gold.columns) <= columns, name


def test_corruptions_are_deterministic_and_present():
    assert load("titanic_corrupt") == load("titanic_corrupt")
    df = pd.read_csv(pd.io.common.BytesIO(load("titanic_corrupt")[0]))
    assert df["Fare"].astype(str).str.startswith("$").all()
    assert (df["Age"] == "unknown").sum() > 0
    assert df.duplicated().sum() > 0
    assert set(df["outcome_label"]) == {"lost", "saved"}
    kinds = {c.name: c.kind for c in _profile("california_corrupt").columns}
    assert kinds["MedInc"] == "numeric_text" and kinds["survey_date"] == "datetime"
    assert kinds["record_id"] == "id"


def test_gold_plan_scores_exact():
    gold = GOLD["iris_corrupt"]
    steps = [PlanStep(action="drop_duplicates", reason="dupes")]
    for column, alternatives in gold.columns.items():
        for action in sorted(alternatives[0]):
            steps.append(PlanStep(column=column, action=action, reason="gold"))
    score = score_plan(Plan(steps=steps, target="species"), gold)
    assert score.exact and score.accuracy == 1.0 and score.silent == []


def test_fallback_misses_the_categorical_leak():
    score = score_plan(fallback_plan(_profile("iris_corrupt"), SETTINGS), GOLD["iris_corrupt"])
    assert score.silent == ["leak column 'species_code' kept"]
    assert not score.exact


def test_bad_imputation_is_a_silent_mistake():
    plan = fallback_plan(_profile("titanic"), SETTINGS)
    steps = [s for s in plan.steps if s.column != "Age"]
    steps.append(PlanStep(column="Age", action="impute_mode", reason="mode"))
    score = score_plan(plan.model_copy(update={"steps": steps}), GOLD["titanic"], _profile("titanic"))
    assert "numeric column 'Age' with missing values imputed with the mode" in score.silent


def test_noop_steps_are_redundant_not_wrong():
    profile = _profile("titanic")
    plan = fallback_plan(profile, SETTINGS)
    extra = [PlanStep(column="SibSp", action="impute_mode", reason="nothing to fill"),
             PlanStep(column="Fare", action="coerce_numeric", reason="already numeric")]
    padded = plan.model_copy(update={"steps": plan.steps + extra})
    score = score_plan(padded, GOLD["titanic"], profile)
    assert score.exact and score.silent == []
    assert sorted(score.noops) == ["coerce_numeric on 'Fare'", "impute_mode on 'SibSp'"]
    assert not score_plan(padded, GOLD["titanic"]).exact


def test_onehot_and_ordinal_are_equivalent():
    plan = fallback_plan(_profile("titanic"), SETTINGS)
    swapped = [s.model_copy(update={"action": "encode_onehot"}) if s.action == "encode_ordinal" else s
               for s in plan.steps]
    assert score_plan(plan.model_copy(update={"steps": swapped}), GOLD["titanic"]).exact


def test_run_plan_records_recovery():
    good = fallback_plan(_profile("iris_corrupt"), SETTINGS)
    no_impute = good.model_copy(update={"steps": [
        s for s in good.steps if not (s.column == "sepal length (cm)" and s.action == "impute_median")]})
    client = LLMClient(MockProvider({"planner": [plan_to_output(no_impute).model_dump_json()]}))
    row = run_plan("iris_corrupt", "A", SETTINGS, client)
    assert row["needed_recovery"] and row["recovered"]
    assert row["debate_rounds"] == 2 and row["first_proposal_score"] < row["plan_accuracy"]
    assert row["silent_mistakes"] == ["leak column 'species_code' kept"]
    assert row["cost_usd"] == 0.0


def test_main_and_summary(tmp_path):
    main(["--provider", "mock", "--variants", "fallback,B", "--datasets", "iris,iris_corrupt",
          "--writer", "--out", str(tmp_path)])
    runs = [json.loads(line) for line in (tmp_path / "runs_mock.jsonl").read_text().splitlines()]
    assert [(r["dataset"], r["variant"]) for r in runs] == [
        ("iris", "fallback"), ("iris", "B"), ("iris_corrupt", "fallback"), ("iris_corrupt", "B")]
    writers = [json.loads(line) for line in (tmp_path / "writer_mock.jsonl").read_text().splitlines()]
    assert all(w["invented_in_final_report"] == 0 for w in writers)
    text = render(tmp_path)
    assert "Hosted model (Gemini): **not run**" in text
    assert "| Rule-based (no LLM) | - | 2 |" in text
    assert "**Silent mistake**, iris_corrupt / fallback" in text
    assert (tmp_path / "results.md").exists()


@pytest.mark.parametrize("name", ["iris_corrupt", "titanic_corrupt", "california_corrupt"])
def test_corrupted_upload_passes_ingestion(name):
    raw, target = load(name)
    data = load_csv(raw, f"{name}.csv", SETTINGS)
    assert target in data.df.columns
