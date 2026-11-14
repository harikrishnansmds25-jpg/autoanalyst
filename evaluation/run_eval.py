"""Evaluate plan quality, recovery, silent mistakes and report grounding.

    python -m evaluation.run_eval --provider ollama --variants fallback,A,B --writer
    python -m evaluation.run_eval --provider hosted --price-in 0.30 --price-out 2.50

Variants: fallback = rule-based plan, no LLM; A = Planner + validators; B = Planner + Critic + validators.
Runs are unattended: the plan that leaves the debate is executed as if the user approved it.
"""

import argparse
import json
import time
from pathlib import Path

from autoanalyst.agents.debate import execute_with_review, plan_with_debate
from autoanalyst.agents.writer import write_narrative
from autoanalyst.config import Settings
from autoanalyst.eda import run_eda
from autoanalyst.ingest import load_csv
from autoanalyst.llm import LLMClient, make_client
from autoanalyst.profiling import build_profile
from autoanalyst.report import build_facts, checked_narrative, unsupported_numbers
from autoanalyst.report.narrative import SECTIONS
from autoanalyst.schemas import PlanStep
from autoanalyst.training import infer_task, train_and_evaluate
from evaluation.corrupt import DATASETS, load
from evaluation.gold import GOLD, score_plan

RESULTS = Path(__file__).parent / "results"


def _errors(checks) -> list[str]:
    return [c.message for c in checks if not c.passed and c.severity == "error"]


def _cost(stats: dict, price_in: float | None, price_out: float | None) -> float | None:
    if stats["provider"] in ("ollama", "mock", "none"):
        return 0.0
    if price_in is None or price_out is None:
        return None
    return round(stats["input_tokens"] / 1e6 * price_in + stats["output_tokens"] / 1e6 * price_out, 5)


def run_plan(dataset: str, variant: str, settings: Settings, client: LLMClient | None,
             price_in=None, price_out=None) -> dict:
    raw, target = load(dataset)
    data = load_csv(raw, f"{dataset}.csv", settings)
    profile = build_profile(data, target, settings)
    use_critic = variant == "B"
    started = time.perf_counter()
    debate = plan_with_debate(profile, client, settings, use_critic=use_critic, cache=None)
    outcome = execute_with_review(data.df, debate.plan, profile, client, settings, use_critic=use_critic)
    seconds = round(time.perf_counter() - started, 2)

    first_plan = debate.rounds[0].plan if debate.rounds else debate.plan
    first_plan_errors = _errors(debate.rounds[0].checks) if debate.rounds else []
    first_exec_errors = _errors(outcome.rounds[0].report.checks)
    needed = bool(first_plan_errors or first_exec_errors)
    passed = outcome.status == "passed"
    gold = GOLD[dataset]
    at_checkpoint = score_plan(debate.plan, gold, profile)
    final = score_plan(outcome.plan, gold, profile)
    stats = client.stats() if client else {"provider": "none", "model": "-", "calls": 0, "failed_calls": 0,
                                            "input_tokens": 0, "output_tokens": 0, "seconds": 0.0}
    return {
        "dataset": dataset,
        "variant": variant,
        "provider": stats["provider"],
        "model": stats["model"],
        "debate_status": debate.status,
        "debate_rounds": len(debate.rounds),
        "first_proposal_score": round(score_plan(first_plan, gold, profile).accuracy, 4),
        "first_plan_errors": first_plan_errors,
        "first_execution_errors": first_exec_errors,
        "execution_status": outcome.status,
        "fix_rounds": len(outcome.rounds),
        "needed_recovery": needed,
        "recovered": needed and passed,
        "plan_accuracy": round(at_checkpoint.accuracy, 4),
        "plan_exact": at_checkpoint.exact,
        "final_accuracy": round(final.accuracy, 4),
        "wrong_columns": final.wrong,
        "silent_mistakes": final.silent if passed else [],
        "redundant_steps": final.noops,
        "final_plan": [s.model_dump() for s in outcome.plan.steps],
        "critic_overruled": sum(e.decision == "Approval overruled" for e in debate.log),
        "critic_objections": sum(len(r.review.issues) for r in debate.rounds if r.review),
        "llm_calls": stats["calls"],
        "llm_failed_calls": stats["failed_calls"],
        "input_tokens": stats["input_tokens"],
        "output_tokens": stats["output_tokens"],
        "seconds": seconds,
        "cost_usd": _cost(stats, price_in, price_out),
    }


def run_writer(dataset: str, settings: Settings, client: LLMClient | None) -> dict:
    raw, target = load(dataset)
    data = load_csv(raw, f"{dataset}.csv", settings)
    profile = build_profile(data, target, settings)
    # The writer needs facts from a valid run: start from the rule-based plan and apply the
    # known-good drops for identifier and leak columns, which the rule-based plan can miss.
    plan = plan_with_debate(profile, None, settings).plan
    traps = {c for c, role in GOLD[dataset].roles.items() if role in ("id", "leak")}
    steps = [s for s in plan.steps if s.column not in traps]
    steps += [PlanStep(column=c, action="drop_column", reason="Known-good plan: identifier or leak")
              for c in sorted(traps)]
    outcome = execute_with_review(data.df, plan.model_copy(update={"steps": steps}), profile, None, settings)
    if outcome.status != "passed":
        return {"dataset": dataset, "skipped": f"fallback plan did not validate: {outcome.explanation}"}
    df = outcome.result.df
    task = infer_task(df[target])
    eda = run_eda(df, target, task, settings)
    model = "decision_tree"
    out = train_and_evaluate(df, target, task, model, outcome.result.steps_by_feature(), settings)
    facts = build_facts(profile, outcome.plan, outcome.report, len(df), eda, out.result)
    started = time.perf_counter()
    result = write_narrative(facts, client, settings)
    clean, removed = checked_narrative(result.narrative, facts)
    invented_final = sum(len(unsupported_numbers(getattr(clean, s), facts)) for s in SECTIONS)
    stats = client.stats() if client else {"calls": 0, "provider": "none", "model": "-"}
    return {
        "dataset": dataset,
        "provider": stats["provider"],
        "model": stats["model"],
        "source": result.source,
        "attempts": result.attempts,
        "invented_first_draft": sum(len(v) for v in result.rejected[0].values()) if result.rejected else 0,
        "invented_numbers_by_attempt": [sum(len(v) for v in r.values()) for r in result.rejected],
        "sentences_removed": sum(len(v) for v in removed.values()),
        "invented_in_final_report": invented_final,
        "llm_calls": stats["calls"],
        "seconds": round(time.perf_counter() - started, 2),
        "narrative": result.narrative.model_dump(),
    }


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", default="ollama", choices=["ollama", "hosted", "mock"])
    parser.add_argument("--variants", default="fallback,A,B")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--writer", action="store_true", help="also evaluate the report writer")
    parser.add_argument("--price-in", type=float, help="USD per million input tokens (hosted)")
    parser.add_argument("--price-out", type=float, help="USD per million output tokens (hosted)")
    parser.add_argument("--out", default=str(RESULTS))
    args = parser.parse_args(argv)

    settings = Settings(llm_provider=args.provider, cv_folds=3, search_iter=3, n_jobs=1, importance_repeats=3)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / f"runs_{args.provider}.jsonl"
    writer_path = out_dir / f"writer_{args.provider}.jsonl"
    runs_path.unlink(missing_ok=True)
    if args.writer:
        writer_path.unlink(missing_ok=True)

    for dataset in args.datasets.split(","):
        for variant in args.variants.split(","):
            client = None if variant == "fallback" else make_client(settings)
            row = run_plan(dataset, variant, settings, client, args.price_in, args.price_out)
            if variant == "fallback":
                row["provider"] = "none"
            with runs_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{dataset:20} {variant:8} acc={row['plan_accuracy']:.2f} exec={row['execution_status']:7} "
                  f"silent={len(row['silent_mistakes'])} calls={row['llm_calls']} {row['seconds']}s", flush=True)
        if args.writer:
            row = run_writer(dataset, settings, make_client(settings))
            with writer_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{dataset:20} writer   first_draft_invented={row.get('invented_first_draft')} "
                  f"attempts={row.get('attempts')} removed={row.get('sentences_removed')}", flush=True)
    return out_dir


if __name__ == "__main__":
    main()
