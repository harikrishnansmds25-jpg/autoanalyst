"""Turn evaluation/results/*.jsonl into results.md: summary tables and a list of failure cases.

    python -m evaluation.summarize
"""

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
_PROFILES: dict = {}


def _profile(dataset: str):
    if dataset not in _PROFILES:
        from autoanalyst.config import Settings
        from autoanalyst.ingest import load_csv
        from autoanalyst.profiling import build_profile
        from evaluation.corrupt import load

        settings = Settings(_env_file=None)
        raw, target = load(dataset)
        _PROFILES[dataset] = build_profile(load_csv(raw, f"{dataset}.csv", settings), target, settings)
    return _PROFILES[dataset]


def judge(run: dict) -> dict:
    """Add no-op-aware scores to a stored run (redundant steps are not counted as mistakes)."""
    from evaluation.gold import GOLD, rescore

    score = rescore(run, GOLD[run["dataset"]], _profile(run["dataset"]))
    passed = run["execution_status"] == "passed"
    return {**run, "accuracy": score.accuracy, "exact": score.exact, "wrong": score.wrong,
            "silent": score.silent if passed else [],
            "redundant": run.get("redundant_steps", score.noops)}
VARIANT_NAMES = {"fallback": "Rule-based (no LLM)", "A": "A: Planner + validators",
                 "B": "B: Planner + Critic + validators"}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def summary_rows(runs: list[dict]) -> list[list]:
    rows = []
    groups: dict[tuple, list[dict]] = {}
    for run in runs:
        groups.setdefault((run["variant"], run["provider"], run["model"]), []).append(run)
    for (variant, provider, model), group in groups.items():
        passed = [r for r in group if r["execution_status"] == "passed"]
        needed = [r for r in group if r["needed_recovery"]]
        silent_runs = [r for r in passed if r["silent"]]
        costs = [0.0 if r["provider"] == "none" else r["cost_usd"] for r in group]
        cost = "n/a" if any(c is None for c in costs) else f"{_mean(costs):.4f}"
        rows.append([
            VARIANT_NAMES.get(variant, variant),
            "-" if provider == "none" else f"{provider} ({model})",
            len(group),
            f"{_mean(r['accuracy'] for r in group):.2f}",
            f"{sum(r['exact'] for r in group)}/{len(group)}",
            f"{sum(r['recovered'] for r in needed)}/{len(needed)}" if needed else "0/0",
            f"{len(passed)}/{len(group)}",
            f"{len(silent_runs)}/{len(passed)}",
            f"{_mean(len(r['redundant']) for r in group):.1f}",
            f"{_mean(r['llm_calls'] for r in group):.1f}",
            f"{_mean(r['llm_failed_calls'] for r in group):.1f}",
            f"{_mean(r['input_tokens'] for r in group):,.0f} / {_mean(r['output_tokens'] for r in group):,.0f}",
            f"{_mean(r['seconds'] for r in group):.1f}",
            cost,
        ])
    return rows


def detail_rows(runs: list[dict]) -> list[list]:
    return [[
        r["dataset"], r["variant"], r["debate_status"], r["debate_rounds"],
        f"{r['first_proposal_score']:.2f}", f"{r['accuracy']:.2f}", r["execution_status"], r["fix_rounds"],
        len(r["silent"]), len(r["redundant"]), r["critic_objections"], r["critic_overruled"], r["llm_calls"],
        r["seconds"],
    ] for r in runs]


def writer_rows(writers: list[dict]) -> list[list]:
    return [[
        w["dataset"], f"{w.get('provider')} ({w.get('model')})", w.get("source", "skipped"), w.get("attempts", "-"),
        w.get("invented_first_draft", "-"), w.get("invented_numbers_by_attempt", "-"),
        w.get("sentences_removed", "-"), w.get("invented_in_final_report", "-"), w.get("seconds", "-"),
    ] for w in writers]


def failure_cases(runs: list[dict], writers: list[dict]) -> list[str]:
    cases = []
    for r in runs:
        label = f"{r['dataset']} / {r['variant']} ({r['provider']})"
        for mistake in r["silent"]:
            cases.append(f"**Silent mistake**, {label}: {mistake}. Validators passed the plan.")
        if r["execution_status"] == "stopped":
            errors = "; ".join(r["first_execution_errors"]) or "see run log"
            cases.append(f"**Stopped**, {label}: validation still failing after {r['fix_rounds']} round(s). "
                         f"First errors: {errors}")
        elif r["needed_recovery"]:
            first = "; ".join(r["first_plan_errors"] + r["first_execution_errors"])
            cases.append(f"**Recovered**, {label}: first proposal failed ({first}); fixed by the loop.")
        if r["critic_overruled"]:
            cases.append(f"**Critic overruled**, {label}: Critic approved {r['critic_overruled']} plan(s) "
                         "with validator errors.")
        if r["debate_status"] == "escalated":
            cases.append(f"**Escalated**, {label}: no agreement after {r['debate_rounds']} rounds.")
        if r["llm_failed_calls"]:
            cases.append(f"**Invalid LLM output**, {label}: {r['llm_failed_calls']} reply(ies) rejected by "
                         "the schema and retried.")
        wrong = [w["column"] for w in r["wrong"]]
        if wrong and r["variant"] != "fallback":
            cases.append(f"**Plan differs from known-good**, {label}: {', '.join(wrong)}")
    for w in writers:
        if w.get("invented_first_draft"):
            cases.append(f"**Invented numbers**, writer on {w['dataset']} ({w['provider']}): "
                         f"{w['invented_numbers_by_attempt']} unsupported numbers per attempt; "
                         f"{w['sentences_removed']} sentence(s) removed from the final report.")
    return cases


def render(out_dir: Path = RESULTS) -> str:
    runs, writers, seen_fallback = [], [], set()
    for path in sorted(out_dir.glob("runs_*.jsonl")):
        for run in read_jsonl(path):
            # The rule-based variant is identical in every provider's file; keep it once.
            if run["variant"] == "fallback":
                if run["dataset"] in seen_fallback:
                    continue
                seen_fallback.add(run["dataset"])
            runs.append(judge(run))
    for path in sorted(out_dir.glob("writer_*.jsonl")):
        writers += read_jsonl(path)
    providers = {r["provider"] for r in runs}

    parts = ["# Evaluation results", ""]
    if "hosted" not in providers:
        parts += ["Hosted model (Gemini): **not run** (no `GEMINI_API_KEY` configured).", ""]
    parts += [
        "## Summary by variant", "",
        "Plan accuracy: share of columns (plus the duplicate-row rule) whose effective actions in the executed "
        "plan match the known-good plan. Steps that cannot change the data (parsing an already numeric column, "
        "imputing a column with nothing missing) are not mistakes; they are counted as redundant steps. "
        "Recovered: runs whose first proposal failed a validator and that ended with a validated plan. "
        "Silent mistakes: validated runs that still keep an identifier or leak column, impute a mostly-missing "
        "column, or mode-impute a numeric column that has missing values. Runs are unattended (no user edits) "
        "and single-shot at temperature 0.1 with a fixed seed.", "",
        _table(["Variant", "Model", "Runs", "Plan acc.", "Exact plans", "Recovered",
                "Validated", "Runs with silent mistakes", "Redundant steps/run", "LLM calls/run",
                "Rejected replies/run",
                "Tokens/run (in / out)", "Time/run (s)", "Cost/run (USD)"], summary_rows(runs)),
        "", "## Per dataset", "",
        _table(["Dataset", "Variant", "Debate", "Rounds", "First proposal acc. (strict)", "Plan acc.",
                "Execution", "Fix rounds", "Silent", "Redundant", "Critic objections", "Critic overruled",
                "LLM calls", "Seconds"], detail_rows(runs)),
    ]
    if writers:
        parts += ["", "## Report writer", "",
                  _table(["Dataset", "Model", "Source", "Attempts", "Invented (1st draft)", "Invented per attempt",
                          "Sentences removed", "Invented in final report", "Seconds"], writer_rows(writers))]
    parts += ["", "## Failure cases found", ""]
    parts += [f"- {c}" for c in failure_cases(runs, writers)] or ["- None."]
    text = "\n".join(parts) + "\n"
    (out_dir / "results.md").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    print(render(Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS))
