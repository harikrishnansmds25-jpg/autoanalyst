from autoanalyst.agents.messages import CriticOutput, plan_to_output
from autoanalyst.agents.planner import profile_for_prompt
from autoanalyst.agents.prompts import CRITIC_RESULTS_INSTRUCTIONS, CRITIC_SYSTEM
from autoanalyst.llm.client import LLMClient
from autoanalyst.schemas import CheckResult, Plan, Profile


def _evidence(checks: list[CheckResult]) -> str:
    if not checks:
        return "No validator issues."
    return "\n".join(
        f"- [{'pass' if c.passed else c.severity}] {c.name}: {c.message}" for c in checks
    )


def critic_prompt(plan: Plan, profile: Profile, checks: list[CheckResult], after_execution: bool = False) -> str:
    parts = [
        f"Target column: {profile.target}",
        f"<profile>\n{profile_for_prompt(profile)}\n</profile>",
        f"<plan>\n{plan_to_output(plan).model_dump_json(indent=1)}\n</plan>",
    ]
    if after_execution:
        parts.append(CRITIC_RESULTS_INSTRUCTIONS)
    parts.append(f"<validator_evidence>\n{_evidence(checks)}\n</validator_evidence>")
    return "\n\n".join(parts)


def review_plan(client: LLMClient, plan: Plan, profile: Profile, checks: list[CheckResult]) -> CriticOutput:
    return client.structured(CRITIC_SYSTEM, critic_prompt(plan, profile, checks), CriticOutput, agent="critic")


def review_results(client: LLMClient, plan: Plan, profile: Profile, checks: list[CheckResult]) -> CriticOutput:
    return client.structured(CRITIC_SYSTEM, critic_prompt(plan, profile, checks, after_execution=True),
                             CriticOutput, agent="critic")
