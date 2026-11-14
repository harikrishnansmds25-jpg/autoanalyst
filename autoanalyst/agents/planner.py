from autoanalyst.agents.messages import PlannerOutput, plan_to_output
from autoanalyst.agents.prompts import PLANNER_SYSTEM
from autoanalyst.llm.client import LLMClient
from autoanalyst.schemas import Plan, Profile


def profile_for_prompt(profile: Profile) -> str:
    """The profile as the agents see it: schema, stats and sample rows; never the data itself."""
    return profile.model_dump_json(exclude={"data_hash", "encoding", "delimiter"}, indent=1)


def planner_prompt(profile: Profile, previous: Plan | None = None, feedback: list[str] | None = None) -> str:
    parts = [f"Target column: {profile.target}", f"<profile>\n{profile_for_prompt(profile)}\n</profile>"]
    if previous is not None:
        parts.append(f"<previous_plan>\n{plan_to_output(previous).model_dump_json(indent=1)}\n</previous_plan>")
    if feedback:
        parts.append("<feedback>\n" + "\n".join(f"- {f}" for f in feedback) + "\n</feedback>")
        parts.append("Return the full revised plan.")
    else:
        parts.append("Return the plan.")
    return "\n\n".join(parts)


def propose_plan(client: LLMClient, profile: Profile, previous: Plan | None = None,
                 feedback: list[str] | None = None) -> tuple[Plan, str]:
    """Ask the Planner for a plan. Returns (plan, notes). Raises LLMError on failure."""
    reply = client.structured(PLANNER_SYSTEM, planner_prompt(profile, previous, feedback),
                              PlannerOutput, agent="planner")
    return reply.to_plan(profile.target), reply.notes
