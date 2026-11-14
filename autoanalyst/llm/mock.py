"""Deterministic offline provider for development and tests.

Default replies are built from the prompt itself (rule-based plan, approval, template narrative),
so the full pipeline runs without a network. Tests can queue scripted replies per agent; a queued
Exception is raised to simulate a provider failure, a callable gets (system, user) and returns text.
"""

import json
import re
from collections import defaultdict, deque
from typing import Callable

from autoanalyst.llm.client import LLMResponse

Reply = str | Exception | Callable[[str, str], str]


def _tagged(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\n(.*?)\n</{tag}>", text, re.S)
    if match is None:
        raise ValueError(f"mock: no <{tag}> block in prompt")
    return match.group(1)


def default_planner(system: str, user: str) -> str:
    from autoanalyst.agents.fallback import fallback_plan
    from autoanalyst.agents.messages import plan_to_output
    from autoanalyst.schemas import Profile

    shown = json.loads(_tagged(user, "profile"))
    # Fields hidden from the agents are not needed for planning.
    profile = Profile.model_validate({"data_hash": "", "encoding": "", "delimiter": "", **shown})
    plan = fallback_plan(profile)
    return plan_to_output(plan, "Mock planner: rule-based plan").model_dump_json()


def default_critic(system: str, user: str) -> str:
    return '{"verdict": "approve", "issues": [], "summary": "Mock critic: no issues found."}'


def default_writer(system: str, user: str) -> str:
    from autoanalyst.report.facts import Facts
    from autoanalyst.report.narrative import template_narrative

    facts = Facts.model_validate_json(_tagged(user, "facts"))
    return template_narrative(facts).model_dump_json()


DEFAULTS = {"planner": default_planner, "critic": default_critic, "writer": default_writer}


class MockProvider:
    name = "mock"
    model = "mock-1"

    def __init__(self, script: dict[str, list[Reply]] | None = None):
        self.queues: dict[str, deque] = defaultdict(deque)
        for agent, replies in (script or {}).items():
            self.queues[agent].extend(replies)
        self.prompts: list[tuple[str, str, str]] = []

    def queue(self, agent: str, *replies: Reply) -> None:
        self.queues[agent].extend(replies)

    def complete(self, system: str, user: str, schema: dict, temperature: float, agent: str) -> LLMResponse:
        self.prompts.append((agent, system, user))
        reply = self.queues[agent].popleft() if self.queues[agent] else DEFAULTS[agent]
        if isinstance(reply, Exception):
            raise reply
        text = reply(system, user) if callable(reply) else reply
        return LLMResponse(text, input_tokens=(len(system) + len(user)) // 4, output_tokens=len(text) // 4)
