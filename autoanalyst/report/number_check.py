"""Verify that every number in a narrative can be traced to facts.json."""

import re
from dataclasses import dataclass

# A number not glued to letters, digits or underscores (so 'c3', '3rd' and 'x_2' are ignored),
# optionally followed by a percent sign or the word 'percent'.
NUMBER = re.compile(
    r"(?<![\w.])(?P<num>-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(?![\w])(?P<pct>\s?%|\s?percent\b)?"
)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z'\"(])")
SKIP_KEYS = {"data_hash", "generated_at", "libraries"}


@dataclass(frozen=True)
class Mention:
    text: str
    value: float
    decimals: int
    percent: bool


def extract_numbers(text: str) -> list[Mention]:
    out = []
    for m in NUMBER.finditer(text):
        raw = m.group("num").replace(",", "")
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        out.append(Mention(m.group(0).strip(), float(raw), decimals, bool(m.group("pct"))))
    return out


def fact_numbers(facts) -> set[float]:
    """All numbers in facts: numeric fields, and numbers written inside text fields."""
    found: set[float] = set()

    def walk(node, key=None):
        if key in SKIP_KEYS:
            return
        if isinstance(node, bool) or node is None:
            return
        if isinstance(node, (int, float)):
            found.add(float(node))
        elif isinstance(node, str):
            found.update(m.value for m in extract_numbers(node))
        elif isinstance(node, dict):
            for k, v in node.items():
                walk(v, k)
        elif isinstance(node, (list, tuple)):
            for v in node:
                walk(v, key)

    walk(facts.model_dump() if hasattr(facts, "model_dump") else facts)
    return found


def is_supported(mention: Mention, numbers: set[float]) -> bool:
    target = abs(mention.value)
    for f in numbers:
        candidates = [abs(f)]
        if mention.percent:
            candidates.append(abs(f) * 100)
        if any(round(c, mention.decimals) == target for c in candidates):
            return True
    return False


def unsupported_numbers(text: str, facts) -> list[Mention]:
    numbers = fact_numbers(facts)
    return [m for m in extract_numbers(text) if not is_supported(m, numbers)]


def strip_unsupported(text: str, facts) -> tuple[str, list[str]]:
    """Drop every sentence that contains a number not found in facts. Returns (text, removed)."""
    numbers = fact_numbers(facts)
    kept, removed = [], []
    for sentence in SENTENCE_END.split(text.strip()):
        if all(is_supported(m, numbers) for m in extract_numbers(sentence)):
            kept.append(sentence)
        else:
            removed.append(sentence)
    return " ".join(kept), removed
