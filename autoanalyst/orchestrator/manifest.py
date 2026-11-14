"""Run manifest: everything needed to audit or reproduce a job."""

from importlib.metadata import PackageNotFoundError, version

from autoanalyst.config import Settings
from autoanalyst.report.facts import library_versions

EXTRA_PACKAGES = ["google-genai", "httpx", "streamlit", "scipy"]
SECRET_FIELDS = {"gemini_api_key"}


def versions() -> dict[str, str]:
    out = library_versions()
    for name in EXTRA_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "not installed"
    return out


def public_config(settings: Settings) -> dict:
    return settings.model_dump(mode="json", exclude=SECRET_FIELDS)


def llm_summary(calls: list[dict]) -> dict:
    by_agent: dict[str, int] = {}
    for call in calls:
        by_agent[call["agent"]] = by_agent.get(call["agent"], 0) + 1
    return {
        "calls": len(calls),
        "failed_calls": sum(not c["ok"] for c in calls),
        "calls_by_agent": by_agent,
        "input_tokens": sum(c["input_tokens"] for c in calls),
        "output_tokens": sum(c["output_tokens"] for c in calls),
        "seconds": round(sum(c["seconds"] for c in calls), 2),
        "providers": sorted({f"{c['provider']}:{c['model']}" for c in calls}),
    }
