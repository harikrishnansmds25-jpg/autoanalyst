import hashlib
import json
from pathlib import Path

from autoanalyst.agents.prompts import PROMPT_VERSION
from autoanalyst.config import Settings
from autoanalyst.schemas import Plan, Profile

# Settings that do not change what plan is produced.
IGNORED = {"gemini_api_key", "runs_dir", "cache_dir", "retention_days", "n_jobs", "soffice_path",
           "pdf_timeout", "llm_timeout"}


def cache_key(profile: Profile, settings: Settings, variant: str = "") -> str:
    config = settings.model_dump(mode="json", exclude=IGNORED)
    payload = json.dumps(
        {"data": profile.data_hash, "target": profile.target, "config": config,
         "prompts": PROMPT_VERSION, "variant": variant},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class PlanCache:
    """Approved plans keyed by data hash + config, so a re-run skips the LLM entirely."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> Plan | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return Plan.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError:
            return None

    def put(self, key: str, plan: Plan) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._path(key).write_text(plan.model_dump_json(indent=1), encoding="utf-8")
