from functools import lru_cache
from pathlib import Path

from autoanalyst.agents import fallback_plan
from autoanalyst.config import Settings
from autoanalyst.ingest import load_csv
from autoanalyst.profiling import build_profile
from autoanalyst.toolbox.executor import ExecutionResult, execute

DATA = Path(__file__).parent / "data"
TARGETS = {"iris": "species", "titanic": "Survived", "california_housing": "MedHouseVal"}


@lru_cache
def prepared(name: str, max_rows: int | None = None) -> ExecutionResult:
    """Load a reference dataset and run the Phase 1 fallback plan on it."""
    settings = Settings(_env_file=None)
    data = load_csv(DATA / f"{name}.csv", settings=settings)
    if max_rows:
        data.df = data.df.sample(max_rows, random_state=0).reset_index(drop=True)
    profile = build_profile(data, target=TARGETS[name], settings=settings)
    return execute(data.df, fallback_plan(profile, settings))
