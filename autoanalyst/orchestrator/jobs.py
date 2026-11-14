"""Job folders: runs/<job_id>/ holds everything a run produces. Old folders are removed after retention."""

import logging
import secrets
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from autoanalyst.orchestrator.state import JobState

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def new_job_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3)


def job_logger(job_dir: Path) -> logging.Logger:
    """Logger writing to runs/<job_id>/job.log (one handler per job, however often it is opened)."""
    logger = logging.getLogger(f"autoanalyst.job.{job_dir.name}")
    path = str((job_dir / "job.log").resolve())
    if not any(getattr(h, "baseFilename", None) == path for h in logger.handlers):
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def close_job_logger(job_dir: Path) -> None:
    logger = logging.getLogger(f"autoanalyst.job.{job_dir.name}")
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def read_state(job_dir: Path) -> JobState | None:
    path = job_dir / "state.json"
    if not path.exists():
        return None
    try:
        return JobState.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def cleanup_runs(runs_dir: Path, retention_days: int, keep: set[str] = frozenset()) -> list[str]:
    """Delete job folders (and cached plans) not touched for `retention_days`. Returns removed job ids."""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    removed = []
    for job_dir in runs_dir.iterdir():
        if not job_dir.is_dir() or job_dir.name.startswith("_") or job_dir.name in keep:
            continue
        state = read_state(job_dir)
        if state is not None and state.busy:
            continue
        updated = (datetime.fromisoformat(state.updated_at) if state
                   else datetime.fromtimestamp(job_dir.stat().st_mtime, timezone.utc))
        if updated < cutoff:
            close_job_logger(job_dir)
            shutil.rmtree(job_dir, ignore_errors=True)
            removed.append(job_dir.name)
    cache = runs_dir / "_cache"
    if cache.exists():
        for entry in cache.glob("*.json"):
            if datetime.fromtimestamp(entry.stat().st_mtime, timezone.utc) < cutoff:
                entry.unlink(missing_ok=True)
    return removed
