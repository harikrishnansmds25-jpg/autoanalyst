"""Runs long transitions off the UI thread. The UI polls state.json; it never waits on a step."""

import threading
from concurrent.futures import Future, ThreadPoolExecutor

from autoanalyst.config import Settings, get_settings
from autoanalyst.orchestrator.machine import Job
from autoanalyst.orchestrator.state import Stage


class JobBusy(RuntimeError):
    pass


class JobRunner:
    def __init__(self, settings: Settings | None = None, inline: bool = False):
        self.settings = settings or get_settings()
        self.inline = inline
        self._pool = None if inline else ThreadPoolExecutor(self.settings.max_workers, "autoanalyst-job")
        self._active: set[str] = set()
        self._lock = threading.Lock()

    def is_busy(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._active

    def submit(self, job_id: str, step: str, **kwargs) -> Future | None:
        """Run `Job.<step>(**kwargs)`, then any follow-up step it implies."""
        with self._lock:
            if job_id in self._active:
                raise JobBusy(f"Job {job_id} is already running a step")
            self._active.add(job_id)
        job = Job.open(job_id, self.settings)
        job.state.busy = True
        job.save()
        if self.inline:
            self._run(job_id, step, kwargs)
            return None
        return self._pool.submit(self._run, job_id, step, kwargs)

    def _run(self, job_id: str, step: str, kwargs: dict) -> None:
        job = Job.open(job_id, self.settings)
        try:
            getattr(job, step)(**kwargs)
            # Approving a plan or choosing a model only changes state; the work follows here.
            if job.state.stage == Stage.EXECUTING:
                job.execute()
            elif job.state.stage == Stage.TRAINING:
                job.train_and_report()
        except Exception as exc:
            job.log.exception("step %s failed", step)
            job.fail(f"{type(exc).__name__}: {exc}")
        finally:
            job.state.busy = False
            job.save()
            with self._lock:
                self._active.discard(job_id)
