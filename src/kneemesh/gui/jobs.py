"""Running GUI jobs in a separate process and waiting for them without blocking the GUI's event loop."""

import asyncio
import multiprocessing
import queue
from collections.abc import Callable
from multiprocessing.context import SpawnContext
from multiprocessing.process import BaseProcess
from typing import Any

from kneemesh.gui.worker import run_job

_POLL_SECONDS = 0.1


class JobError(Exception):
    """A job failed, was cancelled, or its process died."""


class JobRunner:
    """Runs one job at a time in a spawned process, which a cancel terminates."""

    def __init__(self) -> None:
        self._context: SpawnContext = multiprocessing.get_context("spawn")
        self._process: BaseProcess | None = None
        self._cancelled: bool = False

    @property
    def busy(self) -> bool:
        """Whether a job is running."""
        return self._process is not None

    async def run(self, job: dict[str, Any], on_log: Callable[[str, str], None]) -> dict[str, Any]:
        """Run a job (worker.run_job), passing each log record's level and message to on_log, and return its result.

        Raises:
            JobError: If a job is already running, or this one fails, is cancelled, or its process dies.
        """
        if self._process is not None:
            raise JobError("A job is already running")
        reports = self._context.Queue()
        process = self._context.Process(target=run_job, args=(job, reports), daemon=True)
        self._process, self._cancelled = process, False
        process.start()
        try:
            while True:
                try:
                    report = reports.get_nowait()
                except queue.Empty:
                    if not process.is_alive():
                        # Drain anything put just before the process ended.
                        try:
                            report = reports.get(timeout=_POLL_SECONDS)
                        except queue.Empty:
                            if self._cancelled:
                                raise JobError("Cancelled") from None
                            raise JobError(f"The job's process ended unexpectedly with exit code {process.exitcode}") from None
                    else:
                        await asyncio.sleep(_POLL_SECONDS)
                        continue
                match report:
                    case ("log", level, message):
                        on_log(level, message)
                    case ("done", result):
                        return result
                    case ("error", message):
                        raise JobError(message)
                    case _:
                        raise JobError(f"Unexpected report {report!r}")
        finally:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
            reports.close()
            self._process = None

    def cancel(self) -> None:
        """Terminate the running job, if any; run then raises JobError."""
        if self._process is not None and self._process.is_alive():
            self._cancelled = True
            self._process.terminate()
