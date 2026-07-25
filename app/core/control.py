"""Cooperative pause/cancel signalling shared between the job queue and the pipeline.

A single JobControl instance is shared across an entire batch run. The
pipeline calls `checkpoint()` between pages (and the queue calls it between
files), which blocks while paused and raises `Cancelled` once cancel has
been requested - giving pause/cancel effect within roughly one page, not
just between whole documents.
"""

from __future__ import annotations

import threading


class Cancelled(Exception):
    """Raised inside the worker thread to unwind out of an in-progress batch/document."""


class JobControl:
    def __init__(self) -> None:
        self._resume_event = threading.Event()
        self._resume_event.set()
        self._cancel_event = threading.Event()

    def pause(self) -> None:
        self._resume_event.clear()

    def resume(self) -> None:
        self._resume_event.set()

    def cancel(self) -> None:
        self._cancel_event.set()
        self._resume_event.set()  # unblock a paused wait so cancellation is noticed promptly

    def is_paused(self) -> bool:
        return not self._resume_event.is_set()

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def checkpoint(self) -> None:
        """Call between units of work. Blocks while paused; raises Cancelled if cancelled."""
        self._resume_event.wait()
        if self._cancel_event.is_set():
            raise Cancelled()

    def reset(self) -> None:
        """Prepare the control for a fresh batch run."""
        self._cancel_event.clear()
        self._resume_event.set()
