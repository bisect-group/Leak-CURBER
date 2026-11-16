"""Shared, resize-safe terminal progress and component logging utilities.

The small :func:`progress` facade intentionally covers the iterator and manual
counter patterns used by the project's standalone data and benchmark tools.
It keeps all active tasks in one Rich live display, so nested work and log
messages do not fight over the terminal.
"""

from __future__ import annotations

import logging
import inspect
import sys
import threading
from pathlib import Path
from typing import Any, Iterable, Iterator, TypeVar

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

T = TypeVar("T")

_CONSOLE = Console(stderr=True)
_LOCK = threading.RLock()
_ACTIVE_TASKS = 0
_PROGRESS = Progress(
    SpinnerColumn(),
    TextColumn("{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    TextColumn("{task.fields[postfix]}", justify="right"),
    TimeRemainingColumn(),
    console=_CONSOLE,
    transient=False,
    disable=not _CONSOLE.is_terminal,
)


def get_console() -> Console:
    """Return the console shared by progress displays and component loggers."""
    return _CONSOLE


def _start() -> None:
    global _ACTIVE_TASKS
    with _LOCK:
        if _ACTIVE_TASKS == 0:
            _PROGRESS.start()
        _ACTIVE_TASKS += 1


def _stop(task_id: TaskID, transient: bool) -> None:
    global _ACTIVE_TASKS
    with _LOCK:
        if transient:
            _PROGRESS.remove_task(task_id)
        else:
            _PROGRESS.update(task_id, visible=False)
        _ACTIVE_TASKS -= 1
        if _ACTIVE_TASKS == 0:
            _PROGRESS.stop()


class _ProgressTask:
    def __init__(self, description: str = "Working", total: float | None = None, *, unit: str | None = None,
                 transient: bool = False, disable: bool = False, **_: Any) -> None:
        self.description = description
        self.total = total
        self.unit = unit
        self.transient = transient
        self.disable = disable
        self.task_id: TaskID | None = None

    def _open(self) -> None:
        if self.task_id is not None or self.disable:
            return
        _start()
        suffix = f" ({self.unit})" if self.unit else ""
        self.task_id = _PROGRESS.add_task(self.description + suffix, total=self.total, postfix="")

    def __enter__(self) -> "_ProgressTask":
        self._open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __iter__(self) -> Iterator[T]:
        if not hasattr(self, "iterable"):
            raise TypeError("A manual progress task is not iterable")
        self._open()
        try:
            for item in self.iterable:  # type: ignore[attr-defined]
                yield item
                self.update(1)
        finally:
            self.close()

    def update(self, n: float = 1, **kwargs: Any) -> None:
        self._open()
        if self.task_id is not None:
            if "advance" in kwargs:
                n = kwargs.pop("advance")
            _PROGRESS.update(self.task_id, advance=n, **kwargs)

    def set_postfix(self, ordered_dict: Any = None, refresh: bool = True, **kwargs: Any) -> None:
        values = dict(ordered_dict or {}) if isinstance(ordered_dict, dict) else {}
        values.update(kwargs)
        self.set_postfix_str(
            " ".join(f"{key}={value}" for key, value in values.items()), refresh=refresh
        )

    def set_postfix_str(self, text: str, refresh: bool = True) -> None:
        self._open()
        if self.task_id is not None:
            _PROGRESS.update(self.task_id, postfix=text, refresh=refresh)

    def close(self) -> None:
        if self.task_id is not None:
            _stop(self.task_id, self.transient)
            self.task_id = None


def progress(iterable: Iterable[T] | None = None, *args: Any, desc: str | None = None,
             description: str | None = None, total: float | None = None, unit: str | None = None,
             leave: bool = True, disable: bool = False, **kwargs: Any) -> _ProgressTask:
    """Create a Rich progress task, optionally wrapping an iterable.

    ``args`` and unused keyword options are accepted to make migrations from
    older progress call sites explicit while retaining their operational shape.
    """
    del args, kwargs
    if iterable is not None and total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            pass
    task = _ProgressTask(description or desc or "Working", total, unit=unit, transient=not leave, disable=disable)
    if iterable is not None:
        task.iterable = iterable
    return task


def track(iterable: Iterable[T], description: str = "Working", **kwargs: Any) -> Iterator[T]:
    """Iterate with a shared Rich task."""
    return iter(progress(iterable, description=description, **kwargs))


def progress_task(description: str = "Working", total: float | None = None, **kwargs: Any) -> _ProgressTask:
    """Create a manually advanced shared Rich task."""
    return progress(description=description, total=total, **kwargs)


def write(message: str, *, file: Any = None, **_: Any) -> None:
    """Print above any active progress display."""
    if file not in (None, sys.stdout, sys.stderr):
        print(message, file=file)
    else:
        _CONSOLE.print(message, markup=False)


def get_component_logger(name: str, log_file: str | Path, level: int = logging.INFO) -> logging.Logger:
    """Build a stable component logger with plain file output and Rich console output."""
    path = Path(log_file).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    logger_name = f"{name}.{path}".replace("/", ".")
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    if not any(getattr(handler, "_emulator_file", None) == path for handler in logger.handlers):
        handler = logging.FileHandler(path)
        handler._emulator_file = path  # type: ignore[attr-defined]
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    if not any(getattr(handler, "_emulator_rich", False) for handler in logger.handlers):
        handler = RichHandler(console=_CONSOLE, show_time=True, show_level=True, show_path=False, markup=False)
        handler._emulator_rich = True  # type: ignore[attr-defined]
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger


class _ConfiguredLogger:
    """Temporary constructor-shaped adapter used while component constructors migrate."""

    def __init__(self, log_dir: str | Path, log_file_name: str, log_level: int = logging.INFO, **_: Any) -> None:
        caller = inspect.currentframe().f_back
        if caller is not None:
            caller = caller.f_back
        module_name = caller.f_globals.get("__name__", __name__) if caller else __name__
        self.logger = get_component_logger(module_name, Path(log_dir) / log_file_name, log_level)

    def get_logger(self) -> logging.Logger:
        return self.logger


def component_logger(*args: Any, **kwargs: Any) -> _ConfiguredLogger:
    """Configure a component logger from its existing directory/name settings."""
    return _ConfiguredLogger(*args, **kwargs)
