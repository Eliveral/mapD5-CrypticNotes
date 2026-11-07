from __future__ import annotations

import atexit
import faulthandler
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import platform
import sys
import threading
import time
import traceback
from typing import Any, Optional, TextIO


_LOCK = threading.RLock()
_INITIALIZED = False
_SESSION_ID = ""
_DIAGNOSTIC_DIR: Optional[Path] = None
_EVENT_STREAM: Optional[TextIO] = None
_FAULT_STREAM: Optional[TextIO] = None
_STATE_PATH: Optional[Path] = None
_STARTED_AT = 0.0
_UNCAUGHT_EXCEPTION = False
_EXIT_MARKED = False


def runtime_root() -> Path:
    if bool(getattr(sys, "frozen", False)):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def diagnostic_directory() -> Path:
    global _DIAGNOSTIC_DIR
    with _LOCK:
        if _DIAGNOSTIC_DIR is None:
            configured = os.environ.get("IDENTITYV_DIAGNOSTIC_DIR", "").strip()
            _DIAGNOSTIC_DIR = (
                Path(configured).expanduser().resolve()
                if configured
                else runtime_root() / "assets" / "debug" / "runtime_diagnostics"
            )
            _DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
        return _DIAGNOSTIC_DIR


def diagnostic_path(name: str) -> Path:
    return diagnostic_directory() / Path(name).name


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _prune_old_sessions(directory: Path, keep: int = 20) -> None:
    for pattern in ("events_*.jsonl", "native_fault_*.log", "session_*.json"):
        files = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
        for path in files[keep:]:
            try:
                path.unlink()
            except OSError:
                pass


def diagnostic_event(event: str, **details: Any) -> None:
    payload = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "monotonic_ms": round((time.perf_counter() - _STARTED_AT) * 1000.0, 3) if _STARTED_AT else None,
        "process_pid": os.getpid(),
        "thread": threading.current_thread().name,
        "event": str(event),
        **details,
    }
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
    with _LOCK:
        stream = _EVENT_STREAM
        if stream is None:
            return
        try:
            stream.write(line + "\n")
            stream.flush()
        except OSError:
            pass


def _record_uncaught_exception(kind: str, exc_type, exc_value, exc_traceback, **details: Any) -> None:
    global _UNCAUGHT_EXCEPTION
    _UNCAUGHT_EXCEPTION = True
    formatted = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    logging.getLogger(__name__).critical("uncaught %s exception\n%s", kind, formatted)
    diagnostic_event(
        "uncaught_exception",
        exception_kind=kind,
        exception_type=getattr(exc_type, "__name__", str(exc_type)),
        exception=str(exc_value),
        traceback=formatted,
        **details,
    )


def install_tk_exception_handler(root: Any) -> None:
    original = getattr(root, "report_callback_exception", None)

    def report_callback_exception(exc_type, exc_value, exc_traceback) -> None:
        _record_uncaught_exception("tk_callback", exc_type, exc_value, exc_traceback)
        if callable(original):
            try:
                original(exc_type, exc_value, exc_traceback)
            except Exception:
                pass

    root.report_callback_exception = report_callback_exception
    diagnostic_event("tk_exception_handler_installed")


def mark_clean_exit(reason: str = "normal") -> None:
    global _EXIT_MARKED
    with _LOCK:
        if not _INITIALIZED or _STATE_PATH is None or _EXIT_MARKED:
            return
        _EXIT_MARKED = True
        status = "uncaught_exception" if _UNCAUGHT_EXCEPTION else "clean_exit"
        diagnostic_event("process_exit", status=status, reason=reason)
        final_state = {
            "session_id": _SESSION_ID,
            "pid": os.getpid(),
            "status": status,
            "reason": reason,
            "ended_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            _write_json(_STATE_PATH, final_state)
            _write_json(diagnostic_directory() / "latest_session.json", final_state)
        except OSError:
            pass


def initialize_runtime_diagnostics(level: int = logging.INFO) -> Path:
    global _INITIALIZED, _SESSION_ID, _EVENT_STREAM, _FAULT_STREAM, _STATE_PATH, _STARTED_AT
    with _LOCK:
        if _INITIALIZED:
            return diagnostic_directory()

        directory = diagnostic_directory()
        _prune_old_sessions(directory)
        _SESSION_ID = time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
        _STARTED_AT = time.perf_counter()
        event_path = directory / f"events_{_SESSION_ID}.jsonl"
        fault_path = directory / f"native_fault_{_SESSION_ID}.log"
        _STATE_PATH = directory / f"session_{_SESSION_ID}.json"
        _EVENT_STREAM = event_path.open("a", encoding="utf-8", buffering=1)
        _FAULT_STREAM = fault_path.open("a", encoding="utf-8", buffering=1)

        file_handler = RotatingFileHandler(
            directory / "application.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=4,
            encoding="utf-8",
        )
        handlers: list[logging.Handler] = [file_handler]
        if sys.stderr is not None:
            handlers.append(logging.StreamHandler(sys.stderr))
        logging.basicConfig(
            level=level,
            format="%(asctime)s.%(msecs)03d [%(levelname)s] [%(threadName)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=handlers,
            force=True,
        )

        try:
            faulthandler.enable(file=_FAULT_STREAM, all_threads=True)
        except (RuntimeError, OSError):
            logging.getLogger(__name__).exception("failed to enable faulthandler")

        previous_sys_hook = sys.excepthook

        def sys_hook(exc_type, exc_value, exc_traceback) -> None:
            _record_uncaught_exception("main_thread", exc_type, exc_value, exc_traceback)
            try:
                previous_sys_hook(exc_type, exc_value, exc_traceback)
            except Exception:
                pass

        sys.excepthook = sys_hook
        previous_thread_hook = threading.excepthook

        def thread_hook(args: threading.ExceptHookArgs) -> None:
            _record_uncaught_exception(
                "background_thread",
                args.exc_type,
                args.exc_value,
                args.exc_traceback,
                failed_thread=getattr(args.thread, "name", None),
            )
            try:
                previous_thread_hook(args)
            except Exception:
                pass

        threading.excepthook = thread_hook
        _INITIALIZED = True
        state = {
            "session_id": _SESSION_ID,
            "pid": os.getpid(),
            "status": "running",
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "executable": sys.executable,
            "frozen": bool(getattr(sys, "frozen", False)),
            "python": sys.version,
            "platform": platform.platform(),
            "event_log": event_path.name,
            "native_fault_log": fault_path.name,
        }
        _write_json(_STATE_PATH, state)
        _write_json(directory / "latest_session.json", state)
        diagnostic_event("process_start", **state)
        atexit.register(mark_clean_exit, "atexit")
        return directory
