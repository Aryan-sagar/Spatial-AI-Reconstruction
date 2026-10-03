"""Logging plus a stage timer that records start/end/duration and counts."""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path

_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(level: str = "INFO", log_file: str | Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=getattr(logging, level.upper()), format=_FMT, handlers=handlers, force=True)


@contextmanager
def stage(name: str, records: list | None = None, **counts):
    """Time a pipeline stage. `records` (if given) receives a dict per stage."""
    log = logging.getLogger("stage")
    t0 = time.time()
    log.info("start %s %s", name, counts or "")
    info = {"stage": name, "start_time": t0, **counts}
    try:
        yield info
    finally:
        t1 = time.time()
        info.update(end_time=t1, duration_sec=t1 - t0)
        log.info("end %s (%.3fs)", name, t1 - t0)
        if records is not None:
            records.append(info)
