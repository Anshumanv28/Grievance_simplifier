"""
Shared logging setup and stage timer for the pipeline.
"""

import contextlib
import logging
import sys
import time
from pathlib import Path
from typing import Generator

LOG_DIR = "logs"


def setup_logging(log_dir: str = LOG_DIR, log_level: int = logging.INFO) -> None:
    """Configure file and console logging. Call once at pipeline entry."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / "pipeline.log"
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=log_level,
        format=fmt,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


@contextlib.contextmanager
def stage_timer(logger: logging.Logger, stage_name: str) -> Generator[float, None, None]:
    """
    Context manager that logs stage start and end with elapsed time.
    Yields elapsed seconds on exit.
    """
    start = time.monotonic()
    logger.info("%s started.", stage_name)
    try:
        yield 0.0
    finally:
        elapsed = time.monotonic() - start
        logger.info("%s completed in %.2fs", stage_name, elapsed)


def log_stage_samples(
    logger: logging.Logger,
    stage_label: str,
    samples: list,
    max_len: int = 80,
) -> None:
    """Log 2-3 sample entries for a stage. Truncate long strings to max_len."""
    for i, s in enumerate(samples[:3]):
        if isinstance(s, dict):
            parts = [f"{k}: {str(v)[:max_len]}..." if len(str(v)) > max_len else f"{k}: {v}" for k, v in s.items()]
            logger.info("[%s sample %s] %s", stage_label, i + 1, " | ".join(parts))
        else:
            logger.info("[%s sample %s] %s", stage_label, i + 1, str(s)[:max_len * 2])
