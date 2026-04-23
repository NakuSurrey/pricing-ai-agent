"""
central logger config — one place to change how logs look and where they go.
every other module imports `logger` from here instead of calling loguru directly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger

# remove loguru's default handler so we control every sink
logger.remove()

# read config from env, with sensible defaults for local dev
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_TRACE_FILE = Path(os.getenv("TRACE_FILE", "./logs/trace.jsonl"))

# make sure the log dir exists — loguru won't create parents for us
_TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)

# human-readable sink for stderr — what a dev reads in the terminal
logger.add(
    sys.stderr,
    level=_LOG_LEVEL,
    format=(
        "<green>{time:HH:mm:ss}</green> "
        "<level>{level: <8}</level> "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
        "<level>{message}</level>"
    ),
    colorize=True,
    backtrace=False,
    diagnose=False,
)

# structured sink for machine reading — used by the eval harness later
logger.add(
    _TRACE_FILE,
    level=_LOG_LEVEL,
    serialize=True,          # one json object per line
    rotation="10 MB",        # start a new file when this one gets big
    retention="7 days",      # keep a week's worth
    enqueue=True,            # safe from multiple threads / processes
)

__all__ = ["logger"]
