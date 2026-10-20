"""Project logger: `from net.utils.logger import logger; logger.info(...)`.

Prints to the terminal and, once an experiment starts, also to its run.log.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger("net")


def setup_logger(log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """(Re)attach terminal + optional file handlers; uncaught exceptions get logged too."""
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    sys.excepthook = lambda *exc: logger.critical("uncaught exception", exc_info=exc)
    return logger
