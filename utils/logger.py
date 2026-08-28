"""Privacy-friendly event logging for Smart Eye Monitor."""

import logging
from pathlib import Path


class EventLogger:
    """Write state changes to one local text log file."""

    def __init__(self, log_file: Path) -> None:
        log_file.parent.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("smart_eye_monitor.events")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if not self.logger.handlers:
            handler = logging.FileHandler(log_file, encoding="utf-8")
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            self.logger.addHandler(handler)

    def log(self, message: str) -> None:
        """Record a human-readable event without any camera data."""
        self.logger.info(message)
