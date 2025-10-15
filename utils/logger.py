"""
Logging utilities for the project
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class Logger:
    """Custom logger for the project"""

    def __init__(
        self,
        name: str,
        log_dir: Optional[Path] = None,
        level: int = logging.INFO,
        console: bool = True,
        file: bool = True
    ):
        """
        Initialize logger

        Args:
            name: Logger name
            log_dir: Directory to save log files
            level: Logging level
            console: Whether to log to console
            file: Whether to log to file
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if file and log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = log_dir / f"{name}_{timestamp}.log"

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)

    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)

    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)


def get_logger(
    name: str,
    log_dir: Optional[Path] = None,
    level: int = logging.INFO
) -> Logger:
    """
    Get a logger instance

    Args:
        name: Logger name
        log_dir: Directory to save log files
        level: Logging level

    Returns:
        Logger instance
    """
    return Logger(name, log_dir, level)
