"""
Logging configuration for the Core Query Bot.

Provides a centralized logging setup with console and file handlers.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from src.config import settings


class LoggerFactory:
    """Factory for creating configured loggers."""
    
    _loggers: dict = {}
    _initialized: bool = False
    
    @classmethod
    def _ensure_log_directory(cls) -> None:
        """Ensure log directory exists before creating file handler."""
        settings.paths.logs_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_logger(cls, name: str = "cchub_corequery_bot") -> logging.Logger:
        """
        Get or create a configured logger.
        
        Args:
            name: Logger name
            
        Returns:
            Configured logger instance
        """
        if name in cls._loggers:
            return cls._loggers[name]
        
        logger = logging.getLogger(name)
        log_level = getattr(logging, settings.logging.level.upper(), logging.INFO)
        logger.setLevel(log_level)
        
        # Prevent duplicate handlers
        if logger.handlers:
            logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
        
        # File handler (lazy initialization)
        try:
            cls._ensure_log_directory()
            file_handler = logging.FileHandler(settings.paths.log_file)
            file_handler.setLevel(logging.DEBUG)
            file_format = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)
        except (OSError, IOError) as e:
            logger.warning(f"Could not create file handler: {e}")
        
        cls._loggers[name] = logger
        return logger


# Default logger instance
logger = LoggerFactory.get_logger()
