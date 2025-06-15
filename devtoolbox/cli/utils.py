"""
Common utilities for CLI commands
"""
import logging
from typing import Optional


def setup_logging(
    debug: bool = False,
    logger_name: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration for CLI commands

    Args:
        debug: Whether to enable debug mode
        logger_name: Name of the logger, defaults to 'devtoolbox'
        log_format: Custom log format string

    Returns:
        Configured logger instance
    """
    # Set default values
    if logger_name is None:
        logger_name = "devtoolbox"
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure logging
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format=log_format
    )

    # Get and configure logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    if debug:
        logger.debug("Debug mode enabled for %s", logger_name)

    return logger