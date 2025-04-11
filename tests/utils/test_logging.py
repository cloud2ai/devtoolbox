import logging
import sys


def setup_test_logging():
    """Set up standardized logging configuration for all tests.
    
    This function configures logging with a standardized format and appropriate
    log levels for testing. It should be called at the beginning of each test
    module.
    
    Returns:
        logging.Logger: Configured logger instance for the calling module.
    """
    # Create a formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

    # Configure module logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    return logger 