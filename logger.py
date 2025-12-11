"""Logging configuration for API errors."""
import logging
import os

# Configure logger for API errors
def setup_api_error_logger():
    """
    Set up logger for API errors that logs to both file and console.
    
    Returns:
        Logger instance configured for API error logging
    """
    logger = logging.getLogger('api_errors')
    logger.setLevel(logging.ERROR)
    
    # Prevent duplicate handlers if logger is already configured
    if logger.handlers:
        return logger
    
    # Create formatter with timestamp
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler - logs to api_errors.log
    log_file = os.path.join(os.path.dirname(__file__), 'api_errors.log')
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler - logs to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# Create the logger instance
api_error_logger = setup_api_error_logger()
