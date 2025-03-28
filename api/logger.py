import logging
import os
import sys
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure logging


def setup_logger(name, log_file, level=logging.INFO):
    """Function to set up a logger with file and console handlers"""
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create file handler
    file_handler = RotatingFileHandler(
        f"logs/{log_file}",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Create loggers
app_logger = setup_logger('app', 'app.log')
api_logger = setup_logger('api', 'api.log')
db_logger = setup_logger('db', 'db.log')
model_logger = setup_logger('model', 'model.log')
error_logger = setup_logger('error', 'error.log', level=logging.ERROR)

# Performance monitoring


class PerformanceTimer:
    def __init__(self, logger, operation_name):
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting operation: {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        if exc_type:
            self.logger.error(
                f"Operation {self.operation_name} failed after {elapsed_time:.2f} seconds")
            error_logger.error(
                f"Exception in {self.operation_name}: {exc_val}", exc_info=True)
        else:
            self.logger.info(
                f"Operation {self.operation_name} completed in {elapsed_time:.2f} seconds")
