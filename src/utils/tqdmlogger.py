from __future__ import annotations

import logging
from pathlib import Path
from tqdm.auto import tqdm


# Custom TqdmLoggingHandler for console output
class TqdmLoggingHandler(logging.Handler):
    """
    Custom logging handler to integrate with tqdm.

    This handler ensures that log messages do not interfere with tqdm's progress bar
    by using tqdm's `write` method to output log messages.
    """

    def emit(self, record):
        """
        Emit a log record.

        Args:
            record (logging.LogRecord): The log record to be emitted.
        """
        try:
            # Format the log message
            msg = self.format(record)
            # Use tqdm.write to safely write the log message without breaking the progress bar
            tqdm.write(msg)
        except Exception:
            # Handle any errors that occur during logging
            self.handleError(record)


class TqdmLogger:
    """
    A configurable logger class that sets up logging with file and console handlers.

    This class manages logger configuration with customizable log directory, file name,
    and logging levels, integrating with tqdm for progress bar compatibility.
    """

    def __init__(
        self,
        log_dir,
        log_file_name,
        log_level=logging.INFO,
        logger_name="shared_logger",
    ):
        """
        Initialize the logger configuration.

        Args:
            log_dir (str or Path): Directory where the log file will be stored.
            log_file_name (str): Name of the log file.
            log_level (int): Logging level (default: logging.INFO).
            logger_name (str): Name of the logger instance (default: "shared_logger").
        """
        log_dir = Path(log_dir).resolve()
        log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = log_dir / log_file_name
        self.log_level = log_level
        self.logger_name = logger_name
        self.logger = None
        self._setup_logger()

    def _setup_logger(self):
        """
        Sets up the logger with file and console handlers.

        This method configures a logger that writes log messages to both a file
        and the console (integrated with tqdm).
        """
        # Define the log message format
        log_format = "%(asctime)s - %(levelname)s - %(message)s"

        # Create a file handler to write log messages to the log file
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(logging.Formatter(log_format))

        # Create a console handler using the custom TqdmLoggingHandler
        console_handler = TqdmLoggingHandler()
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(logging.Formatter(log_format))

        # Get or create a logger instance named "shared_logger"
        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(self.log_level)  # Set the logging level for the logger
        self.logger.handlers = []  # Clear any existing handlers to avoid duplicate logs
        self.logger.addHandler(file_handler)  # Add the file handler to the logger
        self.logger.addHandler(console_handler)  # Add the console handler to the logger

    def get_logger(self):
        """
        Get the configured logger instance.

        Returns:
            logging.Logger: Configured logger instance.
        """
        return self.logger

    def update_log_level(self, new_level):
        """
        Update the logging level for the logger and all its handlers.

        Args:
            new_level (int): New logging level.
        """
        self.log_level = new_level
        self.logger.setLevel(new_level)
        for handler in self.logger.handlers:
            handler.setLevel(new_level)

    def add_handler(self, handler):
        """
        Add a custom handler to the logger.

        Args:
            handler (logging.Handler): Handler to add to the logger.
        """
        self.logger.addHandler(handler)

    def remove_handlers(self):
        """Remove all handlers from the logger."""
        self.logger.handlers = []

    def __enter__(self):
        """Context manager entry."""
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup handlers."""
        self.remove_handlers()
