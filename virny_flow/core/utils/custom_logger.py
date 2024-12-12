import logging
from colorama import Fore, Style


# Define color mappings for log levels
LOG_COLORS = {
    "DEBUG": Fore.BLUE,
    "INFO": Fore.GREEN,
    "WARNING": Fore.YELLOW,
    "ERROR": Fore.RED,
    "CRITICAL": Fore.MAGENTA,
}

FIELD_COLORS = {
    "asctime": Fore.LIGHTCYAN_EX,
    "filename": Fore.LIGHTCYAN_EX,
    "name": Fore.LIGHTCYAN_EX,
    "lineno": Fore.LIGHTCYAN_EX,
}

class ColorFormatter(logging.Formatter):
    def format(self, record):
        # Add color to log level
        log_color = LOG_COLORS.get(record.levelname, Fore.WHITE)
        record.levelname = f"{log_color}{record.levelname}{Style.RESET_ALL}"

        # Add color to asctime (requires setting it manually if not already set)
        if not hasattr(record, 'asctime'):
            record.asctime = self.formatTime(record, self.datefmt)
        record.asctime = f"{FIELD_COLORS['asctime']}{record.asctime}{Style.RESET_ALL}"

        # Add color to filename
        record.filename = f"{FIELD_COLORS['filename']}{record.filename}{Style.RESET_ALL}"

        # Add color to logger name
        record.name = f"{FIELD_COLORS['name']}{record.name}{Style.RESET_ALL}"

        # Add color to line number
        record.lineno = f"{FIELD_COLORS['lineno']}{record.lineno}{Style.RESET_ALL}"

        # Format the message
        return super().format(record)


def get_logger(logger_name):
    # Define a custom logging format
    log_format = "[%(asctime)s][%(name)s][%(filename)-18s:%(lineno)s][%(levelname)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Set up the logger
    formatter = ColorFormatter(fmt=log_format, datefmt=date_format)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False

    return logger
