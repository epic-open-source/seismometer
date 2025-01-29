import datetime
import logging
from typing import Optional

logging.basicConfig()


def init_logger(name: str = "seismometer") -> None:
    logger = logging.getLogger(name)
    add_log_formatter(logger)

    return logger


def remove_default_handler(logger: Optional[logging.Logger] = None) -> None:
    """
    Removes the default logging handler.

    Parameters
    ----------
    logger : Optional[Logger], optional
        Descriptor of the logger do modify, by default None.
        When None, the root logger is modified.
    """
    logger = logger or logging.getLogger()
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])


def add_log_formatter(logger: logging.Logger):
    """
    Adds a formatter to the logger that includes timestamp info.

    Parameters
    ----------
    logger : logging.Logger
        The logger to add formatting to.
    """
    # Remove root-handler / default
    remove_default_handler()
    # Remove default handler for seismometer - make safe to call multiple times
    remove_default_handler(logger)

    handler = logging.StreamHandler()
    formatter = TimeFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def now() -> str:
    """
    Get the current time in UTC as a string.

    Returns
    -------
    str
        Current UTC time.
    """
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class TimeFormatter(logging.Formatter):
    """
    A logging formatter that adds UTC timestamp information to the log
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Formats the message from the logging record to include a timestamp.

        Parameters
        ----------
        record : logging.LogRecord
            The logging record to format.

        Returns
        -------
        str
            The formatted message.
        """
        formatted_message = f"[{now()} UTC] {record.levelname}: {record.getMessage()}"
        return formatted_message
