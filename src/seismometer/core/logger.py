import datetime
import logging


def set_default_logger_config() -> None:
    """
    Adds the basic configuration for logging.
    """
    logging.basicConfig()


def remove_default_log_handler() -> None:
    """
    Removes the default logging handlers.
    """
    root_log = logging.getLogger()
    while root_log.hasHandlers():
        root_log.removeHandler(root_log.handlers[0])


def add_log_formatter(logger: logging.Logger):
    """
    Adds a formatter to the logger that includes timestamp info.

    Parameters
    ----------
    logger : logging.Logger
        The logger to add formatting to.
    """
    remove_default_log_handler()

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
