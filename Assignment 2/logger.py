import logging
import os
import sys

# This makes the ANSI escape sequence get processed correctly, such as clear line
# https://stackoverflow.com/a/64222858
os.system("")

def addLoggingLevel(levelName, levelNum, methodName=None):
    """
    Comprehensively adds a new logging level to the `logging` module and the
    currently configured logging class.

    `levelName` becomes an attribute of the `logging` module with the value
    `levelNum`. `methodName` becomes a convenience method for both `logging`
    itself and the class returned by `logging.getLoggerClass()` (usually just
    `logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
    used.

    To avoid accidental clobberings of existing attributes, this method will
    raise an `AttributeError` if the level name is already an attribute of the
    `logging` module or if the method name is already present 

    See https://stackoverflow.com/a/56944256/12132063

    Example
    -------
    >>> addLoggingLevel('TRACE', logging.DEBUG - 5)
    >>> logging.getLogger(__name__).setLevel("TRACE")
    >>> logging.getLogger(__name__).trace('that worked')
    >>> logging.trace('so did this')
    >>> logging.TRACE
    5

    """
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName):
       raise AttributeError('{} already defined in logging module'.format(levelName))
    if hasattr(logging, methodName):
       raise AttributeError('{} already defined in logging module'.format(methodName))
    if hasattr(logging.getLoggerClass(), methodName):
       raise AttributeError('{} already defined in logger class'.format(methodName))

    # This method was inspired by the answers to Stack Overflow post
    # http://stackoverflow.com/q/2183233/2988730, especially
    # http://stackoverflow.com/a/13638084/2988730
    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)
    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)

# Add custom levels
addLoggingLevel("PROGRESS", logging.INFO - 5)
addLoggingLevel("STATUS", logging.INFO - 5)

class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    blue = "\x1b[34;20m"
    green = "\x1b[32;20m"
    bold_green = "\x1b[32;1m"
    reset = "\x1b[0m"

    newline = "\n"
    reset_cursor = "\033[K\r"
    clearline = "\x1b[K"

    format = f"{clearline}%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset + newline,
        logging.PROGRESS: green + format + reset + reset_cursor,
        logging.INFO: blue + format + reset + newline,
        logging.STATUS: bold_green + format + reset + newline,
        logging.WARNING: yellow + format + reset + newline,
        logging.ERROR: red + format + reset + newline,
        logging.CRITICAL: bold_red + format + reset + newline
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

# create logger with name of the script initiator without the .py extensionpython log
logger = logging.getLogger(sys.argv[0][:-3])
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.terminator = "" # Remove the newline terminator, the newline sequence is then added back in the custom formatter

ch.setFormatter(CustomFormatter())
logger.addHandler(ch)

if __name__ == "__main__":
    
    import time

    logger.debug("debug message")
    
    logger.status("status message")
    for i in range(5):
        logger.progress(f"progress message {i}")

        time.sleep(1)

    logger.info("info message")

    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")
