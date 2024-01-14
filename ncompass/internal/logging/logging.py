import sys
from typing import Type
import logging as pylog

SIMPLE: str = "simple"
DETAILED: str = "detailed"

def time_fmt():
    return "%Y-%m-%d %H:%M:%S"

def register_new_logger(name: str, log_fmt: str):
    logger = pylog.getLogger(name)
    logger.propagate = False
    handler = pylog.StreamHandler(stream=sys.stdout)
    handler.setLevel(pylog.DEBUG)
    handler.setFormatter(pylog.Formatter(log_fmt, time_fmt()))
    logger.addHandler(handler)

def register_simple_logger():
    log_fmt = "%(levelname)s :: %(message)s"
    register_new_logger(SIMPLE, log_fmt)

def register_detailed_logger():
    log_fmt = "(%(filename)s:%(lineno)d) %(asctime)s :: %(levelname)s :: %(message)s"
    register_new_logger(DETAILED, log_fmt)

def DEBUG(msg: str, logger_type:str = DETAILED):
    logger = pylog.getLogger(logger_type)
    logger.debug(msg)

def INFO(msg: str, logger_type: str = DETAILED):
    logger = pylog.getLogger(logger_type)
    logger.info(msg)

def WARNING(msg: str, logger_type: str = DETAILED):
    logger = pylog.getLogger(logger_type)
    logger.warning(msg)

def ERROR(msg: str, exception: Type[Exception]= None, logger_type: str = DETAILED):
    logger = pylog.getLogger(logger_type)
    logger.error(msg)
    if exception is not None:
        raise exception(msg)
