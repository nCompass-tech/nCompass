import logging as pylog

SIMPLE = "simple"
DETAILED = "detailed"

def time_fmt():
    return "%Y-%m-%d %H:%M:%S"

def register_new_logger(name, log_fmt):
    logger = pylog.getLogger(name)
    handler = pylog.StreamHandler()
    handler.setFormatter(pylog.Formatter(log_fmt, time_fmt()))
    handler.setLevel(pylog.INFO)
    logger.addHandler(handler)

def register_simple_logger():
    log_fmt = "%(levelname)s :: %(message)s"
    register_new_logger(SIMPLE, log_fmt)

def register_detailed_logger():
    log_fmt = "(%(filename)s:%(lineno)d) %(asctime)s :: %(levelname)s :: %(message)s"
    register_new_logger(DETAILED, log_fmt)
