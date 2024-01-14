from .logging import *
import logging as pylog

pylog.basicConfig(level = pylog.INFO)

register_simple_logger()
register_detailed_logger()
