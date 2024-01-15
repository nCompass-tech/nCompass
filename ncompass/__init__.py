import sys
from pathlib import Path

from . import models
from . import loaders
from . import profiling

root_dir = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(root_dir))
