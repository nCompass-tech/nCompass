import sys
import logging
from pathlib import Path
import ncompass.internal.logging as nclog

root_dir = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(root_dir))
