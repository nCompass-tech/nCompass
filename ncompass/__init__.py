import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(root_dir))
