import sys
from pathlib import Path
import diskcache as dc

MAIN_DIR = Path(sys.argv[0]).resolve().parent  
PROJECT_DIR = MAIN_DIR.parent
DEFAULT_CACHE = dc.Cache(str(PROJECT_DIR / "cache"))  
