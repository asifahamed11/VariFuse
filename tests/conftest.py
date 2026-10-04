from __future__ import annotations

import sys
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "src"
PROJECT_ROOT = SOURCE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
