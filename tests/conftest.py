"""Make src/ imports work when pytest runs."""

import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
existing_pythonpath = os.environ.get("PYTHONPATH", "")
if str(SRC_DIR) not in existing_pythonpath.split(os.pathsep):
    os.environ["PYTHONPATH"] = (
        f"{SRC_DIR}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(SRC_DIR)
    )
