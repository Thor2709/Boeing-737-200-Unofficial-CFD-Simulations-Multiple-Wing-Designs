"""Configuration and environment variable defaults for external tools and paths."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_DIR = REPO_ROOT / "src" / "b737wing"
AEROFOIL_DIR = REPO_ROOT / "data" / "aerofoils"
RESULTS_DIR = REPO_ROOT / "results"

# OpenVSP installation directory
B737_OPENVSP_DIR = Path(
    os.getenv("B737_OPENVSP_DIR", r"C:\Apps\OpenVSP-3.53.1-win64")
)

# flow5 executable location
B737_FLOW5_EXE = Path(
    os.getenv("B737_FLOW5_EXE", r"C:\Apps\flow5_v7.57_win64\flow5.exe")
)

# ffmpeg executable location
B737_FFMPEG = Path(
    os.getenv(
        "B737_FFMPEG",
        r"C:\Apps\ffmpeg\bin\ffmpeg.exe",
    )
)

# Ansys root installation directory
AWP_ROOT261 = Path(
    os.getenv("AWP_ROOT261", r"C:\Program Files\ANSYS Inc\ANSYS Student\v261")
)
