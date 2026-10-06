from __future__ import annotations

import os
from pathlib import Path

from b737wing import config


def test_config_defaults():
    assert config.B737_OPENVSP_DIR == Path(r"C:\Apps\OpenVSP-3.53.1-win64")
    assert config.B737_FLOW5_EXE == Path(r"C:\Apps\flow5_v7.57_win64\flow5.exe")
    assert config.B737_FFMPEG == Path(r"C:\Apps\ffmpeg\bin\ffmpeg.exe")
    assert config.AWP_ROOT261 == Path(r"C:\Program Files\ANSYS Inc\ANSYS Student\v261")


def test_config_env_overrides(monkeypatch):
    monkeypatch.setenv("B737_OPENVSP_DIR", "/opt/openvsp")
    monkeypatch.setenv("B737_FLOW5_EXE", "/usr/bin/flow5")
    monkeypatch.setenv("B737_FFMPEG", "/usr/bin/ffmpeg")
    monkeypatch.setenv("AWP_ROOT261", "/opt/ansys/v261")

    import importlib
    importlib.reload(config)

    assert config.B737_OPENVSP_DIR == Path("/opt/openvsp")
    assert config.B737_FLOW5_EXE == Path("/usr/bin/flow5")
    assert config.B737_FFMPEG == Path("/usr/bin/ffmpeg")
    assert config.AWP_ROOT261 == Path("/opt/ansys/v261")

    # Clean up environment and reload to restore defaults
    monkeypatch.delenv("B737_OPENVSP_DIR")
    monkeypatch.delenv("B737_FLOW5_EXE")
    monkeypatch.delenv("B737_FFMPEG")
    monkeypatch.delenv("AWP_ROOT261")
    importlib.reload(config)
