import importlib.util
from pathlib import Path

from b737wing.config import PKG_DIR, REPO_ROOT

ROOT = REPO_ROOT
SCRIPT = PKG_DIR / "cfd" / "mesh" / "mesh_case.py"
TUNE = PKG_DIR / "cfd" / "mesh" / "tune_case.sh"


def load():
    spec = importlib.util.spec_from_file_location("mesh_case_sbes", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sbes_range_and_existing_kinds_unchanged():
    mod = load()
    assert mod.CELL_RANGES["sbes"] == (900_000, 990_000)
    assert mod.CELL_RANGES["prod"] == (800_000, 999_000)
    assert mod.CELL_RANGES["grid"] == (300_000, 999_000)
    assert mod.cell_count_in_range("sbes", 950_000)
    assert not mod.cell_count_in_range("sbes", 991_000)
    assert not mod.cell_count_in_range("sbes", 899_999)


def test_make_parameters_identical_for_all_kinds():
    mod = load()
    params = mod.make_parameters(0.94)
    assert set(params) == {"cmin", "cmax", "ang", "smax", "hex", "fh", "nl", "rate", "wing_face", "boi_size", "boi_box"}


def test_sbes_controls_fixed_sizes():
    mod = load()
    boi, face = mod.sbes_sizing_controls()
    assert boi["BOIExecution"] == "Body Of Influence" and boi["BOIFaceLabelList"] == ["junction_box"]
    assert boi["BOISize"] == 0.025
    assert face["BOIExecution"] == "Face Size" and face["BOIFaceLabelList"] == ["pylon"] and face["BOISize"] == 0.04
    assert mod.SBES_JUNCTION_BOX == (10.9, 4.6, -0.70, 12.2, 5.5, 0.05)


def test_write_boxes_two_zones(tmp_path):
    mod = load()
    mod.write_boxes(tmp_path / "a.msh", [("boi_box", (0, 0, 0, 1, 1, 1)), ("junction_box", (2, 2, 2, 3, 3, 3))])
    text = (tmp_path / "a.msh").read_text()
    assert "(10 (1 1 10 1 3)(" in text and "(39 (2 wall boi_box)())" in text and "(39 (3 wall junction_box)())" in text
    assert "(13 (3 d 18 3 3)(" in text.replace("(13 (3 d 18", "(13 (3 d 18")


def test_sbes_step_only_in_sbes_branch_and_tune_accepts_kind():
    text = SCRIPT.read_text(encoding="utf-8")
    assert text.count('if kind == "sbes":') == 2
    assert "prod|grid|sbes" in TUNE.read_text(encoding="utf-8")
