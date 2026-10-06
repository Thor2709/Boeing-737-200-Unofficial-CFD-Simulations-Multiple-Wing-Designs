import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


from b737wing.config import PKG_DIR, REPO_ROOT

ROOT = REPO_ROOT
MESH_SCRIPT = PKG_DIR / "cfd" / "mesh" / "mesh_case.py"
GEOM_SCRIPT = PKG_DIR / "cfd" / "geom" / "dom_lib.py"


def load_mesh_helpers():
    spec = importlib.util.spec_from_file_location("mesh_case_helpers", MESH_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_geometry_helpers():
    tree = ast.parse(GEOM_SCRIPT.read_text(encoding="utf-8"))
    labels = next(
        node for node in tree.body
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "LABELS" for target in node.targets)
    )
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"require_labels", "check_result_issue_count", "require_zero_issues"}
    ]
    namespace = {"LABELS": ast.literal_eval(labels.value), "np": SimpleNamespace(integer=int)}
    exec(compile(ast.Module(body=[*functions], type_ignores=[]), str(GEOM_SCRIPT), "exec"), namespace)
    return namespace


@pytest.mark.parametrize(
    ("kind", "cells", "accepted"),
    [
        ("prod", 799_999, False),
        ("prod", 800_000, True),
        ("prod", 999_000, True),
        ("prod", 999_001, False),
        ("prod", 1_048_576, False),
        ("grid", 299_999, False),
        ("grid", 300_000, True),
        ("grid", 999_000, True),
        ("grid", 999_001, False),
        ("grid", 1_048_576, False),
    ],
)
def test_mesh_cell_count_ranges(kind, cells, accepted):
    assert load_mesh_helpers().cell_count_in_range(kind, cells) is accepted


def test_case_and_tag_validation():
    mesh = load_mesh_helpers()
    mesh.validate_case_tag("C1", "prod_iH2p5")
    for case, tag in (("C6", "prod"), ("C1", "../prod"), ("C1", "bad-tag"), ("C1", "")):
        with pytest.raises(ValueError):
            mesh.validate_case_tag(case, tag)


def test_resolved_rm_target_stays_inside_case_output(tmp_path):
    mesh = load_mesh_helpers()
    allowed = mesh.safe_output_target(tmp_path, "C1", "prod")
    assert allowed == (tmp_path / "cfd_v2" / "mesh" / "C1" / "prod").resolve()
    case_root = tmp_path / "cfd_v2" / "mesh" / "C1"
    case_root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (case_root / "prod").symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("directory symlinks are unavailable in this environment")
    with pytest.raises(ValueError, match="not inside"):
        mesh.safe_output_target(tmp_path, "C1", "prod")


def test_required_labels_raise_when_fairing_is_missing():
    helpers = load_geometry_helpers()
    groups = {label: [object()] for label in helpers["LABELS"] if label != "fairing"}
    with pytest.raises(RuntimeError, match="fairing"):
        helpers["require_labels"](groups)


def test_none_check_result_is_failure():
    helpers = load_geometry_helpers()
    assert helpers["check_result_issue_count"](None) is None
    with pytest.raises(RuntimeError, match="zero issues"):
        helpers["require_zero_issues"]("find_missing_faces", None)
