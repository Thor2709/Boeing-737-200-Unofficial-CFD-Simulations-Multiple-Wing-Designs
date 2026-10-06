"""Single source of truth for the wing-only B737-200 cases."""
from __future__ import annotations

from copy import deepcopy

ROOT_SECTIONS = [
    {"y": 0.0, "chord": 4.79384, "x": 0.0, "dihedral": 6.0, "twist": 1.0},
    {"y": 4.69852, "chord": 3.73919, "x": 2.45462, "dihedral": 6.0, "twist": 1.0},
    {"y": 9.39704, "chord": 2.68455, "x": 4.90923, "dihedral": 6.0, "twist": 1.0},
    {"y": 14.09556, "chord": 1.62991, "x": 7.36385, "dihedral": 0.0, "twist": 1.0},
]

CRUISE = {"velocity": 224.33, "rho": 0.4582, "nu": 3.249e-5, "target_cl": 0.4895}
TAKEOFF = {"velocity": 74.17, "rho": 1.225, "nu": 1.4607e-5, "target_cl": 1.675}

CASES = {
    "F1": {"foils": ["A0012"] * 4, "twist": "basis", "condition": "cruise", "flight": CRUISE},
    "F2": {"foils": ["AOPT"] * 4, "twist": "basis", "condition": "cruise", "flight": CRUISE},
    "F3": {"foils": ["AOPT"] * 4, "twist": "basis", "condition": "takeoff", "flight": TAKEOFF, "flap": True},
    "F4": {"foils": ["AOPT"] * 4, "twist": "linear", "condition": "cruise", "flight": CRUISE},
    "F5": {"foils": ["B737A", "B737B", "B737C", "B737D"], "twist": "basis", "condition": "cruise", "flight": CRUISE},
}


def case_config(name: str) -> dict:
    key = name.upper()
    if key not in CASES:
        raise ValueError(f"unknown case {name!r}; expected one of {', '.join(CASES)}")
    result = deepcopy(CASES[key])
    result["name"] = key
    result["plane_name"] = f"B737-200 {key} Wing"
    result["polar_name"] = key  # flow5 names output polars after the analysis file stem (verified 2026-10-03)
    result["sections"] = deepcopy(ROOT_SECTIONS)
    for section, foil in zip(result["sections"], result["foils"]):
        section["foil"] = foil
    if result["twist"] == "linear":
        y_tip = ROOT_SECTIONS[-1]["y"]
        for section in result["sections"]:
            section["twist"] = -0.459177 + (0.459175 + 0.459177) * section["y"] / y_tip
    return result


def case_names() -> tuple[str, ...]:
    return tuple(CASES)
