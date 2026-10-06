"""Flight conditions and reference values for C1-C5."""

from __future__ import annotations

import math

from b737wing.flow5.cases import case_config

R_AIR = 287.05
GAMMA_AIR = 1.4
TEMPERATURES_K = {"cruise": 228.7, "takeoff": 288.15}
S_REF_M2 = 91.04
MAC_M = 3.403
X_REF_M = 12.760
Z_REF_M = 0.0


def condition_for_case(case: str) -> dict:
    """Resolve flow5's matching case condition for a CFD case name."""
    key = str(case).upper()
    if len(key) != 2 or key[0] != "C" or key[1] not in "12345":
        raise ValueError(f"unknown case {case!r}; expected C1, C2, C3, C4, or C5")
    source = case_config("F" + key[1])
    condition = source["condition"]
    flight = source["flight"]
    temperature = TEMPERATURES_K[condition]
    velocity = float(flight["velocity"])
    density = float(flight["rho"])
    pressure = density * R_AIR * temperature
    return {
        "case": key,
        "condition": condition,
        "velocity_m_s": velocity,
        "density_kg_m3": density,
        "temperature_K": temperature,
        "pressure_Pa": pressure,
        "mach": velocity / math.sqrt(GAMMA_AIR * R_AIR * temperature),
        "target_cl": float(flight["target_cl"]),
    }


def reference_values(condition: dict) -> dict:
    """Reference values for the half mesh and full-aircraft coefficients."""
    return {
        "area_m2": S_REF_M2 / 2.0,
        "length_m": MAC_M,
        "density_kg_m3": condition["density_kg_m3"],
        "pressure_Pa": condition["pressure_Pa"],
        "temperature_K": condition["temperature_K"],
        "velocity_m_s": condition["velocity_m_s"],
        "moment_center_m": [X_REF_M, 0.0, Z_REF_M],
        "moment_axis": [0.0, 1.0, 0.0],
    }
