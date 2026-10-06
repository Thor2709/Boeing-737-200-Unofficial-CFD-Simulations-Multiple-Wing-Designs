"""Imperial Breguet range for the P7 cruise cases."""
from __future__ import annotations

import math


TSFC_PER_S = 1.662e-4
RHO_SLUG_FT3 = 0.000889
S_FT2 = 979.9
W0_LBF = 115500.0
W1_LBF = 83474.0
FEET_PER_NAUTICAL_MILE = 6076.11549


def breguet_range_nmi(cl: float, cd: float, case: str | None = None) -> dict:
    """Return ``range_nmi`` and an optional reason when range is unavailable."""
    if case is not None and str(case).upper() == "C3":
        return {"range_nmi": None, "reason": "C3 is the take-off case; Breguet range is cruise-only."}
    if not math.isfinite(cl) or not math.isfinite(cd) or cl <= 0 or cd <= 0:
        return {"range_nmi": None, "reason": "CL and CD must be positive finite cruise coefficients."}
    distance_ft = ((1.0 / TSFC_PER_S) * (math.sqrt(cl) / cd)
                   * (2.0 * math.sqrt(2.0) / math.sqrt(RHO_SLUG_FT3 * S_FT2))
                   * (math.sqrt(W0_LBF) - math.sqrt(W1_LBF)))
    return {"range_nmi": distance_ft / FEET_PER_NAUTICAL_MILE, "reason": None}

