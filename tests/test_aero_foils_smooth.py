from pathlib import Path

from b737wing.aero.foils import geometry_metrics, normalise, read_dat, smooth_digitised

from b737wing.config import AEROFOIL_DIR

RAW = AEROFOIL_DIR / "737-200 33% Span Refined.dat"


def test_smooth_digitised_keeps_shape_and_removes_wiggles():
    raw = normalise(read_dat(RAW))
    smooth = smooth_digitised(raw)
    before, after = geometry_metrics(raw), geometry_metrics(smooth)
    assert abs(after["t_c"] - before["t_c"]) < 1e-3
    assert abs(after["camber"] - before["camber"]) < 5e-4
    assert len(smooth) == 201
