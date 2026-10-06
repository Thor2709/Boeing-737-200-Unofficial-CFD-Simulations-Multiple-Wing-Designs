"""Render the newest EnSight snapshot of a transient run to a PNG (pvbatch).

Full aircraft via Reflect about y = 0: walls coloured by static pressure,
streamlines from a seed line ahead of the wing coloured by velocity magnitude.
Reads files only; never touches the running Fluent session.
usage: pvbatch render_latest.py <ensight_dir> <out_png> [frame.encas]
"""
import glob
import os
import sys

from paraview.simple import (Calculator, ColorBy, ExtractBlock, GetActiveViewOrCreate,
                             GetColorTransferFunction, MergeBlocks, Reflect, Render,
                             SaveScreenshot, Show, StreamTracer, EnSightReader)

WALLS = ["mainwing", "horstab", "vertstab", "fuselage", "engine", "nacelle_duct", "pylon", "fairing"]

ens_dir, out_png = sys.argv[1], sys.argv[2]
if len(sys.argv) > 3:
    case = sys.argv[3]
else:
    frames = sorted(glob.glob(os.path.join(ens_dir, "frame_*.encas")))
    if not frames:
        sys.exit("no frames yet")
    case = frames[-2] if len(frames) > 1 else frames[-1]  # newest may still be writing
reader = EnSightReader(CaseFileName=case)

def block(names):
    eb = ExtractBlock(Input=reader, Selectors=["/Root/" + n for n in names])
    return MergeBlocks(Input=eb)

walls = block(WALLS)
air = block(["airair"])  # hierarchy names drop "-"
for src, label in ((walls, "walls"), (air, "air")):
    src.UpdatePipeline()
    print(label, "cells", src.GetDataInformation().GetNumberOfCells(), flush=True)

stream = StreamTracer(Input=air, SeedType="Line")
stream.SeedType.Point1 = [8.0, 0.5, -0.1]
stream.SeedType.Point2 = [8.0, 15.5, 0.3]
stream.SeedType.Resolution = 45
stream.Vectors = ["POINTS", "Velocity"]
stream.MaximumStreamlineLength = 30.0
speed = Calculator(Input=stream, ResultArrayName="speed", Function="mag(Velocity)")

view = GetActiveViewOrCreate("RenderView")
view.ViewSize = [1600, 900]
view.Background = [1, 1, 1]
try:
    view.UseColorPaletteForBackground = 0
except Exception:
    pass
view.OrientationAxesVisibility = 0

for src, field, rng in ((walls, "Static_Pressure", None), (speed, "speed", (0.0, 320.0))):
    full = Reflect(Input=src, Plane="Y Min")
    full.CopyInput = 1
    disp = Show(full, view)
    ColorBy(disp, ("POINTS", field))
    lut = GetColorTransferFunction(field)
    lut.ApplyPreset("Jet" if field == "speed" else "Cool to Warm", True)
    if rng:
        lut.RescaleTransferFunction(*rng)
    else:
        disp.RescaleTransferFunctionToDataRange(False, True)
    disp.SetScalarBarVisibility(view, True)
    if field == "speed":
        disp.LineWidth = 1.5

view.CameraPosition = [-4.0, -19.0, 13.0]
view.CameraFocalPoint = [15.0, 0.0, -0.5]
view.CameraViewUp = [0.25, 0.3, 0.92]
view.CameraViewAngle = 30
Render(view)
SaveScreenshot(out_png, view, ImageResolution=[1600, 900])
print("saved", out_png, "from", os.path.basename(case), flush=True)
