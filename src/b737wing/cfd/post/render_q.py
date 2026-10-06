"""Q-criterion iso-surfaces of one EnSight snapshot (pvbatch), mirrored to the full aircraft.
usage: pvbatch render_q.py <frame.encas> <out_png> [q_level]
"""
import sys

from paraview.simple import (Calculator, ColorBy, Contour, ExtractBlock, GetActiveViewOrCreate,
                             GetColorTransferFunction, Gradient, MergeBlocks, Reflect, Render,
                             SaveScreenshot, Show, EnSightReader, Clip)

WALLS = ["mainwing", "horstab", "vertstab", "fuselage", "engine", "nacelle_duct", "pylon", "fairing"]
case, out_png = sys.argv[1], sys.argv[2]
q_level = float(sys.argv[3]) if len(sys.argv) > 3 else 2000.0
reader = EnSightReader(CaseFileName=case)

def block(names):
    return MergeBlocks(Input=ExtractBlock(Input=reader, Selectors=["/Root/" + n for n in names]))

walls = block(WALLS)
air = block(["airair"])
# keep only the region around and behind the aircraft (speeds up the gradient)
box = Clip(Input=air, ClipType="Box", Invert=1)
box.ClipType.Position = [-2.0, 0.0, -6.0]
box.ClipType.Length = [50.0, 22.0, 12.0]
grad = Gradient(Input=box)
grad.ScalarArray = ["POINTS", "Velocity"]
grad.ComputeQCriterion = 1
grad.ComputeVorticity = 1
iso = Contour(Input=grad, ContourBy=["POINTS", "Q Criterion"], Isosurfaces=[q_level])
iso.ComputeScalars = 1
vort = Calculator(Input=iso, ResultArrayName="vort", Function="mag(Vorticity)")
iso.UpdatePipeline()
print("iso cells", iso.GetDataInformation().GetNumberOfCells(), flush=True)

view = GetActiveViewOrCreate("RenderView")
view.ViewSize = [1920, 1080]
try:
    view.UseColorPaletteForBackground = 0
    view.BackgroundColorMode = "Gradient"
except Exception:
    pass
view.Background = [0.97, 0.97, 0.97]
view.Background2 = [0.55, 0.62, 0.70]
view.OrientationAxesVisibility = 0

wf = Reflect(Input=walls, Plane="Y Min"); wf.CopyInput = 1
wd = Show(wf, view)
ColorBy(wd, None)
wd.DiffuseColor = [0.82, 0.83, 0.86]
wd.Specular = 0.6

vf = Reflect(Input=vort, Plane="Y Min"); vf.CopyInput = 1
vd = Show(vf, view)
ColorBy(vd, ("POINTS", "vort"))
lut = GetColorTransferFunction("vort")
lut.ApplyPreset("Viridis (matplotlib)", True)
lut.RescaleTransferFunction(0.0, 400.0)
vd.Specular = 0.5
vd.SetScalarBarVisibility(view, False)

view.CameraPosition = [42.0, -26.0, 16.0]
view.CameraFocalPoint = [16.0, 0.0, 0.0]
view.CameraViewUp = [-0.25, 0.25, 0.93]
view.CameraViewAngle = 30
Render(view)
SaveScreenshot(out_png, view, ImageResolution=[1920, 1080])
print("saved", out_png, flush=True)
