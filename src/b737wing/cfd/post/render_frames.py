"""Render a transient EnSight sequence with pvbatch."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .video_lib import frame_png_name, select_frames, time_label
except ImportError:
    from video_lib import frame_png_name, select_frames, time_label

WALLS = ["mainwing", "horstab", "vertstab", "fuselage", "engine", "nacelle_duct", "pylon", "fairing"]
NEAR_WALL_SEED_BLOCKS = ["mainwing", "horstab", "vertstab", "engine"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ens-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--style", choices=("q", "surface", "split"), required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--stride", type=int, required=True)
    parser.add_argument("--q-level", type=float, default=2000.0)
    parser.add_argument("--wall-offset", type=float, default=0.05)
    parser.add_argument("--wall-seeds", type=int, default=3000)
    parser.add_argument("--full-seeds-per-line", type=int, default=90)
    parser.add_argument("--speed-min", type=float, default=120.0)
    parser.add_argument("--speed-max", type=float, default=300.0)
    parser.add_argument("--cam-pos", type=float, nargs=3, default=(-6.0, 26.0, 16.0))
    parser.add_argument("--cam-focal", type=float, nargs=3, default=(14.5, 0.0, -0.5))
    parser.add_argument("--cam-up", type=float, nargs=3, default=(0.3, -0.35, 0.89))
    parser.add_argument("--cam-angle", type=float, default=28.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args(argv)


def _block(reader, names, extract_block, merge_blocks):
    extracted = extract_block(Input=reader, Selectors=["/Root/" + name for name in names])
    return merge_blocks(Input=extracted)


def _reflect(source, reflect):
    full = reflect(Input=source, Plane="Y Min")
    full.CopyInput = 1
    return full


def _set_scalar_bar(display, lut, view, title, units, location, get_scalar_bar):
    display.SetScalarBarVisibility(view, True)
    bar = get_scalar_bar(lut, view)
    bar.Title = title
    bar.ComponentTitle = units
    bar.WindowLocation = location
    bar.TitleColor = [0, 0, 0]
    bar.LabelColor = [0, 0, 0]


def _speed_lut(get_color_transfer_function, speed_min, speed_max):
    lut = get_color_transfer_function("speed")
    lut.ApplyPreset("Viridis", True)
    lut.RescaleTransferFunction(speed_min, speed_max)
    lut.AutomaticRescaleRangeMode = "Never"
    return lut


def _build_split(walls, reader, near_wall_blocks, air, view, args, paraview):
    from vtkmodules.vtkCommonCore import vtkMath

    Calculator = paraview["Calculator"]
    Clip = paraview["Clip"]
    ColorBy = paraview["ColorBy"]
    ExtractSurface = paraview["ExtractSurface"]
    GetColorTransferFunction = paraview["GetColorTransferFunction"]
    GetScalarBar = paraview["GetScalarBar"]
    MaskPoints = paraview["MaskPoints"]
    Reflect = paraview["Reflect"]
    Show = paraview["Show"]
    StreamTracerWithCustomSource = paraview["StreamTracerWithCustomSource"]
    SurfaceNormals = paraview["SurfaceNormals"]
    GetOpacityTransferFunction = paraview["GetOpacityTransferFunction"]

    walls.UpdatePipeline()
    pressure = walls.GetDataInformation().GetPointDataInformation().GetArrayInformation("Static_Pressure")
    if pressure is None:
        raise RuntimeError("Static_Pressure point array is missing from the first frame walls")
    pressure_range = pressure.GetComponentRange(0)
    if not all(map(lambda value: value == value, pressure_range)) or pressure_range[0] == pressure_range[1]:
        raise RuntimeError(f"invalid Static_Pressure range: {pressure_range}")
    wall_display = Show(_reflect(walls, Reflect), view)
    ColorBy(wall_display, ("POINTS", "Static_Pressure"))
    pressure_lut = GetColorTransferFunction("Static_Pressure")
    pressure_lut.ApplyPreset("Cool to Warm", True)
    pressure_lut.RescaleTransferFunction(*pressure_range)
    pressure_lut.AutomaticRescaleRangeMode = "Never"
    _set_scalar_bar(wall_display, pressure_lut, view, "Static pressure", "Pa",
                    "Lower Right Corner", GetScalarBar)

    # Four lines seed the complete reflected flow; density is configurable.
    lines = []
    for z in (-2.0, -0.6, 0.6, 2.2):
        line = paraview["Line"](Point1=[7.0, 0.3, z], Point2=[7.0, 15.5, z])
        line.Resolution = args.full_seeds_per_line - 1
        lines.append(line)
    seed_grid = paraview["AppendDatasets"](Input=lines)
    full_flow = StreamTracerWithCustomSource(Input=air, SeedSource=seed_grid)
    full_flow.Vectors = ["POINTS", "Velocity"]
    full_flow.MaximumStreamlineLength = 22.0
    full_speed = Calculator(Input=full_flow, ResultArrayName="speed", Function="mag(Velocity)")
    reflected_flow = Reflect(Input=full_speed, Plane="Y Min")
    reflected_flow.CopyInput = 0

    # ParaView 6.1.1 MaskPoints exposes no per-filter seed property. Seed VTK's
    # global random source before using its verified uniform surface sampler.
    vtkMath.RandomSeed(1)
    near_groups = ((["mainwing"], 3.0, "main-wing"),
                   (["horstab", "vertstab"], 2.0, "empennage"))
    windows = []
    for names, fade_length, label in near_groups:
        group = _block(reader, names, paraview["ExtractBlock"], paraview["MergeBlocks"])
        group.UpdatePipeline()
        bounds = group.GetDataInformation().GetBounds()
        x_min, x_max = bounds[0], bounds[1]
        window = (x_min - fade_length, x_max + fade_length)
        print(f"{label} near-wall x window: {window[0]:.6f} to {window[1]:.6f} m", flush=True)
        windows.append((window, fade_length, label))

    wall_surface = ExtractSurface(Input=near_wall_blocks)
    normals = SurfaceNormals(Input=wall_surface)
    offset = Calculator(Input=normals, ResultArrayName="offset_points",
                        Function=f"coords+{args.wall_offset}*Normals")
    offset.CoordinateResults = 1
    sampled = MaskPoints(Input=offset)
    sampled.RandomSampling = 1
    sampled.RandomSamplingMode = "Uniform Spatial Distribution (Surface Sampling)"
    sampled.MaximumNumberofPoints = args.wall_seeds
    sampled.GenerateVertices = 1
    flow = StreamTracerWithCustomSource(Input=air, SeedSource=sampled)
    flow.Vectors = ["POINTS", "Velocity"]
    flow.IntegrationDirection = "BOTH"
    flow.MaximumStreamlineLength = 22.0
    flow.UpdatePipeline()
    seed_count = sampled.GetDataInformation().GetNumberOfPoints()
    raw_line_count = flow.GetDataInformation().GetNumberOfCells()
    if raw_line_count == 0:
        offset.Function = f"coords-{args.wall_offset}*Normals"
        sampled.UpdatePipeline()
        flow.UpdatePipeline()
        seed_count = sampled.GetDataInformation().GetNumberOfPoints()
        raw_line_count = flow.GetDataInformation().GetNumberOfCells()
    print(f"near-wall untrimmed streamlines: {raw_line_count} from {seed_count} seeds", flush=True)
    if raw_line_count == 0:
        raise RuntimeError("near-wall offset seeds produced no streamlines; normals may point out of the fluid")

    near_flows = []
    streamline_count = 0
    air_bounds = air.GetDataInformation().GetBounds()
    for window, fade_length, label in windows:
        clipped = Clip(Input=flow, ClipType="Box", Invert=1)
        clipped.ClipType.Position = [window[0], air_bounds[2], air_bounds[4]]
        clipped.ClipType.Length = [window[1] - window[0],
                                   air_bounds[3] - air_bounds[2],
                                   air_bounds[5] - air_bounds[4]]
        clipped.UpdatePipeline()
        lines_count = clipped.GetDataInformation().GetNumberOfCells()
        print(f"{label} near-wall streamlines: {lines_count} from {seed_count} seeds", flush=True)
        if lines_count == 0:
            raise RuntimeError(f"no near-wall streamlines intersect the {label} x window")
        streamline_count += lines_count
        fade = Calculator(Input=clipped, ResultArrayName="near_wall_fade",
                          Function=f"min(1,max(0,({window[1]}-coordsX)/{fade_length}))")
        fade_lut = GetOpacityTransferFunction("near_wall_fade")
        fade_lut.RescaleTransferFunction(0.0, 1.0)
        near_flows.append((fade, fade_lut))

    print(f"near-wall total: {streamline_count} from {seed_count} seeds", flush=True)
    if streamline_count == 0:
        raise RuntimeError("near-wall offset seeds produced no streamlines; normals may point out of the fluid")

    lut = _speed_lut(GetColorTransferFunction, args.speed_min, args.speed_max)
    flow_displays = []
    for flow, line_width in ((reflected_flow, 1.5),):
        speed = flow if flow is reflected_flow else Calculator(
            Input=flow, ResultArrayName="speed", Function="mag(Velocity)")
        display = Show(speed, view)
        ColorBy(display, ("POINTS", "speed"))
        display.SetScalarBarVisibility(view, False)
        display.LookupTable = lut
        display.Opacity = 1.0
        display.LineWidth = line_width
        flow_displays.append(display)
    for flow, fade_lut in near_flows:
        speed = Calculator(Input=flow, ResultArrayName="speed", Function="mag(Velocity)")
        display = Show(speed, view)
        ColorBy(display, ("POINTS", "speed"))
        display.SetScalarBarVisibility(view, False)
        display.LookupTable = lut
        display.Opacity = 1.0
        display.LineWidth = 2.5
        display.UseSeparateOpacityArray = 1
        display.OpacityArray = ["POINTS", "near_wall_fade"]
        display.OpacityTransferFunction = fade_lut
        flow_displays.append(display)
    lut.RescaleTransferFunction(args.speed_min, args.speed_max)
    lut.AutomaticRescaleRangeMode = "Never"
    _set_scalar_bar(flow_displays[0], lut, view, "Speed m/s", "",
                    "Lower Left Corner", GetScalarBar)
    return streamline_count, seed_count


def _build(args, first_case):
    # Keep ParaView imports inside execution so CLI parsing remains plain Python.
    from paraview.simple import (  # pylint: disable=import-error
        _DisableFirstRenderCameraReset, Calculator, Clip, ColorBy, Contour, ExtractBlock,
        ExtractSurface, GetActiveViewOrCreate,
        GetColorTransferFunction, GetOpacityTransferFunction, GetScalarBar, Gradient, MergeBlocks, Reflect,
        Render, Show, StreamTracer, StreamTracerWithCustomSource, SurfaceNormals,
        MaskPoints, Line, AppendDatasets, Text, EnSightReader,
    )

    _DisableFirstRenderCameraReset()
    reader = EnSightReader(CaseFileName=str(first_case))
    walls = _block(reader, WALLS, ExtractBlock, MergeBlocks)
    air = _block(reader, ["airair"], ExtractBlock, MergeBlocks)
    view = GetActiveViewOrCreate("RenderView")
    view.ViewSize = [args.width, args.height]
    view.OrientationAxesVisibility = 0
    view.UseColorPaletteForBackground = 0
    view.BackgroundColorMode = "Gradient"
    view.Background = [0.97, 0.97, 0.97]
    view.Background2 = [0.55, 0.62, 0.70]

    if args.style == "q":
        box = Clip(Input=air, ClipType="Box", Invert=1)
        box.ClipType.Position = [-2.0, 0.0, -6.0]
        box.ClipType.Length = [50.0, 22.0, 12.0]
        gradient = Gradient(Input=box)
        gradient.ScalarArray = ["POINTS", "Velocity"]
        gradient.ComputeQCriterion = 1
        gradient.ComputeVorticity = 1
        iso = Contour(Input=gradient, ContourBy=["POINTS", "Q Criterion"],
                      Isosurfaces=[args.q_level])
        iso.ComputeScalars = 1
        vort = Calculator(Input=iso, ResultArrayName="vort", Function="mag(Vorticity)")
        wall_display = Show(_reflect(walls, Reflect), view)
        ColorBy(wall_display, None)
        wall_display.DiffuseColor = [0.82, 0.83, 0.86]
        wall_display.Specular = 0.6
        iso_display = Show(_reflect(vort, Reflect), view)
        ColorBy(iso_display, ("POINTS", "vort"))
        lut = GetColorTransferFunction("vort")
        lut.ApplyPreset("Viridis", True)
        lut.RescaleTransferFunction(0.0, 400.0)
        lut.AutomaticRescaleRangeMode = "Never"
        iso_display.Specular = 0.5
        _set_scalar_bar(iso_display, lut, view, "Vorticity magnitude", "1/s",
                        "Lower Right Corner", GetScalarBar)
    elif args.style == "surface":
        walls.UpdatePipeline()
        pressure = walls.GetDataInformation().GetPointDataInformation().GetArrayInformation("Static_Pressure")
        if pressure is None:
            raise RuntimeError("Static_Pressure point array is missing from the first frame walls")
        pressure_range = pressure.GetComponentRange(0)
        if not all(map(lambda value: value == value, pressure_range)) or pressure_range[0] == pressure_range[1]:
            raise RuntimeError(f"invalid Static_Pressure range: {pressure_range}")
        wall_display = Show(_reflect(walls, Reflect), view)
        ColorBy(wall_display, ("POINTS", "Static_Pressure"))
        pressure_lut = GetColorTransferFunction("Static_Pressure")
        pressure_lut.ApplyPreset("Cool to Warm", True)
        pressure_lut.RescaleTransferFunction(*pressure_range)
        pressure_lut.AutomaticRescaleRangeMode = "Never"
        _set_scalar_bar(wall_display, pressure_lut, view, "Static pressure", "Pa",
                        "Lower Right Corner", GetScalarBar)

        speed_display_for_bar = None
        for z in (-1.0, 0.0, 1.2):
            stream = StreamTracer(Input=air, SeedType="Line")
            stream.SeedType.Point1 = [8.0, 0.3, z]
            stream.SeedType.Point2 = [8.0, 15.5, z]
            stream.SeedType.Resolution = 24
            stream.Vectors = ["POINTS", "Velocity"]
            stream.MaximumStreamlineLength = 30.0
            speed = Calculator(Input=stream, ResultArrayName="speed", Function="mag(Velocity)")
            speed_display = Show(_reflect(speed, Reflect), view)
            ColorBy(speed_display, ("POINTS", "speed"))
            speed_lut = _speed_lut(GetColorTransferFunction, args.speed_min, args.speed_max)
            speed_display.LineWidth = 2.0
            if speed_display_for_bar is None:
                speed_display_for_bar = speed_display
        _set_scalar_bar(speed_display_for_bar, speed_lut, view, "Streamline speed", "m/s",
                        "Lower Left Corner", GetScalarBar)
    else:
        near_wall_blocks = _block(reader, NEAR_WALL_SEED_BLOCKS, ExtractBlock, MergeBlocks)
        _build_split(walls, reader, near_wall_blocks, air, view, args, {
            "Calculator": Calculator,
            "Clip": Clip,
            "ColorBy": ColorBy,
            "ExtractBlock": ExtractBlock,
            "ExtractSurface": ExtractSurface,
            "GetColorTransferFunction": GetColorTransferFunction,
            "GetOpacityTransferFunction": GetOpacityTransferFunction,
            "GetScalarBar": GetScalarBar,
            "MaskPoints": MaskPoints,
            "Reflect": Reflect,
            "Show": Show,
            "StreamTracerWithCustomSource": StreamTracerWithCustomSource,
            "SurfaceNormals": SurfaceNormals,
            "MergeBlocks": MergeBlocks,
            "Line": Line,
            "AppendDatasets": AppendDatasets,
        })

    view.CameraPosition = args.cam_pos
    view.CameraFocalPoint = args.cam_focal
    view.CameraViewUp = args.cam_up
    view.CameraViewAngle = args.cam_angle
    label = Text(Text="")
    label_display = Show(label, view)
    label_display.WindowLocation = "Upper Left Corner"
    label_display.FontSize = 24
    label_display.Color = [0, 0, 0]
    return reader, view, label, Render


def main(argv=None):
    args = parse_args(argv)
    if args.stride <= 0 or args.start > args.stop:
        raise SystemExit("start must be <= stop and stride must be positive")
    frames = select_frames(args.ens_dir, args.start, args.stop, args.stride)
    if not frames:
        raise SystemExit(
            f"no EnSight frames found in {args.ens_dir} for start={args.start}, "
            f"stop={args.stop}, stride={args.stride}; start must lie on the 5-step frame grid"
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reader, view, label, render = _build(args, frames[0])
    from paraview.simple import SaveScreenshot  # pylint: disable=import-error

    done = 0
    for index, case in enumerate(frames):
        png = out_dir / frame_png_name(index)
        if args.skip_existing and png.is_file():
            continue
        reader.CaseFileName = str(case)
        reader.UpdatePipelineInformation()
        reader.UpdatePipeline()
        label.Text = time_label(int(case.stem.split("_")[-1]))
        render(view)
        SaveScreenshot(str(png), view, ImageResolution=[args.width, args.height])
        print(f"rendered {png} from {case}", flush=True)
        done += 1
    print(f"done {done}", flush=True)


if __name__ == "__main__":
    main()
