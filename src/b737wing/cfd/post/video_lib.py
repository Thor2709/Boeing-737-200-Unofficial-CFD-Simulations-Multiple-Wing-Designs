"""Small, ParaView-independent helpers for the transient video pipeline."""

from __future__ import annotations

import re
from pathlib import Path

from b737wing.config import B737_FFMPEG

_FRAME_RE = re.compile(r"^frame_(\d+)\.encas$")
FFMPEG = str(B737_FFMPEG)


def select_frames(ens_dir, start, stop, stride):
    """Return matching EnSight cases in step order, within an inclusive range."""
    if stride <= 0:
        raise ValueError("stride must be a positive integer")
    frames = []
    for path in Path(ens_dir).glob("frame_*.encas"):
        match = _FRAME_RE.match(path.name)
        if match is None:
            continue
        step = int(match.group(1))
        if start <= step <= stop and (step - start) % stride == 0:
            frames.append((step, path))
    return [path for _, path in sorted(frames)]


def time_label(step, dt=1e-4, t0_step=1000):
    """Format a timestep as milliseconds relative to the initial step."""
    milliseconds = (step - t0_step) * dt * 1000.0
    value = f"{milliseconds:.1f}".rstrip("0").rstrip(".")
    return f"t = {value} ms"


def ffmpeg_cmd(frames_dir, pattern, fps, out_mp4, crf=18):
    """Build an ffmpeg argv for an H.264 video with even-sized frames."""
    frame_pattern = str(Path(frames_dir) / pattern)
    scale = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    return [
        FFMPEG, "-y", "-framerate", str(fps), "-i", frame_pattern,
        "-vf", scale, "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-r", str(fps), "-crf", str(crf), str(out_mp4),
    ]


def frame_png_name(index):
    return f"frame_{index:05d}.png"
