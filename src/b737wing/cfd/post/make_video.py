"""Build an MP4 from consecutively numbered PNG frames."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

try:
    from .video_lib import ffmpeg_cmd
except ImportError:
    from video_lib import ffmpeg_cmd


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    frames_dir = Path(args.frames_dir)
    frame_pattern = re.compile(r"^frame_(\d+)\.png$")
    indices = sorted(
        int(match.group(1))
        for path in frames_dir.glob("frame_*.png")
        if (match := frame_pattern.match(path.name)) is not None
    )
    if not indices:
        raise SystemExit(f"no PNG frames found in {frames_dir}")
    index_set = set(indices)
    first_gap = next((index for index in range(indices[-1] + 1) if index not in index_set), None)
    if first_gap is not None:
        raise SystemExit(
            f"PNG frame indices are not contiguous from 0; first gap is frame_{first_gap:05d}.png"
        )
    command = ffmpeg_cmd(frames_dir, "frame_%05d.png", args.fps, args.out)
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise SystemExit(f"ffmpeg executable was not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"ffmpeg failed with exit code {exc.returncode}") from exc


if __name__ == "__main__":
    main()
