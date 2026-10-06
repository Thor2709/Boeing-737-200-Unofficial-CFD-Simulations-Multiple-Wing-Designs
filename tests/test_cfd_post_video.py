from unittest.mock import patch

import pytest

from b737wing.cfd.post import make_video, video_lib


def test_select_frames_filters_range_stride_and_sorts(tmp_path):
    for step in (1010, 1000, 1020, 1005, 1030):
        (tmp_path / f"frame_{step:05d}.encas").touch()
    (tmp_path / "other.encas").touch()
    assert [path.name for path in video_lib.select_frames(tmp_path, 1000, 1020, 10)] == [
        "frame_01000.encas", "frame_01010.encas", "frame_01020.encas"]


@pytest.mark.parametrize("step, expected", [(1000, "t = 0 ms"), (1125, "t = 12.5 ms"),
                                              (3000, "t = 200 ms")])
def test_time_label(step, expected):
    assert video_lib.time_label(step) == expected


def test_ffmpeg_command_has_video_settings_and_output(tmp_path):
    command = video_lib.ffmpeg_cmd(tmp_path, "frame_%05d.png", 25, tmp_path / "out.mp4")
    assert command[command.index("-framerate") + 1] == "25"
    assert command[command.index("-r") + 1] == "25"
    assert "libx264" in command
    assert "yuv420p" in command
    assert "scale=" in command[command.index("-vf") + 1]
    assert command[-1] == str(tmp_path / "out.mp4")


def test_make_video_empty_directory_fails_before_subprocess(tmp_path):
    with patch.object(make_video.subprocess, "run") as run:
        with pytest.raises(SystemExit, match="no PNG frames"):
            make_video.main(["--frames-dir", str(tmp_path), "--out", str(tmp_path / "out.mp4")])
    run.assert_not_called()


def test_make_video_reports_first_missing_frame_before_subprocess(tmp_path):
    (tmp_path / "frame_00000.png").touch()
    (tmp_path / "frame_00002.png").touch()
    with patch.object(make_video.subprocess, "run") as run:
        with pytest.raises(SystemExit, match=r"first gap is frame_00001\.png"):
            make_video.main(["--frames-dir", str(tmp_path), "--out", str(tmp_path / "out.mp4")])
    run.assert_not_called()


def test_make_video_runs_for_contiguous_frames(tmp_path):
    for index in range(3):
        (tmp_path / f"frame_{index:05d}.png").touch()
    with patch.object(make_video.subprocess, "run") as run:
        make_video.main(["--frames-dir", str(tmp_path), "--out", str(tmp_path / "out.mp4")])
    run.assert_called_once()


def test_render_frames_cli_parses_without_paraview_import():
    from b737wing.cfd.post import render_frames as module
    args = module.parse_args(["--ens-dir", "ens", "--out-dir", "out", "--style", "q",
                              "--start", "1000", "--stop", "1010", "--stride", "5"])
    assert (args.style, args.start, args.stop, args.stride, args.q_level) == ("q", 1000, 1010, 5, 2000)


def test_render_frames_split_cli_and_camera_defaults_parse_without_paraview():
    from b737wing.cfd.post import render_frames as module
    args = module.parse_args(["--ens-dir", "ens", "--out-dir", "out", "--style", "split",
                              "--start", "1000", "--stop", "1000", "--stride", "5"])
    assert args.style == "split"
    assert args.cam_pos == (-6.0, 26.0, 16.0)
    assert args.cam_focal == (14.5, 0.0, -0.5)
    assert args.cam_up == (0.3, -0.35, 0.89)
    assert args.cam_angle == 28.0
    assert args.wall_offset == 0.05
    assert args.wall_seeds == 3000
    assert args.full_seeds_per_line == 90
    assert args.speed_min == 120.0
    assert args.speed_max == 300.0

    supplied = module.parse_args(["--ens-dir", "ens", "--out-dir", "out", "--style", "split",
                                  "--start", "1000", "--stop", "1000", "--stride", "5",
                                  "--cam-pos", "1", "2", "3", "--cam-focal", "4", "5", "6",
                                  "--cam-up", "0", "0", "1", "--cam-angle", "35",
                                  "--wall-offset", "0.08", "--wall-seeds", "2500",
                                  "--full-seeds-per-line", "120",
                                  "--speed-min", "90", "--speed-max", "320"])
    assert supplied.cam_pos == [1.0, 2.0, 3.0]
    assert supplied.cam_focal == [4.0, 5.0, 6.0]
    assert supplied.cam_up == [0.0, 0.0, 1.0]
    assert supplied.cam_angle == 35.0
    assert supplied.wall_offset == 0.08
    assert supplied.wall_seeds == 2500
    assert supplied.full_seeds_per_line == 120
    assert supplied.speed_min == 90.0
    assert supplied.speed_max == 320.0
