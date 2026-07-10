from __future__ import annotations

import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from tbam.pipeline import (
    AnimationFormat,
    BackgroundRemovalOptions,
    PipelineError,
    ToolConfig,
    build_encode_animation_cmd,
    build_extract_frame_cmd,
    build_frame_cache_dir,
    build_frame_cache_tag,
    build_ffprobe_duration_cmd,
    build_remove_background_cmd,
    detect_cuda_status,
    format_cuda_status,
    make_animation,
    normalize_output_path,
    resolve_tools,
    run_checked,
    sample_timestamps,
)


def test_sample_timestamps_single_frame_uses_middle() -> None:
    assert sample_timestamps(10.0, 1) == [5.0]


def test_sample_timestamps_two_frames_cover_video_span() -> None:
    assert sample_timestamps(10.0, 2) == pytest.approx([0.0, 5.0])


def test_sample_timestamps_larger_count_is_evenly_spaced() -> None:
    assert sample_timestamps(10.0, 5) == pytest.approx([0.0, 2.0, 4.0, 6.0, 8.0])


def test_sample_timestamps_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="duration"):
        sample_timestamps(0, 1)
    with pytest.raises(ValueError, match="count"):
        sample_timestamps(1, 0)


def test_ffprobe_command() -> None:
    assert build_ffprobe_duration_cmd("ffprobe", Path("input.mp4")) == [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        "input.mp4",
    ]


def test_extract_frame_command() -> None:
    assert build_extract_frame_cmd(
        "ffmpeg",
        Path("input.mp4"),
        1.25,
        Path("frames/frame_000001.png"),
    ) == [
        "ffmpeg",
        "-y",
        "-i",
        "input.mp4",
        "-ss",
        "1.250000",
        "-frames:v",
        "1",
        "-vf",
        "format=rgba",
        "frames/frame_000001.png",
    ]


def test_remove_background_command() -> None:
    assert build_remove_background_cmd(
        "backgroundremover",
        Path("raw.png"),
        Path("transparent.png"),
        "u2netp",
    ) == ["backgroundremover", "-i", "raw.png", "-m", "u2netp", "-o", "transparent.png"]


def test_remove_background_command_with_inherited_backgroundremover_options() -> None:
    assert build_remove_background_cmd(
        "backgroundremover",
        Path("raw.png"),
        Path("transparent.png"),
        BackgroundRemovalOptions(
            model="u2net_human_seg",
            alpha_matting=True,
            alpha_matting_foreground_threshold=230,
            alpha_matting_background_threshold=20,
            alpha_matting_erode_size=15,
            alpha_matting_base_size=800,
            only_mask=True,
            mask_threshold=128,
            background_color=(255, 0, 0),
            background_image=Path("background.png"),
        ),
    ) == [
        "backgroundremover",
        "-i",
        "raw.png",
        "-m",
        "u2net_human_seg",
        "-a",
        "-af",
        "230",
        "-ab",
        "20",
        "-ae",
        "15",
        "-az",
        "800",
        "-om",
        "-mt",
        "128",
        "-bc",
        "255,0,0",
        "-bi",
        "background.png",
        "-o",
        "transparent.png",
    ]


@pytest.mark.parametrize(
    ("output_format", "expected_flags"),
    [
        (AnimationFormat.WEBP, ["-c:v", "libwebp_anim", "-lossless", "1", "-loop", "0"]),
        (AnimationFormat.APNG, ["-c:v", "apng", "-plays", "0"]),
        (AnimationFormat.GIF, ["-gifflags", "+transdiff", "-loop", "0"]),
    ],
)
def test_encode_animation_commands(
    output_format: AnimationFormat,
    expected_flags: list[str],
) -> None:
    command = build_encode_animation_cmd(
        "ffmpeg",
        Path("transparent_frames"),
        Path(f"animation.{output_format.value}"),
        output_format,
        12.0,
    )

    assert command[:5] == [
        "ffmpeg",
        "-y",
        "-framerate",
        "12",
        "-i",
    ]
    assert "transparent_frames/frame_%06d.png" in command
    for flag in expected_flags:
        assert flag in command
    assert command[-1] == f"animation.{output_format.value}"


def test_normalize_output_path_adds_format_suffix_when_missing() -> None:
    assert normalize_output_path(Path("out/animation"), AnimationFormat.APNG) == Path("out/animation.apng")
    assert normalize_output_path(Path("out/animation.webp"), AnimationFormat.APNG) == Path("out/animation.webp")


def test_frame_cache_tag_includes_input_file_and_frame_count(tmp_path: Path) -> None:
    input_video = tmp_path / "Input Video.mp4"
    input_video.write_bytes(b"video")
    options = BackgroundRemovalOptions(model="u2netp")

    first = build_frame_cache_tag(input_video, 12, options)
    second = build_frame_cache_tag(input_video, 12, options)
    different_frame_count = build_frame_cache_tag(input_video, 24, options)

    assert first == second
    assert first.startswith("input-video_frames-12_")
    assert different_frame_count.startswith("input-video_frames-24_")
    assert first != different_frame_count


def test_frame_cache_tag_changes_when_background_options_change(tmp_path: Path) -> None:
    input_video = tmp_path / "input.mp4"
    input_video.write_bytes(b"video")

    default_tag = build_frame_cache_tag(input_video, 12, BackgroundRemovalOptions(model="u2net"))
    model_tag = build_frame_cache_tag(input_video, 12, BackgroundRemovalOptions(model="u2netp"))
    matte_tag = build_frame_cache_tag(input_video, 12, BackgroundRemovalOptions(model="u2net", alpha_matting=True))

    assert default_tag != model_tag
    assert default_tag != matte_tag


def test_run_checked_reports_subprocess_failure() -> None:
    with pytest.raises(PipelineError, match="command failed"):
        run_checked([sys.executable, "-c", "import sys; print('bad', file=sys.stderr); sys.exit(7)"])


def test_detect_cuda_status_handles_partial_torch_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    status = detect_cuda_status()

    assert status.available is False
    assert status.providers == ()
    assert "torch CUDA detection failed" in status.detail
    assert "CUDA available for backgroundremover: no" in format_cuda_status(status)


def test_detect_cuda_status_handles_device_name_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

        @staticmethod
        def get_device_name(_: int) -> str:
            raise RuntimeError("broken device list")

    fake_torch = types.SimpleNamespace(cuda=FakeCuda())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    status = detect_cuda_status()

    assert status.available is True
    assert status.providers == ()
    assert "broken device list" in status.detail


def test_make_animation_smoke_with_real_ffmpeg_and_fake_backgroundremover(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        pytest.skip("ffmpeg and ffprobe are required for the smoke test")

    input_video = tmp_path / "input.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=16x16:d=0.5:r=6",
            "-pix_fmt",
            "yuv420p",
            str(input_video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    fake_backgroundremover = tmp_path / "backgroundremover"
    fake_backgroundremover.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import shutil",
                "import sys",
                "if '-i' not in sys.argv or '-o' not in sys.argv or sys.argv[sys.argv.index('-m') + 1] != 'u2netp':",
                "    raise SystemExit(2)",
                "shutil.copyfile(sys.argv[sys.argv.index('-i') + 1], sys.argv[sys.argv.index('-o') + 1])",
            ]
        ),
        encoding="utf-8",
    )
    os.chmod(fake_backgroundremover, 0o755)

    progress_messages = []
    output_path = tmp_path / "out" / "animation"
    result = make_animation(
        video_path=input_video,
        frames=2,
        output_path=output_path,
        output_format=AnimationFormat.APNG,
        fps=4,
        background_model="u2netp",
        keep_temp=True,
        tools=ToolConfig(ffmpeg=ffmpeg, ffprobe=ffprobe, backgroundremover=str(fake_backgroundremover)),
        progress=progress_messages.append,
    )

    assert result.output_path == output_path.with_suffix(".apng").resolve()
    assert result.output_path.is_file()
    assert len(list(result.raw_frames_dir.glob("*.png"))) == 2
    assert len(list(result.transparent_frames_dir.glob("*.png"))) == 2
    assert any(message.startswith("CUDA available for backgroundremover:") for message in progress_messages)


def test_keep_temp_false_preserves_animation_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands = []

    def fake_run_checked(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[0] == "ffprobe":
            return subprocess.CompletedProcess(command, 0, stdout="1.0\n", stderr="")
        if command[0] == "ffmpeg" and "-frames:v" in command:
            Path(command[-1]).write_bytes(b"raw")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[0] == "backgroundremover":
            Path(command[-1]).write_bytes(b"transparent")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[0] == "ffmpeg" and "-framerate" in command:
            Path(command[-1]).parent.mkdir(parents=True, exist_ok=True)
            Path(command[-1]).write_bytes(b"animation")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("tbam.pipeline.shutil.which", lambda executable: executable)
    monkeypatch.setattr("tbam.pipeline.run_checked", fake_run_checked)

    output_path = tmp_path / "out" / "animation"
    result = make_animation(
        video_path=tmp_path / "input.mp4",
        frames=1,
        output_path=output_path,
        output_format=AnimationFormat.APNG,
        keep_temp=False,
    )

    assert result.output_path == output_path.with_suffix(".apng").resolve()
    assert result.output_path.is_file()
    assert not result.raw_frames_dir.exists()


def test_make_animation_reuses_tagged_transparent_frames_for_new_fps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = []

    def fake_run_checked(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[0] == "ffmpeg" and "-framerate" in command:
            Path(command[-1]).parent.mkdir(parents=True, exist_ok=True)
            Path(command[-1]).write_bytes(b"animation")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("tbam.pipeline.shutil.which", lambda executable: executable)
    monkeypatch.setattr("tbam.pipeline.run_checked", fake_run_checked)

    input_video = tmp_path / "input.mp4"
    input_video.write_bytes(b"video")
    output_path = tmp_path / "out" / "animation"
    options = BackgroundRemovalOptions(model="u2net")
    cache_tag = build_frame_cache_tag(input_video, 2, options)
    transparent_frames_dir = build_frame_cache_dir(output_path.parent.resolve(), cache_tag) / "transparent_frames"
    transparent_frames_dir.mkdir(parents=True)
    for index in range(1, 3):
        (transparent_frames_dir / f"frame_{index:06d}.png").write_bytes(b"transparent")

    progress_messages = []
    result = make_animation(
        video_path=input_video,
        frames=2,
        output_path=output_path,
        output_format=AnimationFormat.APNG,
        fps=24,
        background_options=options,
        progress=progress_messages.append,
    )

    assert result.output_path == output_path.with_suffix(".apng").resolve()
    assert result.transparent_frames_dir == transparent_frames_dir
    assert result.output_path.is_file()
    assert len(commands) == 1
    assert commands[0][0] == "ffmpeg"
    assert commands[0][3] == "24"
    assert any(message.startswith("Reusing transparent frames:") for message in progress_messages)


def test_resolve_tools_only_requires_media_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tbam.pipeline.shutil.which", lambda executable: executable)

    tools = resolve_tools(ToolConfig())

    assert tools.ffmpeg == "ffmpeg"
    assert tools.ffprobe == "ffprobe"
    assert tools.backgroundremover == "backgroundremover"
