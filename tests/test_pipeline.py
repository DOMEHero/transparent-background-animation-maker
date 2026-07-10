from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tbam.pipeline import (
    AnimationFormat,
    GkaOptions,
    PipelineError,
    RembgOptions,
    Renderer,
    ToolConfig,
    build_encode_animation_cmd,
    build_extract_frame_cmd,
    build_ffprobe_duration_cmd,
    build_gka_animation_cmd,
    build_remove_background_cmd,
    make_animation,
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
        "rembg",
        Path("raw.png"),
        Path("transparent.png"),
        "isnet-anime",
    ) == ["rembg", "i", "-m", "isnet-anime", "raw.png", "transparent.png"]


def test_remove_background_command_with_inherited_rembg_options() -> None:
    assert build_remove_background_cmd(
        "rembg",
        Path("raw.png"),
        Path("transparent.png"),
        RembgOptions(
            model="sam",
            alpha_matting=True,
            alpha_matting_foreground_threshold=230,
            alpha_matting_background_threshold=20,
            alpha_matting_erode_size=15,
            only_mask=True,
            post_process_mask=True,
            bgcolor=(255, 0, 0, 128),
            extras='{"sam_prompt": []}',
        ),
    ) == [
        "rembg",
        "i",
        "-m",
        "sam",
        "-a",
        "-af",
        "230",
        "-ab",
        "20",
        "-ae",
        "15",
        "-om",
        "-ppm",
        "-bgc",
        "255",
        "0",
        "0",
        "128",
        "-x",
        '{"sam_prompt": []}',
        "raw.png",
        "transparent.png",
    ]


def test_gka_animation_command() -> None:
    assert build_gka_animation_cmd(
        "gka",
        Path("transparent_frames"),
        Path("gka-output"),
        "canvas",
        12.0,
    ) == [
        "gka",
        "transparent_frames",
        "-t",
        "canvas",
        "-f",
        "0.0833333",
        "-o",
        "gka-output",
    ]


def test_gka_animation_command_with_inherited_gka_options() -> None:
    assert build_gka_animation_cmd(
        "gka",
        Path("transparent_frames"),
        Path("gka-output"),
        GkaOptions(
            template="canvas",
            unique=True,
            crop=True,
            sprites=True,
            algorithm="binary-tree",
            prefix="hero",
            mini=True,
            frame_duration=0.1,
            info=True,
        ),
        12.0,
    ) == [
        "gka",
        "transparent_frames",
        "-t",
        "canvas",
        "-f",
        "0.1",
        "-o",
        "gka-output",
        "-u",
        "-c",
        "-s",
        "-a",
        "binary-tree",
        "-p",
        "hero",
        "-m",
        "-i",
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


def test_run_checked_reports_subprocess_failure() -> None:
    with pytest.raises(PipelineError, match="command failed"):
        run_checked([sys.executable, "-c", "import sys; print('bad', file=sys.stderr); sys.exit(7)"])


def test_make_animation_smoke_with_real_ffmpeg_fake_rembg_and_fake_gka(tmp_path: Path) -> None:
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

    fake_rembg = tmp_path / "rembg"
    fake_rembg.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import shutil",
                "import sys",
                "if len(sys.argv) != 6 or sys.argv[1:4] != ['i', '-m', 'u2netp']:",
                "    raise SystemExit(2)",
                "shutil.copyfile(sys.argv[4], sys.argv[5])",
            ]
        ),
        encoding="utf-8",
    )
    os.chmod(fake_rembg, 0o755)

    fake_gka = tmp_path / "gka"
    fake_gka.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "from pathlib import Path",
                "import sys",
                "if len(sys.argv) != 8 or sys.argv[2] != '-t' or sys.argv[4] != '-f' or sys.argv[6] != '-o':",
                "    raise SystemExit(2)",
                "output = Path(sys.argv[7])",
                "output.mkdir(parents=True, exist_ok=True)",
                "(output / 'gka.html').write_text('ok', encoding='utf-8')",
            ]
        ),
        encoding="utf-8",
    )
    os.chmod(fake_gka, 0o755)

    progress_messages = []
    output_path = tmp_path / "out" / "animation"
    result = make_animation(
        video_path=input_video,
        frames=2,
        output_path=output_path,
        renderer=Renderer.GKA,
        fps=4,
        rembg_model="u2netp",
        gka_template="css",
        keep_temp=True,
        tools=ToolConfig(ffmpeg=ffmpeg, ffprobe=ffprobe, rembg=str(fake_rembg), gka=str(fake_gka)),
        progress=progress_messages.append,
    )

    assert result.output_path == output_path.resolve()
    assert (output_path / "gka.html").is_file()
    assert len(list(result.raw_frames_dir.glob("*.png"))) == 2
    assert len(list(result.transparent_frames_dir.glob("*.png"))) == 2
    assert any(message.startswith("CUDA available for rembg:") for message in progress_messages)


def test_gka_keep_temp_false_preserves_output_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands = []

    def fake_run_checked(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[0] == "ffprobe":
            return subprocess.CompletedProcess(command, 0, stdout="1.0\n", stderr="")
        if command[0] == "ffmpeg" and "-frames:v" in command:
            Path(command[-1]).write_bytes(b"raw")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[0] == "rembg":
            Path(command[-1]).write_bytes(b"transparent")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[0] == "gka":
            output_dir = Path(command[-1])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "gka.html").write_text("ok", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("tbam.pipeline.shutil.which", lambda executable: executable)
    monkeypatch.setattr("tbam.pipeline.run_checked", fake_run_checked)

    output_path = tmp_path / "out" / "animation"
    result = make_animation(
        video_path=tmp_path / "input.mp4",
        frames=1,
        output_path=output_path,
        renderer=Renderer.GKA,
        keep_temp=False,
    )

    assert result.output_path == output_path.resolve()
    assert (output_path / "gka.html").is_file()
    assert not (output_path / "_tbam_intermediates").exists()


def test_resolve_tools_finds_project_local_gka(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    local_gka = tmp_path / "node_modules" / ".bin" / "gka"
    local_gka.parent.mkdir(parents=True)
    local_gka.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    os.chmod(local_gka, 0o755)

    def fake_which(executable: str) -> str | None:
        if executable in {"ffmpeg", "ffprobe", "rembg", "node"}:
            return executable
        return None

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("tbam.pipeline.shutil.which", fake_which)

    tools = resolve_tools(ToolConfig(), Renderer.GKA)

    assert tools.gka == str(local_gka)


def test_resolve_tools_reports_missing_node_for_gka_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_gka = tmp_path / "node_modules" / ".bin" / "gka"
    local_gka.parent.mkdir(parents=True)
    local_gka.write_text("#!/bin/sh\nexec node cli.js \"$@\"\n", encoding="utf-8")
    os.chmod(local_gka, 0o755)

    def fake_which(executable: str) -> str | None:
        if executable in {"ffmpeg", "ffprobe", "rembg"}:
            return executable
        return None

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("tbam.pipeline.shutil.which", fake_which)

    with pytest.raises(PipelineError, match="node is not on PATH"):
        resolve_tools(ToolConfig(), Renderer.GKA)
