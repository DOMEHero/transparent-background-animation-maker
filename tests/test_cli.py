from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import tbam.cli as cli
from tbam.pipeline import PipelineError


def test_cli_rejects_missing_input_file(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "make",
                str(tmp_path / "missing.mp4"),
                "--frames",
                "1",
                "--output",
                str(tmp_path / "out.webp"),
            ]
        )

    assert exc.value.code == 2


def test_cli_rejects_invalid_frame_count(tmp_path: Path) -> None:
    input_video = tmp_path / "input.mp4"
    input_video.write_bytes(b"not a real video")

    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "make",
                str(input_video),
                "--frames",
                "0",
                "--output",
                str(tmp_path / "out.webp"),
            ]
        )

    assert exc.value.code == 2


def test_cli_accepts_input_without_make_subcommand(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_video = tmp_path / "input.mp4"
    output_path = tmp_path / "out"
    input_video.write_bytes(b"not a real video")
    calls = []

    def record_pipeline(**kwargs: object) -> object:
        calls.append(kwargs)
        return SimpleNamespace(
            output_path=output_path,
            kept_intermediates=False,
            raw_frames_dir=tmp_path / "raw",
            transparent_frames_dir=tmp_path / "transparent",
        )

    monkeypatch.setattr(cli, "make_animation", record_pipeline)

    status = cli.main(
        [
            str(input_video),
            "--frames",
            "1",
            "--output",
            str(output_path),
        ]
    )

    assert status == 0
    assert calls[0]["video_path"] == input_video


def test_cli_keeps_make_subcommand_as_backward_compatible_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_video = tmp_path / "input.mp4"
    output_path = tmp_path / "out"
    input_video.write_bytes(b"not a real video")
    calls = []

    def record_pipeline(**kwargs: object) -> object:
        calls.append(kwargs)
        return SimpleNamespace(
            output_path=output_path,
            kept_intermediates=False,
            raw_frames_dir=tmp_path / "raw",
            transparent_frames_dir=tmp_path / "transparent",
        )

    monkeypatch.setattr(cli, "make_animation", record_pipeline)

    status = cli.main(
        [
            "make",
            str(input_video),
            "--frames",
            "1",
            "--output",
            str(output_path),
        ]
    )

    assert status == 0
    assert calls[0]["video_path"] == input_video


def test_cli_passes_inherited_rembg_and_gka_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_video = tmp_path / "input.mp4"
    output_path = tmp_path / "out"
    input_video.write_bytes(b"not a real video")
    calls = []

    def record_pipeline(**kwargs: object) -> object:
        calls.append(kwargs)
        return SimpleNamespace(
            output_path=output_path,
            kept_intermediates=False,
            raw_frames_dir=tmp_path / "raw",
            transparent_frames_dir=tmp_path / "transparent",
        )

    monkeypatch.setattr(cli, "make_animation", record_pipeline)

    status = cli.main(
        [
            str(input_video),
            "--frames",
            "1",
            "--output",
            str(output_path),
            "--rembg-model",
            "sam",
            "--rembg-alpha-matting",
            "--rembg-alpha-matting-foreground-threshold",
            "230",
            "--rembg-alpha-matting-background-threshold",
            "20",
            "--rembg-alpha-matting-erode-size",
            "15",
            "--rembg-only-mask",
            "--rembg-post-process-mask",
            "--rembg-bgcolor",
            "255",
            "0",
            "0",
            "128",
            "--rembg-extras",
            '{"sam_prompt": []}',
            "--gka-template",
            "canvas",
            "--gka-unique",
            "--gka-crop",
            "--gka-sprites",
            "--gka-algorithm",
            "binary-tree",
            "--gka-prefix",
            "hero",
            "--gka-mini",
            "--gka-frame-duration",
            "0.1",
            "--gka-info",
        ]
    )

    assert status == 0
    rembg_options = calls[0]["rembg_options"]
    gka_options = calls[0]["gka_options"]
    assert rembg_options.model == "sam"
    assert rembg_options.alpha_matting is True
    assert rembg_options.alpha_matting_foreground_threshold == 230
    assert rembg_options.alpha_matting_background_threshold == 20
    assert rembg_options.alpha_matting_erode_size == 15
    assert rembg_options.only_mask is True
    assert rembg_options.post_process_mask is True
    assert rembg_options.bgcolor == (255, 0, 0, 128)
    assert rembg_options.extras == '{"sam_prompt": []}'
    assert gka_options.template == "canvas"
    assert gka_options.unique is True
    assert gka_options.crop is True
    assert gka_options.sprites is True
    assert gka_options.algorithm == "binary-tree"
    assert gka_options.prefix == "hero"
    assert gka_options.mini is True
    assert gka_options.frame_duration == 0.1
    assert gka_options.info is True


def test_cli_rejects_unsupported_format(tmp_path: Path) -> None:
    input_video = tmp_path / "input.mp4"
    input_video.write_bytes(b"not a real video")

    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "make",
                str(input_video),
                "--frames",
                "1",
                "--format",
                "mp4",
                "--output",
                str(tmp_path / "out.mp4"),
            ]
        )

    assert exc.value.code == 2


def test_cli_returns_failure_when_pipeline_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_video = tmp_path / "input.mp4"
    input_video.write_bytes(b"not a real video")

    def fail_pipeline(**_: object) -> object:
        raise PipelineError("expected failure")

    monkeypatch.setattr(cli, "make_animation", fail_pipeline)

    status = cli.main(
        [
            str(input_video),
            "--frames",
            "1",
            "--output",
            str(tmp_path / "out.webp"),
        ]
    )

    assert status == 1
    assert "expected failure" in capsys.readouterr().err
