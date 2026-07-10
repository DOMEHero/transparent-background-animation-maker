from __future__ import annotations

import enum
import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


class PipelineError(RuntimeError):
    """Raised when an external processing step fails."""


class AnimationFormat(enum.Enum):
    WEBP = "webp"
    GIF = "gif"
    APNG = "apng"


@dataclass(frozen=True)
class ToolConfig:
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    backgroundremover: str = "backgroundremover"


@dataclass(frozen=True)
class CudaStatus:
    available: bool
    providers: tuple[str, ...]
    detail: str


@dataclass(frozen=True)
class BackgroundRemovalOptions:
    model: str = "u2net"
    alpha_matting: bool = False
    alpha_matting_foreground_threshold: int | None = None
    alpha_matting_background_threshold: int | None = None
    alpha_matting_erode_size: int | None = None
    alpha_matting_base_size: int | None = None
    only_mask: bool = False
    mask_threshold: int | None = None
    background_color: tuple[int, int, int] | None = None
    background_image: Path | None = None


@dataclass(frozen=True)
class MakeResult:
    output_path: Path
    raw_frames_dir: Path
    transparent_frames_dir: Path
    kept_intermediates: bool


def make_animation(
    *,
    video_path: Path,
    frames: int,
    output_path: Path,
    output_format: AnimationFormat = AnimationFormat.APNG,
    fps: float = 12.0,
    background_model: str = "u2net",
    background_options: BackgroundRemovalOptions | None = None,
    keep_temp: bool = True,
    tools: ToolConfig = ToolConfig(),
    progress: Callable[[str], None] | None = None,
) -> MakeResult:
    if frames < 1:
        raise PipelineError("frames must be at least 1")
    if fps <= 0:
        raise PipelineError("fps must be greater than 0")

    emit = progress or (lambda _: None)
    tools = resolve_tools(tools)
    background_options = background_options or BackgroundRemovalOptions(model=background_model)

    video_path = video_path.resolve()
    output_path = normalize_output_path(output_path.resolve(), output_format)
    frame_cache_tag = build_frame_cache_tag(video_path, frames, background_options)
    frame_cache_dir = build_frame_cache_dir(output_path.parent, frame_cache_tag)
    raw_frames_dir = frame_cache_dir / "raw_frames"
    transparent_frames_dir = frame_cache_dir / "transparent_frames"

    cuda_status = detect_cuda_status()
    emit(format_cuda_status(cuda_status))
    emit(f"Using backgroundremover model: {background_options.model}")
    emit(f"Frame cache tag: {frame_cache_tag}")
    emit(f"Output format: {output_format.value}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_frames_dir.mkdir(parents=True, exist_ok=True)
    transparent_frames_dir.mkdir(parents=True, exist_ok=True)

    if has_complete_frame_set(transparent_frames_dir, frames):
        emit(f"Reusing transparent frames: {transparent_frames_dir}")
    else:
        emit(f"Probing video duration: {video_path}")
        duration = probe_duration(video_path, tools)
        timestamps = sample_timestamps(duration, frames)
        emit(f"Sampling {frames} frame(s) across {duration:.3f}s")

        raw_frames = []
        transparent_frames = []
        for index, timestamp in enumerate(timestamps, start=1):
            raw_frame = raw_frames_dir / frame_name(index)
            transparent_frame = transparent_frames_dir / frame_name(index)
            emit(f"[{index}/{frames}] Extracting frame at {timestamp:.3f}s")
            run_checked(build_extract_frame_cmd(tools.ffmpeg, video_path, timestamp, raw_frame))
            if not raw_frame.is_file():
                raise PipelineError(f"ffmpeg did not produce frame: {raw_frame}")
            emit(f"[{index}/{frames}] Removing background")
            run_checked(
                build_remove_background_cmd(
                    tools.backgroundremover,
                    raw_frame,
                    transparent_frame,
                    background_options,
                )
            )
            if not transparent_frame.is_file():
                raise PipelineError(f"backgroundremover did not produce frame: {transparent_frame}")
            raw_frames.append(raw_frame)
            transparent_frames.append(transparent_frame)

        if len(raw_frames) != frames or len(transparent_frames) != frames:
            raise PipelineError("failed to produce the requested number of frames")

    emit(f"Encoding {output_format.value} animation: {output_path}")
    run_checked(
        build_encode_animation_cmd(
            tools.ffmpeg,
            transparent_frames_dir,
            output_path,
            output_format,
            fps,
        )
    )

    kept_intermediates = keep_temp
    if not keep_temp:
        emit(f"Deleting intermediate frames: {frame_cache_dir}")
        shutil.rmtree(frame_cache_dir, ignore_errors=True)
    else:
        emit(f"Keeping raw frames: {raw_frames_dir}")
        emit(f"Keeping transparent frames: {transparent_frames_dir}")

    return MakeResult(
        output_path=output_path,
        raw_frames_dir=raw_frames_dir,
        transparent_frames_dir=transparent_frames_dir,
        kept_intermediates=kept_intermediates,
    )


def normalize_output_path(output_path: Path, output_format: AnimationFormat) -> Path:
    if output_path.suffix:
        return output_path
    return output_path.with_suffix(f".{output_format.value}")


def build_frame_cache_dir(output_parent: Path, cache_tag: str) -> Path:
    return output_parent / "_tbam_frames" / cache_tag


def build_frame_cache_tag(
    video_path: Path,
    frames: int,
    background_options: BackgroundRemovalOptions,
) -> str:
    video_path = video_path.resolve()
    stem = slugify(video_path.stem) or "video"
    payload = {
        "video": file_fingerprint(video_path),
        "frames": frames,
        "background": background_options_fingerprint(background_options),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"{stem}_frames-{frames}_{digest}"


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug[:48].strip("-")


def file_fingerprint(path: Path) -> dict[str, int | str | None]:
    try:
        stat = path.stat()
    except OSError:
        return {
            "path": str(path),
            "size": None,
            "mtime_ns": None,
        }
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def background_options_fingerprint(options: BackgroundRemovalOptions) -> dict[str, object]:
    background_image = None
    if options.background_image is not None:
        background_image = file_fingerprint(options.background_image.resolve())

    return {
        "model": options.model,
        "alpha_matting": options.alpha_matting,
        "alpha_matting_foreground_threshold": options.alpha_matting_foreground_threshold,
        "alpha_matting_background_threshold": options.alpha_matting_background_threshold,
        "alpha_matting_erode_size": options.alpha_matting_erode_size,
        "alpha_matting_base_size": options.alpha_matting_base_size,
        "only_mask": options.only_mask,
        "mask_threshold": options.mask_threshold,
        "background_color": options.background_color,
        "background_image": background_image,
    }


def has_complete_frame_set(frames_dir: Path, frames: int) -> bool:
    return all((frames_dir / frame_name(index)).is_file() for index in range(1, frames + 1))


def resolve_tools(tools: ToolConfig) -> ToolConfig:
    ffmpeg = resolve_required_executable(tools.ffmpeg)
    ffprobe = resolve_required_executable(tools.ffprobe)
    backgroundremover = resolve_required_executable(tools.backgroundremover)

    return ToolConfig(ffmpeg=ffmpeg, ffprobe=ffprobe, backgroundremover=backgroundremover)


def resolve_required_executable(executable: str, local_fallbacks: Sequence[Path] = ()) -> str:
    resolved = shutil.which(executable)
    if resolved is not None:
        return resolved

    for fallback in local_fallbacks:
        if fallback.is_file():
            return str(fallback)

    raise PipelineError(f"missing required executable(s): {executable}")


def detect_cuda_status() -> CudaStatus:
    try:
        import torch
    except Exception as exc:
        return CudaStatus(
            available=False,
            providers=(),
            detail=f"torch is not importable in this environment: {exc}",
        )

    try:
        available = bool(torch.cuda.is_available())
    except Exception as exc:
        return CudaStatus(
            available=False,
            providers=(),
            detail=f"torch CUDA detection failed: {exc}",
        )

    if not available:
        return CudaStatus(
            available=False,
            providers=(),
            detail="torch CUDA is not available",
        )

    try:
        device_count = int(torch.cuda.device_count())
        devices = tuple(torch.cuda.get_device_name(index) for index in range(device_count))
    except Exception as exc:
        return CudaStatus(
            available=True,
            providers=(),
            detail=f"torch CUDA is available, but device names could not be read: {exc}",
        )

    return CudaStatus(available=True, providers=devices, detail="torch CUDA is available")


def format_cuda_status(status: CudaStatus) -> str:
    availability = "yes" if status.available else "no"
    devices = ", ".join(status.providers) if status.providers else "none"
    return f"CUDA available for backgroundremover: {availability} ({status.detail}; devices: {devices})"


def probe_duration(video_path: Path, tools: ToolConfig = ToolConfig()) -> float:
    command = build_ffprobe_duration_cmd(tools.ffprobe, video_path)
    completed = run_checked(command)
    raw_duration = completed.stdout.strip()
    try:
        duration = float(raw_duration)
    except ValueError as exc:
        raise PipelineError(f"ffprobe returned an invalid duration: {raw_duration!r}") from exc
    if duration <= 0:
        raise PipelineError(f"video duration must be greater than 0, got {duration}")
    return duration


def sample_timestamps(duration: float, count: int) -> list[float]:
    if duration <= 0:
        raise ValueError("duration must be greater than 0")
    if count < 1:
        raise ValueError("count must be at least 1")
    if count == 1:
        return [duration / 2]

    step = duration / count
    return [step * index for index in range(count)]


def frame_name(index: int) -> str:
    return f"frame_{index:06d}.png"


def build_ffprobe_duration_cmd(ffprobe: str, video_path: Path) -> list[str]:
    return [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(video_path),
    ]


def build_extract_frame_cmd(
    ffmpeg: str,
    video_path: Path,
    timestamp: float,
    output_path: Path,
) -> list[str]:
    return [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-ss",
        f"{timestamp:.6f}",
        "-frames:v",
        "1",
        "-vf",
        "format=rgba",
        str(output_path),
    ]


def build_remove_background_cmd(
    backgroundremover: str,
    input_path: Path,
    output_path: Path,
    options: BackgroundRemovalOptions | str = BackgroundRemovalOptions(),
) -> list[str]:
    if isinstance(options, str):
        options = BackgroundRemovalOptions(model=options)

    command = [
        backgroundremover,
        "-i",
        str(input_path),
        "-m",
        options.model,
    ]
    if options.alpha_matting:
        command.append("-a")
    if options.alpha_matting_foreground_threshold is not None:
        command.extend(["-af", str(options.alpha_matting_foreground_threshold)])
    if options.alpha_matting_background_threshold is not None:
        command.extend(["-ab", str(options.alpha_matting_background_threshold)])
    if options.alpha_matting_erode_size is not None:
        command.extend(["-ae", str(options.alpha_matting_erode_size)])
    if options.alpha_matting_base_size is not None:
        command.extend(["-az", str(options.alpha_matting_base_size)])
    if options.only_mask:
        command.append("-om")
    if options.mask_threshold is not None:
        command.extend(["-mt", str(options.mask_threshold)])
    if options.background_color is not None:
        command.extend(["-bc", ",".join(str(value) for value in options.background_color)])
    if options.background_image is not None:
        command.extend(["-bi", str(options.background_image)])

    command.extend(["-o", str(output_path)])
    return command


def build_encode_animation_cmd(
    ffmpeg: str,
    frames_dir: Path,
    output_path: Path,
    output_format: AnimationFormat,
    fps: float,
) -> list[str]:
    input_pattern = frames_dir / "frame_%06d.png"
    base = [
        ffmpeg,
        "-y",
        "-framerate",
        format_fps(fps),
        "-i",
        str(input_pattern),
    ]

    if output_format is AnimationFormat.WEBP:
        return [
            *base,
            "-an",
            "-c:v",
            "libwebp_anim",
            "-lossless",
            "1",
            "-loop",
            "0",
            str(output_path),
        ]
    if output_format is AnimationFormat.APNG:
        return [
            *base,
            "-an",
            "-c:v",
            "apng",
            "-plays",
            "0",
            str(output_path),
        ]
    if output_format is AnimationFormat.GIF:
        return [
            *base,
            "-an",
            "-gifflags",
            "+transdiff",
            "-loop",
            "0",
            str(output_path),
        ]

    raise PipelineError(f"unsupported animation format: {output_format}")


def format_fps(fps: float) -> str:
    return f"{fps:g}"


def run_checked(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            list(command),
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise PipelineError(f"executable not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        details = stderr or stdout or f"exit code {exc.returncode}"
        raise PipelineError(f"command failed: {' '.join(command)}\n{details}") from exc
    return completed
