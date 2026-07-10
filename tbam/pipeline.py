from __future__ import annotations

import enum
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


class Renderer(enum.Enum):
    GKA = "gka"
    DIRECT = "direct"


@dataclass(frozen=True)
class ToolConfig:
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    rembg: str = "rembg"
    gka: str = "gka"


@dataclass(frozen=True)
class CudaStatus:
    available: bool
    providers: tuple[str, ...]
    detail: str


@dataclass(frozen=True)
class RembgOptions:
    model: str = "u2net"
    alpha_matting: bool = False
    alpha_matting_foreground_threshold: int | None = None
    alpha_matting_background_threshold: int | None = None
    alpha_matting_erode_size: int | None = None
    only_mask: bool = False
    post_process_mask: bool = False
    bgcolor: tuple[int, int, int, int] | None = None
    extras: str | None = None


@dataclass(frozen=True)
class GkaOptions:
    template: str = "css"
    unique: bool = False
    crop: bool = False
    sprites: bool = False
    algorithm: str | None = None
    prefix: str | None = None
    mini: bool = False
    frame_duration: float | None = None
    info: bool = False


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
    renderer: Renderer = Renderer.GKA,
    output_format: AnimationFormat = AnimationFormat.WEBP,
    fps: float = 12.0,
    rembg_model: str = "u2net",
    gka_template: str = "css",
    rembg_options: RembgOptions | None = None,
    gka_options: GkaOptions | None = None,
    keep_temp: bool = True,
    tools: ToolConfig = ToolConfig(),
    progress: Callable[[str], None] | None = None,
) -> MakeResult:
    if frames < 1:
        raise PipelineError("frames must be at least 1")
    if fps <= 0:
        raise PipelineError("fps must be greater than 0")

    emit = progress or (lambda _: None)
    tools = resolve_tools(tools, renderer)
    rembg_options = rembg_options or RembgOptions(model=rembg_model)
    gka_options = gka_options or GkaOptions(template=gka_template)

    video_path = video_path.resolve()
    output_path = output_path.resolve()
    job_dir = build_job_dir(output_path, renderer)
    raw_frames_dir = job_dir / "raw_frames"
    transparent_frames_dir = job_dir / "transparent_frames"

    cuda_status = detect_cuda_status()
    emit(format_cuda_status(cuda_status))
    emit(f"Using rembg model: {rembg_options.model}")
    emit(f"Renderer: {renderer.value}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_frames_dir.mkdir(parents=True, exist_ok=True)
    transparent_frames_dir.mkdir(parents=True, exist_ok=True)

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
        run_checked(build_remove_background_cmd(tools.rembg, raw_frame, transparent_frame, rembg_options))
        if not transparent_frame.is_file():
            raise PipelineError(f"rembg did not produce frame: {transparent_frame}")
        raw_frames.append(raw_frame)
        transparent_frames.append(transparent_frame)

    if len(raw_frames) != frames or len(transparent_frames) != frames:
        raise PipelineError("failed to produce the requested number of frames")

    if renderer is Renderer.GKA:
        emit(f"Generating GKA animation in: {output_path}")
        run_checked(
            build_gka_animation_cmd(
                tools.gka,
                transparent_frames_dir,
                output_path,
                gka_options,
                fps,
            )
        )
    elif renderer is Renderer.DIRECT:
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
    else:
        raise PipelineError(f"unsupported renderer: {renderer}")

    kept_intermediates = keep_temp
    if not keep_temp:
        emit(f"Deleting intermediate frames: {job_dir}")
        shutil.rmtree(job_dir, ignore_errors=True)
    else:
        emit(f"Keeping raw frames: {raw_frames_dir}")
        emit(f"Keeping transparent frames: {transparent_frames_dir}")

    return MakeResult(
        output_path=output_path,
        raw_frames_dir=raw_frames_dir,
        transparent_frames_dir=transparent_frames_dir,
        kept_intermediates=kept_intermediates,
    )


def build_job_dir(output_path: Path, renderer: Renderer) -> Path:
    if renderer is Renderer.GKA:
        return output_path / "_tbam_intermediates"
    return output_path.parent / output_path.stem


def resolve_tools(tools: ToolConfig, renderer: Renderer = Renderer.GKA) -> ToolConfig:
    ffmpeg = resolve_required_executable(tools.ffmpeg)
    ffprobe = resolve_required_executable(tools.ffprobe)
    rembg = resolve_required_executable(tools.rembg)
    gka = tools.gka

    if renderer is Renderer.GKA:
        gka = resolve_required_executable(
            tools.gka,
            local_fallbacks=[Path.cwd() / "node_modules" / ".bin" / tools.gka],
        )
        if executable_needs_node(gka) and shutil.which("node") is None:
            raise PipelineError(
                f"gka was found at {gka}, but node is not on PATH; install node or pass a working --gka executable"
            )

    return ToolConfig(ffmpeg=ffmpeg, ffprobe=ffprobe, rembg=rembg, gka=gka)


def resolve_required_executable(executable: str, local_fallbacks: Sequence[Path] = ()) -> str:
    resolved = shutil.which(executable)
    if resolved is not None:
        return resolved

    for fallback in local_fallbacks:
        if fallback.is_file():
            return str(fallback)

    raise PipelineError(f"missing required executable(s): {executable}")


def executable_needs_node(executable: str) -> bool:
    path = Path(executable)
    if not path.is_file():
        return False

    try:
        contents = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False

    return "node" in contents[:2048]


def detect_cuda_status() -> CudaStatus:
    try:
        import onnxruntime as ort
    except Exception as exc:
        return CudaStatus(
            available=False,
            providers=(),
            detail=f"onnxruntime is not importable in this environment: {exc}",
        )

    providers = tuple(ort.get_available_providers())
    available = "CUDAExecutionProvider" in providers
    detail = "CUDAExecutionProvider is available" if available else "CUDAExecutionProvider is not available"
    return CudaStatus(available=available, providers=providers, detail=detail)


def format_cuda_status(status: CudaStatus) -> str:
    availability = "yes" if status.available else "no"
    providers = ", ".join(status.providers) if status.providers else "none"
    return f"CUDA available for rembg: {availability} ({status.detail}; providers: {providers})"


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
    rembg: str,
    input_path: Path,
    output_path: Path,
    options: RembgOptions | str = RembgOptions(),
) -> list[str]:
    if isinstance(options, str):
        options = RembgOptions(model=options)

    command = [
        rembg,
        "i",
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
    if options.only_mask:
        command.append("-om")
    if options.post_process_mask:
        command.append("-ppm")
    if options.bgcolor is not None:
        command.append("-bgc")
        command.extend(str(value) for value in options.bgcolor)
    if options.extras is not None:
        command.extend(["-x", options.extras])

    command.extend([str(input_path), str(output_path)])
    return command


def build_gka_animation_cmd(
    gka: str,
    frames_dir: Path,
    output_path: Path,
    options: GkaOptions | str,
    fps: float,
) -> list[str]:
    if isinstance(options, str):
        options = GkaOptions(template=options)

    command = [
        gka,
        str(frames_dir),
        "-t",
        options.template,
        "-f",
        format_gka_frame_duration_from_options(fps, options),
        "-o",
        str(output_path),
    ]
    if options.unique:
        command.append("-u")
    if options.crop:
        command.append("-c")
    if options.sprites:
        command.append("-s")
    if options.algorithm is not None:
        command.extend(["-a", options.algorithm])
    if options.prefix is not None:
        command.extend(["-p", options.prefix])
    if options.mini:
        command.append("-m")
    if options.info:
        command.append("-i")

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


def format_gka_frame_duration(fps: float) -> str:
    return f"{1 / fps:g}"


def format_gka_frame_duration_from_options(fps: float, options: GkaOptions) -> str:
    if options.frame_duration is not None:
        return f"{options.frame_duration:g}"
    return format_gka_frame_duration(fps)


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
