from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tbam.pipeline import (
    AnimationFormat,
    GkaOptions,
    PipelineError,
    RembgOptions,
    Renderer,
    ToolConfig,
    make_animation,
)


REMBG_MODELS = [
    "birefnet-general",
    "birefnet-general-lite",
    "birefnet-portrait",
    "birefnet-dis",
    "birefnet-hrsod",
    "birefnet-cod",
    "birefnet-massive",
    "isnet-anime",
    "dis_custom",
    "isnet-general-use",
    "sam",
    "silueta",
    "u2net_cloth_seg",
    "u2net_custom",
    "u2net_human_seg",
    "u2net",
    "u2netp",
    "bria-rmbg",
    "ben_custom",
]

GKA_ALGORITHMS = ["top-down", "left-right", "binary-tree", "diagonal", "alt-diagonal"]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    normalized_argv = list(sys.argv[1:] if argv is None else argv)
    if normalized_argv[:1] == ["make"]:
        normalized_argv = normalized_argv[1:]

    args = parser.parse_args(normalized_argv)
    return run_make_command(args, parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tbam",
        description="Extract video frames, remove backgrounds, and render transparent animations.",
    )
    parser.add_argument("input", type=Path, help="Input video path.")
    parser.add_argument(
        "--frames",
        type=parse_positive_int,
        required=True,
        help="Number of evenly sampled frames to extract.",
    )
    parser.add_argument(
        "--format",
        choices=[item.value for item in AnimationFormat],
        default=AnimationFormat.WEBP.value,
        help="Output animation format for --renderer direct. Defaults to webp.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output animation path.",
    )
    parser.add_argument(
        "--fps",
        type=parse_positive_float,
        default=12.0,
        help="Playback frame rate for the rendered animation. Defaults to 12.",
    )
    parser.add_argument(
        "--keep-temp",
        type=parse_bool,
        default=True,
        metavar="true|false",
        help="Keep raw and transparent intermediate frames. Defaults to true.",
    )
    parser.add_argument(
        "--no-keep-temp",
        action="store_false",
        dest="keep_temp",
        help="Delete intermediate frames after the animation is written.",
    )
    parser.add_argument(
        "--renderer",
        choices=[item.value for item in Renderer],
        default=Renderer.GKA.value,
        help="Animation renderer. Defaults to gka.",
    )
    parser.add_argument(
        "--rembg-model",
        choices=REMBG_MODELS,
        default="u2net",
        help="rembg model name. Defaults to u2net.",
    )
    parser.add_argument(
        "--rembg-alpha-matting",
        action="store_true",
        help="Pass rembg -a/--alpha-matting.",
    )
    parser.add_argument(
        "--rembg-alpha-matting-foreground-threshold",
        type=int,
        help="Pass rembg -af/--alpha-matting-foreground-threshold.",
    )
    parser.add_argument(
        "--rembg-alpha-matting-background-threshold",
        type=int,
        help="Pass rembg -ab/--alpha-matting-background-threshold.",
    )
    parser.add_argument(
        "--rembg-alpha-matting-erode-size",
        type=int,
        help="Pass rembg -ae/--alpha-matting-erode-size.",
    )
    parser.add_argument(
        "--rembg-only-mask",
        action="store_true",
        help="Pass rembg -om/--only-mask.",
    )
    parser.add_argument(
        "--rembg-post-process-mask",
        action="store_true",
        help="Pass rembg -ppm/--post-process-mask.",
    )
    parser.add_argument(
        "--rembg-bgcolor",
        type=parse_color_channel,
        nargs=4,
        metavar=("R", "G", "B", "A"),
        help="Pass rembg -bgc/--bgcolor as RGBA channels from 0 to 255.",
    )
    parser.add_argument(
        "--rembg-extras",
        help="Pass rembg -x/--extras JSON/text, for SAM or custom model parameters.",
    )
    parser.add_argument(
        "--gka-template",
        default="css",
        help="GKA output template, for example css, canvas, svg, or wechat-svg. Defaults to css.",
    )
    parser.add_argument(
        "--gka-unique",
        action="store_true",
        help="Pass gka -u/--unique to remove duplicate frames.",
    )
    parser.add_argument(
        "--gka-crop",
        action="store_true",
        help="Pass gka -c/--crop to crop images.",
    )
    parser.add_argument(
        "--gka-sprites",
        action="store_true",
        help="Pass gka -s/--sprites to generate sprite images.",
    )
    parser.add_argument(
        "--gka-algorithm",
        choices=GKA_ALGORITHMS,
        help="Pass gka -a/--algorithm for sprite layout.",
    )
    parser.add_argument(
        "--gka-prefix",
        help="Pass gka -p/--prefix to rename generated assets with a prefix.",
    )
    parser.add_argument(
        "--gka-mini",
        action="store_true",
        help="Pass gka -m/--mini to minify images.",
    )
    parser.add_argument(
        "--gka-frame-duration",
        type=parse_positive_float,
        help="Pass gka -f/--frameduration directly. Overrides --fps for GKA.",
    )
    parser.add_argument(
        "--gka-info",
        action="store_true",
        help="Pass gka -i/--info to print image info.",
    )
    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        help="ffmpeg executable path or name. Defaults to ffmpeg.",
    )
    parser.add_argument(
        "--ffprobe",
        default="ffprobe",
        help="ffprobe executable path or name. Defaults to ffprobe.",
    )
    parser.add_argument(
        "--rembg",
        default="rembg",
        help="rembg executable path or name. Defaults to rembg.",
    )
    parser.add_argument(
        "--gka",
        default="gka",
        help="gka executable path or name. Defaults to gka.",
    )

    return parser


def run_make_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if not args.input.is_file():
        parser.error(f"input video does not exist or is not a file: {args.input}")

    try:
        result = make_animation(
            video_path=args.input,
            frames=args.frames,
            output_path=args.output,
            renderer=Renderer(args.renderer),
            output_format=AnimationFormat(args.format),
            fps=args.fps,
            rembg_model=args.rembg_model,
            gka_template=args.gka_template,
            rembg_options=RembgOptions(
                model=args.rembg_model,
                alpha_matting=args.rembg_alpha_matting,
                alpha_matting_foreground_threshold=args.rembg_alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=args.rembg_alpha_matting_background_threshold,
                alpha_matting_erode_size=args.rembg_alpha_matting_erode_size,
                only_mask=args.rembg_only_mask,
                post_process_mask=args.rembg_post_process_mask,
                bgcolor=tuple(args.rembg_bgcolor) if args.rembg_bgcolor is not None else None,
                extras=args.rembg_extras,
            ),
            gka_options=GkaOptions(
                template=args.gka_template,
                unique=args.gka_unique,
                crop=args.gka_crop,
                sprites=args.gka_sprites,
                algorithm=args.gka_algorithm,
                prefix=args.gka_prefix,
                mini=args.gka_mini,
                frame_duration=args.gka_frame_duration,
                info=args.gka_info,
            ),
            keep_temp=args.keep_temp,
            tools=ToolConfig(
                ffmpeg=args.ffmpeg,
                ffprobe=args.ffprobe,
                rembg=args.rembg,
                gka=args.gka,
            ),
            progress=print,
        )
    except PipelineError as exc:
        print(f"tbam: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote animation: {result.output_path}")
    if result.kept_intermediates:
        print(f"Raw frames: {result.raw_frames_dir}")
        print(f"Transparent frames: {result.transparent_frames_dir}")
    return 0


def parse_positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("must be true or false")


def parse_color_channel(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0 or parsed > 255:
        raise argparse.ArgumentTypeError("must be between 0 and 255")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
