from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tbam.pipeline import (
    AnimationFormat,
    BackgroundRemovalOptions,
    PipelineError,
    ToolConfig,
    make_animation,
)


BACKGROUNDREMOVER_MODELS = [
    "u2net",
    "u2netp",
    "u2net_human_seg",
]


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
        default=AnimationFormat.APNG.value,
        help="Output animation format. Defaults to apng.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output root directory. Defaults to output.",
    )
    parser.add_argument(
        "--fps",
        type=parse_positive_float,
        default=12.0,
        help="Playback frame rate for the rendered animation. Defaults to 12.",
    )
    parser.add_argument(
        "--spritesheet-rows",
        type=parse_positive_int,
        help="Number of rows for --format spritesheet. Defaults to automatic layout.",
    )
    parser.add_argument(
        "--spritesheet-columns",
        type=parse_positive_int,
        help="Number of columns for --format spritesheet. Defaults to automatic layout.",
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
        "--model",
        choices=BACKGROUNDREMOVER_MODELS,
        default="u2net",
        help="backgroundremover model name. Defaults to u2net.",
    )
    parser.add_argument(
        "--alpha-matting",
        action="store_true",
        help="Pass backgroundremover -a/--alpha-matting.",
    )
    parser.add_argument(
        "--alpha-matting-foreground-threshold",
        type=int,
        help="Pass backgroundremover -af/--alpha-matting-foreground-threshold.",
    )
    parser.add_argument(
        "--alpha-matting-background-threshold",
        type=int,
        help="Pass backgroundremover -ab/--alpha-matting-background-threshold.",
    )
    parser.add_argument(
        "--alpha-matting-erode-size",
        type=int,
        help="Pass backgroundremover -ae/--alpha-matting-erode-size.",
    )
    parser.add_argument(
        "--alpha-matting-base-size",
        type=int,
        help="Pass backgroundremover -az/--alpha-matting-base-size.",
    )
    parser.add_argument(
        "--only-mask",
        action="store_true",
        help="Pass backgroundremover -om/--only-mask.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=parse_color_channel,
        help="Pass backgroundremover -mt/--mask-threshold as a value from 0 to 255.",
    )
    parser.add_argument(
        "--background-color",
        type=parse_color_channel,
        nargs=3,
        metavar=("R", "G", "B"),
        help="Pass backgroundremover -bc/--background-color as RGB channels from 0 to 255.",
    )
    parser.add_argument(
        "--background-image",
        type=Path,
        help="Pass backgroundremover -bi/--backgroundimage.",
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
        "--backgroundremover",
        default="backgroundremover",
        help="backgroundremover executable path or name. Defaults to backgroundremover.",
    )

    return parser


def run_make_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if not args.input.is_file():
        parser.error(f"input video does not exist or is not a file: {args.input}")

    try:
        result = make_animation(
            video_path=args.input,
            frames=args.frames,
            output_dir=args.output_dir,
            output_format=AnimationFormat(args.format),
            fps=args.fps,
            spritesheet_rows=args.spritesheet_rows,
            spritesheet_columns=args.spritesheet_columns,
            background_model=args.model,
            background_options=BackgroundRemovalOptions(
                model=args.model,
                alpha_matting=args.alpha_matting,
                alpha_matting_foreground_threshold=args.alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=args.alpha_matting_background_threshold,
                alpha_matting_erode_size=args.alpha_matting_erode_size,
                alpha_matting_base_size=args.alpha_matting_base_size,
                only_mask=args.only_mask,
                mask_threshold=args.mask_threshold,
                background_color=tuple(args.background_color) if args.background_color is not None else None,
                background_image=args.background_image,
            ),
            keep_temp=args.keep_temp,
            tools=ToolConfig(
                ffmpeg=args.ffmpeg,
                ffprobe=args.ffprobe,
                backgroundremover=args.backgroundremover,
            ),
            progress=print,
        )
    except PipelineError as exc:
        print(f"tbam: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote output: {result.output_path}")
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
