# transparent-background-animation-maker
A tool that extracts the main character in your video and converts it into animation.

## CLI usage

This project provides a `tbam` command that:

1. Samples a fixed number of frames evenly across a video with `ffmpeg`.
2. Removes each sampled frame background with `backgroundremover`.
3. Encodes the transparent frames into an `apng`, `webp`, or `gif` animation, or a WebP spritesheet, with `ffmpeg`.

```bash
uv sync
uv run tbam input.mp4 --frames 12
```

`apng` is the default output format. Outputs are written under `output/` by default:

```text
output/
└─ input-12frames-<hash>/
   ├─ raw_frames/
   │  ├─ frame_000001.png
   │  └─ frame_000002.png
   ├─ transparent_frames/
   │  ├─ frame_000001.png
   │  └─ frame_000002.png
   └─ animations/
      └─ animation-12fps.apng
```

Choose another format with `--format`:

```bash
uv run tbam input.mp4 --frames 12 --format webp --fps 48
uv run tbam input.mp4 --frames 12 --format spritesheet
uv run tbam input.mp4 --frames 12 --format spritesheet --spritesheet-columns 6 --spritesheet-rows 2
```

These commands create files like:

```text
output/input-12frames-<hash>/animations/animation-48fps.webp
output/input-12frames-<hash>/animations/spritesheet.webp
```

The job directory tag is based on the input video, requested frame count, and background removal options. It does not include animation FPS or render format, so you can re-render the same transparent frames at a different playback speed or as a spritesheet:

```bash
uv run tbam input.mp4 --frames 12 --fps 12
uv run tbam input.mp4 --frames 12 --fps 24 --format webp
uv run tbam input.mp4 --frames 12 --format spritesheet
```

The later commands reuse the job's `transparent_frames` cache when it is complete.

Use a different output root with:

```bash
uv run tbam input.mp4 --frames 12 --output-dir renders
```

Delete intermediates after rendering with:

```bash
uv run tbam input.mp4 --frames 12 --keep-temp false
```

## Options

- `--frames`: Required. The number of frames to sample evenly across the video.
- `--model`: backgroundremover model name. Defaults to `u2net`.
- `--fps`: Playback frame rate for the generated animation. Defaults to `12`.
- `--format`: `apng`, `webp`, `gif`, or `spritesheet`. Defaults to `apng`.
- `--output-dir`: Output root directory. Defaults to `output`.
- `--spritesheet-columns`, `--spritesheet-rows`: Optional layout controls for `--format spritesheet`. If only one is provided, the other is computed.
- `--ffmpeg`, `--ffprobe`, `--backgroundremover`: Override executable paths when needed.

GIF transparency is limited by the GIF format. Use `apng` or `webp` when high-quality alpha is important.

`backgroundremover` uses PyTorch. The CLI reports whether PyTorch can see CUDA before removing backgrounds. If it reports `CUDA available for backgroundremover: no`, background removal is running on CPU in that environment.

## Upstream options

The CLI exposes the useful `backgroundremover` image options:

- `--alpha-matting`
- `--alpha-matting-foreground-threshold`
- `--alpha-matting-background-threshold`
- `--alpha-matting-erode-size`
- `--alpha-matting-base-size`
- `--only-mask`
- `--mask-threshold`
- `--background-color R G B`
- `--background-image PATH`

Example:

```bash
uv run tbam input.mp4 --frames 12 \
  --model u2netp \
  --alpha-matting \
  --alpha-matting-erode-size 5
```

On first real use, `backgroundremover` may download its U2NET model weights into `~/.u2net`.

## Development

```bash
uv sync --dev
uv run pytest
```
