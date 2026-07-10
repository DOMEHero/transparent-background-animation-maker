# transparent-background-animation-maker
A tool that extracts the main character in your video and converts it into animation.

## CLI usage

This project provides a `tbam` command that:

1. Samples a fixed number of frames evenly across a video with `ffmpeg`.
2. Removes each sampled frame background with `backgroundremover`.
3. Encodes the transparent frames into an `apng`, `webp`, or `gif` animation with `ffmpeg`.

```bash
uv sync
uv run tbam input.mp4 --frames 12 --output out/animation
```

`apng` is the default output format. If `--output` has no extension, the selected format is appended:

```text
out/animation.apng
```

Choose another format with `--format`:

```bash
uv run tbam input.mp4 --frames 12 --format webp --output out/animation.webp
```

Intermediate files are kept by default:

```text
out/_tbam_frames/<input-tag>_frames-12_<hash>/raw_frames/
out/_tbam_frames/<input-tag>_frames-12_<hash>/transparent_frames/
```

The cache tag is based on the input video, requested frame count, and background removal options. It does not include animation FPS or output filename, so you can re-render the same transparent frames at a different playback speed:

```bash
uv run tbam input.mp4 --frames 12 --fps 12 --output out/animation-12fps
uv run tbam input.mp4 --frames 12 --fps 24 --output out/animation-24fps
```

The second command reuses the tagged `transparent_frames` cache when it is complete.

Delete intermediates after rendering with:

```bash
uv run tbam input.mp4 --frames 12 --output out/animation --keep-temp false
```

## Options

- `--frames`: Required. The number of frames to sample evenly across the video.
- `--model`: backgroundremover model name. Defaults to `u2net`.
- `--fps`: Playback frame rate for the generated animation. Defaults to `12`.
- `--format`: `apng`, `webp`, or `gif`. Defaults to `apng`.
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
uv run tbam input.mp4 --frames 12 --output out/animation \
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
