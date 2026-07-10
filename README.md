# transparent-background-animation-maker
A tool that extracts the main character in your video and converts it into animation.

## CLI usage

This project provides a `tbam` command that:

1. Samples a fixed number of frames evenly across a video with `ffmpeg`.
2. Removes each sampled frame background with `rembg`.
3. Feeds the transparent frames into `gka` to generate the final animation files.

```bash
uv sync
uv run tbam input.mp4 --frames 12 --output out/animation
```

GKA is an npm tool and must be available separately:

```bash
pnpm add gka
```

If `gka` is not on `PATH`, pass it explicitly:

```bash
uv run tbam input.mp4 --frames 12 --output out/animation --gka /path/to/gka
```

Intermediate files are kept by default:

```text
out/animation/_tbam_intermediates/raw_frames/
out/animation/_tbam_intermediates/transparent_frames/
```

Delete intermediates after rendering with:

```bash
uv run tbam input.mp4 --frames 12 --output out/animation --keep-temp false
```

## Options

- `--frames`: Required. The number of frames to sample evenly across the video.
- `--renderer`: `gka` or `direct`. Defaults to `gka`.
- `--gka-template`: GKA template name. Defaults to `css`; common values include `css`, `canvas`, `svg`, and `wechat-svg`.
- `--rembg-model`: rembg model name. Defaults to `u2net`.
- `--fps`: Playback frame rate for the generated animation. Defaults to `12`.
- `--format`: `webp`, `gif`, or `apng` when using `--renderer direct`.
- `--ffmpeg`, `--ffprobe`, `--rembg`, `--gka`: Override executable paths when needed.

Direct rendering is still available for simple transparent animation files:

```bash
uv run tbam input.mp4 --frames 12 --renderer direct --format webp --output out/animation.webp
```

GIF transparency is limited by the GIF format. Use GKA, `webp`, or `apng` when high-quality alpha is important.

To make CUDA available to rembg, the current Python environment needs an onnxruntime CUDA provider. If this command reports `CUDA available for rembg: no`, rembg is running on CPU in that environment.

## Upstream options

The CLI exposes the useful `rembg i` options with a `--rembg-*` prefix:

- `--rembg-alpha-matting`
- `--rembg-alpha-matting-foreground-threshold`
- `--rembg-alpha-matting-background-threshold`
- `--rembg-alpha-matting-erode-size`
- `--rembg-only-mask`
- `--rembg-post-process-mask`
- `--rembg-bgcolor R G B A`
- `--rembg-extras TEXT`

Example:

```bash
uv run tbam input.mp4 --frames 12 --output out/animation \
  --rembg-model sam \
  --rembg-extras '{"sam_prompt": [{"type": "point", "data": [724, 740], "label": 1}]}'
```

The CLI also exposes GKA options with a `--gka-*` prefix:

- `--gka-unique`
- `--gka-crop`
- `--gka-sprites`
- `--gka-algorithm top-down|left-right|binary-tree|diagonal|alt-diagonal`
- `--gka-prefix TEXT`
- `--gka-mini`
- `--gka-frame-duration SECONDS`
- `--gka-info`

Example:

```bash
uv run tbam input.mp4 --frames 12 --output out/animation \
  --gka-template canvas \
  --gka-sprites \
  --gka-algorithm binary-tree \
  --gka-mini
```

## Development

```bash
uv sync --dev
uv run pytest
```
