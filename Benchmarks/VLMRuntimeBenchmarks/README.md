# VLM Runtime Benchmarks

This benchmark target measures two MLX VLM runtime behaviors that matter for
real prompt preparation costs:

- `PaliGemma.prepare(..., windowSize:)` full-sequence versus chunked multimodal
  prefill
- `SmolVLM2` video prompt preparation under different `video_sampling.max_frames`
  caps

## Build

```bash
swift build -c release --product VLMRuntimeBenchmarks
```

## Run

```bash
./.build/arm64-apple-macosx/release/VLMRuntimeBenchmarks --runs 5
```

Useful flags:

- `--runs <n>`
- `--paligemma-text-tokens <n>`
- `--paligemma-window-size <n>`
- `--smol-low-max-frames <n>`
- `--smol-high-max-frames <n>`
- `--video-path <path>`

Defaults:

- video path: `Tests/MLXLMTests/Resources/1080p_30.mov`
- PaliGemma text tokens: `1024`
- PaliGemma chunk window: `256`
- SmolVLM2 frame-cap comparison: `8` versus `20`

## Output

The executable prints Markdown-friendly tables with:

- mean and median runtime
- mean and max MLX peak-memory deltas
- effective SmolVLM2 frame counts and prompt-token counts

## Metal Library Note

Under pure SwiftPM CLI execution, MLX may require `mlx.metallib` or
`Resources/default.metallib` to be staged beside the built executable. If the
benchmark exits immediately with a Metal library lookup failure, stage the MLX
metal library next to the release binary before running the benchmark.
