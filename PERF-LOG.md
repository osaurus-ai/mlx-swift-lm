# Performance Optimization Log — mlx-swift-lm MoE Speed

## Target
- Gemma4 26B MoE: **80 tok/s** (Python VMLX / osa-jang baseline)
- NemotronH 30B-A3B: **143 tok/s** (osa-jang baseline)

## Current: ~50 tok/s Gemma4, ~48 tok/s NemotronH

## What Worked

| Fix | Impact | Commit | Notes |
|-----|--------|--------|-------|
| bfloat16 MoE conversion | +38-75% | pre-existing | Prevents Metal float16→float32 promotion on mixed-dtype MoE ops |
| Compiled SwiGLU/GeGLU activations | +5% | pre-existing | Fuses `silu(gate)*x` into single Metal dispatch |
| Memory.clearCache() every 256 tokens | +32% (37→49) | 91c3537 | Reduces Metal allocator fragmentation from expert weight cycling |
| Memory.clearCache() after prefill chunks | included above | 91c3537 | Matches Python mlx-lm |
| Memory.withWiredLimit (upstream API) | ~0% | 285fa64 | Async version goes through WiredMemoryManager, overhead negates benefit |
| Compiled forward (non-Gemma) | ~19% on Qwen | 89d2d21 | compile(shapeless:true) on model forward. Disabled for Gemma (crashes), hybrid, TQ |
| Eager cache state eval in asyncEval | ~0% | 89d2d21 | Rebuilds innerState() per step, no benefit |

## What Didn't Work / Didn't Help

| Attempt | Result | Why |
|---------|--------|-----|
| Stream.withNewDefaultStream per step() | **-40% SLOWER** (49→28) | Creates/destroys Metal stream per token. Context leak errors. |
| Flat decode loop (fastGenerate) bypassing TokenIterator | **0% gain** (49.9 vs 49.6) | Proves Swift overhead is NOT the bottleneck. GPU execution time dominates. |
| Pre-computed _cacheStateArrays (one allocation) | **0% gain** | Same as above — Swift allocation overhead is negligible vs GPU time |
| True double-buffered pipeline in TestRunner | **0% gain** | Build next graph before materializing current — no measurable overlap |
| Memory.cacheLimit = physMem/4 | **0% measurable** | May help under memory pressure but no benchmark difference |

## What Osa-Jang Does That We Can't (Yet)

| Feature | Osa-Jang Code | Impact (est.) | Blocker |
|---------|---------------|---------------|---------|
| Direct `mlx_set_wired_limit` | line 327, calls C API directly | **+30-50%** | Not exposed usably in upstream mlx-swift (sync version is no-op) |
| Persistent generation stream | NOT USED (osa-jang doesn't use a separate stream!) | N/A | Red herring — osa-jang runs on default stream |
| Compiled forward for non-Gemma | line 1192 | +19% on Qwen/Mistral | Already implemented, excluded for Gemma |
| `compiledCategoricalSample` | line 1235 | +2-3% | Not ported yet |
| Metal cache limit tuning | line 332 | unknown | Needs `Memory.cacheLimit` + wired limit together |

## Key Insight

The 50→143 gap on NemotronH is almost entirely **wired memory**. Osa-jang pins 75% of physical RAM as wired GPU memory (`mlx_set_wired_limit`), preventing macOS from paging model weights to SSD. Without this, every token requires fetching ~3B parameters from SSD/swap, adding massive latency.

The upstream mlx-swift `Memory.withWiredLimit` async API wraps this in WiredMemoryManager coordination overhead that negates the benefit. The osaurus-ai/mlx-swift fork adds `Memory.setWiredLimit()` as a direct public wrapper around `mlx_set_wired_limit`.

## Fork Status (osaurus-ai/mlx-swift)

Two commits on top of upstream:
1. `2224855` — `Stream.withDefaultStream(_:body:)` — reuse persistent stream (NOT helpful, see above)
2. `3e10c33` — `Memory.setWiredLimit()` — direct wired limit (CRITICAL, not tested yet)

### Metal Library Issue
The TestRunner executable needs `mlx.metallib` next to the binary. When switching mlx-swift versions, stale build caches cause "Failed to load the default metallib" errors. Fix: copy metallib from an Xcode build: `cp .../mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib .build/release/mlx.metallib`

## Next Steps

1. Use osaurus-ai/mlx-swift fork with `Memory.setWiredLimit()`
2. Call `Memory.setWiredLimit(physMem * 3/4)` at model load time (not per-generate)
3. Test: if wired limit alone closes the gap, the fork is justified
4. Port `compiledCategoricalSample` from osa-jang for non-greedy sampling
5. Investigate whether Gemma4 can use compiled forward with MoE routing workaround
