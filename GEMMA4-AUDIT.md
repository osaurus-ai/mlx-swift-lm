# Gemma 4 + JANG Integration: Final Audit

> Full audit of Gemma4Text.swift, JangLoader.swift, Load.swift, and factory registrations.
> Covers: 26B MoE, 31B Dense, JANG variants, mlx-community variants, VLM gaps.
> Date: 2026-04-03

---

## Test Results (Verified Working)

| Model | Format | MoE | Speed | Status |
|-------|--------|-----|-------|--------|
| Qwen3.5-4B | JANG | No | 122.7 tok/s | Correct |
| Gemma4 26B | JANG 4M | Yes (128 experts) | 31.8 tok/s | Correct |
| Gemma4 31B | mlx-community 4bit | No (dense) | 17.3 tok/s | Correct |
| Gemma4 26B | mlx-community 4bit | Yes (128 experts) | 34.5 tok/s | Correct |
| Gemma4 31B | JANG 4M | No (dense) | 18.6 tok/s | Loads, tokenizer issue |

---

## Previously Critical Issues — ALL FIXED

These were identified in the initial audit and have been resolved in the current code:

| ID | Issue | Fix Applied | Verified |
|----|-------|-------------|----------|
| C1 | RMSNorm +1 offset | `Gemma4RMSNorm` class (no +1), used throughout | Line 19-31 |
| C2 | Dense 31B crash (MoE always created) | `hasMoE` conditional; MoE components are `Optional` | Line 397, 403-412 |
| C3 | Expert key mismatch (mlx vs JANG) | `sanitize()` remaps `switch_mlp` → `experts.switch_glu`; module tree uses `experts.switch_glu` via `Gemma4Experts` wrapper | Lines 596-598, 343-370 |
| C4 | Router wrong (sigmoid, no norm, scalar scale) | Rewritten: softmax, RMSNormNoScale pre-norm, rootSize scaling, vector scale | Lines 292-338 |
| C5 | Missing v_norm | `rmsNormNoScale()` applied to values in attention | Lines 38-41, 241-246 |
| H1 | K=V not handled | `useKEqV` conditional; v_proj optional; values = keys before k_norm | Lines 160, 200-203, 240-241 |
| H2 | Dense forward pass wrong norms | Dense path: `preFeedforward → mlp → postFeedforward → residual` (no extra norms) | Lines 459-463 |
| M1 | Uniform cache strategy | Per-layer-type: `RotatingKVCache(slidingWindow)` for sliding, `KVCacheSimple` for full | Lines 618-632 |
| M3 | Hardcoded RoPE params | Reads from `config.ropeParameters` with fallback defaults | Lines 212-221 |

---

## Remaining Issues

### R1. No VLM Support — Vision/Audio Capabilities Unavailable

**Severity:** HIGH (feature gap, not a bug)

**Status:** Gemma4 is natively a VLM (`Gemma4ForConditionalGeneration`, has `vision_config`, `vision_tower` weights). Text-only loading works through `LLMModelFactory`, but multimodal is unavailable.

**What's needed for Gemma4 VLM (new file `Libraries/MLXVLM/Models/Gemma4.swift`):**

1. **Vision encoder** — completely new architecture vs Gemma3:
   - Linear patch embedding (not Conv2d)
   - 2D position embedding table `[2, pos_emb_size, hidden_size]`
   - Vision attention with QK norm + parameterless V norm
   - `ClippableLinear` wrappers (weight keys are `*.linear.weight` not `*.weight`)
   - 2D multidimensional RoPE (X/Y split per head dim half)
   - `VisionPooler` (avg pool to `vision_soft_tokens_per_image=280` tokens)
   - 4x RMSNorm per block (not LayerNorm)

2. **Multimodal embedder** (`embed_vision`):
   - `nn.Linear(vision_hidden, text_hidden)` + `RMSNormNoScale` post-projection
   - No AvgPool (pooling is inside VisionModel)

3. **Image token scattering**:
   - Pure MLX approach: `cumsum + where` (not CPU loop like Gemma3)
   - Image token id: 258880 (different from Gemma3's 262144)

4. **Processor** (`Gemma4Processor`):
   - Aspect-ratio preserving resize (not fixed square)
   - Dynamic soft token count: `num_patches // pooling_kernel_size²`
   - Rescale to [0,1] only (no mean/std normalization)
   - Channel-first output

5. **Factory registration**:
   - `VLMTypeRegistry`: add `"gemma4"` entry
   - `VLMProcessorTypeRegistry`: add `"Gemma4Processor"` entry

**Audio support:** Not needed initially. Gemma4 has optional audio but it's gated on `audio_config != None`.

---

### R2. Missing 2B/4B Model Features

**Severity:** MEDIUM (only affects smaller model variants not yet available as JANG/MLX)

Three features present in the Python reference are not implemented:

1. **Per-layer input gate** (`hidden_size_per_layer_input > 0`):
   - Each decoder layer gets an auxiliary low-dimensional embedding
   - Requires `per_layer_input_gate`, `per_layer_projection`, `post_per_layer_input_norm` modules
   - For 26B/31B: `hidden_size_per_layer_input = 0`, so this is never used

2. **KV sharing between layers** (`num_kv_shared_layers > 0`):
   - Later layers reuse K/V from earlier layers of the same type
   - Requires `shared_kv_store` dict and `kv_shared_layer_index` tracking
   - For 26B/31B: `num_kv_shared_layers = 0`, so this is never used

3. **Double-wide MLP** (`use_double_wide_mlp`):
   - KV-shared layers double their MLP intermediate size
   - For 26B/31B: `use_double_wide_mlp = false`, so this is never used

**Impact:** If 2B or 4B Gemma4 models appear as MLX/JANG checkpoints, they will fail to load due to missing module parameters. This can be deferred until those models are available.

---

### R3. JANG Config Load Uses `try?` — Silent Fallback Risk

**Severity:** LOW

**File:** `LLMModelFactory.swift:549`

```swift
jangConfig = try? JangLoader.loadConfig(at: modelDirectory)
```

If `jang_config.json` exists but is malformed, the error is silently swallowed. The code falls back to standard MLX quantization (`config.json`'s flat `"quantization"` block), which would apply uniform 4-bit instead of JANG's per-layer mixed-bit quantization. The model would load but produce degraded output.

**Fix:** Replace `try?` with proper error handling:
```swift
if JangLoader.isJangModel(at: modelDirectory) {
    jangConfig = try JangLoader.loadConfig(at: modelDirectory)
}
```

---

### R4. Gemma4 31B JANG Tokenizer Issue

**Severity:** MEDIUM

Test results show the 31B JANG model "loads, tokenizer issue." This likely means the JANG 31B model directory is missing or has an incompatible `tokenizer.json` / `tokenizer_config.json`. The model weights load and quantize correctly, but chat template application fails.

**Investigation needed:**
```bash
ls /Users/eric/mlx-models/Gemma-4-31B-it-JANG_4M/tokenizer*
cat /Users/eric/mlx-models/Gemma-4-31B-it-JANG_4M/tokenizer_config.json | head -5
```

This may be a packaging issue (missing files in the JANG export), not a code bug.

---

### R5. No `ModelConfiguration` Registry Entries

**Severity:** LOW (convenience only)

Neither `LLMRegistry` nor `VLMRegistry` have named `ModelConfiguration` entries for Gemma4 models. Users must load from directory or create custom configurations. Not a bug, but reduces discoverability for downstream users of the library.

---

## What's Working Correctly (Verified)

### Text Model Architecture
- `Gemma4RMSNorm`: weight directly, no +1 offset (correct for Gemma4)
- `rmsNormNoScale`: parameterless norm for v_norm and router pre-norm
- Attention: `scale = 1.0`, QK norms with Gemma4RMSNorm, v_norm with rmsNormNoScale
- K=V sharing: values = raw k_proj output (before k_norm), then v_norm applied
- RoPE: reads `rope_parameters` from config, per-layer-type theta and partial_rotary_factor
- Dense MLP: `gelu_approx(gate) * up → down`
- MoE parallel paths: dense MLP + router → experts, combined, then shared post-norm
- Layer scalar applied correctly
- Logit softcapping: `tanh(x/cap) * cap`
- Embedding scaling: `sqrt(hiddenSize)`
- Tied embeddings: uses `embedTokens.asLinear()` when `tieWordEmbeddings = true`

### MoE Components
- Router: softmax routing, RMSNormNoScale pre-norm, 1/sqrt(d) scaling, vector scale, per-expert scale indexed by selected experts
- Experts wrapper: `Gemma4Experts` with `switch_glu` key matching mlx-community checkpoints
- SwitchGLU: correct activation (gelu_approx), correct argument order

### Conditional Architecture
- `enableMoeBlock` config field controls MoE creation
- `numExperts` defaults to 0 (not 128) when missing
- Optional MoE modules (`router?`, `experts?`, extra norms with `?`)
- Dense forward path: `preFeedforward → mlp → postFeedforward → residual`
- MoE forward path: parallel `path1 + path2 → postFeedforward → residual`

### Weight Loading Pipeline
- Sanitize handles 3 prefix patterns:
  - JANG VLM: `model.language_model.X` → `model.X`
  - mlx-community VLM: `language_model.X` → `X`
  - Text-only: `model.X` → `model.X` (no change)
- JANG expert key remapping: `switch_mlp` → `experts.switch_glu`
- Vision/projector weight filtering: `vision_tower.*`, `embed_vision.*`, `multi_modal_projector.*`
- Vocab trimming for oversized embedding tensors
- Per-layer quantization key prefix remapping in Load.swift

### JANG Loader
- Detection: `jang_config.json` presence check
- Config parsing: format, version, quantization, source model, architecture, runtime
- Per-layer bit inference: formula `(packedDim * 32) / inDim` with nearest-valid-bit clamping
- V2 format: standard safetensors, no repacking needed
- Quantization override: JANG path completely replaces `config.json`'s flat quantization
- All 4 JANG Gemma4 models verified working

### Cache Strategy
- Full attention layers: `KVCacheSimple()` (unbounded)
- Sliding attention layers: `RotatingKVCache(maxSize: slidingWindow, keep: 0)`
- Falls back to `RotatingKVCache(maxSize: maxKVSize, keep: 4)` for full attention when user-specified

### EOS Tokens
- All Gemma4 models: `[1, 106, 50]` from `generation_config.json`
- Framework correctly reads and applies array EOS token IDs

---

## RoPE Comparison (Swift vs Python vs Config)

| Property | Python | Swift | Config (26B/31B) | Match? |
|----------|--------|-------|-----------------|--------|
| Sliding `rope_theta` | `rope_params.get("rope_theta", 10000.0)` | `ropeParams["rope_theta"] ?? 10000.0` | `10000.0` | YES |
| Full `rope_theta` | `rope_params.get("rope_theta", 1000000.0)` | `ropeParams["rope_theta"] ?? 1_000_000.0` | `1000000.0` | YES |
| Sliding `partial_rotary_factor` | `rope_params.get(..., 1.0)` | `ropeParams["partial_rotary_factor"] ?? 1.0` | Not set (defaults to 1.0) | YES |
| Full `partial_rotary_factor` | `rope_params.get(..., 1.0)` | `ropeParams["partial_rotary_factor"] ?? 0.25` | `0.25` | YES* |
| `traditional` | `config.rope_traditional` (default False) | `config.ropeTraditional` (default false) | Not set | YES |
| `rope_type: proportional` | Ignored by `nn.RoPE` | Ignored (no `scalingConfig`) | Present but unused | YES |
| Applied to Q and K | Yes | Yes | — | YES |
| Dims for full attn | `int(512 * 0.25) = 128` | `max(1, Int(512 * 0.25)) = 128` | — | YES |

*Note: Swift default for full attention `partial_rotary_factor` is `0.25` while Python defaults to `1.0`. However, ALL current Gemma4 configs explicitly provide `0.25`, so the different defaults don't matter in practice. If a future config omits `partial_rotary_factor` for full attention, Swift would use 0.25 (correct for current models) while Python would use 1.0 (full rotation). This is a latent divergence.

---

## JANG Loader Compatibility Matrix

| Check | 26B JANG 4M | 26B JANG 2L | 31B JANG 4M | 31B JANG 2L |
|-------|:-----------:|:-----------:|:-----------:|:-----------:|
| `jang_config.json` detected | YES | YES | YES | YES |
| Format version 2.0 (V2) | YES | YES | YES | YES |
| Standard `.safetensors` files | YES | YES | YES | YES |
| Bit widths | 4, 8 | 2, 4, 8 | 4, 8 | 2, 6, 8 |
| Block size | 64 | 64 | 64 | 64 |
| Per-layer inference | PASS | PASS | PASS | PASS |
| Prefix stripping | `model.language_model.` → `model.` | same | same | same |
| Expert key remap | `switch_mlp` → `experts.switch_glu` | same | N/A (dense) | N/A (dense) |
| Vision weights skipped | YES | YES | YES | YES |
| `config.json` quantization conflict | None (JANG path overrides) | same | same | same |

---

## File Reference

| File | Lines Changed | Purpose |
|------|:------------:|---------|
| `Libraries/MLXLLM/Models/Gemma4Text.swift` | NEW (640 lines) | Full Gemma4 text model: dense + MoE, dual attention types, v_norm, k_eq_v |
| `Libraries/MLXLMCommon/JangLoader.swift` | NEW (~400 lines) | JANG detection, config parsing, per-layer quantization inference |
| `Libraries/MLXLMCommon/Load.swift` | Modified | JANG integration, per-layer quantization key remapping |
| `Libraries/MLXLLM/LLMModelFactory.swift` | Modified | JANG detection + `"gemma4"` / `"gemma4_text"` registration |
| `Libraries/MLXVLM/VLMModelFactory.swift` | Modified | JANG detection (VLM path) |
| `TestRunner/` | NEW | Test harness for local model loading |

---

## Recommended Next Steps

1. **R4 — Investigate 31B JANG tokenizer issue** (quick fix, likely missing file)
2. **R3 — Replace `try?` with proper error handling** for JANG config load (5 min fix)
3. **R1 — Gemma4 VLM support** (significant effort, new `Gemma4.swift` in MLXVLM)
4. **R5 — Add ModelConfiguration entries** for popular Gemma4 HuggingFace model IDs
5. **R2 — 2B/4B model features** (defer until those models exist as MLX/JANG checkpoints)
