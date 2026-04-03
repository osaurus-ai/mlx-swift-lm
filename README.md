# MLX Swift LM

**by [Osaurus](https://osaurus.ai)** | Fork of [ml-explore/mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm)

A Swift package for building applications with large language models (LLMs) and vision language models (VLMs) on Apple Silicon, powered by [MLX Swift](https://github.com/ml-explore/mlx-swift).

This fork adds native support for [JANG](https://jangq.ai) mixed-precision quantized models and **Gemma 4** (26B MoE + 31B Dense) on top of the full upstream model library. Everything from the original mlx-swift-lm works as before — existing apps don't need to change anything.

## What's New

### JANG Model Support

[JANG](https://jangq.ai) models use per-layer mixed-precision quantization — attention layers at 6-8 bit, MLP/expert layers at 2-4 bit — for better quality at the same memory footprint as uniform quantization. This fork loads them natively:

- Automatic detection via `jang_config.json`
- Per-layer bit width inference from tensor shapes
- Zero code changes for app developers — just point at a JANG model directory
- Works with both JANG v1 (legacy) and v2 (current) formats

```swift
// Loading a JANG model is identical to loading any other model
let modelDirectory = URL(filePath: "/path/to/Gemma-4-26B-A4B-it-JANG_4M")
let container = try await loadModelContainer(
    from: modelDirectory,
    using: TokenizersLoader()
)
```

### Gemma 4

Full support for Google's Gemma 4 architecture, including both variants:

| Variant | Parameters | Architecture | Key Features |
|---------|-----------|-------------|--------------|
| **26B (A4B)** | 26B total, 4B active | MoE (128 experts, top-8) | Parallel dense MLP + expert branches, sigmoid-free softmax routing |
| **31B** | 31B | Dense | Standard transformer with mixed sliding/full attention |

Both share: dual attention types (sliding window + full), per-layer RoPE configuration, QK norms, V norm (parameterless), K=V sharing for full attention, logit softcapping, per-layer scaling.

**Tested models:**

| Model | Format | Speed |
|-------|--------|-------|
| Gemma 4 26B MoE | JANG 4M | 31.8 tok/s |
| Gemma 4 26B MoE | mlx-community 4bit | 34.5 tok/s |
| Gemma 4 31B Dense | mlx-community 4bit | 17.3 tok/s |
| Gemma 4 31B Dense | JANG 4M | 18.6 tok/s |

*Speeds measured on Apple Silicon. Your results will vary by chip and memory bandwidth.*

## Supported Models

### LLMs (50+ architectures)

Llama, Mistral, Phi, Phi-3, Phi-MoE, Gemma, Gemma 2, Gemma 3, Gemma 3n, **Gemma 4**, Qwen2, Qwen3, Qwen3-MoE, Qwen3.5, Qwen3.5-MoE, DeepSeek-V3, Cohere, OpenELM, InternLM2, Starcoder2, MiniCPM, Granite, Granite-MoE-Hybrid, MiMo, MiMo-V2-Flash, MiniMax, GLM-4, GLM-4-MoE, Falcon-H1, Bitnet, SmolLM3, ERNIE 4.5, LFM2, LFM2-MoE, Baichuan-M1, Exaone4, GPT-OSS, Lille-130m, OLMoE, OLMo2, OLMo3, Bailing-MoE, NanoChat, Nemotron-H, AF-MoE, Jamba, Mistral3, Apertus, and more.

### VLMs (15+ architectures)

PaliGemma, Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3.5, Qwen3.5-MoE, Gemma 3, SmolVLM2, FastVLM, Pixtral, Mistral3, LFM2-VL, GLM-OCR, Idefics3, and more.

### Embedders

Sentence Transformers, BERT, and other popular embedding models.

## Usage

Add the package to your `Package.swift`:

```swift
.package(url: "https://github.com/osaurus-ai/mlx-swift-lm", branch: "main"),
```

Then add your preferred tokenizer and downloader integrations:

```swift
.package(url: "https://github.com/DePasqualeOrg/swift-tokenizers-mlx", from: "0.1.0"),
.package(url: "https://github.com/DePasqualeOrg/swift-hf-api-mlx", from: "0.1.0"),
```

And add the libraries to your target:

```swift
.target(
    name: "YourTargetName",
    dependencies: [
        .product(name: "MLXLLM", package: "mlx-swift-lm"),
        .product(name: "MLXLMTokenizers", package: "swift-tokenizers-mlx"),
        .product(name: "MLXLMHuggingFace", package: "swift-hf-api-mlx"),
    ]),
```

### Quick Start

```swift
import MLXLLM
import MLXLMHuggingFace
import MLXLMTokenizers

let model = try await loadModel(
    from: HubClient.default,
    using: TokenizersLoader(),
    id: "mlx-community/Qwen3-4B-4bit"
)
let session = ChatSession(model)
print(try await session.respond(to: "What are two things to see in San Francisco?"))
```

### Loading a Local Model (Standard or JANG)

```swift
import MLXLLM
import MLXLMTokenizers

// Works for any model — standard MLX quantized, JANG, or unquantized
let modelDirectory = URL(filePath: "/path/to/model")
let container = try await loadModelContainer(
    from: modelDirectory,
    using: TokenizersLoader()
)
```

JANG models are detected automatically. No special flags or configuration needed.

### Tokenizer and Downloader Integrations

MLX Swift LM focuses on model implementations. Tokenization and model downloading are handled by separate packages:

| Downloader package | Adapter |
|-|-|
| [huggingface/swift-huggingface](https://github.com/huggingface/swift-huggingface) | [DePasqualeOrg/swift-huggingface-mlx](https://github.com/DePasqualeOrg/swift-huggingface-mlx) |
| [DePasqualeOrg/swift-hf-api](https://github.com/DePasqualeOrg/swift-hf-api) | [DePasqualeOrg/swift-hf-api-mlx](https://github.com/DePasqualeOrg/swift-hf-api-mlx) |

| Tokenizer package | Adapter |
|-|-|
| [DePasqualeOrg/swift-tokenizers](https://github.com/DePasqualeOrg/swift-tokenizers) | [DePasqualeOrg/swift-tokenizers-mlx](https://github.com/DePasqualeOrg/swift-tokenizers-mlx) |
| [huggingface/swift-transformers](https://github.com/huggingface/swift-transformers) | [DePasqualeOrg/swift-transformers-mlx](https://github.com/DePasqualeOrg/swift-transformers-mlx) |

> **Note:** The adapters are offered for convenience and are not required. You can also use tokenizer and downloader packages directly by setting up the required protocol conformance. See the integration packages for examples.

## How JANG Loading Works

For developers integrating JANG support into their own apps, here's what happens under the hood:

1. **Detection** — The factory checks for `jang_config.json` in the model directory.
2. **Config parsing** — `JangLoader` reads the JANG profile (bit widths, block size, source model info).
3. **Weight loading** — Standard `.safetensors` files are loaded normally (JANG v2 is MLX-native).
4. **Sanitize** — Model-specific weight key remapping runs (VLM prefix stripping, expert key normalization).
5. **Quantization inference** — Per-layer bit widths are inferred from `weight.shape` vs `scales.shape` for each tensor.
6. **Apply** — The inferred per-layer quantization replaces any uniform quantization from `config.json`.

The entire pipeline is transparent. If `jang_config.json` doesn't exist, the standard MLX loading path runs unchanged.

## Roadmap

Planned additions to this fork:

- **Gemma 4 VLM** — Full vision-language support (vision encoder, processor, image token scattering)
- **Native TurboQuant** — Quantization-aware weight format for faster loading
- **Paged KV Cache** — Memory-efficient caching for long contexts
- **Hybrid SSM** — Support for state-space model layers (Mamba/Jamba-style)
- **Prefix Caching** — Reuse KV cache across prompts with shared prefixes
- **Async L2 Disk Cache** — Spill KV cache to disk for very long contexts
- **Re-derive** — Dynamic re-quantization at load time

## Migrating from Upstream

If you're switching from `ml-explore/mlx-swift-lm`, just change your package URL:

```swift
// Before
.package(url: "https://github.com/ml-explore/mlx-swift-lm", branch: "main"),

// After
.package(url: "https://github.com/osaurus-ai/mlx-swift-lm", branch: "main"),
```

Everything else stays the same. All upstream APIs, model architectures, and integrations are preserved. You gain JANG support and Gemma 4 for free.

If you're migrating from upstream 2.x to this fork (which is based on 3.x), see the [version 3 migration guide](#migrating-to-version-3) below.

## Migrating to Version 3

Version 3 decouples the tokenizer and downloader implementations. The key changes:

### New dependencies

```swift
// Before (2.x) -- single dependency
.package(url: "https://github.com/ml-explore/mlx-swift-lm/", from: "2.30.0"),

// After (3.x) -- core + adapters
.package(url: "https://github.com/osaurus-ai/mlx-swift-lm/", branch: "main"),
.package(url: "https://github.com/DePasqualeOrg/swift-tokenizers-mlx/", from: "0.1.0"),
.package(url: "https://github.com/DePasqualeOrg/swift-hf-api-mlx/", from: "0.1.0"),
```

### New imports

```swift
// Before (2.x)
import MLXLLM

// After (3.x)
import MLXLLM
import MLXLMHuggingFace  // Downloader adapter
import MLXLMTokenizers   // Tokenizer adapter
```

### Loading API changes

- `hub:` parameter is now `from:` (accepts any `Downloader` or local `URL`)
- `HubApi` is now `HubClient`
- `decode(tokens:)` is renamed to `decode(tokenIds:)`

```swift
// Before (2.x)
let container = try await loadModelContainer(id: "mlx-community/Qwen3-4B-4bit")

// After (3.x)
let container = try await loadModelContainer(
    from: HubClient.default,
    id: "mlx-community/Qwen3-4B-4bit"
)
```

## Documentation

- [Porting and implementing models](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/porting)
- [MLXLMCommon](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon): Common API for LLM and VLM
- [MLXLLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxllm): Large language model implementations
- [MLXVLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxvlm): Vision language model implementations
- [MLXEmbedders](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxembedders): Embedding model implementations

## Files Changed (vs. Upstream)

| File | Change | Purpose |
|------|--------|---------|
| `Libraries/MLXLLM/Models/Gemma4Text.swift` | New | Full Gemma 4 model (dense + MoE, dual attention, v_norm, K=V) |
| `Libraries/MLXLMCommon/JangLoader.swift` | New | JANG detection, config parsing, per-layer quantization inference |
| `Libraries/MLXLMCommon/Load.swift` | Modified | JANG integration, per-layer quantization key remapping |
| `Libraries/MLXLLM/LLMModelFactory.swift` | Modified | JANG detection, `gemma4` / `gemma4_text` registration |
| `Libraries/MLXVLM/VLMModelFactory.swift` | Modified | JANG detection for VLM path |

## License

MIT License. See [LICENSE](LICENSE) for details.

Based on [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) by Apple's ML Explore team.

## Acknowledgments

- [Apple ML Explore](https://github.com/ml-explore) for MLX and mlx-swift-lm
- [JANG](https://jangq.ai) mixed-precision quantization format
- [Google DeepMind](https://deepmind.google) for the Gemma 4 architecture
