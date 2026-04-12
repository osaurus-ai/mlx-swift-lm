import BenchmarkHelpers
import Foundation
import MLX
import MLXLMCommon
import MLXVLM

struct BenchmarkOptions {
    var runs = 5
    var paligemmaTextTokens = 1024
    var paligemmaWindowSize = 256
    var smolLowMaxFrames = 8
    var smolHighMaxFrames = 20
    var videoPath = "Tests/MLXLMTests/Resources/1080p_30.mov"

    init(arguments: [String]) {
        var index = 0
        while index < arguments.count {
            let argument = arguments[index]
            func nextValue() -> String? {
                guard index + 1 < arguments.count else { return nil }
                index += 1
                return arguments[index]
            }

            switch argument {
            case "--runs":
                if let value = nextValue(), let parsed = Int(value) {
                    runs = parsed
                }
            case "--paligemma-text-tokens":
                if let value = nextValue(), let parsed = Int(value) {
                    paligemmaTextTokens = parsed
                }
            case "--paligemma-window-size":
                if let value = nextValue(), let parsed = Int(value) {
                    paligemmaWindowSize = parsed
                }
            case "--smol-low-max-frames":
                if let value = nextValue(), let parsed = Int(value) {
                    smolLowMaxFrames = parsed
                }
            case "--smol-high-max-frames":
                if let value = nextValue(), let parsed = Int(value) {
                    smolHighMaxFrames = parsed
                }
            case "--video-path":
                if let value = nextValue() {
                    videoPath = value
                }
            default:
                break
            }

            index += 1
        }
    }
}

struct SampleSummary {
    let stats: BenchmarkStats
    let meanPeakMiB: Double
    let maxPeakMiB: Double
}

struct SmolSummary {
    let stats: BenchmarkStats
    let meanPeakMiB: Double
    let maxPeakMiB: Double
    let meanFrameCount: Double
    let meanPromptTokens: Double
}

private struct BenchmarkTokenizer: Tokenizer {
    let bosToken: String? = nil
    let eosToken: String? = "<eos>"
    let unknownToken: String? = "<unk>"

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        let count = max(1, text.count / 4)
        return (0 ..< count).map { 1 + ($0 % 255) }
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        "User: Describe the video."
    }

    func convertTokenToId(_ token: String) -> Int? {
        switch token {
        case "<eos>":
            return 999
        case "<unk>":
            return 998
        default:
            return 1
        }
    }

    func convertIdToToken(_ id: Int) -> String? {
        "tok\(id)"
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        [1, 2, 3, 4]
    }
}

@main
struct VLMRuntimeBenchmarks {
    static func main() async throws {
        let options = BenchmarkOptions(arguments: Array(CommandLine.arguments.dropFirst()))
        printEnvironment(options: options)

        let paligemma = try benchmarkPaliGemma(options: options)
        printPaliGemma(paligemma, windowSize: options.paligemmaWindowSize)

        let smol = try await benchmarkSmolVLM(options: options)
        printSmolVLM(smol, options: options)
    }

    private static func printEnvironment(options: BenchmarkOptions) {
        print("# VLM Runtime Benchmarks")
        print("")
        print("- runs: \(options.runs)")
        print("- paligemma text tokens: \(options.paligemmaTextTokens)")
        print("- paligemma chunk window: \(options.paligemmaWindowSize)")
        print("- smol frame caps: \(options.smolLowMaxFrames) vs \(options.smolHighMaxFrames)")
        print("- video path: \(options.videoPath)")
        print("")
    }

    private static func benchmarkPaliGemma(options: BenchmarkOptions) throws
        -> (full: SampleSummary, chunked: SampleSummary)
    {
        let config = try makePaliGemmaConfiguration()
        let model = PaliGemma(config)
        let input = makePaliGemmaInput(config: config, textTokens: options.paligemmaTextTokens)

        try warmPaliGemma(model: model, input: input, windowSize: nil)
        try warmPaliGemma(model: model, input: input, windowSize: options.paligemmaWindowSize)

        let full = try samplePaliGemma(
            label: "Full prefill",
            model: model,
            input: input,
            windowSize: nil,
            runs: options.runs
        )
        let chunked = try samplePaliGemma(
            label: "Chunked prefill",
            model: model,
            input: input,
            windowSize: options.paligemmaWindowSize,
            runs: options.runs
        )

        return (full, chunked)
    }

    private static func warmPaliGemma(
        model: PaliGemma,
        input: LMInput,
        windowSize: Int?
    ) throws {
        let cache = model.newCache(parameters: nil)
        _ = try evaluatePaliGemma(model: model, input: input, cache: cache, windowSize: windowSize)
        Memory.clearCache()
    }

    private static func samplePaliGemma(
        label: String,
        model: PaliGemma,
        input: LMInput,
        windowSize: Int?,
        runs: Int
    ) throws -> SampleSummary {
        var times = [Double]()
        var peaks = [Double]()

        for run in 1 ... runs {
            let cache = model.newCache(parameters: nil)
            let startActive = Memory.activeMemory
            Memory.peakMemory = 0
            let start = CFAbsoluteTimeGetCurrent()
            _ = try evaluatePaliGemma(
                model: model,
                input: input,
                cache: cache,
                windowSize: windowSize
            )
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
            let peakActive = max(Memory.peakMemory, startActive)
            let peakDeltaMiB = bytesToMiB(max(0, peakActive - startActive))
            times.append(elapsed)
            peaks.append(peakDeltaMiB)
            print(
                "\(label) run \(run): \(format(elapsed))ms, peak delta \(format(peakDeltaMiB)) MiB"
            )
            Memory.clearCache()
        }

        return SampleSummary(
            stats: BenchmarkStats(times: times),
            meanPeakMiB: peaks.reduce(0, +) / Double(peaks.count),
            maxPeakMiB: peaks.max() ?? 0
        )
    }

    private static func evaluatePaliGemma(
        model: PaliGemma,
        input: LMInput,
        cache: [KVCache],
        windowSize: Int?
    ) throws -> PrepareResult {
        let result = try model.prepare(input, cache: cache, windowSize: windowSize)
        switch result {
        case .tokens(let text):
            MLX.eval(text.tokens)
        case .logits(let output):
            MLX.eval(output.logits)
        }
        let cacheState = cache.flatMap(\.state)
        if !cacheState.isEmpty {
            MLX.eval(cacheState)
        }
        return result
    }

    private static func benchmarkSmolVLM(options: BenchmarkOptions) async throws
        -> (low: SmolSummary, high: SmolSummary)
    {
        let videoURL = resolveVideoURL(path: options.videoPath)
        let input = UserInput(
            chat: [
                .user("Describe the video in detail.", videos: [.url(videoURL)])
            ]
        )
        let tokenizer = BenchmarkTokenizer()

        let lowProcessor = try makeSmolProcessor(
            maxFrames: options.smolLowMaxFrames,
            tokenizer: tokenizer
        )
        let highProcessor = try makeSmolProcessor(
            maxFrames: options.smolHighMaxFrames,
            tokenizer: tokenizer
        )

        _ = try await lowProcessor.prepare(input: input)
        _ = try await highProcessor.prepare(input: input)
        Memory.clearCache()

        let low = try await sampleSmol(
            label: "SmolVLM2 \(options.smolLowMaxFrames) frames",
            processor: lowProcessor,
            input: input,
            runs: options.runs
        )
        let high = try await sampleSmol(
            label: "SmolVLM2 \(options.smolHighMaxFrames) frames",
            processor: highProcessor,
            input: input,
            runs: options.runs
        )

        return (low, high)
    }

    private static func sampleSmol(
        label: String,
        processor: SmolVLMProcessor,
        input: UserInput,
        runs: Int
    ) async throws -> SmolSummary {
        var times = [Double]()
        var peaks = [Double]()
        var frameCounts = [Double]()
        var promptTokens = [Double]()

        for run in 1 ... runs {
            let startActive = Memory.activeMemory
            Memory.peakMemory = 0
            let start = CFAbsoluteTimeGetCurrent()
            let prepared = try await processor.prepare(input: input)
            MLX.eval(prepared.text.tokens)
            if let image = prepared.image {
                MLX.eval(image.pixels)
            }
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
            let peakActive = max(Memory.peakMemory, startActive)
            let peakDeltaMiB = bytesToMiB(max(0, peakActive - startActive))
            let frames = Double(prepared.image?.frames?.count ?? prepared.image?.pixels.dim(0) ?? 0)
            let tokens = Double(prepared.text.tokens.dim(1))

            times.append(elapsed)
            peaks.append(peakDeltaMiB)
            frameCounts.append(frames)
            promptTokens.append(tokens)

            print(
                "\(label) run \(run): \(format(elapsed))ms, frames \(Int(frames)), "
                    + "prompt tokens \(Int(tokens)), peak delta \(format(peakDeltaMiB)) MiB"
            )
            Memory.clearCache()
        }

        return SmolSummary(
            stats: BenchmarkStats(times: times),
            meanPeakMiB: peaks.reduce(0, +) / Double(peaks.count),
            maxPeakMiB: peaks.max() ?? 0,
            meanFrameCount: frameCounts.reduce(0, +) / Double(frameCounts.count),
            meanPromptTokens: promptTokens.reduce(0, +) / Double(promptTokens.count)
        )
    }

    private static func printPaliGemma(
        _ results: (full: SampleSummary, chunked: SampleSummary),
        windowSize: Int
    ) {
        print("## PaliGemma Prefill")
        print("")
        print("| Mode | Mean ms | Median ms | Mean peak delta MiB | Max peak delta MiB |")
        print("| --- | ---: | ---: | ---: | ---: |")
        print(
            "| Full prefill | \(format(results.full.stats.mean)) | \(format(results.full.stats.median)) | "
                + "\(format(results.full.meanPeakMiB)) | \(format(results.full.maxPeakMiB)) |")
        print(
            "| Chunked (\(windowSize)) | \(format(results.chunked.stats.mean)) | "
                + "\(format(results.chunked.stats.median)) | \(format(results.chunked.meanPeakMiB)) | "
                + "\(format(results.chunked.maxPeakMiB)) |")
        print("")
    }

    private static func printSmolVLM(
        _ results: (low: SmolSummary, high: SmolSummary),
        options: BenchmarkOptions
    ) {
        print("## SmolVLM2 Video Processing")
        print("")
        print(
            "| Frame cap | Mean ms | Median ms | Mean frames | Mean prompt tokens | "
                + "Mean peak delta MiB |"
        )
        print("| --- | ---: | ---: | ---: | ---: | ---: |")
        print(
            "| \(options.smolLowMaxFrames) | \(format(results.low.stats.mean)) | "
                + "\(format(results.low.stats.median)) | \(format(results.low.meanFrameCount)) | "
                + "\(format(results.low.meanPromptTokens)) | \(format(results.low.meanPeakMiB)) |")
        print(
            "| \(options.smolHighMaxFrames) | \(format(results.high.stats.mean)) | "
                + "\(format(results.high.stats.median)) | \(format(results.high.meanFrameCount)) | "
                + "\(format(results.high.meanPromptTokens)) | "
                + "\(format(results.high.meanPeakMiB)) |"
        )
        print("")
    }

    private static func makePaliGemmaConfiguration() throws -> PaliGemmaConfiguration {
        let json = """
            {
              "text_config": {
                "model_type": "gemma",
                "hidden_size": 256,
                "num_hidden_layers": 6,
                "intermediate_size": 1024,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "vocab_size": 4096
              },
              "vision_config": {
                "model_type": "siglip_vision_model",
                "hidden_size": 256,
                "num_hidden_layers": 4,
                "intermediate_size": 1024,
                "num_attention_heads": 8,
                "patch_size": 4,
                "projection_dim": 256,
                "image_size": 64,
                "num_channels": 3
              },
              "model_type": "paligemma",
              "vocab_size": 4096,
              "ignore_index": -100,
              "image_token_index": 4095,
              "hidden_size": 256,
              "pad_token_id": 0
            }
            """

        return try JSONDecoder().decode(PaliGemmaConfiguration.self, from: Data(json.utf8))
    }

    private static func makePaliGemmaInput(
        config: PaliGemmaConfiguration,
        textTokens: Int
    ) -> LMInput {
        let patchSize = config.visionConfiguration.patchSize
        let imageSize = config.visionConfiguration.imageSize
        let imageTokens = (imageSize / patchSize) * (imageSize / patchSize)
        let imageTokenId = Int32(config.imageTokenIndex)
        let textTokenIds = (0 ..< textTokens).map { Int32(1 + ($0 % 2048)) }
        let tokenIds = Array(repeating: imageTokenId, count: imageTokens) + textTokenIds
        let inputIds = MLXArray(tokenIds)[.newAxis, .ellipsis]
        let mask = ones(like: inputIds)
        let pixels = MLXArray.zeros([1, 3, imageSize, imageSize], type: Float.self)
        return LMInput(
            text: .init(tokens: inputIds, mask: mask),
            image: .init(pixels: pixels)
        )
    }

    private static func makeSmolProcessor(
        maxFrames: Int,
        tokenizer: some Tokenizer
    ) throws -> SmolVLMProcessor {
        let json = """
            {
              "image_mean": [0.5, 0.5, 0.5],
              "image_std": [0.5, 0.5, 0.5],
              "size": { "longest_edge": 2048 },
              "max_image_size": { "longest_edge": 384 },
              "video_sampling": { "fps": 3, "max_frames": \(maxFrames) },
              "image_seq_len": 64
            }
            """

        let config = try JSONDecoder().decode(
            SmolVLMProcessorConfiguration.self, from: Data(json.utf8))
        return SmolVLMProcessor(config, tokenizer: tokenizer)
    }

    private static func resolveVideoURL(path: String) -> URL {
        let direct = URL(fileURLWithPath: path)
        if FileManager.default.fileExists(atPath: direct.path) {
            return direct
        }

        let packageRoot = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        return packageRoot.appending(path: path)
    }

    private static func bytesToMiB(_ bytes: Int) -> Double {
        Double(bytes) / 1_048_576
    }

    private static func format(_ value: Double) -> String {
        String(format: "%.2f", value)
    }
}
