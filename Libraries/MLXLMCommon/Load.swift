// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Load model weights.
///
/// This is typically called via ``ModelFactory/load(from:configuration:progressHandler:)``.
/// This function loads all `safetensor` files in the given `modelDirectory`,
/// calls ``LanguageModel/sanitize(weights:metadata:)`` to allow per-model preprocessing,
/// applies optional quantization, and updates the model with the weights.
///
/// When a JANG model is detected (via `jangConfig`), per-layer bit widths are
/// inferred from tensor shapes automatically. Standard MLX models are unaffected.
public func loadWeights(
    modelDirectory: URL, model: LanguageModel,
    quantization: BaseConfiguration.Quantization? = nil,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil,
    jangConfig: JangConfig? = nil
) throws {
    // load the weights and collect metadata from the first safetensor file
    var weights = [String: MLXArray]()
    var metadata = [String: String]()

    // JANG v1 models use .jang.safetensors files that need uint8->uint32 repacking
    if let jangConfig, !jangConfig.isV2, JangLoader.hasV1Weights(at: modelDirectory) {
        weights = try JangLoader.loadV1Weights(at: modelDirectory)
    } else {
        let enumerator = FileManager.default.enumerator(
            at: modelDirectory, includingPropertiesForKeys: nil)!
        for case let url as URL in enumerator {
            if url.pathExtension == "safetensors" {
                let (w, m) = try loadArraysAndMetadata(url: url)
                for (key, value) in w {
                    weights[key] = value
                }
                if metadata.isEmpty {
                    metadata = m
                }
            }
        }
    }

    // per-model cleanup (models can inspect metadata to customize behavior)
    weights = model.sanitize(weights: weights, metadata: metadata)

    // Determine quantization: JANG models infer per-layer bit widths from tensor shapes.
    // Standard MLX models use the quantization from config.json as before.
    let effectivePerLayerQuantization: BaseConfiguration.PerLayerQuantization?
    if let jangConfig {
        effectivePerLayerQuantization = JangLoader.inferPerLayerQuantization(
            weights: weights, jangConfig: jangConfig)
    } else if let perLayerQuantization {
        // Remap perLayerQuantization keys to match sanitized weight paths.
        // Config.json may use VLM-prefixed keys like "language_model.model.layers.0..."
        // but sanitize() strips these to "model.layers.0..."
        var remappedPerLayer = [String: BaseConfiguration.QuantizationOption]()
        for (key, value) in perLayerQuantization.perLayerQuantization {
            var newKey = key
            if newKey.hasPrefix("language_model.model.") {
                newKey = String(newKey.dropFirst("language_model.".count))
            } else if newKey.hasPrefix("language_model.") {
                newKey = String(newKey.dropFirst("language_model.".count))
            }
            remappedPerLayer[newKey] = value
        }
        effectivePerLayerQuantization = BaseConfiguration.PerLayerQuantization(
            quantization: perLayerQuantization.quantization,
            perLayerQuantization: remappedPerLayer
        )
    } else {
        effectivePerLayerQuantization = nil
    }

    // quantize if needed
    if quantization != nil || effectivePerLayerQuantization != nil {
        quantize(model: model) { path, module in
            if weights["\(path).scales"] != nil {
                if let effectivePerLayerQuantization {
                    return effectivePerLayerQuantization.quantization(layer: path)?.asTuple
                } else {
                    return quantization?.asTuple
                }
            } else {
                return nil
            }
        }
    }

    // apply the loaded weights
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [.all])

    eval(model)
}
