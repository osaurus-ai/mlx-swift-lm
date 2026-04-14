import MLX
import MLXLMCommon
import XCTest
@testable import MLXVLM

final class VLMPrepareUtilsTests: XCTestCase {

    func testSquareMaskPrefixHelpersSliceLeadingAndTrailingBlocks() {
        let mask = MLXArray(0 ..< 16).reshaped([1, 1, 4, 4])

        let prefix = vlmTakeSquareMaskPrefix(mask, count: 2)
        XCTAssertEqual(prefix.shape.map { Int($0) }, [1, 1, 2, 2])
        XCTAssertEqual(prefix.asArray(Int32.self), [0, 1, 4, 5])

        let trailing = vlmDropSquareMaskPrefix(mask, count: 2)
        XCTAssertEqual(trailing.shape.map { Int($0) }, [1, 1, 2, 2])
        XCTAssertEqual(trailing.asArray(Int32.self), [10, 11, 14, 15])
    }

    func testPaliGemmaPrepareHonorsChunkedWindowSize() throws {
        let config = try makeTinyPaliGemmaConfiguration()
        let model = PaliGemma(config)

        let imageToken = Int32(config.imageTokenIndex)
        let inputIds = MLXArray([imageToken, imageToken, imageToken, imageToken, 1, 2])[
            .newAxis, .ellipsis]
        let mask = ones(like: inputIds)
        let pixels = MLXArray.zeros([1, 3, 4, 4], type: Float.self)
        let input = LMInput(
            text: .init(tokens: inputIds, mask: mask),
            image: .init(pixels: pixels)
        )

        let cache = model.newCache(parameters: nil)
        let result = try model.prepare(input, cache: cache, windowSize: 2)

        switch result {
        case .logits(let output):
            XCTAssertEqual(output.logits.dim(0), 1)
            XCTAssertEqual(output.logits.dim(1), 2)
            XCTAssertEqual(output.logits.dim(2), config.vocabularySize)
        case .tokens:
            XCTFail("Expected prepare() to return logits")
        }

        XCTAssertEqual(cache.count, config.textConfiguration.hiddenLayers)
        XCTAssertTrue(cache.allSatisfy { $0.offset == 6 })
    }

    private func makeTinyPaliGemmaConfiguration() throws -> PaliGemmaConfiguration {
        let json = """
            {
              "text_config": {
                "model_type": "gemma",
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 64
              },
              "vision_config": {
                "model_type": "siglip_vision_model",
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "patch_size": 2,
                "projection_dim": 8,
                "image_size": 4,
                "num_channels": 3
              },
              "model_type": "paligemma",
              "vocab_size": 64,
              "ignore_index": -100,
              "image_token_index": 63,
              "hidden_size": 8,
              "pad_token_id": 0
            }
            """

        return try JSONDecoder().decode(PaliGemmaConfiguration.self, from: Data(json.utf8))
    }
}
