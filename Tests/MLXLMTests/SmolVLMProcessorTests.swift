import XCTest
@testable import MLXVLM

final class SmolVLMProcessorTests: XCTestCase {

    func testProcessorUsesConfiguredMaxVideoFrames() {
        let config = SmolVLMProcessorConfiguration(
            imageMean: [0.5, 0.5, 0.5],
            imageStd: [0.5, 0.5, 0.5],
            size: .init(longestEdge: 2048),
            maxImageSize: .init(longestEdge: 384),
            videoSampling: .init(fps: 3, maxFrames: 12),
            imageSequenceLength: 64
        )
        let processor = SmolVLMProcessor(config, tokenizer: TestTokenizer())

        XCTAssertEqual(processor.maxVideoFrames, 12)
        XCTAssertEqual(processor.targetVideoFPS, 3)
    }
}
