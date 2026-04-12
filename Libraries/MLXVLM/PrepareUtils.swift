import MLX
import MLXLMCommon

func vlmSequenceLength(_ array: MLXArray) -> Int {
    switch array.ndim {
    case 1:
        return array.dim(0)
    case 2, 3:
        return array.dim(1)
    default:
        fatalError("Unsupported VLM prefill rank: \(array.ndim)")
    }
}

func vlmEmbeddingSequenceLength(_ array: MLXArray) -> Int {
    switch array.ndim {
    case 2:
        return array.dim(0)
    case 3:
        return array.dim(1)
    default:
        fatalError("Unsupported VLM embedding rank: \(array.ndim)")
    }
}

func vlmTakePrefix(_ array: MLXArray, count: Int) -> MLXArray {
    switch array.ndim {
    case 1:
        return array[..<count]
    case 2:
        return array[0..., ..<count]
    case 3:
        return array[0..., ..<count, 0...]
    default:
        fatalError("Unsupported VLM prefill rank: \(array.ndim)")
    }
}

func vlmDropPrefix(_ array: MLXArray, count: Int) -> MLXArray {
    switch array.ndim {
    case 1:
        return array[count...]
    case 2:
        return array[0..., count...]
    case 3:
        return array[0..., count..., 0...]
    default:
        fatalError("Unsupported VLM prefill rank: \(array.ndim)")
    }
}

func vlmTakeEmbeddingPrefix(_ array: MLXArray, count: Int) -> MLXArray {
    switch array.ndim {
    case 2:
        return array[..<count, 0...]
    case 3:
        return array[0..., ..<count, 0...]
    default:
        fatalError("Unsupported VLM embedding rank: \(array.ndim)")
    }
}

func vlmDropEmbeddingPrefix(_ array: MLXArray, count: Int) -> MLXArray {
    switch array.ndim {
    case 2:
        return array[count..., 0...]
    case 3:
        return array[0..., count..., 0...]
    default:
        fatalError("Unsupported VLM embedding rank: \(array.ndim)")
    }
}

func vlmTakeSequenceSlice(_ array: MLXArray, offset: Int, count: Int) -> MLXArray {
    switch array.ndim {
    case 1:
        return array[offset ..< (offset + count)]
    case 2:
        return array[0..., offset ..< (offset + count)]
    case 3:
        return array[0..., offset ..< (offset + count), 0...]
    default:
        fatalError("Unsupported VLM sequence slice rank: \(array.ndim)")
    }
}

func vlmTrueCount(_ array: MLXArray) -> Int {
    array.asType(.bool).asArray(Bool.self).reduce(into: 0) { partial, value in
        if value { partial += 1 }
    }
}

func vlmTakeLeadingFeatures(_ array: MLXArray, count: Int) -> MLXArray {
    switch array.ndim {
    case 2:
        return array[..<count, 0...]
    case 3:
        if array.dim(0) == 1 {
            return array[0..., ..<count, 0...]
        } else {
            return array[..<count, 0..., 0...]
        }
    default:
        fatalError("Unsupported VLM feature slice rank: \(array.ndim)")
    }
}

func vlmDropLeadingFeatures(_ array: MLXArray, count: Int) -> MLXArray {
    switch array.ndim {
    case 2:
        return array[count..., 0...]
    case 3:
        if array.dim(0) == 1 {
            return array[0..., count..., 0...]
        } else {
            return array[count..., 0..., 0...]
        }
    default:
        fatalError("Unsupported VLM feature slice rank: \(array.ndim)")
    }
}

func vlmChunkedEmbeddedPrefill(
    tokens: MLXArray? = nil,
    embeddings: MLXArray,
    windowSize: Int?,
    prefill: (_ tokens: MLXArray?, _ embeddings: MLXArray) -> Void
) -> (tokens: MLXArray?, embeddings: MLXArray) {
    let prefillStepSize = windowSize ?? 512
    var remainingTokens = tokens
    var remainingEmbeddings = embeddings

    while vlmEmbeddingSequenceLength(remainingEmbeddings) > prefillStepSize {
        let chunkTokens = remainingTokens.map { vlmTakePrefix($0, count: prefillStepSize) }
        let chunkEmbeddings = vlmTakeEmbeddingPrefix(remainingEmbeddings, count: prefillStepSize)
        prefill(chunkTokens, chunkEmbeddings)
        remainingTokens = remainingTokens.map { vlmDropPrefix($0, count: prefillStepSize) }
        remainingEmbeddings = vlmDropEmbeddingPrefix(remainingEmbeddings, count: prefillStepSize)
        Memory.clearCache()
    }

    return (remainingTokens, remainingEmbeddings)
}

func vlmChunkedTextPrefill(
    _ text: LMInput.Text,
    windowSize: Int?,
    prefill: (_ text: LMInput.Text) -> Void
) -> LMInput.Text {
    let prefillStepSize = windowSize ?? 512
    var remaining = text

    while vlmSequenceLength(remaining.tokens) > prefillStepSize {
        let chunk = LMInput.Text(
            tokens: vlmTakePrefix(remaining.tokens, count: prefillStepSize),
            mask: remaining.mask.map { vlmTakePrefix($0, count: prefillStepSize) })
        prefill(chunk)
        remaining = LMInput.Text(
            tokens: vlmDropPrefix(remaining.tokens, count: prefillStepSize),
            mask: remaining.mask.map { vlmDropPrefix($0, count: prefillStepSize) })
        Memory.clearCache()
    }

    return remaining
}
