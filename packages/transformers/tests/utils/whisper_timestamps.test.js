import { WhisperForConditionalGeneration } from "../../src/models/whisper/modeling_whisper.js";
import { Tensor } from "../../src/utils/tensor.js";

function extractTimestamps(numFrames, medianFilterWidth = 7) {
  const outputs = {
    cross_attentions: [[new Tensor("float32", Float32Array.of(1, 0, 0, 0, 0.75, 0.25, 0, 0, 0, 0.25, 0.25, 0.5), [1, 1, 3, 4])]],
    sequences: new Tensor("int64", new BigInt64Array(4), [1, 4]),
  };
  return WhisperForConditionalGeneration.prototype._extract_token_timestamps.call({ config: { decoder_layers: 1, median_filter_width: medianFilterWidth } }, outputs, [[0, 0]], numFrames);
}

describe("Whisper token timestamps", () => {
  it.each([
    [1, [0, 0, 0, 0]],
    [2, [0, 0, 0.02, 0.02]],
    [3, [0, 0, 0.02, 0.02]],
  ])("skips smoothing when cropped input has %i frames", (numFrames, expected) => {
    const timestamps = extractTimestamps(numFrames);
    expect(timestamps.dims).toEqual([1, 4]);
    expect(timestamps.data).toEqual(Float32Array.from(expected));
  });

  it("still smooths when the input is longer than the half window", () => {
    expect(extractTimestamps(4).data).toEqual(Float32Array.of(0, 0, 0, 0));
  });

  it("rejects an even filter width on short inputs", () => {
    expect(() => extractTimestamps(3, 6)).toThrow("Window size must be a positive odd number");
  });
});
