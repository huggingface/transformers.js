import { AutoFeatureExtractor, NemotronAsrStreamingFeatureExtractor } from "../../../src/transformers.js";

import { MAX_FEATURE_EXTRACTOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

/**
 * A linear chirp sweeping 50 Hz -> 50 + 4000 * duration Hz, covering every mel band.
 * @param {number} num_samples
 */
function make_chirp(num_samples, sampling_rate = 16000) {
  const audio = new Float32Array(num_samples);
  for (let i = 0; i < num_samples; ++i) {
    const t = i / sampling_rate;
    audio[i] = 0.5 * Math.sin(2 * Math.PI * (50 * t + 2000 * t * t));
  }
  return audio;
}

export default () => {
  // NemotronAsrStreamingFeatureExtractor
  describe("NemotronAsrStreamingFeatureExtractor", () => {
    const model_id = "onnx-internal-testing/tiny-random-Nemotron3DiarizationForAudioFrameClassification";

    /** @type {NemotronAsrStreamingFeatureExtractor} */
    let feature_extractor;
    beforeAll(async () => {
      feature_extractor = await AutoFeatureExtractor.from_pretrained(model_id);
    }, MAX_FEATURE_EXTRACTOR_LOAD_TIME);

    it(
      "centered windows",
      async () => {
        const { input_features, attention_mask } = await feature_extractor(make_chirp(32000));
        expect(input_features.dims).toEqual([1, 201, 128]);
        expect(attention_mask.dims).toEqual([1, 201]);

        // The last frame's analysis window reaches past the audio: it is masked and zeroed
        const mask = Array.from(attention_mask.data, Number);
        expect(mask.slice(0, 200).every((x) => x === 1)).toBe(true);
        expect(mask[200]).toBe(0);
        expect(input_features.data.subarray(200 * 128).every((x) => x === 0)).toBe(true);

        expect(input_features.mean().item()).toBeCloseTo(-15.28091812133789, 3);
        expect(input_features.data[0]).toBeCloseTo(-3.4887149333953857, 3);
        expect(input_features.data[1]).toBeCloseTo(-3.390415668487549, 3);
        expect(input_features.data[128]).toBeCloseTo(-5.29766845703125, 3);
        expect(input_features.data[6472]).toBeCloseTo(3.2587783336639404, 3);
        expect(input_features.data[12900]).toBeCloseTo(4.219216823577881, 3);
        expect(input_features.data[19316]).toBeCloseTo(4.175293445587158, 3);
        expect(input_features.data[25599]).toBeCloseTo(2.7701821327209473, 3);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "uncentered windows",
      async () => {
        const { input_features, attention_mask } = await feature_extractor(make_chirp(32000).slice(4000, 20000), { center: false });
        expect(input_features.dims).toEqual([1, 97, 128]);
        expect(Array.from(attention_mask.data, Number).every((x) => x === 1)).toBe(true);

        expect(input_features.mean().item()).toBeCloseTo(-15.221309661865234, 3);
        expect(input_features.data[0]).toBeCloseTo(-16.59052085876465, 3);
        expect(input_features.data[46]).toBeCloseTo(2.3588690757751465, 3);
        expect(input_features.data[3144]).toBeCloseTo(3.502058267593384, 3);
        expect(input_features.data[6232]).toBeCloseTo(3.943904161453247, 3);
        expect(input_features.data[9315]).toBeCloseTo(4.25022029876709, 3);
        expect(input_features.data[12268]).toBeCloseTo(4.376149654388428, 3);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "uncentered chunks reproduce a centered pass",
      async () => {
        const audio = make_chirp(32000);
        const { n_fft, hop_length } = feature_extractor.config;
        const full = await feature_extractor(audio);

        // A chunk starting `n_fft / 2` samples before frame 50 yields frames 50, 51, ...
        const frame_idx = 50;
        const start = frame_idx * hop_length - Math.floor(n_fft / 2);
        const chunk = await feature_extractor(audio.slice(start, start + 16000), { center: false });
        const num_frames = chunk.input_features.dims[1];
        const expected = full.input_features.slice(null, [frame_idx, frame_idx + num_frames], null);

        let max_diff = 0;
        for (let i = 0; i < expected.data.length; ++i) {
          max_diff = Math.max(max_diff, Math.abs(chunk.input_features.data[i] - expected.data[i]));
        }
        expect(max_diff).toBeLessThan(1e-4);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "uncentered windows need at least n_fft samples",
      async () => {
        await expect(feature_extractor(new Float32Array(511), { center: false })).rejects.toThrow();
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
