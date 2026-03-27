import { AutoFeatureExtractor, CohereAsrFeatureExtractor } from "../../../src/transformers.js";

import { load_cached_audio } from "../../asset_cache.js";
import { MAX_FEATURE_EXTRACTOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  describe("CohereAsrFeatureExtractor", () => {
    const model_id = "onnx-community/cohere-transcribe-03-2026-ONNX";

    /** @type {CohereAsrFeatureExtractor} */
    let feature_extractor;
    beforeAll(async () => {
      feature_extractor = await AutoFeatureExtractor.from_pretrained(model_id);
    }, MAX_FEATURE_EXTRACTOR_LOAD_TIME);

    it(
      "default",
      async () => {
        const audio = await load_cached_audio("mlk");
        const { input_features, attention_mask } = await feature_extractor(audio);

        // Shape: [1, num_frames, 128] (transposed mel spectrogram)
        expect(input_features.dims).toEqual([1, 1301, 128]);

        // Attention mask: [1, num_frames], 1300 valid frames
        expect(attention_mask.dims).toEqual([1, 1301]);
        const mask_sum = attention_mask.data.reduce((a, b) => a + b, 0n);
        expect(Number(mask_sum)).toEqual(1300);

        // Check feature values against JS-computed reference
        expect(input_features.mean().item()).toBeCloseTo(0.0, 4);
        expect(input_features.data[0]).toBeCloseTo(1.9016, 4); // [0,0,0]
        expect(input_features.data[1]).toBeCloseTo(1.4604, 4); // [0,0,1]
        expect(input_features.data[128]).toBeCloseTo(1.6361, 4); // [0,1,0]
        expect(input_features.data[127]).toBeCloseTo(-0.8963, 4); // [0,0,127]
        expect(input_features.data[12800]).toBeCloseTo(0.9839, 4); // [0,100,0]
        expect(input_features.data[64050]).toBeCloseTo(-0.6128, 4); // [0,500,50]
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "short audio",
      async () => {
        const audio = await load_cached_audio("mlk");
        const { input_features, attention_mask } = await feature_extractor(audio.slice(0, 16000));

        // 1 second of audio at 16kHz: ~100 frames
        expect(input_features.dims[0]).toEqual(1);
        expect(input_features.dims[2]).toEqual(128);
        expect(input_features.dims[1]).toBeGreaterThan(90);
        expect(input_features.dims[1]).toBeLessThan(110);

        expect(attention_mask.dims[0]).toEqual(1);
        expect(attention_mask.dims[1]).toEqual(input_features.dims[1]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "split_audio for long audio",
      async () => {
        const audio = await load_cached_audio("mlk");
        // mlk is ~13 seconds, below 35s threshold, should not split
        const chunks = feature_extractor.split_audio(audio);
        expect(chunks.length).toEqual(1);
        expect(chunks[0].length).toEqual(audio.length);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "split_audio triggers for very long audio",
      async () => {
        // Create a fake long audio (>35s at 16kHz = 560000 samples)
        const long_audio = new Float32Array(600000);
        for (let i = 0; i < long_audio.length; ++i) {
          long_audio[i] = Math.sin(i / 100) * 0.1;
        }
        const chunks = feature_extractor.split_audio(long_audio);
        expect(chunks.length).toBeGreaterThan(1);

        // All chunks together should cover the full audio
        const total = chunks.reduce((acc, c) => acc + c.length, 0);
        expect(total).toEqual(long_audio.length);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
