import { AutoFeatureExtractor } from "../../../src/transformers.js";

import { load_cached_audio } from "../../asset_cache.js";
import { MAX_FEATURE_EXTRACTOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  describe("CohereAsrFeatureExtractor", () => {
    const model_id = "onnx-community/cohere-transcribe-03-2026-ONNX";

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

        // Check feature values against Python reference
        // NOTE: Small differences (~1e-3) are expected due to mel filter precision
        // (librosa float32 vs JS float64) and different dithering PRNGs.
        expect(input_features.mean().item()).toBeCloseTo(0.0, 3);
        expect(input_features.data[0]).toBeCloseTo(1.9019224644, 2); // [0,0,0]
        expect(input_features.data[1]).toBeCloseTo(1.4606336355, 2); // [0,0,1]
        expect(input_features.data[128]).toBeCloseTo(1.6364065409, 2); // [0,1,0]
        expect(input_features.data[127]).toBeCloseTo(-0.8954101205, 2); // [0,0,127]
        expect(input_features.data[12800]).toBeCloseTo(0.9838520288, 2); // [0,100,0]
        expect(input_features.data[64050]).toBeCloseTo(-0.6117327809, 2); // [0,500,50]
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "short audio",
      async () => {
        const audio = await load_cached_audio("mlk");
        const { input_features, attention_mask } = await feature_extractor(audio.slice(0, 16000));

        expect(input_features.dims).toEqual([1, 101, 128]);
        expect(attention_mask.dims).toEqual([1, 101]);

        const mask_sum = attention_mask.data.reduce((a, b) => a + b, 0n);
        expect(Number(mask_sum)).toEqual(100);

        expect(input_features.mean().item()).toBeCloseTo(0.0, 3);
        expect(input_features.data[0]).toBeCloseTo(1.5188870430, 2); // [0,0,0]
        expect(input_features.data[1]).toBeCloseTo(1.1131993532, 2); // [0,0,1]
        expect(input_features.data[128]).toBeCloseTo(1.2305405140, 2); // [0,1,0]
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
