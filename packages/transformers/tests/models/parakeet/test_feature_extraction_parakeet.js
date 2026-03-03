import { ParakeetFeatureExtractor } from "../../../src/transformers.js";

import { MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  describe("ParakeetFeatureExtractor", () => {
    const config = {
      feature_size: 80,
      sampling_rate: 16000,
      n_fft: 512,
      win_length: 400,
      hop_length: 160,
      preemphasis: 0.97,
    };

    /** @type {ParakeetFeatureExtractor} */
    let feature_extractor;
    beforeAll(() => {
      feature_extractor = new ParakeetFeatureExtractor(config);
    });

    it(
      "extracts features and mask from synthetic audio",
      async () => {
        const duration_s = 1.0;
        const total = Math.floor(config.sampling_rate * duration_s);
        const audio = Float32Array.from({ length: total }, (_, i) => Math.sin((2 * Math.PI * 220 * i) / config.sampling_rate));

        const { input_features, attention_mask } = await feature_extractor(audio);
        try {
          expect(input_features.dims[0]).toBe(1);
          expect(input_features.dims[2]).toBe(config.feature_size);
          expect(attention_mask.dims).toEqual([1, input_features.dims[1]]);

          const validFrames = attention_mask.data.reduce((acc, x) => acc + Number(x), 0);
          expect(validFrames).toBeGreaterThan(0);
          expect(validFrames).toBeLessThanOrEqual(input_features.dims[1]);

          const preview = Array.from(input_features.data.slice(0, 256));
          expect(preview.every(Number.isFinite)).toBe(true);
        } finally {
          input_features.dispose();
          attention_mask.dispose();
        }
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
