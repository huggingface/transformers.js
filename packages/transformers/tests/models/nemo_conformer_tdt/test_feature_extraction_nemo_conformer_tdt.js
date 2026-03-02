import { NemoConformerTDTFeatureExtractor } from "../../../src/transformers.js";

import { MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  describe("NemoConformerTDTFeatureExtractor", () => {
    const base = {
      sampling_rate: 16000,
      n_fft: 512,
      win_length: 400,
      hop_length: 160,
      preemphasis: 0.97,
    };

    const audio = Float32Array.from({ length: 16000 }, (_, i) => Math.sin((2 * Math.PI * 220 * i) / 16000));

    it(
      "supports 80 mel bins",
      async () => {
        const extractor = new NemoConformerTDTFeatureExtractor({ ...base, feature_size: 80 });
        const { input_features, attention_mask } = await extractor(audio);
        expect(input_features.dims[0]).toBe(1);
        expect(input_features.dims[2]).toBe(80);
        expect(attention_mask.dims).toEqual([1, input_features.dims[1]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "supports 128 mel bins",
      async () => {
        const extractor = new NemoConformerTDTFeatureExtractor({ ...base, feature_size: 128 });
        const { input_features, attention_mask } = await extractor(audio);
        expect(input_features.dims[0]).toBe(1);
        expect(input_features.dims[2]).toBe(128);
        expect(attention_mask.dims).toEqual([1, input_features.dims[1]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "supports concatenated delta and delta-delta features",
      async () => {
        const extractor = new NemoConformerTDTFeatureExtractor({
          ...base,
          feature_size: 128,
          delta_order: 2,
          delta_window: 2,
          delta_concatenate: true,
        });
        const { input_features } = await extractor(audio);
        expect(input_features.dims[2]).toBe(128 * 3);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "uses feature cache when enabled",
      async () => {
        const extractor = new NemoConformerTDTFeatureExtractor({
          ...base,
          feature_size: 80,
          use_feature_cache: true,
          feature_cache_max_entries: 8,
          feature_cache_max_size_mb: 8,
        });
        const first = await extractor(audio);
        const second = await extractor(audio);

        expect(first).toBe(second);
        expect(extractor.get_cache_stats().entries).toBe(1);
        extractor.clear_cache();
        expect(extractor.get_cache_stats().entries).toBe(0);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "validates preemphasis range",
      async () => {
        const invalidHigh = new NemoConformerTDTFeatureExtractor({ ...base, feature_size: 80, preemphasis: 1 });
        await expect(invalidHigh(audio)).rejects.toThrow("preemphasis");

        const invalidLow = new NemoConformerTDTFeatureExtractor({ ...base, feature_size: 80, preemphasis: -0.1 });
        await expect(invalidLow(audio)).rejects.toThrow("preemphasis");
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
