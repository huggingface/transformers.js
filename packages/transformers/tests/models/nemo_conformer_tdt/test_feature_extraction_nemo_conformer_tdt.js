import { NemoConformerTDTFeatureExtractor, Tensor } from "../../../src/transformers.js";
import { NEMO_FEATURE_OUTPUT_OWNERSHIP } from "../../../src/models/nemo_conformer_tdt/feature_extraction_nemo_conformer_tdt.js";

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
        try {
          expect(input_features.dims[0]).toBe(1);
          expect(input_features.dims[2]).toBe(80);
          expect(attention_mask.dims).toEqual([1, input_features.dims[1]]);
        } finally {
          input_features.dispose();
          attention_mask.dispose();
        }
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "supports 128 mel bins",
      async () => {
        const extractor = new NemoConformerTDTFeatureExtractor({ ...base, feature_size: 128 });
        const { input_features, attention_mask } = await extractor(audio);
        try {
          expect(input_features.dims[0]).toBe(1);
          expect(input_features.dims[2]).toBe(128);
          expect(attention_mask.dims).toEqual([1, input_features.dims[1]]);
        } finally {
          input_features.dispose();
          attention_mask.dispose();
        }
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
        try {
          expect(input_features.dims[2]).toBe(128 * 3);
        } finally {
          input_features.dispose();
        }
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "supports non-concatenated delta and delta-delta features",
      async () => {
        const extractor = new NemoConformerTDTFeatureExtractor({
          ...base,
          feature_size: 80,
          delta_order: 2,
          delta_window: 2,
          delta_concatenate: false,
        });
        const { input_features, delta_features, delta_delta_features, attention_mask } = await extractor(audio);
        try {
          expect(input_features.dims[0]).toBe(1);
          expect(input_features.dims[2]).toBe(80);
          expect(delta_features).toBeDefined();
          expect(delta_delta_features).toBeDefined();
          expect(delta_features.dims).toEqual(input_features.dims);
          expect(delta_delta_features.dims).toEqual(input_features.dims);
          expect(attention_mask.dims).toEqual([1, input_features.dims[1]]);
        } finally {
          input_features.dispose();
          delta_features?.dispose();
          delta_delta_features?.dispose();
          attention_mask.dispose();
        }
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "disposes replaced base features when concatenated delta output is used",
      async () => {
        const extractor = new NemoConformerTDTFeatureExtractor({
          ...base,
          feature_size: 80,
          delta_order: 1,
          delta_window: 2,
          delta_concatenate: true,
        });

        const originalDispose = Tensor.prototype.dispose;
        let disposeCalls = 0;
        Tensor.prototype.dispose = function () {
          disposeCalls += 1;
          return originalDispose.call(this);
        };

        let input_features;
        try {
          ({ input_features } = await extractor(audio));
          expect(input_features.dims[2]).toBe(80 * 2);
        } finally {
          Tensor.prototype.dispose = originalDispose;
          input_features?.dispose();
        }

        // One dispose from computeTemporalDeltas intermediate tensor, one from replacing base features tensor.
        expect(disposeCalls).toBe(2);
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
        try {
          const first = await extractor(audio);
          const second = await extractor(audio);

          expect(first).not.toBe(second);
          expect(first.input_features).toBe(second.input_features);
          expect(first.attention_mask).toBe(second.attention_mask);
          expect(extractor.get_cache_stats().entries).toBe(1);
        } finally {
          extractor.clear_cache();
        }
        expect(extractor.get_cache_stats().entries).toBe(0);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "marks uncached outputs as caller-owned",
      async () => {
        const extractor = new NemoConformerTDTFeatureExtractor({ ...base, feature_size: 80 });
        const outputs = await extractor(audio);
        try {
          expect(outputs[NEMO_FEATURE_OUTPUT_OWNERSHIP]).toBe(false);
        } finally {
          outputs.input_features.dispose();
          outputs.attention_mask.dispose();
        }
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "marks cached outputs as cache-owned",
      async () => {
        const extractor = new NemoConformerTDTFeatureExtractor({
          ...base,
          feature_size: 80,
          use_feature_cache: true,
          feature_cache_max_entries: 8,
          feature_cache_max_size_mb: 8,
        });
        try {
          const first = await extractor(audio);
          const second = await extractor(audio);

          expect(first[NEMO_FEATURE_OUTPUT_OWNERSHIP]).toBe(true);
          expect(second[NEMO_FEATURE_OUTPUT_OWNERSHIP]).toBe(true);
          expect(first.input_features).toBe(second.input_features);
        } finally {
          extractor.clear_cache();
        }
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "marks skipped-cache outputs as caller-owned",
      async () => {
        const extractor = new NemoConformerTDTFeatureExtractor({
          ...base,
          feature_size: 80,
          use_feature_cache: true,
          feature_cache_max_entries: 0,
          feature_cache_max_size_mb: 8,
        });
        const outputs = await extractor(audio);
        try {
          expect(outputs[NEMO_FEATURE_OUTPUT_OWNERSHIP]).toBe(false);
          expect(extractor.get_cache_stats().entries).toBe(0);
        } finally {
          outputs.input_features.dispose();
          outputs.attention_mask.dispose();
        }
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "marks oversized-cache outputs as caller-owned",
      async () => {
        const extractor = new NemoConformerTDTFeatureExtractor({
          ...base,
          feature_size: 80,
          use_feature_cache: true,
          feature_cache_max_entries: 8,
          feature_cache_max_size_mb: 0.000001,
        });
        const outputs = await extractor(audio);
        try {
          expect(outputs[NEMO_FEATURE_OUTPUT_OWNERSHIP]).toBe(false);
          expect(extractor.get_cache_stats().entries).toBe(0);
        } finally {
          outputs.input_features.dispose();
          outputs.attention_mask.dispose();
        }
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "uses feature cache when enabled for non-concatenated delta outputs",
      async () => {
        const extractor = new NemoConformerTDTFeatureExtractor({
          ...base,
          feature_size: 80,
          delta_order: 2,
          delta_window: 2,
          delta_concatenate: false,
          use_feature_cache: true,
          feature_cache_max_entries: 8,
          feature_cache_max_size_mb: 8,
        });
        try {
          const first = await extractor(audio);
          const second = await extractor(audio);

          expect(first).not.toBe(second);
          expect(first.input_features).toBe(second.input_features);
          expect(first.attention_mask).toBe(second.attention_mask);
          expect(first.delta_features).toBe(second.delta_features);
          expect(first.delta_delta_features).toBe(second.delta_delta_features);
          expect(extractor.get_cache_stats().entries).toBe(1);
        } finally {
          extractor.clear_cache();
        }
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

    it("validates delta_window at construction time", () => {
      expect(
        () => new NemoConformerTDTFeatureExtractor({ ...base, feature_size: 80, delta_order: 1, delta_window: 0 }),
      ).toThrow("delta_window");
      expect(
        () =>
          new NemoConformerTDTFeatureExtractor({
            ...base,
            feature_size: 80,
            delta_order: 1,
            delta_window: 1.5,
          }),
      ).toThrow("delta_window");
    });

    it("validates n_fft and win_length at construction time", () => {
      expect(() => new NemoConformerTDTFeatureExtractor({ ...base, feature_size: 80, n_fft: 0 })).toThrow("n_fft");
      expect(() => new NemoConformerTDTFeatureExtractor({ ...base, feature_size: 80, win_length: 0 })).toThrow(
        "win_length",
      );
      expect(() => new NemoConformerTDTFeatureExtractor({ ...base, feature_size: 80, win_length: 1024 })).toThrow(
        "win_length",
      );
    });
  });
};
