import { AutoProcessor, Gemma4AudioFeatureExtractor } from "../../../src/transformers.js";

import { load_cached_audio } from "../../asset_cache.js";
import { MAX_PROCESSOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  describe.skip("Gemma4AudioFeatureExtractor", () => {
    const model_id = "onnx-community/gemma-4-E2B-it-ONNX";

    /** @type {Gemma4AudioFeatureExtractor} */
    let feature_extractor;
    beforeAll(async () => {
      const processor = await AutoProcessor.from_pretrained(model_id);
      feature_extractor = processor.feature_extractor;
    }, MAX_PROCESSOR_LOAD_TIME);

    it(
      "full audio",
      async () => {
        const audio = await load_cached_audio("mlk");
        const { input_features, input_features_mask } = await feature_extractor(audio);

        // Python ref: shape=(1299, 128), mean=-1.606543, mask_sum=1299
        expect(input_features.dims).toEqual([1, 1299, 128]);
        expect(input_features_mask.dims).toEqual([1, 1299]);

        expect(input_features.mean().item()).toBeCloseTo(-1.6065434217453, 5);
        expect(input_features.data[0]).toBeCloseTo(-6.907755374908447, 5);
        expect(input_features.data[1]).toBeCloseTo(-1.892147541046143, 5);
        expect(input_features.data[127]).toBeCloseTo(-3.07513427734375, 5);
        expect(input_features.data[128]).toBeCloseTo(-6.907755374908447, 5);
        expect(input_features.data[1000]).toBeCloseTo(-1.210815072059631, 5);
        expect(input_features.data[input_features.data.length - 1]).toBeCloseTo(-3.489276647567749, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "short audio (1 second)",
      async () => {
        const audio = await load_cached_audio("mlk");
        const { input_features, input_features_mask } = await feature_extractor(audio.slice(0, 16000));

        // Python ref: shape=(99, 128), mean=-1.938566
        expect(input_features.dims).toEqual([1, 99, 128]);
        expect(input_features_mask.dims).toEqual([1, 99]);

        expect(input_features.mean().item()).toBeCloseTo(-1.938566446304321, 5);
        expect(input_features.data[0]).toBeCloseTo(-6.907755374908447, 5);
        expect(input_features.data[input_features.data.length - 1]).toBeCloseTo(-2.342944622039795, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "very short audio (50ms)",
      async () => {
        const audio = await load_cached_audio("mlk");
        const { input_features, input_features_mask } = await feature_extractor(audio.slice(0, 800));

        // Python ref: shape=(5, 128), mean=-1.612146, mask=[T,T,T,T,F]
        expect(input_features.dims).toEqual([1, 5, 128]);
        expect(input_features_mask.dims).toEqual([1, 5]);

        expect(input_features.mean().item()).toBeCloseTo(-1.612145781517029, 5);
        expect(input_features.data[0]).toBeCloseTo(-6.907755374908447, 5);
        // Last frame is masked (zeroed out)
        expect(input_features.data[input_features.data.length - 1]).toBeCloseTo(0.0, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "3 second audio",
      async () => {
        const audio = await load_cached_audio("mlk");
        const { input_features, input_features_mask } = await feature_extractor(audio.slice(0, 48000));

        // Python ref: shape=(299, 128), mean=-1.686108
        expect(input_features.dims).toEqual([1, 299, 128]);
        expect(input_features_mask.dims).toEqual([1, 299]);

        expect(input_features.mean().item()).toBeCloseTo(-1.686108350753784, 5);
        expect(input_features.data[0]).toBeCloseTo(-6.907755374908447, 5);
        expect(input_features.data[1000]).toBeCloseTo(-1.210815072059631, 5);
        expect(input_features.data[input_features.data.length - 1]).toBeCloseTo(-2.798041820526123, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
