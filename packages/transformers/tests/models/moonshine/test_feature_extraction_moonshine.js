import { AutoFeatureExtractor, MoonshineFeatureExtractor } from "../../../src/transformers.js";

import { load_cached_audio } from "../../asset_cache.js";
import { MAX_FEATURE_EXTRACTOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  // MoonshineFeatureExtractor
  describe("MoonshineFeatureExtractor", () => {
    const model_id = "onnx-community/moonshine-tiny-ONNX";

    /** @type {MoonshineFeatureExtractor} */
    let feature_extractor;
    beforeAll(async () => {
      feature_extractor = await AutoFeatureExtractor.from_pretrained(model_id);
    }, MAX_FEATURE_EXTRACTOR_LOAD_TIME);

    it(
      "default",
      async () => {
        const audio = await load_cached_audio("mlk");
        const { input_values } = await feature_extractor(audio);
        expect(input_values.dims).toEqual([1, 208000]);
        expect(input_values.mean().item()).toBeCloseTo(-1.5654930507480458e-7, 6);
        expect(input_values.data[0]).toBeCloseTo(0.0067138671875, 6);
        expect(input_values.data.at(-1)).toBeCloseTo(-0.013427734375, 6);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
