import { AutoFeatureExtractor, WhisperFeatureExtractor } from "../../../src/transformers.js";

import { load_cached_audio } from "../../asset_cache.js";
import { MAX_FEATURE_EXTRACTOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  // WhisperFeatureExtractor
  describe("WhisperFeatureExtractor", () => {
    const model_id = "Xenova/whisper-tiny.en";

    /** @type {WhisperFeatureExtractor} */
    let feature_extractor;
    beforeAll(async () => {
      feature_extractor = await AutoFeatureExtractor.from_pretrained(model_id);
    }, MAX_FEATURE_EXTRACTOR_LOAD_TIME);

    it(
      "default",
      async () => {
        const audio = await load_cached_audio("mlk");
        const { input_features } = await feature_extractor(audio);
        const { dims, data } = input_features;
        expect(dims).toEqual([1, 80, 3000]);
        expect(input_features.mean().item()).toBeCloseTo(-0.2813588131551941);
        expect(data[0]).toBeCloseTo(0.33168578147888184);
        expect(data[1]).toBeCloseTo(0.30986475944519043);
        expect(data[81]).toBeCloseTo(0.10727232694625854);
        expect(data[3001]).toBeCloseTo(0.2555035352706909);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
