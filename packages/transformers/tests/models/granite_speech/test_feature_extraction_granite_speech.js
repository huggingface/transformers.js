import { AutoFeatureExtractor, GraniteSpeechFeatureExtractor } from "../../../src/transformers.js";

import { load_cached_audio } from "../../asset_cache.js";
import { MAX_FEATURE_EXTRACTOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  describe("GraniteSpeechFeatureExtractor", () => {
    const model_id = "onnx-community/granite-4.0-1b-speech-ONNX";

    /** @type {GraniteSpeechFeatureExtractor} */
    let feature_extractor;
    beforeAll(async () => {
      feature_extractor = await AutoFeatureExtractor.from_pretrained(model_id);
    }, MAX_FEATURE_EXTRACTOR_LOAD_TIME);

    it(
      "full audio",
      async () => {
        const audio = await load_cached_audio("mlk");
        const { input_features } = await feature_extractor(audio);

        // Shape: [1, 650, 160]
        expect(input_features.dims).toEqual([1, 650, 160]);

        expect(input_features.mean().item()).toBeCloseTo(0.709274411201477, 3);
        expect(input_features.data[0]).toBeCloseTo(0.726300835609436, 3);
        expect(input_features.data[1]).toBeCloseTo(0.683963894844055, 3);
        expect(input_features.data[159]).toBeCloseTo(0.264412879943848, 3);
        expect(input_features.data[160]).toBeCloseTo(0.544037103652954, 3);
        expect(input_features.data[1000]).toBeCloseTo(0.773821651935577, 3);
        expect(input_features.data[10000]).toBeCloseTo(-0.2131507396698, 3);
        expect(input_features.data[input_features.data.length - 1]).toBeCloseTo(0.233379304409027, 3);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "short audio",
      async () => {
        const audio = await load_cached_audio("mlk");
        const { input_features } = await feature_extractor(audio.slice(0, 1000));

        // Shape: [1, 3, 160]
        expect(input_features.dims).toEqual([1, 3, 160]);

        expect(input_features.mean().item()).toBeCloseTo(0.634890496730804, 3);
        expect(input_features.data[0]).toBeCloseTo(0.726300835609436, 3);
        expect(input_features.data[input_features.data.length - 1]).toBeCloseTo(0.301110625267029, 3);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
