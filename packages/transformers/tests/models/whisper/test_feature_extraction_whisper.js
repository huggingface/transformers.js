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

    it(
      "max_length (max_length < audio.length < max_num_samples)",
      async () => {
        const audio = await load_cached_audio("mlk");
        const { input_features } = await feature_extractor(audio, { max_length: 5 * 16000 });
        const { dims, data } = input_features;
        expect(dims).toEqual([1, 80, 500]);
        expect(input_features.mean().item()).toBeCloseTo(0.20474646985530853);
        expect(data[0]).toBeCloseTo(0.33168578147888184);
        expect(data[1]).toBeCloseTo(0.30986475944519043);
        expect(data[81]).toBeCloseTo(0.10727238655090332);
        expect(data[3001]).toBeCloseTo(0.4018087387084961);
        expect(data.at(-1)).toBeCloseTo(-0.41003990173339844);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "max_length (audio.length < max_length < max_num_samples)",
      async () => {
        const audio = await load_cached_audio("mlk");
        const { input_features } = await feature_extractor(audio, { max_length: 25 * 16000 });
        const { dims, data } = input_features;
        expect(dims).toEqual([1, 80, 2500]);
        expect(input_features.mean().item()).toBeCloseTo(-0.20426231622695923);
        expect(data[0]).toBeCloseTo(0.33168578147888184);
        expect(data[1]).toBeCloseTo(0.30986475944519043);
        expect(data[81]).toBeCloseTo(0.10727238655090332);
        expect(data[3001]).toBeCloseTo(0.18040966987609863);
        expect(data.at(-1)).toBeCloseTo(-0.6668410897254944);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
