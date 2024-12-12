import { AutoFeatureExtractor, ClapFeatureExtractor } from "../../../src/transformers.js";

import { load_cached_audio } from "../../asset_cache.js";
import { MAX_FEATURE_EXTRACTOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  // ClapFeatureExtractor
  describe("ClapFeatureExtractor", () => {
    const model_id = "Xenova/clap-htsat-unfused";

    /** @type {ClapFeatureExtractor} */
    let feature_extractor;
    beforeAll(async () => {
      feature_extractor = await AutoFeatureExtractor.from_pretrained(model_id);
    }, MAX_FEATURE_EXTRACTOR_LOAD_TIME);

    it(
      "truncation",
      async () => {
        const audio = await load_cached_audio("mlk");

        // Since truncation uses a random strategy, we override
        // Math.random to ensure that the test is deterministic
        const originalRandom = Math.random;
        Math.random = () => 0.5;

        let long_audio = new Float32Array(500000);
        long_audio.set(audio);
        long_audio.set(audio, long_audio.length - audio.length);

        const { input_features } = await feature_extractor(long_audio);
        const { dims, data } = input_features;
        expect(dims).toEqual([1, 1, 1001, 64]);

        expect(input_features.mean().item()).toBeCloseTo(-37.94569396972656);
        expect(data[0]).toBeCloseTo(-53.32647705078125);
        expect(data[1]).toBeCloseTo(-47.76755142211914);
        expect(data[65]).toBeCloseTo(-36.32261276245117);
        expect(data[1002]).toBeCloseTo(-28.0314884185791);
        expect(data[10000]).toBeCloseTo(-21.905902862548828);
        expect(data[60000]).toBeCloseTo(-14.877863883972168);
        expect(data[64062]).toBeCloseTo(-37.9784049987793);
        expect(data[64063]).toBeCloseTo(-37.73963928222656);

        // Reset Math.random
        Math.random = originalRandom;
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "padding",
      async () => {
        const audio = await load_cached_audio("mlk");
        const { input_features } = await feature_extractor(audio);
        const { data, dims } = input_features;
        expect(dims).toEqual([1, 1, 1001, 64]);

        expect(input_features.mean().item()).toBeCloseTo(-34.99049377441406);
        expect(data[0]).toBeCloseTo(-21.32573890686035);
        expect(data[1]).toBeCloseTo(-26.168411254882812);
        expect(data[65]).toBeCloseTo(-29.716018676757812);
        expect(data[1002]).toBeCloseTo(-32.16273498535156);
        expect(data[10000]).toBeCloseTo(-19.9283390045166);

        // padded values
        expect(data[60000]).toBeCloseTo(-100.0);
        expect(data[64062]).toBeCloseTo(-100.0);
        expect(data[64063]).toBeCloseTo(-100.0);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
