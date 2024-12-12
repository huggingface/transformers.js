import { AutoFeatureExtractor, ASTFeatureExtractor } from "../../../src/transformers.js";

import { load_cached_audio } from "../../asset_cache.js";
import { MAX_FEATURE_EXTRACTOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  // ASTFeatureExtractor
  describe("ASTFeatureExtractor", () => {
    const model_id = "Xenova/ast-finetuned-audioset-10-10-0.4593";

    /** @type {ASTFeatureExtractor} */
    let feature_extractor;
    beforeAll(async () => {
      feature_extractor = await AutoFeatureExtractor.from_pretrained(model_id);
    }, MAX_FEATURE_EXTRACTOR_LOAD_TIME);

    it(
      "truncation",
      async () => {
        const audio = await load_cached_audio("mlk");
        const { input_values } = await feature_extractor(audio);
        expect(input_values.dims).toEqual([1, 1024, 128]);

        expect(input_values.mean().item()).toBeCloseTo(-0.04054912979309085);
        expect(input_values.data[0]).toBeCloseTo(-0.5662586092948914);
        expect(input_values.data[1]).toBeCloseTo(-1.0300861597061157);
        expect(input_values.data[129]).toBeCloseTo(-1.084834098815918);
        expect(input_values.data[1025]).toBeCloseTo(-1.1204065084457397);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "padding",
      async () => {
        const audio = await load_cached_audio("mlk");
        const { input_values } = await feature_extractor(audio.slice(0, 1000));
        expect(input_values.dims).toEqual([1, 1024, 128]); // [1, 4, 128] -> (padded to) -> [1, 1024, 128]

        expect(input_values.mean().item()).toBeCloseTo(0.4647964835166931);
        expect(input_values.data[0]).toBeCloseTo(-0.5662586092948914);
        expect(input_values.data[1]).toBeCloseTo(-1.0300861597061157);
        expect(input_values.data[129]).toBeCloseTo(-1.084834098815918);

        // padded values
        expect(input_values.data[1025]).toBeCloseTo(0.46703237295150757);
        expect(input_values.data[2049]).toBeCloseTo(0.46703237295150757);
        expect(input_values.data[10000]).toBeCloseTo(0.46703237295150757);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
