import { AutoFeatureExtractor, SeamlessM4TFeatureExtractor } from "../../../src/transformers.js";

import { load_cached_audio } from "../../asset_cache.js";
import { MAX_FEATURE_EXTRACTOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

const sum = (array) => Number(array.reduce((a, b) => a + b, array instanceof BigInt64Array ? 0n : 0));

export default () => {
  // SeamlessM4TFeatureExtractor
  describe("SeamlessM4TFeatureExtractor", () => {
    const model_id = "Xenova/wav2vec2-bert-CV16-en";

    /** @type {SeamlessM4TFeatureExtractor} */
    let feature_extractor;
    beforeAll(async () => {
      feature_extractor = await AutoFeatureExtractor.from_pretrained(model_id);
    }, MAX_FEATURE_EXTRACTOR_LOAD_TIME);

    it(
      "default",
      async () => {
        const audio = await load_cached_audio("mlk");

        const { input_features, attention_mask } = await feature_extractor(audio);
        const { dims, data } = input_features;
        expect(dims).toEqual([1, 649, 160]);
        expect(attention_mask.dims).toEqual([1, 649]);

        expect(input_features.mean().item()).toBeCloseTo(-2.938903875815413e-8);
        expect(data[0]).toBeCloseTo(1.1939343214035034);
        expect(data[1]).toBeCloseTo(0.7874255180358887);
        expect(data[160]).toBeCloseTo(-0.712975025177002);
        expect(data[161]).toBeCloseTo(0.045802414417266846);
        expect(data.at(-1)).toBeCloseTo(-1.3328346014022827);

        expect(sum(attention_mask.data)).toEqual(649);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "padding (pad_to_multiple_of=2)",
      async () => {
        const audio = await load_cached_audio("mlk");

        const { input_features, attention_mask } = await feature_extractor(audio.slice(0, 10000));
        const { dims, data } = input_features;

        // [1, 61, 80] -> [1, 62, 80] -> [1, 31, 160]
        expect(dims).toEqual([1, 31, 160]);
        expect(attention_mask.dims).toEqual([1, 31]);

        expect(input_features.mean().item()).toBeCloseTo(0.01612919569015503);
        expect(data[0]).toBeCloseTo(0.9657132029533386);
        expect(data[1]).toBeCloseTo(0.12912897765636444);
        expect(data[160]).toBeCloseTo(-1.2364212274551392);
        expect(data[161]).toBeCloseTo(-0.9703778028488159);
        expect(data.at(-1)).toBeCloseTo(1); // padding value

        expect(sum(attention_mask.data)).toEqual(30);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
