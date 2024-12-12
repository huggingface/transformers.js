import { AutoFeatureExtractor, WeSpeakerFeatureExtractor } from "../../../src/transformers.js";

import { MAX_FEATURE_EXTRACTOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  // WeSpeakerFeatureExtractor
  describe("WeSpeakerFeatureExtractor", () => {
    const model_id = "onnx-community/wespeaker-voxceleb-resnet34-LM";

    /** @type {WeSpeakerFeatureExtractor} */
    let feature_extractor;
    beforeAll(async () => {
      feature_extractor = await AutoFeatureExtractor.from_pretrained(model_id);
    }, MAX_FEATURE_EXTRACTOR_LOAD_TIME);

    it(
      "default",
      async () => {
        const audio = new Float32Array(16000).map((_, i) => Math.sin(i / 100));
        const { input_features } = await feature_extractor(audio);
        const { dims, data } = input_features;
        expect(dims).toEqual([1, 98, 80]);

        expect(input_features.mean().item()).toBeCloseTo(5.461731689138105e-8);
        expect(data[0]).toBeCloseTo(-0.19300270080566406);
        expect(data[1]).toBeCloseTo(-0.05825042724609375);
        expect(data[78]).toBeCloseTo(0.2683420181274414);
        expect(data[79]).toBeCloseTo(0.26250171661376953);
        expect(data[80]).toBeCloseTo(0.19062232971191406);
        expect(data.at(-2)).toBeCloseTo(-0.43694400787353516);
        expect(data.at(-1)).toBeCloseTo(-0.4266204833984375);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "pad to `min_num_frames`",
      async () => {
        const audio = new Float32Array(3).map((_, i) => Math.sin(i / 100));
        const { input_features } = await feature_extractor(audio);
        const { dims, data } = input_features;
        expect(dims).toEqual([1, 9, 80]);

        expect(input_features.mean().item()).toBeCloseTo(-0.0000010093053181966146);
        expect(data[0]).toBeCloseTo(20.761859893798828);
        expect(data[1]).toBeCloseTo(21.02924346923828);
        expect(data[78]).toBeCloseTo(19.083993911743164);
        expect(data[79]).toBeCloseTo(18.003454208374023);
        expect(data[80]).toBeCloseTo(-2.595233917236328);
        expect(data.at(-2)).toBeCloseTo(-2.385499954223633);
        expect(data.at(-1)).toBeCloseTo(-2.2504329681396484);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
