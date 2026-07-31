import { AutoFeatureExtractor, VoxtralRealtimeFeatureExtractor } from "../../../src/transformers.js";

import { load_cached_audio } from "../../asset_cache.js";
import { MAX_FEATURE_EXTRACTOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  // VoxtralRealtimeFeatureExtractor
  describe("VoxtralRealtimeFeatureExtractor", () => {
    const model_id = "onnx-community/Voxtral-Mini-4B-Realtime-2602-ONNX";

    /** @type {VoxtralRealtimeFeatureExtractor} */
    let feature_extractor;
    beforeAll(async () => {
      feature_extractor = await AutoFeatureExtractor.from_pretrained(model_id);
    }, MAX_FEATURE_EXTRACTOR_LOAD_TIME);

    it(
      "full audio",
      async () => {
        const audio = await load_cached_audio("mlk");
        const { input_features } = await feature_extractor(audio);
        expect(input_features.dims).toEqual([1, 128, 1300]);

        expect(input_features.mean().item()).toBeCloseTo(0.193456813693047, 3);
        expect(input_features.data[0]).toBeCloseTo(0.255926549434662, 3);
        expect(input_features.data[1]).toBeCloseTo(0.23410552740097, 3);
        expect(input_features.data[128]).toBeCloseTo(-0.2862229347229, 3);
        expect(input_features.data[129]).toBeCloseTo(-0.625, 3);
        expect(input_features.data[1000]).toBeCloseTo(0.053786396980286, 3);
        expect(input_features.data[10000]).toBeCloseTo(0.320511281490326, 3);
        expect(input_features.data[100000]).toBeCloseTo(0.251729607582092, 3);
        expect(input_features.data[input_features.data.length - 1]).toBeCloseTo(-0.436606526374817, 3);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "short audio",
      async () => {
        const audio = await load_cached_audio("mlk");
        const { input_features } = await feature_extractor(audio.slice(0, 1000));
        expect(input_features.dims).toEqual([1, 128, 6]);

        expect(input_features.mean().item()).toBeCloseTo(0.120253048837185, 3);
        expect(input_features.data[0]).toBeCloseTo(0.255926549434662, 3);
        expect(input_features.data[1]).toBeCloseTo(0.23410552740097, 3);
        expect(input_features.data[128]).toBeCloseTo(0.317371904850006, 3);
        expect(input_features.data[input_features.data.length - 1]).toBeCloseTo(-0.322449326515198, 3);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
