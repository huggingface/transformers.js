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
        expect(input_features.data[1]).toBeCloseTo(0.234105527400970, 3);
        expect(input_features.data[128]).toBeCloseTo(-0.286222934722900, 3);
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
        expect(input_features.data[1]).toBeCloseTo(0.234105527400970, 3);
        expect(input_features.data[128]).toBeCloseTo(0.317371904850006, 3);
        expect(input_features.data[input_features.data.length - 1]).toBeCloseTo(-0.322449326515198, 3);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "prepare_streaming",
      async () => {
        const audio = await load_cached_audio("mlk");
        const model_config = { audio_length_per_tok: 8, default_num_delay_tokens: 6 };
        const { input_ids, input_features } = await feature_extractor.prepare_streaming(audio, model_config);

        // Verify input_ids: BOS + 38 [STREAMING_PAD] tokens
        expect(input_ids.dims).toEqual([1, 39]);
        expect(Number(input_ids.data[0])).toBe(1); // BOS
        for (let i = 1; i < 39; ++i) {
          expect(Number(input_ids.data[i])).toBe(32); // [STREAMING_PAD]
        }

        // Collect chunks
        const chunks = [];
        for await (const chunk of input_features) {
          chunks.push(chunk);
        }

        // Verify first chunk: left-padded with silence, matching Python processor output
        const first = chunks[0];
        expect(first.dims).toEqual([1, 128, 312]);
        expect(first.mean().item()).toBeCloseTo(-0.489270150661469, 3);
        expect(first.data[0]).toBeCloseTo(-0.625, 3); // silence (left-pad)
        expect(first.data[1]).toBeCloseTo(-0.625, 3);
        expect(first.data[127 * 312]).toBeCloseTo(-0.625, 3); // last mel bin, first frame
        expect(first.data[256]).toBeCloseTo(0.192136287689209, 3); // first audio frame
        expect(first.data[311]).toBeCloseTo(-0.084741473197937, 3); // last audio frame
        expect(first.data[first.data.length - 1]).toBeCloseTo(-0.246982932090759, 3);

        // Verify second chunk
        const second = chunks[1];
        expect(second.dims).toEqual([1, 128, 8]);
        expect(second.mean().item()).toBeCloseTo(0.130928903818130, 3);
        expect(second.data[0]).toBeCloseTo(-0.090936064720154, 3);
        expect(second.data[second.data.length - 1]).toBeCloseTo(-0.198794126510620, 3);

        // Verify total chunk count: 1 first + 155 subsequent = 156
        expect(chunks.length).toBe(156);

        // Verify all subsequent chunks have shape [1, 128, 8]
        for (let i = 1; i < chunks.length; ++i) {
          expect(chunks[i].dims).toEqual([1, 128, 8]);
        }
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
