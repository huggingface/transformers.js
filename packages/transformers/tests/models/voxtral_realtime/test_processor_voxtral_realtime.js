import { AutoProcessor, VoxtralRealtimeProcessor } from "../../../src/transformers.js";

import { load_cached_audio } from "../../asset_cache.js";
import { MAX_FEATURE_EXTRACTOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  // VoxtralRealtimeProcessor
  describe("VoxtralRealtimeProcessor", () => {
    const model_id = "onnx-community/Voxtral-Mini-4B-Realtime-2602-ONNX";

    /** @type {VoxtralRealtimeProcessor} */
    let processor;
    beforeAll(async () => {
      processor = await AutoProcessor.from_pretrained(model_id);
    }, MAX_FEATURE_EXTRACTOR_LOAD_TIME);

    it(
      "streaming first chunk",
      async () => {
        const audio = await load_cached_audio("mlk");

        const first_audio = audio.subarray(0, processor.num_samples_first_audio_chunk);
        const { input_ids, input_features } = await processor(first_audio, {
          is_streaming: true,
          is_first_audio_chunk: true,
        });

        // Verify input_ids: BOS + 38 [STREAMING_PAD] tokens
        expect(input_ids.dims).toEqual([1, 39]);
        expect(Number(input_ids.data[0])).toBe(1); // BOS
        for (let i = 1; i < 39; ++i) {
          expect(Number(input_ids.data[i])).toBe(32); // [STREAMING_PAD]
        }

        // Verify first chunk: left-padded with silence, matching Python processor output
        expect(input_features.dims).toEqual([1, 128, 312]);
        expect(input_features.mean().item()).toBeCloseTo(-0.489270150661469, 3);
        expect(input_features.data[0]).toBeCloseTo(-0.625, 3); // silence (left-pad)
        expect(input_features.data[1]).toBeCloseTo(-0.625, 3);
        expect(input_features.data[127 * 312]).toBeCloseTo(-0.625, 3); // last mel bin, first frame
        expect(input_features.data[256]).toBeCloseTo(0.192136287689209, 3); // first audio frame
        expect(input_features.data[311]).toBeCloseTo(-0.084741473197937, 3); // last audio frame
        expect(input_features.data[input_features.data.length - 1]).toBeCloseTo(-0.246982932090759, 3);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "streaming subsequent chunks",
      async () => {
        const audio = await load_cached_audio("mlk");

        const { hop_length, n_fft } = processor.feature_extractor.config;
        const win_half = Math.floor(n_fft / 2);

        // Collect all subsequent chunks
        const chunks = [];
        let mel_frame_idx = processor.num_mel_frames_first_audio_chunk;
        let start_idx = mel_frame_idx * hop_length - win_half;

        while (start_idx + processor.num_samples_per_audio_chunk < audio.length) {
          const end_idx = start_idx + processor.num_samples_per_audio_chunk;
          const chunk = await processor(audio.slice(start_idx, end_idx), {
            is_streaming: true,
            is_first_audio_chunk: false,
          });
          chunks.push(chunk.input_features);

          mel_frame_idx += processor.audio_length_per_tok;
          start_idx = mel_frame_idx * hop_length - win_half;
        }

        // Verify second chunk (first subsequent)
        const second = chunks[0];
        expect(second.dims).toEqual([1, 128, 8]);
        expect(second.mean().item()).toBeCloseTo(0.13092890381813, 3);
        expect(second.data[0]).toBeCloseTo(-0.090936064720154, 3);
        expect(second.data[second.data.length - 1]).toBeCloseTo(-0.19879412651062, 3);

        // Verify total chunk count: 155 subsequent chunks
        expect(chunks.length).toBe(155);

        // Verify all subsequent chunks have shape [1, 128, 8]
        for (let i = 0; i < chunks.length; ++i) {
          expect(chunks[i].dims).toEqual([1, 128, 8]);
        }
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
