import { VoxtralRealtimeForConditionalGeneration, VoxtralRealtimeProcessor } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  describe("VoxtralRealtimeForConditionalGeneration", () => {
    const model_id = "onnx-internal-testing/tiny-random-VoxtralRealtimeForConditionalGeneration";

    /** @type {VoxtralRealtimeForConditionalGeneration} */
    let model;
    /** @type {VoxtralRealtimeProcessor} */
    let processor;

    beforeAll(async () => {
      model = await VoxtralRealtimeForConditionalGeneration.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      processor = await VoxtralRealtimeProcessor.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    // Helper: generate a 1-second 440Hz sine wave
    function syntheticAudio() {
      const audio = new Float32Array(16000);
      for (let i = 0; i < audio.length; ++i) {
        audio[i] = Math.sin((2 * Math.PI * 440 * i) / 16000) * 0.5;
      }
      return audio;
    }

    // Helper: build input_features generator using the processor's __call__ API
    async function* buildInputFeaturesGenerator(audio) {
      const { hop_length, n_fft } = processor.feature_extractor.config;
      const win_half = Math.floor(n_fft / 2);

      // First chunk
      const first_chunk = await processor(audio.subarray(0, processor.num_samples_first_audio_chunk), { is_streaming: true, is_first_audio_chunk: true });
      yield first_chunk.input_features;

      // Subsequent chunks
      let mel_frame_idx = processor.num_mel_frames_first_audio_chunk;
      let start_idx = mel_frame_idx * hop_length - win_half;

      while (start_idx + processor.num_samples_per_audio_chunk < audio.length) {
        const end_idx = start_idx + processor.num_samples_per_audio_chunk;
        const chunk = await processor(audio.slice(start_idx, end_idx), { is_streaming: true, is_first_audio_chunk: false });
        yield chunk.input_features;

        mel_frame_idx += processor.audio_length_per_tok;
        start_idx = mel_frame_idx * hop_length - win_half;
      }
    }

    it(
      "streaming generation",
      async () => {
        const audio = syntheticAudio();

        // Get input_ids from first chunk call
        const first_chunk = await processor(audio.subarray(0, processor.num_samples_first_audio_chunk), { is_streaming: true, is_first_audio_chunk: true });

        const outputs = await model.generate({
          input_ids: first_chunk.input_ids,
          input_features: buildInputFeaturesGenerator(audio),
          max_new_tokens: 10,
        });

        expect(outputs.tolist()).toEqual([
          [
            /* Input (39 tokens: BOS + 38 STREAMING_PAD) */
            1n,
            32n,
            32n,
            32n,
            32n,
            32n,
            32n,
            32n,
            32n,
            32n,
            32n,
            32n,
            32n,
            32n,
            32n,
            32n,
            32n,
            32n,
            32n,
            32n,
            32n,
            32n,
            32n,
            32n,
            32n,
            32n,
            32n,
            32n,
            32n,
            32n,
            32n,
            32n,
            32n,
            32n,
            32n,
            32n,
            32n,
            32n,
            32n,
            /* Generated */
            28478n,
            28478n,
            28478n,
            28478n,
            28478n,
            28478n,
            98356n,
          ],
        ]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "streaming generation with delayed chunks",
      async () => {
        const audio = syntheticAudio();

        const first_chunk = await processor(audio.subarray(0, processor.num_samples_first_audio_chunk), { is_streaming: true, is_first_audio_chunk: true });

        // Simulate receiving chunks with 100ms delays
        async function* delayedChunks() {
          for await (const chunk of buildInputFeaturesGenerator(audio)) {
            await new Promise((r) => setTimeout(r, 100));
            yield chunk;
          }
        }

        const outputs = await model.generate({
          input_ids: first_chunk.input_ids,
          input_features: delayedChunks(),
          max_new_tokens: 10,
        });

        // Same output as without delays — generation is deterministic
        expect(outputs.tolist()).toEqual([[1n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 28478n, 28478n, 28478n, 28478n, 28478n, 28478n, 98356n]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
