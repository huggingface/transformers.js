import { jest } from "@jest/globals";

import { VoxtralRealtimeForConditionalGeneration, VoxtralRealtimeProcessor, Tensor } from "../../../src/transformers.js";

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
            /* </s>: the audio ran out */
            2n,
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
        expect(outputs.tolist()).toEqual([[1n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 32n, 28478n, 28478n, 28478n, 28478n, 28478n, 28478n, 2n]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "ends with </s> when the audio runs out, without decoding a step that has no audio",
      async () => {
        const audio = syntheticAudio();
        const first_chunk = await processor(audio.subarray(0, processor.num_samples_first_audio_chunk), { is_streaming: true, is_first_audio_chunk: true });
        const decoder_run = jest.spyOn(model.sessions["decoder_model_merged"], "run");

        const outputs = await model.generate({
          input_ids: first_chunk.input_ids,
          input_features: buildInputFeaturesGenerator(audio),
          max_new_tokens: 100,
        });

        // 1 s of audio is 44 audio tokens. The prefill consumes 39 of them and samples the first token,
        // the next five tokens consume one each, and the step after that has no audio left: it must end
        // the sequence rather than decode the text embedding alone.
        const generated = outputs.tolist()[0].slice(first_chunk.input_ids.dims[1]);
        expect(generated).toHaveLength(7);
        expect(generated.at(-1)).toBe(2n);
        expect(decoder_run).toHaveBeenCalledTimes(generated.length - 1);
        decoder_run.mockRestore();
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "ends with </s> even when min_new_tokens would mask it",
      async () => {
        const audio = syntheticAudio();
        const first_chunk = await processor(audio.subarray(0, processor.num_samples_first_audio_chunk), { is_streaming: true, is_first_audio_chunk: true });

        const outputs = await model.generate({
          input_ids: first_chunk.input_ids,
          input_features: buildInputFeaturesGenerator(audio),
          max_new_tokens: 100,
          // Past the seven tokens this audio yields: the processor that masks </s>
          // until then must not be able to keep generation decoding audio-less steps.
          min_new_tokens: 20,
        });

        const generated = outputs.tolist()[0].slice(first_chunk.input_ids.dims[1]);
        expect(generated).toHaveLength(7);
        expect(generated.at(-1)).toBe(2n);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "ends with </s> when sampling",
      async () => {
        const audio = syntheticAudio();
        const first_chunk = await processor(audio.subarray(0, processor.num_samples_first_audio_chunk), { is_streaming: true, is_first_audio_chunk: true });

        const outputs = await model.generate({
          input_ids: first_chunk.input_ids,
          input_features: buildInputFeaturesGenerator(audio),
          max_new_tokens: 100,
          do_sample: true,
          temperature: 0.8,
          // Sampling reads the same logits: `min_new_tokens` masks </s> here too,
          // and a masked row is a uniform distribution over the vocabulary.
          min_new_tokens: 20,
        });

        const generated = outputs.tolist()[0].slice(first_chunk.input_ids.dims[1]);
        expect(generated).toHaveLength(7);
        expect(generated.at(-1)).toBe(2n);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "ends on a per-call eos_token_id",
      async () => {
        const audio = syntheticAudio();
        const first_chunk = await processor(audio.subarray(0, processor.num_samples_first_audio_chunk), { is_streaming: true, is_first_audio_chunk: true });

        // The model's own </s> is 2; this call ends on 3 instead, and the forced
        // step must end on the token this call named, not on the stored config's.
        const outputs = await model.generate({
          input_ids: first_chunk.input_ids,
          input_features: buildInputFeaturesGenerator(audio),
          max_new_tokens: 100,
          eos_token_id: 3,
        });

        const generated = outputs.tolist()[0].slice(first_chunk.input_ids.dims[1]);
        expect(generated.at(-1)).toBe(3n);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "ends on an eos_token_id from a per-call generation_config",
      async () => {
        const audio = syntheticAudio();
        const first_chunk = await processor(audio.subarray(0, processor.num_samples_first_audio_chunk), { is_streaming: true, is_first_audio_chunk: true });

        const outputs = await model.generate({
          input_ids: first_chunk.input_ids,
          input_features: buildInputFeaturesGenerator(audio),
          max_new_tokens: 100,
          generation_config: { eos_token_id: 4 },
        });

        const generated = outputs.tolist()[0].slice(first_chunk.input_ids.dims[1]);
        expect(generated.at(-1)).toBe(4n);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "generates without an eos_token_id when max_new_tokens comes first",
      async () => {
        const audio = syntheticAudio();
        const first_chunk = await processor(audio.subarray(0, processor.num_samples_first_audio_chunk), { is_streaming: true, is_first_audio_chunk: true });

        // Only a step whose audio has run out needs an end-of-sequence token. A call
        // that stops on `max_new_tokens` with audio to spare never takes that step,
        // so it must not be rejected for the lack of one.
        const outputs = await model.generate({
          input_ids: first_chunk.input_ids,
          input_features: buildInputFeaturesGenerator(audio),
          max_new_tokens: 1,
          generation_config: { eos_token_id: null },
        });

        const generated = outputs.tolist()[0].slice(first_chunk.input_ids.dims[1]);
        expect(generated).toHaveLength(1);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "ends on a per-call eos_token_id when the model stores none",
      async () => {
        const audio = syntheticAudio();
        const first_chunk = await processor(audio.subarray(0, processor.num_samples_first_audio_chunk), { is_streaming: true, is_first_audio_chunk: true });

        // Ending a step whose audio ran out has to read the eos_token_id of this call,
        // not the model's stored one: a model that stores none still ends correctly
        // when the caller passes one.
        const storedGenerationEos = model.generation_config.eos_token_id;
        const storedTextEos = model.config.text_config.eos_token_id;
        model.generation_config.eos_token_id = null;
        model.config.text_config.eos_token_id = null;
        try {
          const outputs = await model.generate({
            input_ids: first_chunk.input_ids,
            input_features: buildInputFeaturesGenerator(audio),
            max_new_tokens: 100,
            eos_token_id: 3,
          });

          const generated = outputs.tolist()[0].slice(first_chunk.input_ids.dims[1]);
          expect(generated.at(-1)).toBe(3n);
        } finally {
          model.generation_config.eos_token_id = storedGenerationEos;
          model.config.text_config.eos_token_id = storedTextEos;
        }
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "rejects a config without an eos_token_id once the audio runs out",
      async () => {
        const audio = syntheticAudio();
        const first_chunk = await processor(audio.subarray(0, processor.num_samples_first_audio_chunk), { is_streaming: true, is_first_audio_chunk: true });

        // A step whose audio has run out can only end on an end-of-sequence token.
        // Without one, generation fails at that step instead of decoding positions
        // that have no audio embedding. `max_new_tokens` is what carries it to that
        // step: the default budget would stop long before the audio runs out.
        await expect(
          model.generate({
            input_ids: first_chunk.input_ids,
            input_features: buildInputFeaturesGenerator(audio),
            max_new_tokens: 100,
            generation_config: { eos_token_id: null },
          }),
        ).rejects.toThrow(/eos_token_id/);

        // The model is still usable afterwards.
        const outputs = await model.generate({
          input_ids: first_chunk.input_ids,
          input_features: buildInputFeaturesGenerator(audio),
          max_new_tokens: 100,
        });
        const generated = outputs.tolist()[0].slice(first_chunk.input_ids.dims[1]);
        expect(generated).toHaveLength(7);
        expect(generated.at(-1)).toBe(2n);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "forward without generate (no encoder state)",
      async () => {
        const input_ids = new Tensor("int64", [1n, 32n, 32n], [1, 3]);
        const attention_mask = new Tensor("int64", [1n, 1n, 1n], [1, 3]);

        const outputs = await model.forward({ input_ids, attention_mask, past_key_values: null });
        expect(outputs.logits.dims).toEqual([1, 3, expect.any(Number)]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
