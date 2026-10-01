import { AutoProcessor, Nemotron3DiarizationProcessor, Tensor } from "../../../src/transformers.js";

import { MAX_PROCESSOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

/**
 * Two alternating tones: 440 Hz for the first half, 220 Hz for the second.
 * @param {number} num_samples
 */
function make_audio(num_samples, sampling_rate = 16000) {
  const audio = new Float32Array(num_samples);
  for (let i = 0; i < num_samples; ++i) {
    audio[i] = i < Math.floor(num_samples / 2) ? 0.5 * Math.sin((2 * Math.PI * 440 * i) / sampling_rate) : 0.3 * Math.sin((2 * Math.PI * 220 * i) / sampling_rate);
  }
  return audio;
}

export default () => {
  // Nemotron3DiarizationProcessor
  describe("Nemotron3DiarizationProcessor", () => {
    const model_id = "onnx-internal-testing/tiny-random-Nemotron3DiarizationForAudioFrameClassification";

    /** @type {Nemotron3DiarizationProcessor} */
    let processor;
    beforeAll(async () => {
      processor = await AutoProcessor.from_pretrained(model_id);
    }, MAX_PROCESSOR_LOAD_TIME);

    it(
      "streaming modes",
      async () => {
        expect(processor).toBeInstanceOf(Nemotron3DiarizationProcessor);
        expect(processor.streaming_mode).toBe("low_latency");
        expect(Object.keys(processor.streaming_modes).sort()).toEqual(["low_latency", "ultra_low_latency", "very_low_latency"]);
        expect(() => processor.set_streaming_mode("offline")).toThrow('Unknown `streaming_mode` "offline"');

        const expected = {
          low_latency: [1040, 104, 72, 16680, 17040, 11264],
          very_low_latency: [640, 64, 48, 10280, 10640, 7424],
          ultra_low_latency: [320, 32, 24, 5160, 5520, 3584],
        };
        for (const [mode, values] of Object.entries(expected)) {
          processor.set_streaming_mode(mode);
          expect([processor.streaming_latency_ms, processor.num_mel_frames_per_audio_chunk, processor.num_mel_frames_per_step, processor.num_samples_first_audio_chunk, processor.num_samples_per_audio_chunk, processor.audio_chunk_start(processor.num_mel_frames_per_step)]).toEqual(values);
        }
        processor.set_streaming_mode("low_latency");
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "call modes",
      async () => {
        const audio = new Float32Array(4 * 16000);

        // Offline outputs are the features alone
        const offline = await processor(audio);
        expect(Object.keys(offline).sort()).toEqual(["attention_mask", "input_features"]);
        await expect(processor(audio, { is_first_audio_chunk: false })).rejects.toThrow("In non-streaming mode");
        await expect(processor(audio, { is_last_audio_chunk: true })).rejects.toThrow("In non-streaming mode");

        // Every streaming chunk but the last carries `num_lookahead_frames`, the chunk sizes following the mode
        processor.set_streaming_mode("very_low_latency");
        const first = await processor(audio.subarray(0, processor.num_samples_first_audio_chunk), { is_streaming: true });
        expect(first.num_lookahead_frames).toBe(2);
        expect(first.input_features.dims).toEqual([1, 64, 128]);
        expect(first.attention_mask.dims).toEqual([1, 64]);
        processor.set_streaming_mode("low_latency");
        const later = await processor(audio.subarray(0, processor.num_samples_per_audio_chunk), { is_streaming: true, is_first_audio_chunk: false });
        expect(later.num_lookahead_frames).toBe(4);
        expect(later.input_features.dims).toEqual([1, 104, 128]);

        // A chunk of the wrong size is rejected unless it is the last one
        await expect(processor(audio.subarray(0, 16000), { is_streaming: true })).rejects.toThrow("A `low_latency` chunk must hold 104 mel frames, got 100");
        const last = await processor(audio.subarray(0, 16000), { is_streaming: true, is_first_audio_chunk: false, is_last_audio_chunk: true });
        expect(last.num_lookahead_frames).toBeUndefined();
        expect(last.input_features.dims).toEqual([1, 98, 128]); // zero-padded by `n_fft / 2 - hop_length` samples
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "streaming chunks reproduce a full-utterance pass",
      async () => {
        /**
         * @param {Tensor} chunk_features The features of a chunk.
         * @param {Tensor} full_features The features of the full pass.
         * @param {number} frame_idx The frame of the full pass the chunk starts at.
         */
        const expectFramesCloseTo = (chunk_features, full_features, frame_idx) => {
          const expected = full_features.slice(null, [frame_idx, frame_idx + chunk_features.dims[1]], null);
          let max_diff = 0;
          for (let i = 0; i < expected.data.length; ++i) {
            max_diff = Math.max(max_diff, Math.abs(chunk_features.data[i] - expected.data[i]));
          }
          expect(max_diff).toBeLessThan(1e-4);
        };

        const { n_fft, hop_length } = processor.feature_extractor.config;
        // The window of the last frame of the full pass ends `max_overhang - audio.length % hop_length` samples past the
        // audio: the last chunk needs end padding for that frame with a multiple of `hop_length` samples, and none with
        // `max_overhang` more.
        const max_overhang = Math.floor(n_fft / 2) - hop_length;
        for (const num_audio_samples of [400 * hop_length, 400 * hop_length + max_overhang]) {
          const audio = make_audio(num_audio_samples);
          const full = await processor(audio);
          const num_valid_frames = Array.from(full.attention_mask.data, Number).reduce((a, b) => a + b, 0);

          let frame_idx = 0;
          let start = 0;
          while (true) {
            const is_first = frame_idx === 0;
            start = is_first ? 0 : processor.audio_chunk_start(frame_idx);
            const num_samples = is_first ? processor.num_samples_first_audio_chunk : processor.num_samples_per_audio_chunk;
            if (start + num_samples > audio.length) break;

            const chunk = await processor(audio.subarray(start, start + num_samples), { is_streaming: true, is_first_audio_chunk: is_first });
            expect(chunk.input_features.dims).toEqual([1, processor.num_mel_frames_per_audio_chunk, 128]);
            expectFramesCloseTo(chunk.input_features, full.input_features, frame_idx);
            frame_idx += processor.num_mel_frames_per_step;
          }
          expect(frame_idx).toBeGreaterThan(processor.num_mel_frames_per_step);

          // The last chunk holds the rest of the audio, down to the last frame of the full pass
          const last = await processor(audio.subarray(start), { is_streaming: true, is_first_audio_chunk: false, is_last_audio_chunk: true });
          expect(frame_idx + last.input_features.dims[1]).toBe(num_valid_frames);
          expectFramesCloseTo(last.input_features, full.input_features, frame_idx);
        }
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "extract_speaker_dict",
      async () => {
        const logits = new Tensor("float32", new Float32Array(2 * 10 * 3).fill(-5), [2, 10, 3]);
        const activate = (b, t0, t1, s) => {
          for (let t = t0; t < t1; ++t) logits.data[(b * 10 + t) * 3 + s] = 5;
        };
        activate(0, 2, 5, 0); // speaker 0: frames 2-4
        activate(0, 4, 6, 1); // speaker 1: frames 4-5, overlapping speaker 0
        activate(0, 8, 10, 0); // speaker 0 again, until the end
        activate(1, 0, 10, 2); // speaker 2 all along, but the sample is 6 frames long
        const attention_mask = new Tensor(
          "int64",
          BigInt64Array.from({ length: 20 }, (_, i) => (i >= 16 ? 0n : 1n)),
          [2, 10],
        );

        expect(processor.extract_speaker_dict(logits, attention_mask)).toEqual([
          [
            { Start: 0.02, End: 0.05, Speaker: 0 },
            { Start: 0.04, End: 0.06, Speaker: 1 },
            { Start: 0.08, End: 0.1, Speaker: 0 },
          ],
          [{ Start: 0.0, End: 0.06, Speaker: 2 }],
        ]);
        // without the mask, the padded frames of the second sample count
        expect(processor.extract_speaker_dict(logits)[1]).toEqual([{ Start: 0.0, End: 0.1, Speaker: 2 }]);
        // a stricter threshold silences everyone
        expect(processor.extract_speaker_dict(logits, null, 1.0)).toEqual([[], []]);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
