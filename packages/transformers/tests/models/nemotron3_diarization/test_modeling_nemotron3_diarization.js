import { AutoProcessor, AutoModelForAudioFrameClassification, Nemotron3DiarizationForAudioFrameClassification, Nemotron3DiarizationProcessor, Nemotron3DiarizationSpeakerCache, Tensor, cat } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

/**
 * features[b, t, m] = 3 * sin(0.37 t + 0.11 m + 1.3 b) + cos(0.05 t m / 7) - 4
 * @param {number} batch_size
 * @param {number} num_frames
 * @param {number} num_mel_bins
 */
function make_features(batch_size, num_frames, num_mel_bins) {
  const data = new Float32Array(batch_size * num_frames * num_mel_bins);
  for (let b = 0; b < batch_size; ++b) {
    for (let t = 0; t < num_frames; ++t) {
      for (let m = 0; m < num_mel_bins; ++m) {
        data[(b * num_frames + t) * num_mel_bins + m] = 3 * Math.sin(0.37 * t + 0.11 * m + 1.3 * b) + Math.cos((0.05 * t * m) / 7) - 4;
      }
    }
  }
  return new Tensor("float32", data, [batch_size, num_frames, num_mel_bins]);
}

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

/**
 * @param {Tensor} logits
 * @param {{ dims: number[], mean: number, first: number[], last: number[] }} expected
 */
function expectLogitsCloseTo(logits, { dims, mean, first, last }) {
  expect(logits.dims).toEqual(dims);
  expect(logits.mean().item()).toBeCloseTo(mean, 4);
  const data = /** @type {Float32Array} */ (logits.data);
  first.forEach((x, i) => expect(data[i]).toBeCloseTo(x, 4));
  last.forEach((x, i) => expect(data[data.length - last.length + i]).toBeCloseTo(x, 4));
}

const OFFLINE_FEATURES_LOGITS = {
  dims: [1, 256, 4],
  mean: 0.43180719017982483,
  first: [0.969059407711029, -0.10112547874450684, -0.5775530338287354, 1.8899556398391724],
  last: [0.7135230302810669, 0.5973988175392151, -0.9728816747665405, 0.9967129826545715],
};

export default () => {
  describe("Nemotron3DiarizationForAudioFrameClassification", () => {
    const model_id = "onnx-internal-testing/tiny-random-Nemotron3DiarizationForAudioFrameClassification";

    /** @type {Nemotron3DiarizationForAudioFrameClassification} */
    let model;
    /** @type {Nemotron3DiarizationProcessor} */
    let processor;
    beforeAll(async () => {
      model = await AutoModelForAudioFrameClassification.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      processor = await AutoProcessor.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "offline",
      async () => {
        expect(model).toBeInstanceOf(Nemotron3DiarizationForAudioFrameClassification);
        const { logits, speaker_cache } = await model({ input_features: make_features(1, 256, 128) });
        expectLogitsCloseTo(logits, OFFLINE_FEATURES_LOGITS);
        expect(speaker_cache).toBeNull();
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "streaming steps reproduce the offline forward",
      async () => {
        // The tiny model uses the same FIFO sizes offline and streaming, so feeding the offline chunks one forward at
        // a time reproduces the offline forward
        const input_features = make_features(1, 256, 128);
        const { chunk_length, chunk_right_context } = /** @type {any} */ (model.config);
        const factor = /** @type {any} */ (model.config).audio_config.subsampling_factor;
        const chunk_frames = chunk_length * factor;
        const lookahead_frames = chunk_right_context * factor;
        const num_frames = input_features.dims[1];

        const step_logits = [];
        let speaker_cache = null;
        let start = 0;
        while (start + chunk_frames + lookahead_frames <= num_frames) {
          const outputs = await model({
            input_features: input_features.slice(null, [start, start + chunk_frames + lookahead_frames], null),
            speaker_cache,
            num_lookahead_frames: chunk_right_context,
          });
          // the chunk's frames, without its look-ahead
          expect(outputs.logits.dims).toEqual([1, chunk_frames, 4]);
          expect(outputs.speaker_cache).toBeInstanceOf(Nemotron3DiarizationSpeakerCache);
          step_logits.push(outputs.logits);
          speaker_cache = outputs.speaker_cache;
          start += chunk_frames;
        }
        // The last call: the remaining frames are the last (partial) chunk, none of them is look-ahead
        const last = await model({ input_features: input_features.slice(null, [start, num_frames], null), speaker_cache });
        expect(last.speaker_cache).toBe(speaker_cache);
        step_logits.push(last.logits);

        expectLogitsCloseTo(cat(step_logits, 1), OFFLINE_FEATURES_LOGITS);
        expect(speaker_cache.num_cache_frames).toBe(12);
        expect(speaker_cache.num_fifo_frames).toBe(4);
        expect(speaker_cache.is_compressed).toBe(true);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "padded batch",
      async () => {
        // The second sample is 200 frames long
        const attention_mask = new Tensor(
          "int64",
          BigInt64Array.from({ length: 2 * 256 }, (_, i) => (i >= 256 + 200 ? 0n : 1n)),
          [2, 256],
        );
        const { logits } = await model({ input_features: make_features(2, 256, 128), attention_mask });
        expect(logits.dims).toEqual([2, 256, 4]);
        expectLogitsCloseTo(logits.slice(0, null, null), { ...OFFLINE_FEATURES_LOGITS, dims: [256, 4] });
        expectLogitsCloseTo(logits.slice(1, [0, 200], null), {
          dims: [200, 4],
          mean: 0.41830694675445557,
          first: [0.9303014278411865, 0.2020755112171173, -0.8907040357589722, 1.661102056503296],
          last: [0.8144288063049316, 0.5871553421020508, -1.0155444145202637, 0.9329540133476257],
        });
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "offline audio",
      async () => {
        const inputs = await processor(make_audio(48000));
        const { logits } = await model(inputs);
        expectLogitsCloseTo(logits, {
          dims: [1, 301, 4],
          mean: 0.41537535190582275,
          first: [0.7759349346160889, 0.532768189907074, -0.991628110408783, 1.2060028314590454],
          last: [0.9027887582778931, 0.04079252481460571, -0.4897063672542572, 1.7132656574249268],
        });
        expect(processor.extract_speaker_dict(logits, inputs.attention_mask)).toEqual([
          [
            { Start: 0.0, End: 3.0, Speaker: 0 },
            { Start: 0.0, End: 3.0, Speaker: 1 },
            { Start: 0.0, End: 3.0, Speaker: 3 },
          ],
        ]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "streaming audio",
      async () => {
        const audio = make_audio(48000);
        processor.set_streaming_mode("low_latency");

        const chunks = [await processor(audio.subarray(0, processor.num_samples_first_audio_chunk), { is_streaming: true })];
        let mel_frame_idx = processor.num_mel_frames_per_step;
        let start_idx = processor.audio_chunk_start(mel_frame_idx);
        while (start_idx + processor.num_samples_per_audio_chunk <= audio.length) {
          chunks.push(await processor(audio.subarray(start_idx, start_idx + processor.num_samples_per_audio_chunk), { is_streaming: true, is_first_audio_chunk: false }));
          mel_frame_idx += processor.num_mel_frames_per_step;
          start_idx = processor.audio_chunk_start(mel_frame_idx);
        }
        chunks.push(await processor(audio.subarray(start_idx), { is_streaming: true, is_first_audio_chunk: false, is_last_audio_chunk: true }));

        const step_logits = [];
        let speaker_cache = null;
        for (const inputs of chunks) {
          const outputs = await model({ ...inputs, speaker_cache });
          step_logits.push(outputs.logits);
          speaker_cache = outputs.speaker_cache;
        }
        expect(step_logits.map((x) => x.dims[1])).toEqual([72, 72, 72, 83]);
        expect(speaker_cache.num_cache_frames).toBe(12);
        expect(speaker_cache.num_fifo_frames).toBe(4);
        expect(speaker_cache.is_compressed).toBe(true);

        const logits = cat(step_logits, 1);
        expectLogitsCloseTo(logits, {
          dims: [1, 299, 4],
          mean: 0.41420215368270874,
          first: [0.7759641408920288, 0.5326508283615112, -0.9915345907211304, 1.2059553861618042],
          last: [0.9267847537994385, -0.005907908082008362, -0.48862895369529724, 1.5985972881317139],
        });
        expect(processor.extract_speaker_dict(logits)).toEqual([
          [
            { Start: 0.0, End: 2.99, Speaker: 0 },
            { Start: 0.0, End: 2.96, Speaker: 1 },
            { Start: 0.0, End: 2.99, Speaker: 3 },
          ],
        ]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "rejects an invalid look-ahead",
      async () => {
        const input_features = make_features(1, 256, 128);
        await expect(model({ input_features, num_lookahead_frames: -1 })).rejects.toThrow();
        // nothing but look-ahead
        await expect(model({ input_features: input_features.slice(null, [0, 8], null), num_lookahead_frames: 1 })).rejects.toThrow();
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
