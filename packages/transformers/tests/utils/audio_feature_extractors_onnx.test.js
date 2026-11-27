import { spectrogram, window_function } from "../../src/utils/audio.js";
import { SeamlessM4TFeatureExtractor } from "../../src/models/seamless_m4t/feature_extraction_seamless_m4t.js";
import { WeSpeakerFeatureExtractor } from "../../src/models/wespeaker/feature_extraction_wespeaker.js";
import { ASTFeatureExtractor } from "../../src/models/audio_spectrogram_transformer/feature_extraction_audio_spectrogram_transformer.js";
import { ParakeetFeatureExtractor } from "../../src/models/parakeet/feature_extraction_parakeet.js";
import { Wav2Vec2FeatureExtractor } from "../../src/models/wav2vec2/feature_extraction_wav2vec2.js";
import { init } from "../init.js";

init();

// Offline checks that the ONNX-based post-processing of the audio feature extractors matches the plain JS
// definitions they replaced (written out below as float64 references).

/** @param {number} length */
function make_signal(length) {
  const out = new Float32Array(length);
  let seed = 4321;
  for (let i = 0; i < length; ++i) {
    seed = (seed * 1103515245 + 12345) % 2147483648;
    out[i] = 0.02 + 0.4 * Math.sin((2 * Math.PI * 300 * i) / 16000) + 0.1 * (seed / 2147483648 - 0.5);
  }
  return out;
}

function expect_close(actual, expected, tolerance) {
  expect(actual.length).toBe(expected.length);
  let max_diff = 0;
  for (let i = 0; i < expected.length; ++i) {
    max_diff = Math.max(max_diff, Math.abs(actual[i] - expected[i]));
  }
  expect(max_diff).toBeLessThanOrEqual(tolerance);
}

/**
 * Normalizes each column of a row-major [rows, cols] matrix over its first `count` rows.
 * @returns {Float64Array}
 */
function normalize_columns(data, rows, cols, { count = rows, ddof, eps_inside = 0, eps_outside = 0, scale = true }) {
  const out = Float64Array.from(data);
  for (let j = 0; j < cols; ++j) {
    let sum = 0;
    for (let i = 0; i < count; ++i) sum += data[i * cols + j];
    const mean = sum / count;
    let sq = 0;
    for (let i = 0; i < count; ++i) sq += (data[i * cols + j] - mean) ** 2;
    const std = Math.sqrt(sq / Math.max(count - ddof, 1) + eps_inside) + eps_outside;
    for (let i = 0; i < count; ++i) out[i * cols + j] = scale ? (data[i * cols + j] - mean) / std : data[i * cols + j] - mean;
  }
  return out;
}

const KALDI_OPTIONS = { fft_length: 512, power: 2.0, center: false, preemphasis: 0.97, log_mel: "log", mel_floor: 1.192092955078125e-7, remove_dc_offset: true, transpose: true };

describe("Audio feature extractors (ONNX post-processing)", () => {
  it("SeamlessM4T: 16-bit scaling folded into the window, and per-mel normalization", async () => {
    const config = { sampling_rate: 16000, num_mel_bins: 80, feature_size: 80, padding_value: 1, stride: 2 };
    const fe = new SeamlessM4TFeatureExtractor(config);
    const audio = make_signal(16000);

    // The scaled window must give the same features as scaling the waveform.
    const window = window_function(400, "povey", { periodic: false });
    const raw = await spectrogram(
      audio.map((x) => x * 32768),
      window,
      400,
      160,
      { ...KALDI_OPTIONS, mel_filters: fe.mel_filters },
    );
    const features = await fe._extract_fbank_features(audio, config.max_length);
    expect(features.dims).toEqual(raw.dims);
    expect_close(features.data, raw.data, 1e-3);

    const [frames, mels] = raw.dims;
    const expected = normalize_columns(raw.data, frames, mels, { ddof: 1, eps_inside: 1e-7 });
    const { input_features } = await fe(audio, { padding: false });
    expect(input_features.dims).toEqual([1, frames / 2, mels * 2]);
    expect_close(input_features.data, expected, 1e-3);
  });

  it("WeSpeaker: centering over all frames, including the `min_num_frames` padding", async () => {
    const config = { sampling_rate: 16000, num_mel_bins: 80, min_num_frames: 200, fbank_centering_span: null };
    const fe = new WeSpeakerFeatureExtractor(config);
    const audio = make_signal(8000);

    const window = window_function(400, "hamming", { periodic: false });
    const raw = await spectrogram(
      audio.map((x) => x * 32768),
      window,
      400,
      160,
      { ...KALDI_OPTIONS, mel_filters: fe.mel_filters, min_num_frames: 200 },
    );
    const [frames, mels] = raw.dims;
    expect(frames).toBe(200);

    const { input_features } = await fe(audio);
    expect(input_features.dims).toEqual([1, frames, mels]);
    expect_close(input_features.data, normalize_columns(raw.data, frames, mels, { ddof: 0, scale: false }), 1e-3);
  });

  it("AST: normalization to mean 0, std 0.5", async () => {
    const config = { sampling_rate: 16000, num_mel_bins: 128, max_length: 256, do_normalize: true, mean: -4.27, std: 4.57 };
    const fe = new ASTFeatureExtractor(config);
    const audio = make_signal(16000);

    const raw = await fe._extract_fbank_features(audio, config.max_length);
    const { input_values } = await fe(audio);
    expect(input_values.dims).toEqual([1, ...raw.dims]);
    expect_close(
      input_values.data,
      Array.from(raw.data, (x) => (x - config.mean) / (config.std * 2)),
      1e-5,
    );
  });

  describe("Parakeet", () => {
    const config = { sampling_rate: 16000, feature_size: 128, n_fft: 512, win_length: 400, hop_length: 160, preemphasis: 0.97 };

    it("preemphasis over the whole waveform", async () => {
      const fe = new ParakeetFeatureExtractor(config);
      const identity = new ParakeetFeatureExtractor({ ...config, preemphasis: 0 });
      const audio = make_signal(4000);

      const emphasized = new Float64Array(audio);
      for (let j = emphasized.length - 1; j >= 1; --j) emphasized[j] -= config.preemphasis * emphasized[j - 1];

      const actual = await fe._extract_fbank_features(audio);
      const expected = await identity._extract_fbank_features(emphasized);
      expect(actual.dims).toEqual(expected.dims);
      expect_close(actual.data, expected.data, 1e-3);
    });

    it("normalization over the valid frames, with zeros after them", async () => {
      const fe = new ParakeetFeatureExtractor(config);
      const audio = make_signal(4000);

      const raw = await fe._extract_fbank_features(audio);
      const [frames, mels] = raw.dims;
      const count = Math.floor(audio.length / config.hop_length);
      expect(count).toBeLessThan(frames);

      const expected = normalize_columns(raw.data, frames, mels, { count, ddof: 1, eps_outside: 1e-5 });
      expected.fill(0, count * mels);

      const { input_features, attention_mask } = await fe(audio);
      expect(input_features.dims).toEqual([1, frames, mels]);
      expect_close(input_features.data, expected, 1e-3);
      expect(Array.from(input_features.data.subarray(count * mels)).every((x) => x === 0)).toBe(true);
      expect(Array.from(attention_mask.data, Number)).toEqual(Array.from({ length: frames }, (_, i) => (i < count ? 1 : 0)));
    });

    it("audio too short for a valid frame gives zeros", async () => {
      const fe = new ParakeetFeatureExtractor(config);
      const { input_features, attention_mask } = await fe(make_signal(100));
      expect(input_features.type).toBe("float32");
      expect(input_features.dims).toEqual([1, 1, config.feature_size]);
      expect(Array.from(input_features.data).every((x) => x === 0)).toBe(true);
      expect(Array.from(attention_mask.data, Number)).toEqual([0]);
    });

    it("does not modify the input", async () => {
      const fe = new ParakeetFeatureExtractor(config);
      const audio = make_signal(4000);
      const copy = audio.slice();
      await fe(audio);
      expect(audio).toEqual(copy);
    });
  });

  it("Wav2Vec2: zero-mean unit-variance normalization", async () => {
    const fe = new Wav2Vec2FeatureExtractor({ do_normalize: true, sampling_rate: 16000 });
    const audio = make_signal(48000);
    const copy = audio.slice();

    const { input_values, attention_mask } = await fe(audio);
    expect(input_values.dims).toEqual([1, audio.length]);
    expect(attention_mask.dims).toEqual([1, audio.length]);
    expect_close(input_values.data, normalize_columns(audio, audio.length, 1, { ddof: 0, eps_inside: 1e-7 }), 1e-4);
    expect(audio).toEqual(copy);
  });
});
