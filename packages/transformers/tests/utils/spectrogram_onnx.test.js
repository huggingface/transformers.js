import { jest } from "@jest/globals";
import { spectrogram, spectrogram_kernel, window_function, mel_filter_bank } from "../../src/utils/audio.js";
import { TensorOpRegistry } from "../../src/ops/registry.js";
import { init } from "../init.js";

init();

/**
 * Deterministic test signal: a few sines plus a DC offset and a pseudo-random component.
 * @param {number} length
 */
function make_signal(length) {
  const out = new Float32Array(length);
  let seed = 1234;
  for (let i = 0; i < length; ++i) {
    seed = (seed * 1103515245 + 12345) % 2147483648;
    const noise = seed / 2147483648 - 0.5;
    out[i] = 0.3 + 0.5 * Math.sin((2 * Math.PI * 440 * i) / 16000) + 0.2 * Math.sin((2 * Math.PI * 3100 * i) / 16000) + 0.1 * noise;
  }
  return out;
}

/**
 * Float64 reference for `spectrogram()`: a naive DFT per frame, written independently of the implementation.
 */
function reference_spectrogram(waveform, window, frame_length, hop_length, options) {
  const { fft_length = frame_length, power = 1.0, center = true, pad_mode = "reflect", onesided = true, preemphasis = null, mel_filters, mel_floor = 1e-10, mel_offset = 0, mel_floor_mode = "clamp", log_mel = null, max_log_mel = null, reference = 1.0, min_value = 1e-10, db_range = null, remove_dc_offset = null, min_num_frames = null, max_num_frames = null, do_pad = true, transpose = false } = options;

  let x = Array.from(waveform);
  if (center) {
    const p = Math.floor(frame_length / 2);
    if (pad_mode === "reflect") {
      // Periodic reflection (period 2 * (length - 1)), so padding may exceed the signal length.
      const w = x.length - 1;
      const at = (i) => {
        const m = ((i % (2 * w)) + 2 * w) % (2 * w);
        return x[m > w ? 2 * w - m : m];
      };
      const left = [];
      const right = [];
      for (let i = p; i >= 1; --i) left.push(at(-i));
      for (let i = 1; i <= p; ++i) right.push(at(w + i));
      x = [...left, ...x, ...right];
    } else if (pad_mode === "constant") {
      x = [...new Array(p).fill(0), ...x, ...new Array(p).fill(0)];
    } else if (pad_mode === "semicausal") {
      x = [...new Array(p).fill(0), ...x];
    }
  }

  let num_frames = Math.floor(1 + Math.floor((x.length - frame_length) / hop_length));
  if (min_num_frames !== null && num_frames < min_num_frames) num_frames = min_num_frames;
  let d1 = num_frames;
  let d1Max = num_frames;
  if (max_num_frames !== null) {
    if (max_num_frames > num_frames) {
      if (do_pad) d1Max = max_num_frames;
    } else {
      d1Max = d1 = max_num_frames;
    }
  }
  const bins = onesided ? Math.floor(fft_length / 2) + 1 : fft_length;
  const n_mels = mel_filters.length;
  const out = new Float64Array(n_mels * d1Max);
  const cos = new Float64Array(fft_length);
  const sin = new Float64Array(fft_length);
  for (let m = 0; m < fft_length; ++m) {
    cos[m] = Math.cos((2 * Math.PI * m) / fft_length);
    sin[m] = Math.sin((2 * Math.PI * m) / fft_length);
  }

  for (let t = 0; t < d1; ++t) {
    const offset = t * hop_length;
    const size = Math.max(0, Math.min(x.length - offset, frame_length));
    const frame = new Float64Array(frame_length);
    for (let j = 0; j < size; ++j) frame[j] = x[offset + j];
    if (remove_dc_offset && size > 0) {
      let mean = 0;
      for (let j = 0; j < size; ++j) mean += frame[j];
      mean /= size;
      for (let j = 0; j < size; ++j) frame[j] -= mean;
    }
    if (preemphasis !== null && size > 0) {
      for (let j = size - 1; j >= 1; --j) frame[j] -= preemphasis * frame[j - 1];
      frame[0] *= 1 - preemphasis;
    }
    const spectrum = new Float64Array(bins);
    for (let k = 0; k < bins; ++k) {
      let re = 0;
      let im = 0;
      for (let n = 0; n < frame_length; ++n) {
        const v = frame[n] * window[n];
        const m = (k * n) % fft_length;
        re += v * cos[m];
        im -= v * sin[m];
      }
      spectrum[k] = (re * re + im * im) ** (power / 2);
    }
    for (let m = 0; m < n_mels; ++m) {
      let sum = 0;
      for (let k = 0; k < bins; ++k) sum += mel_filters[m][k] * spectrum[k];
      out[m * d1Max + t] = sum;
    }
  }
  let data = out;
  let dims = [n_mels, d1Max];
  if (transpose) {
    data = new Float64Array(out.length);
    for (let m = 0; m < n_mels; ++m) {
      for (let t = 0; t < d1Max; ++t) data[t * n_mels + m] = out[m * d1Max + t];
    }
    dims = [d1Max, n_mels];
  }
  for (let i = 0; i < data.length; ++i) {
    data[i] = mel_floor_mode === "add" ? mel_offset + data[i] + mel_floor : mel_offset + Math.max(mel_floor, data[i]);
  }
  // Same regions as `spectrogram()`: the log covers the first `d1 * n_mels` values; dB and the max cover all of them.
  const o = Math.min(data.length, d1 * n_mels);
  const max_of = (arr) => arr.reduce((a, b) => Math.max(a, b), -Infinity);
  if (log_mel === "log") for (let i = 0; i < o; ++i) data[i] = Math.log(data[i]);
  if (log_mel === "log10") for (let i = 0; i < o; ++i) data[i] = Math.log10(data[i]);
  if (log_mel === "log10_max_norm") {
    for (let i = 0; i < o; ++i) data[i] = Math.log10(data[i]);
    const threshold = (max_log_mel ?? max_of(data)) - 8;
    for (let i = 0; i < o; ++i) data[i] = (Math.max(data[i], threshold) + 4) / 4;
  }
  if (log_mel === "dB") {
    const factor = power === 1 ? 20 : 10;
    const log_reference = Math.log10(Math.max(min_value, reference));
    for (let i = 0; i < data.length; ++i) data[i] = factor * Math.log10(Math.max(min_value, data[i]) - log_reference);
    if (db_range !== null) {
      const floor = max_of(data) - db_range;
      for (let i = 0; i < data.length; ++i) data[i] = Math.max(data[i], floor);
    }
  }
  return { data, dims };
}

const whisper_filters = mel_filter_bank(201, 80, 0.0, 8000.0, 16000, "slaney", "slaney");
const kaldi_filters = mel_filter_bank(257, 80, 20, Math.floor(16000 / 2), 16000, null, "kaldi", true);
const twosided_filters = mel_filter_bank(256, 40, 0, 16000, 16000, "slaney", "slaney").map((row) => row.slice(0, 256));

const CASES = [
  {
    name: "Whisper-style (reflect padding, power 2)",
    length: 16000,
    frame_length: 400,
    hop_length: 160,
    window: window_function(400, "hann"),
    options: { power: 2.0, mel_filters: whisper_filters },
  },
  {
    name: "longer than one op chunk (> 1024 frames), max_num_frames padding and transpose",
    length: 16000 * 12,
    frame_length: 400,
    hop_length: 160,
    window: window_function(400, "hann"),
    options: { power: 2.0, mel_filters: whisper_filters, max_num_frames: 1300, transpose: true },
  },
  {
    name: "max_num_frames truncation",
    length: 16000 * 2,
    frame_length: 400,
    hop_length: 160,
    window: window_function(400, "hann"),
    options: { power: 2.0, mel_filters: whisper_filters, max_num_frames: 50 },
  },
  {
    name: "Kaldi-style (fft_length > frame_length, DC offset, preemphasis, no centering)",
    length: 8000,
    frame_length: 400,
    hop_length: 160,
    window: window_function(400, "povey", { periodic: false }),
    options: { fft_length: 512, power: 2.0, center: false, preemphasis: 0.97, remove_dc_offset: true, mel_filters: kaldi_filters },
  },
  {
    name: "min_num_frames past the end of the signal with DC offset and preemphasis",
    length: 2000,
    frame_length: 400,
    hop_length: 160,
    window: window_function(400, "povey", { periodic: false }),
    options: { fft_length: 512, power: 2.0, center: false, preemphasis: 0.97, remove_dc_offset: true, mel_filters: kaldi_filters, min_num_frames: 30 },
  },
  {
    name: "amplitude spectrogram with constant padding",
    length: 6000,
    frame_length: 512,
    hop_length: 128,
    window: window_function(512, "hann"),
    options: { power: 1.0, pad_mode: "constant", mel_filters: mel_filter_bank(257, 64, 0, 8000, 16000, "slaney", "slaney") },
  },
  {
    name: "semicausal padding",
    length: 6000,
    frame_length: 400,
    hop_length: 160,
    window: window_function(400, "hann"),
    options: { power: 2.0, pad_mode: "semicausal", mel_filters: whisper_filters },
  },
  {
    name: "window zero-padded to frame_length (Parakeet-style)",
    length: 16000 * 3,
    frame_length: 512,
    hop_length: 160,
    window: window_function(400, "hann", { frame_length: 512 }),
    options: { power: 2.0, mel_filters: mel_filter_bank(257, 128, 0, 8000, 16000, "slaney", "slaney") },
  },
  {
    name: "zero-padded window with DC offset and preemphasis",
    length: 6000,
    frame_length: 512,
    hop_length: 160,
    window: window_function(400, "hann", { frame_length: 512 }),
    options: { power: 2.0, preemphasis: 0.97, remove_dc_offset: true, mel_filters: kaldi_filters, min_num_frames: 50 },
  },
  {
    name: "log10_max_norm with do_pad (log over the first d1 * n_mels values)",
    length: 4000,
    frame_length: 400,
    hop_length: 160,
    window: window_function(400, "hann"),
    options: { power: 2.0, mel_filters: whisper_filters, log_mel: "log10_max_norm", max_num_frames: 60 },
  },
  {
    name: "log10 with transpose",
    length: 6000,
    frame_length: 400,
    hop_length: 160,
    window: window_function(400, "hann"),
    options: { power: 2.0, mel_filters: whisper_filters, log_mel: "log10", transpose: true },
  },
  {
    name: "power dB with db_range (CLAP-style)",
    length: 8000,
    frame_length: 400,
    hop_length: 160,
    window: window_function(400, "hann"),
    options: { power: 2.0, mel_filters: whisper_filters, log_mel: "dB", db_range: 80, transpose: true },
  },
  {
    name: "amplitude dB",
    length: 6000,
    frame_length: 400,
    hop_length: 160,
    window: window_function(400, "hann"),
    options: { power: 1.0, mel_filters: whisper_filters, log_mel: "dB" },
  },
  {
    name: "additive mel_floor with mel_offset and log (Gemma 4-style)",
    length: 6000,
    frame_length: 400,
    hop_length: 160,
    window: window_function(400, "hann"),
    options: { power: 2.0, pad_mode: "semicausal", mel_filters: whisper_filters, log_mel: "log", mel_floor: 1e-3, mel_floor_mode: "add", mel_offset: 0.5, transpose: true },
  },
  {
    name: "reflect padding longer than the signal",
    length: 120,
    frame_length: 400,
    hop_length: 40,
    window: window_function(400, "hann"),
    options: { power: 2.0, mel_filters: whisper_filters, log_mel: "log" },
  },
  {
    name: "two-sided spectrum",
    length: 4000,
    frame_length: 256,
    hop_length: 64,
    window: window_function(256, "hann"),
    options: { power: 2.0, onesided: false, mel_filters: twosided_filters },
  },
];

describe("spectrogram (ONNX Runtime ops)", () => {
  const OPS = ["conv1d_channels_last", "pow", "matmul", "matmul_transpose_a"];
  let getters;
  beforeEach(() => {
    getters = Object.fromEntries(OPS.map((name) => [name, jest.spyOn(TensorOpRegistry, name, "get")]));
  });
  afterEach(() => {
    Object.values(getters).forEach((getter) => getter.mockRestore());
  });

  it.each(CASES)("$name", async ({ length, frame_length, hop_length, window, options }) => {
    const waveform = make_signal(length);
    const output = await spectrogram(waveform, window, frame_length, hop_length, options);
    const expected = reference_spectrogram(waveform, window, frame_length, hop_length, options);

    // The ONNX Runtime ops must have run.
    expect(getters.conv1d_channels_last).toHaveBeenCalled();
    expect(getters.pow).toHaveBeenCalled();
    expect(getters[options.transpose ? "matmul_transpose_a" : "matmul"]).toHaveBeenCalled();

    expect(output.dims).toEqual(expected.dims);
    let peak = 0;
    for (const v of expected.data) peak = Math.max(peak, v);
    for (let i = 0; i < expected.data.length; ++i) {
      const e = expected.data[i];
      const a = output.data[i];
      // float32 accumulation: relative error for significant bins, absolute (vs. peak) for tiny ones
      // Log-scaled outputs: float32 rounding of the mel values becomes a small absolute error after the log.
      const tolerance = Math.max(1e-4 * Math.abs(e), 1e-6 * peak, options.log_mel ? 1e-3 : 0);
      expect(Math.abs(a - e)).toBeLessThanOrEqual(tolerance);
    }
  });

  describe("precomputed kernel", () => {
    const window = window_function(400, "povey", { periodic: false });
    const options = { fft_length: 512, power: 2.0, center: false, preemphasis: 0.97, remove_dc_offset: true, mel_filters: kaldi_filters };
    const kernel = spectrogram_kernel(window, 160, { fft_length: 512, preemphasis: 0.97, remove_dc_offset: true });

    it("gives the same result as building the kernel on each call", async () => {
      const waveform = make_signal(8000);
      const built = await spectrogram(waveform, window, 400, 160, options);
      const precomputed = await spectrogram(waveform, window, 400, 160, { ...options, kernel });
      expect(precomputed.dims).toEqual(built.dims);
      expect(Array.from(precomputed.data)).toEqual(Array.from(built.data));
    });

    it("rejects a kernel built for different settings", async () => {
      const waveform = make_signal(8000);
      await expect(spectrogram(waveform, window, 400, 160, { ...options, preemphasis: 0.9, kernel })).rejects.toThrow("`kernel`");
      await expect(spectrogram(waveform, window, 400, 100, { ...options, kernel })).rejects.toThrow("`kernel`");
      await expect(spectrogram(waveform, Float64Array.from(window), 400, 160, { ...options, kernel })).rejects.toThrow("`kernel`");
    });
  });
});
