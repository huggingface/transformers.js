import { RawAudio, spectrogram, window_function, mel_filter_bank } from "../../src/utils/audio.js";
import { init } from "../init.js";

init();

/**
 * Helper function to generate a sine wave.
 * @param {number} length Length of the audio in samples.
 * @param {number} freq Frequency of the sine wave.
 * @param {number} sampling_rate Sampling rate.
 * @returns {Float32Array} The generated sine wave.
 */
function generateSineWave(length, freq, sampling_rate) {
  const audio = new Float32Array(length);
  for (let i = 0; i < length; ++i) {
    audio[i] = Math.sin((2 * Math.PI * freq * i) / sampling_rate);
  }
  return audio;
}

/**
 * Zero-pad a window to a given frame length, centering the window.
 * Replicates Python's window_function zero-padding behavior.
 * @param {Float64Array} win The window to pad.
 * @param {number} frameLength The target frame length.
 * @returns {Float64Array} The zero-padded window.
 */
function zeroPadWindow(win, frameLength) {
  const padded = new Float64Array(frameLength);
  const offset = Math.floor((frameLength - win.length) / 2);
  padded.set(win, offset);
  return padded;
}

/**
 * Generate a deterministic waveform of a given length.
 * @param {number} length Length of the waveform.
 * @returns {Float32Array} The generated waveform.
 */
function generateDeterministicWaveform(length) {
  const audio = new Float32Array(length);
  const freqs = [220, 440, 880, 1760];
  const sr = 16000;
  for (let i = 0; i < length; ++i) {
    let val = 0;
    for (const f of freqs) {
      val += Math.sin((2 * Math.PI * f * i) / sr);
    }
    audio[i] = val / freqs.length;
  }
  return audio;
}

/**
 * Create an identity mel filter bank of shape (numBins, numBins).
 * This lets us call spectrogram without changing the output shape,
 * since the JS implementation requires mel_filters.
 * @param {number} numBins Number of frequency bins.
 * @returns {number[][]} Identity matrix as mel filters.
 */
function identityMelFilters(numBins) {
  return Array.from({ length: numBins }, (_, i) => {
    const row = new Array(numBins).fill(0);
    row[i] = 1;
    return row;
  });
}

describe("Audio utilities", () => {
  describe("RawAudio", () => {
    it("should create RawAudio from a single Float32Array", () => {
      const sampling_rate = 16000;
      const audioData = generateSineWave(1000, 440, sampling_rate);
      const rawAudio = new RawAudio(audioData, sampling_rate);

      expect(rawAudio.sampling_rate).toBe(sampling_rate);
      expect(rawAudio.data).toBeInstanceOf(Float32Array);
      expect(rawAudio.data).toEqual(audioData);
      expect(rawAudio.data.length).toBe(1000);
    });

    it("should create RawAudio from multiple Float32Array chunks", () => {
      const sampling_rate = 16000;
      const chunk1 = generateSineWave(500, 440, sampling_rate);
      const chunk2 = generateSineWave(500, 880, sampling_rate);
      const rawAudio = new RawAudio([chunk1, chunk2], sampling_rate);

      expect(rawAudio.sampling_rate).toBe(sampling_rate);
      expect(rawAudio.data).toBeInstanceOf(Float32Array);
      expect(rawAudio.data.length).toBe(1000);

      // Check if concatenation is correct
      const combined = new Float32Array(1000);
      combined.set(chunk1, 0);
      combined.set(chunk2, 500);
      expect(rawAudio.data).toEqual(combined);
    });

    it("should handle empty array of chunks", () => {
      const rawAudio = new RawAudio([], 16000);
      expect(rawAudio.data).toBeInstanceOf(Float32Array);
      expect(rawAudio.data.length).toBe(0);
    });

    it("should convert to Blob (WAV)", () => {
      const sampling_rate = 16000;
      const audioData = generateSineWave(1000, 440, sampling_rate);
      const rawAudio = new RawAudio(audioData, sampling_rate);

      const blob = rawAudio.toBlob();
      expect(blob).toBeInstanceOf(Blob);
      expect(blob.type).toBe("audio/wav");

      // WAV header is 44 bytes
      // 1000 samples * 4 bytes/sample (float32) = 4000 bytes
      expect(blob.size).toBe(4044);
    });

    it("should convert to Blob (WAV) from chunks", () => {
      const sampling_rate = 16000;
      const chunk1 = generateSineWave(500, 440, sampling_rate);
      const chunk2 = generateSineWave(500, 880, sampling_rate);
      const rawAudio = new RawAudio([chunk1, chunk2], sampling_rate);

      const blob = rawAudio.toBlob();
      expect(blob).toBeInstanceOf(Blob);
      expect(blob.type).toBe("audio/wav");
      expect(blob.size).toBe(4044); // 44 header + 4000 data
    });
  });

  describe("spectrogram", () => {
    it("should compute spectrogram of impulse signal", async () => {
      const waveform = new Float32Array(40);
      waveform[9] = 1.0;

      const win = window_function(12, "hann");
      const paddedWin = zeroPadWindow(win, 16);
      const numBins = 9; // onesided: fft_length/2 + 1 = 16/2 + 1

      const spec = await spectrogram(waveform, paddedWin, 16, 4, {
        power: 1.0,
        center: true,
        pad_mode: "reflect",
        onesided: true,
        mel_filters: identityMelFilters(numBins),
      });

      expect(spec.dims).toEqual([9, 11]);

      const expected = [0.0, 0.0669873, 0.9330127, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
      expect(Array.from(spec.data.slice(0, 11))).toBeCloseToNested(expected, 4);
    });

    it("should work with window_function frame_length zero-padding", async () => {
      // Should zero-pad the window to length 16, matching Python behavior.
      // See https://github.com/huggingface/transformers.js/issues/1387.
      const waveform = new Float32Array(40);
      waveform[9] = 1.0;

      const win = window_function(12, "hann", { frame_length: 16 });
      expect(win.length).toBe(16);

      const numBins = 9;
      const spec = await spectrogram(waveform, win, 16, 4, {
        power: 1.0,
        center: true,
        pad_mode: "reflect",
        onesided: true,
        mel_filters: identityMelFilters(numBins),
      });

      expect(spec.dims).toEqual([9, 11]);

      const expected = [0.0, 0.0669873, 0.9330127, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
      expect(Array.from(spec.data.slice(0, 11))).toBeCloseToNested(expected, 4);
    });

    describe("shapes", () => {
      const waveform = generateDeterministicWaveform(93680);

      it("should produce correct shape with default params", async () => {
        const numBins = 201; // 400/2 + 1
        const spec = await spectrogram(waveform, window_function(400, "hann"), 400, 128, {
          power: 1.0,
          center: true,
          pad_mode: "reflect",
          onesided: true,
          mel_filters: identityMelFilters(numBins),
        });
        expect(spec.dims).toEqual([201, 732]);
      });

      it("should produce correct shape with center=false", async () => {
        const numBins = 201;
        const spec = await spectrogram(waveform, window_function(400, "hann"), 400, 128, {
          power: 1.0,
          center: false,
          pad_mode: "reflect",
          onesided: true,
          mel_filters: identityMelFilters(numBins),
        });
        expect(spec.dims).toEqual([201, 729]);
      });

      it("should produce correct shape with fft_length", async () => {
        const numBins = 257; // 512/2 + 1
        const spec = await spectrogram(waveform, window_function(400, "hann"), 400, 128, {
          fft_length: 512,
          power: 1.0,
          center: true,
          pad_mode: "reflect",
          onesided: true,
          mel_filters: identityMelFilters(numBins),
        });
        expect(spec.dims).toEqual([257, 732]);
      });

      it("should produce correct shape with onesided=false and frame_length=512 (padded window)", async () => {
        const win = window_function(400, "hann");
        const paddedWin = zeroPadWindow(win, 512);
        const numBins = 512; // onesided=false
        const spec = await spectrogram(waveform, paddedWin, 512, 64, {
          power: 1.0,
          center: true,
          pad_mode: "reflect",
          onesided: false,
          mel_filters: identityMelFilters(numBins),
        });
        expect(spec.dims).toEqual([512, 1464]);
      });

      it("should produce correct shape with onesided=false and window_length=512", async () => {
        const numBins = 512;
        const spec = await spectrogram(waveform, window_function(512, "hann"), 512, 64, {
          power: 1.0,
          center: true,
          pad_mode: "reflect",
          onesided: false,
          mel_filters: identityMelFilters(numBins),
        });
        expect(spec.dims).toEqual([512, 1464]);
      });

      it("should produce correct shape with large hop_length", async () => {
        const numBins = 512;
        const spec = await spectrogram(waveform, window_function(512, "hann"), 512, 512, {
          power: 1.0,
          center: true,
          pad_mode: "reflect",
          onesided: false,
          mel_filters: identityMelFilters(numBins),
        });
        expect(spec.dims).toEqual([512, 183]);
      });
    });

    describe("center padding", () => {
      const waveform = generateDeterministicWaveform(93680);
      const numBins = 257; // 512/2 + 1

      it("should handle reflect padding", async () => {
        const spec = await spectrogram(waveform, window_function(512, "hann"), 512, 128, {
          center: true,
          pad_mode: "reflect",
          mel_filters: identityMelFilters(numBins),
        });
        expect(spec.dims).toEqual([257, 732]);
      });

      it("should handle constant padding", async () => {
        const spec = await spectrogram(waveform, window_function(512, "hann"), 512, 128, {
          center: true,
          pad_mode: "constant",
          mel_filters: identityMelFilters(numBins),
        });
        expect(spec.dims).toEqual([257, 732]);
      });

      it("should handle no centering", async () => {
        const spec = await spectrogram(waveform, window_function(512, "hann"), 512, 128, {
          center: false,
          mel_filters: identityMelFilters(numBins),
        });
        expect(spec.dims).toEqual([257, 728]);
      });
    });

    describe("mel spectrogram", () => {
      const waveform = generateDeterministicWaveform(93680);

      it("should produce correct shape without mel filters (using identity)", async () => {
        const numBins = 513; // 1024/2 + 1
        const win = window_function(800, "hann");
        const paddedWin = zeroPadWindow(win, 1024);
        const spec = await spectrogram(waveform, paddedWin, 1024, 128, {
          power: 2.0,
          mel_filters: identityMelFilters(numBins),
        });
        expect(spec.dims).toEqual([513, 732]);
      });

      it("should produce correct shape with mel filters", async () => {
        const mel_filters = mel_filter_bank(513, 13, 100, 4000, 16000, null, "htk");

        const win = window_function(800, "hann");
        const paddedWin = zeroPadWindow(win, 1024);
        const spec = await spectrogram(waveform, paddedWin, 1024, 128, {
          power: 2.0,
          mel_filters,
        });
        expect(spec.dims).toEqual([13, 732]);
      });
    });

    describe("power", () => {
      const waveform = generateDeterministicWaveform(93680);

      it("should compute amplitude spectrogram (power=1.0)", async () => {
        const numBins = 257;
        const win = window_function(400, "hann");
        const paddedWin = zeroPadWindow(win, 512);
        const spec = await spectrogram(waveform, paddedWin, 512, 128, {
          power: 1.0,
          mel_filters: identityMelFilters(numBins),
        });
        expect(spec.dims).toEqual([257, 732]);
      });

      it("should compute power spectrogram (power=2.0)", async () => {
        const numBins = 257;
        const win = window_function(400, "hann");
        const paddedWin = zeroPadWindow(win, 512);
        const spec = await spectrogram(waveform, paddedWin, 512, 128, {
          power: 2.0,
          mel_filters: identityMelFilters(numBins),
        });
        expect(spec.dims).toEqual([257, 732]);
      });

      it("power=2 values should be square of power=1 values", async () => {
        const numBins = 257;
        const win = window_function(400, "hann");
        const paddedWin = zeroPadWindow(win, 512);

        const spec1 = await spectrogram(waveform, paddedWin, 512, 128, {
          power: 1.0,
          mel_filters: identityMelFilters(numBins),
        });
        const spec2 = await spectrogram(waveform, paddedWin, 512, 128, {
          power: 2.0,
          mel_filters: identityMelFilters(numBins),
        });

        // Check a slice: power=2 values should equal power=1 values squared
        const slice1 = Array.from(spec1.data.slice(0, 20));
        const slice2 = Array.from(spec2.data.slice(0, 20));
        const expectedSquared = slice1.map((v) => v * v);
        expect(slice2).toBeCloseToNested(expectedSquared, 4);
      });
    });
  });
});
