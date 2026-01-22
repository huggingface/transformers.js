import { RawAudio } from "../../src/utils/audio.js";

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
});
