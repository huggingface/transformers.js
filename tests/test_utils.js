export async function loadAudio(url) {
  // NOTE: Since the Web Audio API is not available in Node.js, we will need to use the `wavefile` library to obtain the raw audio data.
  // For more information, see: https://huggingface.co/docs/transformers.js/guides/node-audio-processing
  let wavefile = (await import("wavefile")).default;

  // Load audio data
  let buffer = Buffer.from(await fetch(url).then((x) => x.arrayBuffer()));

  // Read .wav file and convert it to required format
  let wav = new wavefile.WaveFile(buffer);
  wav.toBitDepth("32f"); // Pipeline expects input as a Float32Array
  wav.toSampleRate(16000); // Whisper expects audio with a sampling rate of 16000
  let audioData = wav.getSamples();
  if (Array.isArray(audioData)) {
    if (audioData.length > 1) {
      const SCALING_FACTOR = Math.sqrt(2);

      // Merge channels (into first channel to save memory)
      for (let i = 0; i < audioData[0].length; ++i) {
        audioData[0][i] = (SCALING_FACTOR * (audioData[0][i] + audioData[1][i])) / 2;
      }
    }

    // Select first channel
    audioData = audioData[0];
  }
  return audioData;
}
/**
 * Deep equality test (for arrays and objects) with tolerance for floating point numbers
 * @param {any} val1 The first item
 * @param {any} val2 The second item
 * @param {number} tol Tolerance for floating point numbers
 */
export function compare(val1, val2, tol = 0.1) {
  if (val1 !== null && val2 !== null && typeof val1 === "object" && typeof val2 === "object") {
    // Both are non-null objects

    if (Array.isArray(val1) && Array.isArray(val2)) {
      expect(val1).toHaveLength(val2.length);

      for (let i = 0; i < val1.length; ++i) {
        compare(val1[i], val2[i], tol);
      }
    } else {
      expect(Object.keys(val1)).toHaveLength(Object.keys(val2).length);

      for (let key in val1) {
        compare(val1[key], val2[key], tol);
      }
    }
  } else {
    // At least one of them is not an object
    // First check that both have the same type
    expect(typeof val1).toEqual(typeof val2);

    if (typeof val1 === "number" && (!Number.isInteger(val1) || !Number.isInteger(val2))) {
      // If both are numbers and at least one of them is not an integer
      expect(val1).toBeCloseTo(val2, -Math.log10(tol));
    } else {
      // Perform equality test
      expect(val1).toEqual(val2);
    }
  }
}
