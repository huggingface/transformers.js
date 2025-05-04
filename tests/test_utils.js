import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

import { distance } from "fastest-levenshtein";

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

/**
 * Compare two strings adding some tolerance for variation between model outputs.
 * 
 * Similarity score is computing using Levenshtein distance (n_diff) between the two strings, as a fraction of the first string's length:
 *   similarity score = 1 - n_diff / str1.length.
 *
 * @param {string} str1 The first string
 * @param {string} str2 The second string
 * @param {number} tol Tolerance score for similarity between strings, from -Infinity to 1.0 (100% match).
 */
export function compareString(str1, str2, tol = 0.9) {
  const dist = distance(str1, str2);
  const score = 1 - dist / (str1.length ?? 1);
  expect(score).toBeGreaterThanOrEqual(tol);
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const models_dir = path.join(__dirname, "models");
const pipelines_dir = path.join(__dirname, "pipelines");

/**
 * Helper function to collect all unit tests, which can be found in files
 * of the form: `tests/models/<model_type>/test_<filename>_<model_type>.js`.
 * @param {string} filename
 * @returns {Promise<[string, Function][]>}
 */
export async function collect_tests(filename) {
  const model_types = fs.readdirSync(models_dir);
  const all_tests = [];
  for (const model_type of model_types) {
    const dir = path.join(models_dir, model_type);

    if (!fs.existsSync(dir) || !fs.statSync(dir).isDirectory()) {
      continue;
    }

    const file = path.join(dir, `test_${filename}_${model_type}.js`);
    if (!fs.existsSync(file)) {
      continue;
    }

    const items = await import(file);
    all_tests.push([model_type, items]);
  }
  return all_tests;
}

/**
 * Helper function to collect and execute all unit tests, which can be found in files
 * of the form: `tests/models/<model_type>/test_<filename>_<model_type>.js`.
 * @param {string} title The title of the test
 * @param {string} filename The name of the test
 */
export async function collect_and_execute_tests(title, filename) {
  // 1. Collect all tests
  const all_tests = await collect_tests(filename);

  // 2. Execute tests
  describe(title, () => all_tests.forEach(([name, test]) => describe(name, test.default)));
}

/**
 * Helper function to collect all pipeline tests, which can be found in files
 * of the form: `tests/pipelines/test_pipeline_<pipeline_id>.js`.
 */
export async function collect_and_execute_pipeline_tests(title) {
  // 1. Collect all tests
  const all_tests = [];
  const pipeline_types = fs.readdirSync(pipelines_dir);
  for (const filename of pipeline_types) {
    const file = path.join(pipelines_dir, filename);
    const items = await import(file);
    all_tests.push(items);
  }

  // 2. Execute tests
  describe(title, () => all_tests.forEach((test) => test.default()));
}
