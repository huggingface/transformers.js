import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

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
