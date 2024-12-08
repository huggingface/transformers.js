import { MarianTokenizer, MarianMTModel } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  describe("MarianMTModel", () => {
    const model_id = "onnx-community/tiny-random-MarianMTModel";

    /** @type {MarianMTModel} */
    let model;
    /** @type {MarianTokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await MarianMTModel.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      tokenizer = await MarianTokenizer.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "batch_size=1",
      async () => {
        const inputs = tokenizer("hello");
        const outputs = await model.generate({
          ...inputs,
          max_length: 10,
        });
        expect(outputs.tolist()).toEqual([[3n, 40672n, 8358n, 32810n, 32810n, 32810n, 32810n, 35687n, 33073n, 6870n]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "batch_size>1",
      async () => {
        const inputs = tokenizer(["hello", "hello world"], { padding: true });
        const outputs = await model.generate({
          ...inputs,
          max_length: 10,
        });
        expect(outputs.tolist()).toEqual([
          [3n, 40672n, 8358n, 32810n, 32810n, 32810n, 32810n, 35687n, 33073n, 6870n],
          [3n, 40672n, 8358n, 32810n, 32810n, 32810n, 32810n, 35687n, 33073n, 6870n],
        ]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
