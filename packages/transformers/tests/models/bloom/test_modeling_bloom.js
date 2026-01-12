import { BloomTokenizer, BloomForCausalLM } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  describe("BloomForCausalLM", () => {
    const model_id = "hf-internal-testing/tiny-random-BloomForCausalLM";
    /** @type {BloomForCausalLM} */
    let model;
    /** @type {BloomTokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await BloomForCausalLM.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      tokenizer = await BloomTokenizer.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "batch_size=1",
      async () => {
        const inputs = tokenizer("hello");
        const outputs = await model.generate({
          ...inputs,
          max_length: 10,
        });
        expect(outputs.tolist()).toEqual([[198n, 803n, 82n, 82n, 82n, 82n, 82n, 82n, 82n, 82n]]);
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
          [3n, 3n, 198n, 803n, 82n, 82n, 82n, 82n, 82n, 82n],
          [198n, 803n, 82n, 209n, 753n, 753n, 753n, 753n, 753n, 753n],
        ]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
