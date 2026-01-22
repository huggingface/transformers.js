import { CodeGenTokenizer, CodeGenForCausalLM } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  describe("CodeGenForCausalLM", () => {
    const model_id = "hf-internal-testing/tiny-random-CodeGenForCausalLM";
    /** @type {CodeGenForCausalLM} */
    let model;
    /** @type {CodeGenTokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await CodeGenForCausalLM.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      tokenizer = await CodeGenTokenizer.from_pretrained(model_id);
      tokenizer.padding_side = "left";
    }, MAX_MODEL_LOAD_TIME);

    it(
      "batch_size=1",
      async () => {
        const inputs = tokenizer("hello");
        const outputs = await model.generate({
          ...inputs,
          max_length: 10,
        });
        expect(outputs.tolist()).toEqual([[258n, 863n, 79n, 437n, 334n, 450n, 294n, 621n, 375n, 385n]]);
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
          [0n, 0n, 258n, 863n, 79n, 437n, 334n, 450n, 294n, 621n],
          [258n, 863n, 79n, 269n, 813n, 759n, 113n, 295n, 574n, 987n],
        ]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
