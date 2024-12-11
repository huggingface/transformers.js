import { GPT2Tokenizer, GPTNeoForCausalLM } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  describe("GPTNeoForCausalLM", () => {
    const model_id = "hf-internal-testing/tiny-random-GPTNeoForCausalLM";
    /** @type {GPTNeoForCausalLM} */
    let model;
    /** @type {GPT2Tokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await GPTNeoForCausalLM.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      tokenizer = await GPT2Tokenizer.from_pretrained(model_id);
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
        expect(outputs.tolist()).toEqual([[258n, 863n, 79n, 79n, 79n, 949n, 949n, 949n, 949n, 949n]]);
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
          [0n, 0n, 258n, 863n, 79n, 79n, 79n, 949n, 949n, 949n],
          [258n, 863n, 79n, 269n, 813n, 849n, 849n, 849n, 849n, 849n],
        ]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
