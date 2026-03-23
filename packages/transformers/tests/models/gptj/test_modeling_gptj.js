import { GPTNeoXTokenizer, GPTJForCausalLM } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  describe("GPTJForCausalLM", () => {
    const model_id = "hf-internal-testing/tiny-random-GPTJForCausalLM";
    /** @type {GPTJForCausalLM} */
    let model;
    /** @type {GPTNeoXTokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await GPTJForCausalLM.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      tokenizer = await GPTNeoXTokenizer.from_pretrained(model_id);
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
        expect(outputs.tolist()).toEqual([[258n, 863n, 79n, 102n, 401n, 773n, 889n, 159n, 957n, 869n]]);
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
          [0n, 0n, 258n, 863n, 79n, 102n, 401n, 773n, 889n, 159n],
          [258n, 863n, 79n, 269n, 813n, 879n, 175n, 39n, 141n, 1000n],
        ]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
