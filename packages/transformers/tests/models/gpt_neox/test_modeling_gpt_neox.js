import { GPTNeoXTokenizer, GPTNeoXForCausalLM } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  describe("GPTNeoXForCausalLM", () => {
    const model_id = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM";
    /** @type {GPTNeoXForCausalLM} */
    let model;
    /** @type {GPTNeoXTokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await GPTNeoXForCausalLM.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
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
        expect(outputs.tolist()).toEqual([[259n, 864n, 80n, 881n, 502n, 895n, 938n, 668n, 502n, 895n]]);
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
          [0n, 0n, 259n, 864n, 80n, 881n, 502n, 895n, 938n, 668n],
          [259n, 864n, 80n, 270n, 814n, 522n, 112n, 268n, 503n, 468n],
        ]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
