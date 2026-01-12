import { LlamaTokenizer, MistralForCausalLM } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  describe("MistralForCausalLM", () => {
    const model_id = "hf-internal-testing/tiny-random-MistralForCausalLM";
    /** @type {MistralForCausalLM} */
    let model;
    /** @type {LlamaTokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await MistralForCausalLM.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      tokenizer = await LlamaTokenizer.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "batch_size=1",
      async () => {
        const inputs = tokenizer("hello");
        const outputs = await model.generate({
          ...inputs,
          max_length: 10,
        });
        expect(outputs.tolist()).toEqual([[1n, 6312n, 28709n, 24704n, 8732n, 1310n, 9808n, 13771n, 27309n, 4779n]]);
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
          [2n, 1n, 6312n, 28709n, 24704n, 8732n, 1310n, 9808n, 13771n, 27309n],
          [1n, 6312n, 28709n, 1526n, 8687n, 5690n, 1770n, 30811n, 12501n, 3325n],
        ]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
