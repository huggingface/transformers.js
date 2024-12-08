import { GemmaTokenizer, GemmaForCausalLM } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  describe("GemmaForCausalLM", () => {
    const model_id = "Xenova/tiny-random-GemmaForCausalLM";
    /** @type {GemmaForCausalLM} */
    let model;
    /** @type {GemmaTokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await GemmaForCausalLM.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      tokenizer = await GemmaTokenizer.from_pretrained(model_id);
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
        expect(outputs.tolist()).toEqual([[2n, 17534n, 254059n, 254059n, 254059n, 254059n, 254059n, 254059n, 254059n, 254059n]]);
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
          [0n, 2n, 17534n, 254059n, 254059n, 254059n, 254059n, 254059n, 254059n, 254059n],
          [2n, 17534n, 2134n, 71055n, 71055n, 71055n, 71055n, 71055n, 71055n, 71055n],
        ]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
