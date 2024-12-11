import { GemmaTokenizer, Gemma2ForCausalLM } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  describe("Gemma2ForCausalLM", () => {
    const model_id = "hf-internal-testing/tiny-random-Gemma2ForCausalLM";
    /** @type {Gemma2ForCausalLM} */
    let model;
    /** @type {GemmaTokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await Gemma2ForCausalLM.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
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
        expect(outputs.tolist()).toEqual([[2n, 17534n, 127534n, 160055n, 160055n, 160055n, 160055n, 160055n, 160055n, 160055n]]);
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
          [0n, 2n, 17534n, 127534n, 127534n, 215341n, 215341n, 215341n, 215341n, 215341n],
          [2n, 17534n, 2134n, 107508n, 160055n, 160055n, 160055n, 160055n, 160055n, 160055n],
        ]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
