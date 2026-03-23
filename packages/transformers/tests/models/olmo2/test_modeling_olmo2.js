import { GPT2Tokenizer, Olmo2ForCausalLM } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  describe("Olmo2ForCausalLM", () => {
    const model_id = "hf-internal-testing/tiny-random-Olmo2ForCausalLM";
    /** @type {Olmo2ForCausalLM} */
    let model;
    /** @type {GPT2Tokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await Olmo2ForCausalLM.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
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
        expect(outputs.tolist()).toEqual([[15339n, 50957n, 43410n, 77030n, 91444n, 99516n, 80720n, 4608n, 90428n, 22806n]]);
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
          [100277n, 15339n, 50957n, 43410n, 77030n, 91444n, 99516n, 80720n, 4608n, 90428n],
          [15339n, 1917n, 12095n, 21350n, 61586n, 19306n, 39486n, 91527n, 59768n, 31934n],
        ]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
