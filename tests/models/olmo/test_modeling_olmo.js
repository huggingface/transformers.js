import { GPTNeoXTokenizer, OlmoForCausalLM } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  describe("OlmoForCausalLM", () => {
    const model_id = "onnx-community/tiny-random-olmo-hf";
    /** @type {OlmoForCausalLM} */
    let model;
    /** @type {GPTNeoXTokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await OlmoForCausalLM.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
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
        expect(outputs.tolist()).toEqual([[25521n, 10886n, 44936n, 38777n, 33038n, 18557n, 1810n, 33853n, 9517n, 28892n]]);
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
          [1n, 25521n, 10886n, 44936n, 38777n, 33038n, 18557n, 1810n, 33853n, 9517n],
          [25521n, 1533n, 37199n, 27362n, 30594n, 39261n, 8824n, 19175n, 8545n, 29335n],
        ]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
