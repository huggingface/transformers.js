import { PreTrainedTokenizer, GraniteMoeHybridForCausalLM } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  describe("GraniteMoeHybridForCausalLM", () => {
    const model_id = "onnx-internal-testing/tiny-random-GraniteMoeHybridForCausalLM";
    /** @type {GraniteMoeHybridForCausalLM} */
    let model;
    /** @type {PreTrainedTokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await GraniteMoeHybridForCausalLM.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      tokenizer = await PreTrainedTokenizer.from_pretrained(model_id);
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
        expect(outputs.tolist()).toEqual([[15339n, 70197n, 8290n, 70197n, 8290n, 70197n, 8290n, 70197n, 8290n, 70197n]]);
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
          [100256n, 15339n, 70197n, 8290n, 70197n, 8290n, 70197n, 8290n, 70197n, 8290n],
          [15339n, 1917n, 88135n, 6324n, 88135n, 6324n, 88135n, 6324n, 88135n, 6324n],
        ]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
