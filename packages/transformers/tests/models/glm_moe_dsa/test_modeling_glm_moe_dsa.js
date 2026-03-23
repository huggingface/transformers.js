import { PreTrainedTokenizer, GlmMoeDsaForCausalLM } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  describe("GlmMoeDsaForCausalLM", () => {
    const model_id = "onnx-internal-testing/tiny-random-GlmMoeDsaForCausalLM";
    /** @type {GlmMoeDsaForCausalLM} */
    let model;
    /** @type {PreTrainedTokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await GlmMoeDsaForCausalLM.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
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
        expect(outputs.tolist()).toEqual([[14978n, 147266n, 147266n, 147266n, 147266n, 147266n, 147266n, 147266n, 147266n, 147266n]]);
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
          [154820n, 14978n, 147266n, 147266n, 147266n, 147266n, 147266n, 147266n, 147266n, 147266n],
          [14978n, 1879n, 114263n, 57689n, 61053n, 61053n, 61053n, 61053n, 61053n, 61053n],
        ]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
