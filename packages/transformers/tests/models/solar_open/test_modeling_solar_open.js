import { PreTrainedTokenizer, SolarOpenForCausalLM } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  describe("SolarOpenForCausalLM", () => {
    const model_id = "onnx-internal-testing/tiny-random-SolarOpenForCausalLM";
    /** @type {SolarOpenForCausalLM} */
    let model;
    /** @type {PreTrainedTokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await SolarOpenForCausalLM.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
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
        expect(outputs.tolist()).toEqual([[27090n, 134662n, 105184n, 152304n, 69736n, 67149n, 165462n, 142691n, 181896n, 67149n]]);
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
          [2n, 27090n, 134662n, 105184n, 152304n, 69736n, 67149n, 165462n, 142691n, 181896n],
          [27090n, 8358n, 188200n, 94175n, 188200n, 94175n, 195168n, 195168n, 195168n, 113354n],
        ]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
