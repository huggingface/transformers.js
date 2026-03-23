import { PreTrainedTokenizer, FalconH1ForCausalLM } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  describe("FalconH1ForCausalLM", () => {
    const model_id = "onnx-internal-testing/tiny-random-FalconH1ForCausalLM";
    /** @type {FalconH1ForCausalLM} */
    let model;
    /** @type {PreTrainedTokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await FalconH1ForCausalLM.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
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
        expect(outputs.tolist()).toEqual([[30939n, 175290n, 186817n, 203743n, 186817n, 203743n, 186817n, 203743n, 186817n, 203743n]]);
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
          [0n, 30939n, 175290n, 186817n, 203743n, 186817n, 203743n, 186817n, 203743n, 186817n],
          [30939n, 3870n, 194940n, 195605n, 240105n, 240105n, 240105n, 240105n, 240105n, 240105n],
        ]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
