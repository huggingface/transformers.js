import { PreTrainedTokenizer, Lfm2ForCausalLM } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  describe("Lfm2ForCausalLM", () => {
    const model_id = "onnx-internal-testing/tiny-random-Lfm2ForCausalLM";
    /** @type {Lfm2ForCausalLM} */
    let model;
    /** @type {PreTrainedTokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await Lfm2ForCausalLM.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
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
        expect(outputs.tolist()).toEqual([[1n, 52572n, 38892n, 6902n, 53329n, 33092n, 13656n, 49822n, 6902n, 52520n]]);
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
          [0n, 1n, 52572n, 60239n, 57205n, 6790n, 58292n, 30935n, 5959n, 6902n],
          [1n, 52572n, 2031n, 59572n, 43345n, 42427n, 31142n, 41100n, 5321n, 5816n],
        ]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
