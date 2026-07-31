import { PreTrainedTokenizer, NemotronHForCausalLM } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  describe("NemotronHForCausalLM", () => {
    const model_id = "onnx-internal-testing/tiny-random-NemotronHForCausalLM";
    /** @type {NemotronHForCausalLM} */
    let model;
    /** @type {PreTrainedTokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await NemotronHForCausalLM.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
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
        expect(outputs.tolist()).toEqual([[29706n, 97336n, 121288n, 82722n, 25950n, 99280n, 131027n, 20440n, 94437n, 28124n]]);
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
          [11n, 29706n, 97336n, 121288n, 82722n, 25950n, 99280n, 131027n, 20440n, 94437n],
          [29706n, 4304n, 128810n, 63172n, 70343n, 47180n, 52275n, 46732n, 66887n, 62566n],
        ]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
