import { EuroBertModel, EuroBertForMaskedLM, EuroBertForSequenceClassification, EuroBertForTokenClassification, AutoTokenizer } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  describe("EuroBertModel", () => {
    const model_id = "onnx-internal-testing/tiny-random-EuroBertModel";

    /** @type {EuroBertModel} */
    let model;
    /** @type {import("../../../src/transformers.js").PreTrainedTokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await EuroBertModel.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      tokenizer = await AutoTokenizer.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "batch_size=1",
      async () => {
        const inputs = tokenizer("hello");
        const { last_hidden_state } = await model(inputs);
        expect(last_hidden_state.dims).toEqual([1, 2, 32]);
        expect(last_hidden_state.mean().item()).toBeCloseTo(-0.1891579031944275, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "batch_size>1",
      async () => {
        const inputs = tokenizer(["hello", "hello world"], { padding: true });
        const { last_hidden_state } = await model(inputs);
        expect(last_hidden_state.dims).toEqual([2, 3, 32]);
        expect(last_hidden_state.mean().item()).toBeCloseTo(-0.13678674399852753, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
