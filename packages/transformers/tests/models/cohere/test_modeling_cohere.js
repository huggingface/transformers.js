import { CohereTokenizer, CohereModel, CohereForCausalLM } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  describe("CohereModel", () => {
    const model_id = "hf-internal-testing/tiny-random-CohereModel";
    /** @type {CohereModel} */
    let model;
    /** @type {CohereTokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await CohereModel.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      tokenizer = await CohereTokenizer.from_pretrained(model_id);
      tokenizer.padding_side = "left";
    }, MAX_MODEL_LOAD_TIME);

    it(
      "batch_size=1",
      async () => {
        const inputs = tokenizer("hello");
        const { last_hidden_state } = await model(inputs);
        expect(last_hidden_state.dims).toEqual([1, 4, 32]);
        expect(last_hidden_state.mean().item()).toBeCloseTo(0.0, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "batch_size>1",
      async () => {
        const inputs = tokenizer(["hello", "hello world"], { padding: true });
        const { last_hidden_state } = await model(inputs);
        expect(last_hidden_state.dims).toEqual([2, 6, 32]);
        expect(last_hidden_state.mean().item()).toBeCloseTo(9.934107758624577e-9, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });

  describe("CohereForCausalLM", () => {
    const model_id = "hf-internal-testing/tiny-random-CohereForCausalLM";
    /** @type {CohereForCausalLM} */
    let model;
    /** @type {CohereTokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await CohereForCausalLM.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      tokenizer = await CohereTokenizer.from_pretrained(model_id);
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
        expect(outputs.tolist()).toEqual([[5n, 203n, 790n, 87n, 87n, 87n, 87n, 87n, 87n, 87n]]);
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
          [0n, 0n, 5n, 203n, 790n, 87n, 87n, 87n, 87n, 87n],
          [5n, 203n, 790n, 87n, 214n, 741n, 741n, 741n, 741n, 741n],
        ]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
