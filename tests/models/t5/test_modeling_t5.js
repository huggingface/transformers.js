import { T5Tokenizer, T5Model, T5ForConditionalGeneration } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  describe("T5Model", () => {
    const model_id = "hf-internal-testing/tiny-random-T5Model";

    /** @type {T5Model} */
    let model;
    /** @type {T5Tokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await T5Model.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      tokenizer = await T5Tokenizer.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "forward",
      async () => {
        // Example adapted from https://huggingface.co/google-t5/t5-small#how-to-get-started-with-the-model
        const inputs = tokenizer("Studies have been shown that owning a dog is good for you");
        const { input_ids: decoder_input_ids } = tokenizer("Studies show that");

        const { last_hidden_state } = await model({ ...inputs, decoder_input_ids });
        expect(last_hidden_state.dims).toEqual([1, 4, 32]);
        expect(last_hidden_state.mean().item()).toBeCloseTo(7.492632721550763e-5, 8);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
  describe("T5ForConditionalGeneration", () => {
    const model_id = "hf-internal-testing/tiny-random-T5ForConditionalGeneration";

    /** @type {T5ForConditionalGeneration} */
    let model;
    /** @type {T5Tokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await T5ForConditionalGeneration.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      tokenizer = await T5Tokenizer.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "forward",
      async () => {
        // Example adapted from https://huggingface.co/google-t5/t5-small#how-to-get-started-with-the-model
        const inputs = tokenizer("Studies have been shown that owning a dog is good for you");
        const { input_ids: decoder_input_ids } = tokenizer("Studies show that");

        const model = await T5ForConditionalGeneration.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
        const outputs = await model({ ...inputs, decoder_input_ids });
        expect(outputs.logits.dims).toEqual([1, 4, 32100]);
        expect(outputs.logits.mean().item()).toBeCloseTo(8.867568901393952e-9, 12);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "batch_size=1",
      async () => {
        const inputs = tokenizer("hello");
        const outputs = await model.generate({
          ...inputs,
          max_length: 10,
        });
        expect(outputs.tolist()).toEqual([[0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n]]);
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
          [0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n],
          [0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n],
        ]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
