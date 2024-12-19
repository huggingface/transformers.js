import { PreTrainedTokenizer, ModernBertModel, ModernBertForMaskedLM, ModernBertForSequenceClassification, ModernBertForTokenClassification } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  describe("ModernBertModel", () => {
    const model_id = "hf-internal-testing/tiny-random-ModernBertModel";

    /** @type {ModernBertModel} */
    let model;
    /** @type {PreTrainedTokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await ModernBertModel.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      tokenizer = await PreTrainedTokenizer.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "batch_size=1",
      async () => {
        const inputs = tokenizer("hello");
        const { last_hidden_state } = await model(inputs);
        expect(last_hidden_state.dims).toEqual([1, 3, 32]);
        expect(last_hidden_state.mean().item()).toBeCloseTo(-0.08922556787729263, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "batch_size>1",
      async () => {
        const inputs = tokenizer(["hello", "hello world"], { padding: true });
        const { last_hidden_state } = await model(inputs);
        expect(last_hidden_state.dims).toEqual([2, 4, 32]);
        expect(last_hidden_state.mean().item()).toBeCloseTo(0.048988230526447296, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });

  describe("ModernBertForMaskedLM", () => {
    const model_id = "hf-internal-testing/tiny-random-ModernBertForMaskedLM";

    const texts = ["The goal of life is [MASK].", "Paris is the [MASK] of France."];

    /** @type {ModernBertForMaskedLM} */
    let model;
    /** @type {PreTrainedTokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await ModernBertForMaskedLM.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      tokenizer = await PreTrainedTokenizer.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "batch_size=1",
      async () => {
        const inputs = tokenizer(texts[0]);
        const { logits } = await model(inputs);
        expect(logits.dims).toEqual([1, 9, 50368]);
        expect(logits.mean().item()).toBeCloseTo(0.0053214821964502335, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "batch_size>1",
      async () => {
        const inputs = tokenizer(texts, { padding: true });
        const { logits } = await model(inputs);
        expect(logits.dims).toEqual([2, 9, 50368]);
        expect(logits.mean().item()).toBeCloseTo(0.009154772385954857, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });

  describe("ModernBertForSequenceClassification", () => {
    const model_id = "hf-internal-testing/tiny-random-ModernBertForSequenceClassification";

    /** @type {ModernBertForSequenceClassification} */
    let model;
    /** @type {PreTrainedTokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await ModernBertForSequenceClassification.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      tokenizer = await PreTrainedTokenizer.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "batch_size=1",
      async () => {
        const inputs = tokenizer("hello");
        const { logits } = await model(inputs);
        const target = [[-0.7050137519836426, 2.343430519104004]];
        expect(logits.dims).toEqual([1, 2]);
        expect(logits.tolist()).toBeCloseToNested(target, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "batch_size>1",
      async () => {
        const inputs = tokenizer(["hello", "hello world"], { padding: true });
        const { logits } = await model(inputs);
        const target = [
          [-0.7050137519836426, 2.343430519104004],
          [-2.6860175132751465, 3.993380546569824],
        ];
        expect(logits.dims).toEqual([2, 2]);
        expect(logits.tolist()).toBeCloseToNested(target, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });

  describe("ModernBertForTokenClassification", () => {
    const model_id = "hf-internal-testing/tiny-random-ModernBertForTokenClassification";

    /** @type {ModernBertForTokenClassification} */
    let model;
    /** @type {PreTrainedTokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await ModernBertForTokenClassification.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      tokenizer = await PreTrainedTokenizer.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "batch_size=1",
      async () => {
        const inputs = tokenizer("hello");
        const { logits } = await model(inputs);
        expect(logits.dims).toEqual([1, 3, 2]);
        expect(logits.mean().item()).toBeCloseTo(1.0337047576904297, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "batch_size>1",
      async () => {
        const inputs = tokenizer(["hello", "hello world"], { padding: true });
        const { logits } = await model(inputs);
        expect(logits.dims).toEqual([2, 4, 2]);
        expect(logits.mean().item()).toBeCloseTo(-1.3397092819213867, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
