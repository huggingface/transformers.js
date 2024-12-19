import { BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, BertForTokenClassification, BertForQuestionAnswering } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  describe("BertModel", () => {
    const model_id = "hf-internal-testing/tiny-random-BertModel";

    /** @type {BertModel} */
    let model;
    /** @type {BertTokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await BertModel.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      tokenizer = await BertTokenizer.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "batch_size=1",
      async () => {
        const inputs = tokenizer("hello");
        const { last_hidden_state } = await model(inputs);
        expect(last_hidden_state.dims).toEqual([1, 7, 32]);
        expect(last_hidden_state.mean().item()).toBeCloseTo(0.0, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "batch_size>1",
      async () => {
        const inputs = tokenizer(["hello", "hello world"], { padding: true });
        const { last_hidden_state } = await model(inputs);
        expect(last_hidden_state.dims).toEqual([2, 12, 32]);
        expect(last_hidden_state.mean().item()).toBeCloseTo(1.4901161193847656e-8, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });

  describe("BertForMaskedLM", () => {
    const model_id = "hf-internal-testing/tiny-random-BertForMaskedLM";

    const texts = ["The goal of life is [MASK].", "Paris is the [MASK] of France."];

    /** @type {BertForMaskedLM} */
    let model;
    /** @type {BertTokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await BertForMaskedLM.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      tokenizer = await BertTokenizer.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "batch_size=1",
      async () => {
        const inputs = tokenizer(texts[0]);
        const { logits } = await model(inputs);
        expect(logits.dims).toEqual([1, 19, 1124]);
        expect(logits.mean().item()).toBeCloseTo(0.0016587056452408433, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "batch_size>1",
      async () => {
        const inputs = tokenizer(texts, { padding: true });
        const { logits } = await model(inputs);
        expect(logits.dims).toEqual([2, 22, 1124]);
        expect(logits.mean().item()).toBeCloseTo(0.0017160633578896523, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });

  describe("BertForSequenceClassification", () => {
    const model_id = "hf-internal-testing/tiny-random-BertForSequenceClassification";

    /** @type {BertForSequenceClassification} */
    let model;
    /** @type {BertTokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await BertForSequenceClassification.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      tokenizer = await BertTokenizer.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "batch_size=1",
      async () => {
        const inputs = tokenizer("hello");
        const { logits } = await model(inputs);
        const target = [[0.00043986947275698185, -0.030218850821256638]];
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
          [0.00043986947275698185, -0.030218850821256638],
          [0.0003853091038763523, -0.03022204339504242],
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

  describe("BertForTokenClassification", () => {
    const model_id = "hf-internal-testing/tiny-random-BertForTokenClassification";

    /** @type {BertForTokenClassification} */
    let model;
    /** @type {BertTokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await BertForTokenClassification.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      tokenizer = await BertTokenizer.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "batch_size=1",
      async () => {
        const inputs = tokenizer("hello");
        const { logits } = await model(inputs);
        expect(logits.dims).toEqual([1, 7, 2]);
        expect(logits.mean().item()).toBeCloseTo(0.07089076191186905, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "batch_size>1",
      async () => {
        const inputs = tokenizer(["hello", "hello world"], { padding: true });
        const { logits } = await model(inputs);
        expect(logits.dims).toEqual([2, 12, 2]);
        expect(logits.mean().item()).toBeCloseTo(0.04702216014266014, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });

  describe("BertForQuestionAnswering", () => {
    const model_id = "hf-internal-testing/tiny-random-BertForQuestionAnswering";

    /** @type {BertForQuestionAnswering} */
    let model;
    /** @type {BertTokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await BertForQuestionAnswering.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      tokenizer = await BertTokenizer.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "batch_size=1",
      async () => {
        const inputs = tokenizer("hello");
        const { start_logits, end_logits } = await model(inputs);
        expect(start_logits.dims).toEqual([1, 7]);
        expect(start_logits.mean().item()).toBeCloseTo(0.12772157788276672, 5);
        expect(end_logits.dims).toEqual([1, 7]);
        expect(end_logits.mean().item()).toBeCloseTo(0.11811424791812897, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "batch_size>1",
      async () => {
        const inputs = tokenizer(["hello", "hello world"], { padding: true });
        const { start_logits, end_logits } = await model(inputs);
        expect(start_logits.dims).toEqual([2, 12]);
        expect(start_logits.mean().item()).toBeCloseTo(0.12843115627765656, 5);
        expect(end_logits.dims).toEqual([2, 12]);
        expect(end_logits.mean().item()).toBeCloseTo(0.11745202541351318, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
