import { LayoutLMv3Model, LayoutLMv3ForSequenceClassification, LayoutLMv3ForTokenClassification, LayoutLMv3Tokenizer, Tensor } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

const WORDS = ["Hello", "world"];
const BOXES = [
  [100, 100, 200, 200],
  [300, 300, 400, 400],
];

/**
 * Create zero-valued pixel values matching the tiny LayoutLMv3 models.
 * @returns {Tensor} The pixel values tensor.
 */
function createPixelValues() {
  return new Tensor("float32", new Float32Array(3 * 4 * 4), [1, 3, 4, 4]);
}

export default () => {
  describe("LayoutLMv3Model", () => {
    const model_id = "onnx-internal-testing/tiny-random-LayoutLMv3Model-ONNX";

    /** @type {LayoutLMv3Model} */
    let model;
    /** @type {LayoutLMv3Tokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await LayoutLMv3Model.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      tokenizer = await LayoutLMv3Tokenizer.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "batch_size=1",
      async () => {
        const inputs = tokenizer(WORDS, { boxes: BOXES });
        const { last_hidden_state } = await model({ ...inputs, pixel_values: createPixelValues() });
        expect(last_hidden_state.dims).toEqual([1, 12, 36]);
        expect(last_hidden_state.mean().item()).toBeCloseTo(3.999081954475514e-9, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });

  describe("LayoutLMv3ForTokenClassification", () => {
    const model_id = "onnx-internal-testing/tiny-random-LayoutLMv3ForTokenClassification-ONNX";

    /** @type {LayoutLMv3ForTokenClassification} */
    let model;
    /** @type {LayoutLMv3Tokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await LayoutLMv3ForTokenClassification.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      tokenizer = await LayoutLMv3Tokenizer.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "batch_size=1",
      async () => {
        const inputs = tokenizer(WORDS, { boxes: BOXES });
        const { logits } = await model({ ...inputs, pixel_values: createPixelValues() });
        expect(logits.dims).toEqual([1, 7, 2]);
        expect(logits.mean().item()).toBeCloseTo(0.017555471509695053, 5);
        expect(logits.tolist()[0][0]).toBeCloseToNested([-0.012317213229835033, 0.01551758497953415], 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "batch_size>1",
      async () => {
        const inputs = tokenizer([WORDS, ["hello"]], {
          boxes: [BOXES, [[0, 0, 100, 100]]],
          padding: true,
        });
        const { logits } = await model({
          ...inputs,
          pixel_values: new Tensor("float32", new Float32Array(2 * 3 * 4 * 4), [2, 3, 4, 4]),
        });
        expect(logits.dims).toEqual([2, 7, 2]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });

  describe("LayoutLMv3ForSequenceClassification", () => {
    const model_id = "onnx-internal-testing/tiny-random-LayoutLMv3ForSequenceClassification-ONNX";

    /** @type {LayoutLMv3ForSequenceClassification} */
    let model;
    /** @type {LayoutLMv3Tokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await LayoutLMv3ForSequenceClassification.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      tokenizer = await LayoutLMv3Tokenizer.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "batch_size=1",
      async () => {
        const inputs = tokenizer(WORDS, { boxes: BOXES });
        const { logits } = await model({ ...inputs, pixel_values: createPixelValues() });
        expect(logits.dims).toEqual([1, 2]);
        expect(logits.mean().item()).toBeCloseTo(0.016887376084923744, 5);
        expect(logits.tolist()).toBeCloseToNested([[0.004811071790754795, 0.028963681310415268]], 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
