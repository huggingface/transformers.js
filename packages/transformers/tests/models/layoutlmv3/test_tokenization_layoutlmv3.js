import { LayoutLMv3Tokenizer } from "../../../src/models/tokenizers.js";
import { MAX_TOKENIZER_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export const TOKENIZER_CLASS = LayoutLMv3Tokenizer;
export const TEST_CONFIG = {};

const model_id = "onnx-internal-testing/tiny-random-LayoutLMv3ForTokenClassification-ONNX";
const WORDS = ["Hello", "world"];
const BOXES = [
  [100, 100, 200, 200],
  [300, 300, 400, 400],
];

export const CUSTOM_TESTS = () => {
  describe("word-level layout inputs", () => {
    /** @type {LayoutLMv3Tokenizer} */
    let tokenizer;
    beforeAll(async () => {
      tokenizer = await LayoutLMv3Tokenizer.from_pretrained(model_id);
    }, MAX_TOKENIZER_LOAD_TIME);

    it(
      "encodes words and boxes",
      () => {
        const output = tokenizer(WORDS, { boxes: BOXES, return_tensor: false });
        expect(output.input_ids).toEqual([0, 349, 483, 83, 273, 817, 2]);
        expect(output.attention_mask).toEqual([1, 1, 1, 1, 1, 1, 1]);
        expect(output.bbox).toEqual([
          [0, 0, 0, 0],
          [100, 100, 200, 200],
          [100, 100, 200, 200],
          [100, 100, 200, 200],
          [300, 300, 400, 400],
          [300, 300, 400, 400],
          [0, 0, 0, 0],
        ]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "returns tensors with a batch dimension",
      () => {
        const output = tokenizer(WORDS, { boxes: BOXES });
        expect(output.input_ids.dims).toEqual([1, 7]);
        expect(output.attention_mask.dims).toEqual([1, 7]);
        expect(output.bbox.dims).toEqual([1, 7, 4]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "creates word labels for the first subword only",
      () => {
        const output = tokenizer(WORDS, {
          boxes: BOXES,
          word_labels: [1, 2],
          return_tensor: false,
        });
        expect(output.labels).toEqual([-100, 1, -100, -100, 2, -100, -100]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "pads batches with the padding box and label",
      () => {
        const output = tokenizer([WORDS, ["hello"]], {
          boxes: [BOXES, [[0, 0, 100, 100]]],
          word_labels: [[1, 2], [3]],
          padding: true,
          return_tensor: false,
        });
        expect(output.input_ids).toEqual([
          [0, 349, 483, 83, 273, 817, 2],
          [0, 398, 867, 83, 2, 1, 1],
        ]);
        expect(output.bbox[1]).toEqual([
          [0, 0, 0, 0],
          [0, 0, 100, 100],
          [0, 0, 100, 100],
          [0, 0, 100, 100],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
        ]);
        expect(output.labels[1]).toEqual([-100, 3, -100, -100, -100, -100, -100]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "encodes a first sequence without boxes",
      () => {
        const output = tokenizer("What is it?", {
          text_pair: WORDS,
          boxes: BOXES,
          return_tensor: false,
        });
        expect(output.input_ids).toEqual([0, 379, 76, 275, 368, 389, 35, 2, 2, 349, 483, 83, 273, 817, 2]);
        expect(output.bbox.slice(0, 9)).toEqual(new Array(9).fill([0, 0, 0, 0]));
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "truncates word tokens and keeps special tokens",
      () => {
        const output = tokenizer(WORDS, {
          boxes: BOXES,
          truncation: true,
          max_length: 4,
          return_tensor: false,
        });
        expect(output.input_ids).toEqual([0, 349, 483, 2]);
        expect(output.bbox).toEqual([
          [0, 0, 0, 0],
          [100, 100, 200, 200],
          [100, 100, 200, 200],
          [0, 0, 0, 0],
        ]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it("throws when boxes are missing", () => {
      expect(() => tokenizer(WORDS)).toThrow("You must provide corresponding bounding boxes.");
    });
  });
};
