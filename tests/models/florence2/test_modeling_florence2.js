import { Florence2Processor, Florence2ForConditionalGeneration, RawImage, full } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  const texts = ["Describe with a paragraph what is shown in the image.", "Locate the objects with category name in the image."];

  // Empty white image
  const dims = [224, 224, 3];
  const image = new RawImage(new Uint8ClampedArray(dims[0] * dims[1] * dims[2]).fill(255), ...dims);

  describe("Florence2ForConditionalGeneration", () => {
    const model_id = "Xenova/tiny-random-Florence2ForConditionalGeneration";

    /** @type {Florence2ForConditionalGeneration} */
    let model;
    /** @type {Florence2Processor} */
    let processor;
    beforeAll(async () => {
      model = await Florence2ForConditionalGeneration.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      processor = await Florence2Processor.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "forward",
      async () => {
        const inputs = await processor(image, texts[0]);

        const { logits } = await model({
          ...inputs,
          decoder_input_ids: full([1, 1], 2n),
        });
        expect(logits.dims).toEqual([1, 1, 51289]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "batch_size=1",
      async () => {
        {
          const text_inputs = processor.tokenizer(texts[0]);
          const generate_ids = await model.generate({ ...text_inputs, max_new_tokens: 10 });
          expect(generate_ids.tolist()).toEqual([[2n, 0n, 0n, 0n, 1n, 0n, 0n, 2n]]);
        }
        {
          const inputs = await processor(image, texts[0]);
          const generate_ids = await model.generate({ ...inputs, max_new_tokens: 10 });
          expect(generate_ids.tolist()).toEqual([[2n, 0n, 48n, 48n, 48n, 48n, 48n, 48n, 48n, 48n, 2n]]);
        }
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "batch_size>1",
      async () => {
        {
          const text_inputs = processor.tokenizer(texts, { padding: true });
          const generate_ids = await model.generate({ ...text_inputs, max_new_tokens: 10 });
          expect(generate_ids.tolist()).toEqual([
            [2n, 0n, 0n, 0n, 1n, 0n, 0n, 2n],
            [2n, 0n, 0n, 0n, 1n, 0n, 0n, 2n],
          ]);
        }
        {
          const inputs = await processor([image, image], texts, { padding: true });

          const generate_ids = await model.generate({ ...inputs, max_new_tokens: 10 });
          expect(generate_ids.tolist()).toEqual([
            [2n, 0n, 48n, 48n, 48n, 48n, 48n, 48n, 48n, 48n, 2n],
            [2n, 0n, 48n, 48n, 48n, 48n, 48n, 48n, 48n, 48n, 2n],
          ]);
        }
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
