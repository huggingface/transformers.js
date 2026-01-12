import { AutoProcessor, Phi3VProcessor } from "../../../src/transformers.js";

import { load_cached_image } from "../../asset_cache.js";
import { MAX_PROCESSOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  const model_id = "onnx-community/Phi-3.5-vision-instruct";

  describe("Phi3VProcessor", () => {
    /** @type {Phi3VProcessor} */
    let processor;
    let images = {};

    beforeAll(async () => {
      processor = await AutoProcessor.from_pretrained(model_id, {
        // Use legacy to match python version
        legacy: true,
      });
      images = {
        white_image: await load_cached_image("white_image"),
      };
    }, MAX_PROCESSOR_LOAD_TIME);

    const create_prompt = (text, images = []) => {
      const placeholder = images.map((_, i) => `<|image_${i + 1}|>\n`).join("");
      const messages = [{ role: "user", content: placeholder + text }];
      const prompt = processor.tokenizer.apply_chat_template(messages, { tokenize: false, add_generation_prompt: true });
      return prompt;
    };

    it(
      "Text-only",
      async () => {
        const prompt = create_prompt("Hi there.");
        const { input_ids, pixel_values } = await processor(prompt);
        expect(input_ids.dims).toEqual([1, 11]);
        expect(pixel_values).toBeUndefined();
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "Single image & text",
      async () => {
        const imgs = [images.white_image];
        const prompt = create_prompt("Describe this image.", imgs);
        const { input_ids, attention_mask, pixel_values, image_sizes } = await processor(prompt, imgs);
        expect(input_ids.dims).toEqual([1, /* 773 */ 770]);
        expect(attention_mask.dims).toEqual(input_ids.dims);
        expect(pixel_values.dims).toEqual([1, 5, 3, 336, 336]);
        expect(image_sizes.tolist()).toEqual([[672n, 672n]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "Single image (num_crops=16) & text",
      async () => {
        const imgs = [images.white_image];
        const prompt = create_prompt("Describe this image.", imgs);
        const { input_ids, attention_mask, pixel_values, image_sizes } = await processor(prompt, imgs, { num_crops: 16 });
        expect(input_ids.dims).toEqual([1, /* 2525 */ 2522]);
        expect(attention_mask.dims).toEqual(input_ids.dims);
        expect(pixel_values.dims).toEqual([1, 17, 3, 336, 336]);
        expect(image_sizes.tolist()).toEqual([[1344n, 1344n]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "Multiple images & text",
      async () => {
        const imgs = [images.white_image, images.white_image];
        const prompt = create_prompt("Describe these images.", imgs);
        const { input_ids, attention_mask, pixel_values, image_sizes } = await processor(prompt, imgs);
        expect(input_ids.dims).toEqual([1, /* 1533 */ 1527]);
        expect(attention_mask.dims).toEqual(input_ids.dims);
        expect(pixel_values.dims).toEqual([2, 5, 3, 336, 336]);
        expect(image_sizes.tolist()).toEqual([
          [672n, 672n],
          [672n, 672n],
        ]);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
