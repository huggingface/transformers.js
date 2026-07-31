import { AutoProcessor, Glm46VProcessor, RawImage } from "../../../src/transformers.js";

import { MAX_PROCESSOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  describe("Glm46VProcessor", () => {
    const model_id = "onnx-community/GLM-OCR-ONNX";

    /** @type {Glm46VProcessor} */
    let processor;
    let image;

    beforeAll(async () => {
      processor = await AutoProcessor.from_pretrained(model_id);
      // Create a test image of size 666 (height) x 540 (width)
      image = new RawImage(new Uint8ClampedArray(666 * 540 * 3), 540, 666, 3);
    }, MAX_PROCESSOR_LOAD_TIME);

    it(
      "Image and text",
      async () => {
        const conversation = [
          {
            role: "user",
            content: [{ type: "image" }, { type: "text", text: "Text Recognition:" }],
          },
        ];

        const text = processor.apply_chat_template(conversation, {
          add_generation_prompt: true,
        });
        const { input_ids, attention_mask, pixel_values, image_grid_thw } = await processor(text, image);

        expect(input_ids.dims).toEqual([1, 468]);
        expect(attention_mask.dims).toEqual([1, 468]);
        expect(pixel_values.dims).toEqual([1824, 1176]);
        expect(image_grid_thw.dims).toEqual([1, 3]);
        expect(image_grid_thw.tolist()).toEqual([[1n, 48n, 38n]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
