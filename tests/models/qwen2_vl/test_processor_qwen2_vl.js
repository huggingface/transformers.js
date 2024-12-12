import { AutoProcessor, Qwen2VLProcessor } from "../../../src/transformers.js";

import { load_cached_image } from "../../asset_cache.js";
import { MAX_PROCESSOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  describe("Qwen2VLProcessor", () => {
    const model_id = "hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration";

    /** @type {Qwen2VLProcessor} */
    let processor;
    let images = {};

    beforeAll(async () => {
      processor = await AutoProcessor.from_pretrained(model_id);
      images = {
        white_image: await load_cached_image("white_image"),
      };
    }, MAX_PROCESSOR_LOAD_TIME);

    it(
      "Image and text",
      async () => {
        const conversation = [
          {
            role: "user",
            content: [{ type: "image" }, { type: "text", text: "Describe this image." }],
          },
        ];

        const text = processor.apply_chat_template(conversation, {
          add_generation_prompt: true,
        });
        const { input_ids, attention_mask, pixel_values, image_grid_thw } = await processor(text, images.white_image);

        expect(input_ids.dims).toEqual([1, 89]);
        expect(attention_mask.dims).toEqual([1, 89]);
        expect(pixel_values.dims).toEqual([256, 1176]);
        expect(image_grid_thw.dims).toEqual([1, 3]);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
