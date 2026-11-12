import { AutoProcessor, Qwen2VLProcessor, RawImage } from "../../../src/transformers.js";

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

    it(
      "Multiple images and text",
      async () => {
        const img1 = images.white_image;
        const img2 = new RawImage(new Uint8ClampedArray(480 * 640 * 3).fill(128), 640, 480, 3);

        const conversation = [
          {
            role: "user",
            content: [{ type: "image" }, { type: "text", text: "First image." }, { type: "image" }, { type: "text", text: "Second image." }],
          },
        ];

        const text = processor.apply_chat_template(conversation, {
          add_generation_prompt: true,
        });
        const { input_ids, attention_mask, pixel_values, image_grid_thw } = await processor(text, [img1, img2]);

        const merge_length = processor.image_processor.config.merge_size ** 2;
        const grids = image_grid_thw.tolist();
        const expected_tokens_1 = Number(grids[0].reduce((a, b) => a * b, 1n)) / merge_length;
        const expected_tokens_2 = Number(grids[1].reduce((a, b) => a * b, 1n)) / merge_length;

        expect(expected_tokens_1).toBe(64);
        expect(expected_tokens_2).toBe(391);

        expect(pixel_values.dims).toEqual([256 + 1564, 1176]);
        expect(image_grid_thw.dims).toEqual([2, 3]);

        const ids = input_ids.tolist()[0].map(Number);
        const image_pad_token_id = processor.tokenizer.encode(Qwen2VLProcessor.image_token)[0];
        const vision_start_token_id = processor.tokenizer.encode("<|vision_start|>")[0];
        const vision_end_token_id = processor.tokenizer.encode("<|vision_end|>")[0];

        const first_start = ids.indexOf(vision_start_token_id);
        const first_end = ids.indexOf(vision_end_token_id);
        const second_start = ids.indexOf(vision_start_token_id, first_end + 1);
        const second_end = ids.indexOf(vision_end_token_id, second_start + 1);

        const first_placeholder_tokens = ids.slice(first_start + 1, first_end);
        const second_placeholder_tokens = ids.slice(second_start + 1, second_end);

        expect(first_placeholder_tokens.length).toBe(expected_tokens_1);
        expect(first_placeholder_tokens.every((x) => x === image_pad_token_id)).toBe(true);

        expect(second_placeholder_tokens.length).toBe(expected_tokens_2);
        expect(second_placeholder_tokens.every((x) => x === image_pad_token_id)).toBe(true);

        const total_image_pad_tokens = ids.filter((x) => x === image_pad_token_id).length;
        expect(total_image_pad_tokens).toBe(expected_tokens_1 + expected_tokens_2);
        expect(input_ids.dims).toEqual([1, ids.length]);
        expect(attention_mask.dims).toEqual([1, ids.length]);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
