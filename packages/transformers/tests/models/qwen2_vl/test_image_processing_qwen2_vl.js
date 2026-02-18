import { AutoImageProcessor, Qwen2VLImageProcessor } from "../../../src/transformers.js";

import { load_cached_image } from "../../asset_cache.js";
import { MAX_PROCESSOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  // Qwen2VLImageProcessor
  // - custom image processing (min_pixels, max_pixels)
  describe("Qwen2VLImageProcessor", () => {
    const model_id = "hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration";

    /** @type {Qwen2VLImageProcessor} */
    let processor;
    beforeAll(async () => {
      processor = await AutoImageProcessor.from_pretrained(model_id);
    }, MAX_PROCESSOR_LOAD_TIME);

    it(
      "custom image processing",
      async () => {
        const image = await load_cached_image("white_image");
        const { pixel_values, image_grid_thw, original_sizes, reshaped_input_sizes } = await processor(image);

        expect(pixel_values.dims).toEqual([256, 1176]);
        expect(pixel_values.mean().item()).toBeCloseTo(2.050372362136841, 6);
        expect(image_grid_thw.tolist()).toEqual([[1n, 16n, 16n]]);

        expect(original_sizes).toEqual([[224, 224]]);
        expect(reshaped_input_sizes).toEqual([[224, 224]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
