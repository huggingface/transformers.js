import { AutoImageProcessor, NougatImageProcessor } from "../../../src/transformers.js";

import { load_cached_image } from "../../asset_cache.js";
import { MAX_PROCESSOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  // NougatImageProcessor
  //  - tests padding after normalization (image_mean != 0.5, image_std != 0.5)
  describe("NougatImageProcessor", () => {
    const model_id = "Xenova/nougat-small";

    /** @type {NougatImageProcessor} */
    let processor;
    beforeAll(async () => {
      processor = await AutoImageProcessor.from_pretrained(model_id);
    }, MAX_PROCESSOR_LOAD_TIME);

    it(
      "padding after normalization",
      async () => {
        const image = await load_cached_image("paper");
        const { pixel_values, original_sizes, reshaped_input_sizes } = await processor(image);

        expect(pixel_values.dims).toEqual([1, 3, 896, 672]);
        expect(pixel_values.mean().item()).toBeCloseTo(1.8447155005897355, 6);

        expect(original_sizes).toEqual([[850, 685]]);
        expect(reshaped_input_sizes).toEqual([[833, 672]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
