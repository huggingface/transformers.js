import { AutoImageProcessor, ViTFeatureExtractor } from "../../../src/transformers.js";

import { load_cached_image } from "../../asset_cache.js";
import { MAX_PROCESSOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  describe("ViTFeatureExtractor", () => {
    const model_id = "Xenova/vit-base-patch16-224";

    /** @type {ViTFeatureExtractor} */
    let processor;
    beforeAll(async () => {
      processor = await AutoImageProcessor.from_pretrained(model_id);
    }, MAX_PROCESSOR_LOAD_TIME);

    it(
      "default",
      async () => {
        const image = await load_cached_image("tiger");
        const { pixel_values, original_sizes, reshaped_input_sizes } = await processor(image);

        expect(pixel_values.dims).toEqual([1, 3, 224, 224]);
        expect(pixel_values.mean().item()).toBeCloseTo(-0.22706867939852762, 6);

        expect(original_sizes).toEqual([[408, 612]]);
        expect(reshaped_input_sizes).toEqual([[224, 224]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
