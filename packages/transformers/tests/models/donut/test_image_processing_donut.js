import { AutoImageProcessor, DonutFeatureExtractor } from "../../../src/transformers.js";

import { load_cached_image } from "../../asset_cache.js";
import { MAX_PROCESSOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  // DonutFeatureExtractor
  //  - tests thumbnail resizing (do_thumbnail=true, size=[960, 1280])
  //  - tests padding after normalization (image_mean=image_std=0.5)
  describe("DonutFeatureExtractor", () => {
    const model_id = "Xenova/donut-base-finetuned-cord-v2";

    /** @type {DonutFeatureExtractor} */
    let processor;
    beforeAll(async () => {
      processor = await AutoImageProcessor.from_pretrained(model_id);
    }, MAX_PROCESSOR_LOAD_TIME);

    it(
      "default",
      async () => {
        const image = await load_cached_image("receipt");
        const { pixel_values, original_sizes, reshaped_input_sizes } = await processor(image);

        expect(pixel_values.dims).toEqual([1, 3, 1280, 960]);
        expect(pixel_values.mean().item()).toBeCloseTo(0.1229388610053704, 6);

        expect(original_sizes).toEqual([[864, 576]]);
        expect(reshaped_input_sizes).toEqual([[1280, 853]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
