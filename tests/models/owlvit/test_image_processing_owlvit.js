import { AutoImageProcessor, OwlViTFeatureExtractor } from "../../../src/transformers.js";

import { load_cached_image } from "../../asset_cache.js";
import { MAX_PROCESSOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  describe("OwlViTFeatureExtractor", () => {
    const model_id = "Xenova/owlvit-base-patch32";

    /** @type {OwlViTFeatureExtractor} */
    let processor;
    beforeAll(async () => {
      processor = await AutoImageProcessor.from_pretrained(model_id);
    }, MAX_PROCESSOR_LOAD_TIME);

    it(
      "default",
      async () => {
        const image = await load_cached_image("cats");
        const { pixel_values, original_sizes, reshaped_input_sizes } = await processor(image);

        expect(pixel_values.dims).toEqual([1, 3, 768, 768]);
        expect(pixel_values.mean().item()).toBeCloseTo(0.250620447910435, 6);

        expect(original_sizes).toEqual([[480, 640]]);
        expect(reshaped_input_sizes).toEqual([[768, 768]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
