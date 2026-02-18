import { AutoImageProcessor, GLPNFeatureExtractor } from "../../../src/transformers.js";

import { load_cached_image } from "../../asset_cache.js";
import { MAX_PROCESSOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  // GLPNFeatureExtractor
  //  - tests `size_divisor` and no size (size_divisor=32)
  describe("GLPNFeatureExtractor", () => {
    const model_id = "Xenova/glpn-kitti";

    /** @type {GLPNFeatureExtractor} */
    let processor;
    beforeAll(async () => {
      processor = await AutoImageProcessor.from_pretrained(model_id);
    }, MAX_PROCESSOR_LOAD_TIME);

    it(
      "multiple of size_divisor",
      async () => {
        const image = await load_cached_image("cats");
        const { pixel_values, original_sizes, reshaped_input_sizes } = await processor(image);
        expect(pixel_values.dims).toEqual([1, 3, 480, 640]);
        expect(pixel_values.mean().item()).toBeCloseTo(0.5186172404123327, 6);

        expect(original_sizes).toEqual([[480, 640]]);
        expect(reshaped_input_sizes).toEqual([[480, 640]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "non-multiple of size_divisor",
      async () => {
        // Tests input which is not a multiple of 32 ([408, 612] -> [384, 608])
        const image = await load_cached_image("tiger");
        const { pixel_values, original_sizes, reshaped_input_sizes } = await processor(image);

        expect(pixel_values.dims).toEqual([1, 3, 384, 608]);
        expect(pixel_values.mean().item()).toBeCloseTo(0.38628831535989555, 6);

        expect(original_sizes).toEqual([[408, 612]]);
        expect(reshaped_input_sizes).toEqual([[384, 608]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
