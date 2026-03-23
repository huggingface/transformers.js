import { AutoImageProcessor, DPTFeatureExtractor, DPTImageProcessor } from "../../../src/transformers.js";

import { load_cached_image } from "../../asset_cache.js";
import { MAX_PROCESSOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  // DPTFeatureExtractor
  describe("DPTFeatureExtractor", () => {
    const model_id = "Xenova/dpt-hybrid-midas";

    /** @type {DPTFeatureExtractor} */
    let processor;
    beforeAll(async () => {
      processor = await AutoImageProcessor.from_pretrained(model_id);
    }, MAX_PROCESSOR_LOAD_TIME);

    it(
      "grayscale images",
      async () => {
        const image = await load_cached_image("cats");
        const { pixel_values, original_sizes, reshaped_input_sizes } = await processor(image);

        expect(pixel_values.dims).toEqual([1, 3, 384, 384]);
        expect(pixel_values.mean().item()).toBeCloseTo(0.0372855559389454, 6);

        expect(original_sizes).toEqual([[480, 640]]);
        expect(reshaped_input_sizes).toEqual([[384, 384]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });

  // DPTImageProcessor
  //  - tests ensure_multiple_of
  //  - tests keep_aspect_ratio
  //  - tests bankers rounding
  describe("DPTImageProcessor", () => {
    const model_id = "Xenova/depth-anything-small-hf";

    /** @type {DPTImageProcessor} */
    let processor;
    beforeAll(async () => {
      processor = await AutoImageProcessor.from_pretrained(model_id);
    }, MAX_PROCESSOR_LOAD_TIME);

    it(
      "ensure_multiple_of w/ normal rounding",
      async () => {
        const image = await load_cached_image("cats");
        const { pixel_values, original_sizes, reshaped_input_sizes } = await processor(image);

        expect(pixel_values.dims).toEqual([1, 3, 518, 686]);
        expect(pixel_values.mean().item()).toBeCloseTo(0.30337387323379517, 3);

        expect(original_sizes).toEqual([[480, 640]]);
        expect(reshaped_input_sizes).toEqual([[518, 686]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "ensure_multiple_of w/ bankers rounding",
      async () => {
        const image = await load_cached_image("checkerboard_64x32");
        const { pixel_values, original_sizes, reshaped_input_sizes } = await processor(image);

        // NOTE: without bankers rounding, this would be [1, 3, 266, 518]
        expect(pixel_values.dims).toEqual([1, 3, 252, 518]);
        expect(pixel_values.mean().item()).toBeCloseTo(0.2267402559518814, 1);

        expect(original_sizes).toEqual([[32, 64]]);
        expect(reshaped_input_sizes).toEqual([[252, 518]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
