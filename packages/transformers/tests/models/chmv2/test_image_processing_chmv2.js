import { AutoImageProcessor, CHMv2ImageProcessor } from "../../../src/transformers.js";

import { load_cached_image } from "../../asset_cache.js";
import { MAX_PROCESSOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  describe("CHMv2ImageProcessor", () => {
    const model_id = "onnx-community/dinov3-vitl16-chmv2-dpt-head-ONNX";

    /** @type {CHMv2ImageProcessor} */
    let processor;
    beforeAll(async () => {
      processor = await AutoImageProcessor.from_pretrained(model_id);
    }, MAX_PROCESSOR_LOAD_TIME);

    it(
      "default (no resize, pad to multiple of 16)",
      async () => {
        // cats.jpg is 640x480, already a multiple of 16 — no padding needed
        const image = await load_cached_image("cats");
        const { pixel_values, original_sizes, reshaped_input_sizes } = await processor(image);

        expect(pixel_values.dims).toEqual([1, 3, 480, 640]);
        expect(pixel_values.mean().item()).toBeCloseTo(0.8060207962989807, 3);

        expect(original_sizes).toEqual([[480, 640]]);
        expect(reshaped_input_sizes).toEqual([[480, 640]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "pads to multiple of size_divisor",
      async () => {
        // tiger.jpg is 612x408 — padded to 624x416
        const image = await load_cached_image("tiger");
        const { pixel_values, original_sizes, reshaped_input_sizes } = await processor(image);

        expect(pixel_values.dims).toEqual([1, 3, 416, 624]);
        expect(pixel_values.mean().item()).toBeCloseTo(0.07290177792310715, 3);

        expect(original_sizes).toEqual([[408, 612]]);
        expect(reshaped_input_sizes).toEqual([[408, 612]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
