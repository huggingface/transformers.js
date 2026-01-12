import { AutoImageProcessor, Swin2SRImageProcessor } from "../../../src/transformers.js";

import { load_cached_image } from "../../asset_cache.js";
import { MAX_PROCESSOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  // Swin2SRImageProcessor
  //  - tests when padding is a number (do_pad=true, pad_size=8)
  describe("Swin2SRImageProcessor", () => {
    const model_id = "Xenova/swin2SR-classical-sr-x2-64";

    /** @type {Swin2SRImageProcessor} */
    let processor;
    beforeAll(async () => {
      processor = await AutoImageProcessor.from_pretrained(model_id);
    }, MAX_PROCESSOR_LOAD_TIME);

    it(
      "Pad to multiple of 8 (3x3 -> 8x8)",
      async () => {
        const image = await load_cached_image("pattern_3x3");
        const { pixel_values } = await processor(image);

        expect(pixel_values.dims).toEqual([1, 3, 8, 8]);
        expect(pixel_values.mean().item()).toBeCloseTo(0.5458333368102709, 6);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "Do not pad if already a multiple of 8 (8x8 -> 8x8)",
      async () => {
        const image = await load_cached_image("checkerboard_8x8");
        const { pixel_values } = await processor(image);
        expect(pixel_values.dims).toEqual([1, 3, 8, 8]);
        expect(pixel_values.mean().item()).toBeCloseTo(0.5, 6);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
