import { AutoImageProcessor, DetrFeatureExtractor } from "../../../src/transformers.js";

import { load_cached_image } from "../../asset_cache.js";
import { MAX_PROCESSOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  describe("DetrFeatureExtractor", () => {
    const model_id = "Xenova/detr-resnet-50";

    /** @type {DetrFeatureExtractor} */
    let processor;
    beforeAll(async () => {
      processor = await AutoImageProcessor.from_pretrained(model_id);
    }, MAX_PROCESSOR_LOAD_TIME);

    it(
      "default",
      async () => {
        const image = await load_cached_image("tiger");
        const { pixel_values, original_sizes, reshaped_input_sizes, pixel_mask } = await processor(image);

        expect(pixel_values.dims).toEqual([1, 3, 888, 1333]);
        expect(pixel_values.mean().item()).toBeCloseTo(-0.27840224131001773, 6);

        expect(original_sizes).toEqual([[408, 612]]);
        expect(reshaped_input_sizes).toEqual([[888, 1333]]);

        expect(pixel_mask.dims).toEqual([1, 64, 64]);
        expect(pixel_mask.to("float32").mean().item()).toBeCloseTo(1.0, 6);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
