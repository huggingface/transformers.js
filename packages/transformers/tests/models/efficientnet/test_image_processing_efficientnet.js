import { EfficientNetImageProcessor } from "../../../src/transformers.js";

import { load_cached_image } from "../../asset_cache.js";
import { MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  // EfficientNetImageProcessor
  //  - tests include_top
  describe("EfficientNetImageProcessor", () => {
    /** @type {EfficientNetImageProcessor} */
    const processor = new EfficientNetImageProcessor({
      crop_size: {
        height: 289,
        width: 289,
      },
      do_center_crop: false,
      do_normalize: true,
      do_rescale: true,
      do_resize: true,
      image_mean: [0.485, 0.456, 0.406],
      image_processor_type: "EfficientNetImageProcessor",
      image_std: [0.47853944, 0.4732864, 0.47434163],
      include_top: true,
      resample: 0,
      rescale_factor: 0.00392156862745098,
      rescale_offset: false,
      size: {
        height: 224,
        width: 224,
      },
    });

    it(
      "default",
      async () => {
        const image = await load_cached_image("cats");
        const { pixel_values, original_sizes, reshaped_input_sizes } = await processor(image);
        expect(pixel_values.dims).toEqual([1, 3, 224, 224]);
        expect(pixel_values.mean().item()).toBeCloseTo(0.3015307230282871, 6);
        expect(original_sizes).toEqual([[480, 640]]);
        expect(reshaped_input_sizes).toEqual([[224, 224]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
