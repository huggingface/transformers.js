import { AutoImageProcessor, MobileViTFeatureExtractor, MobileViTImageProcessor } from "../../../src/transformers.js";

import { load_cached_image } from "../../asset_cache.js";
import { MAX_PROCESSOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  // MobileViTFeatureExtractor
  describe("MobileViTFeatureExtractor (default)", () => {
    const model_id = "Xenova/mobilevit-small";

    /** @type {MobileViTFeatureExtractor} */
    let processor;
    beforeAll(async () => {
      processor = await AutoImageProcessor.from_pretrained(model_id);
    }, MAX_PROCESSOR_LOAD_TIME);

    it(
      "default",
      async () => {
        const image = await load_cached_image("tiger");
        const { pixel_values, original_sizes, reshaped_input_sizes } = await processor(image);

        expect(pixel_values.dims).toEqual([1, 3, 256, 256]);
        expect(pixel_values.mean().item()).toBeCloseTo(0.4599160496887033, 6);

        expect(original_sizes).toEqual([[408, 612]]);
        expect(reshaped_input_sizes).toEqual([[256, 256]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });

  // MobileViTFeatureExtractor
  //  - tests not converting to rgb (do_convert_rgb=false)
  describe("MobileViTFeatureExtractor (do_convert_rgb=false)", () => {
    const model_id = "Xenova/quickdraw-mobilevit-small";

    /** @type {MobileViTFeatureExtractor} */
    let processor;
    beforeAll(async () => {
      processor = await AutoImageProcessor.from_pretrained(model_id);
    }, MAX_PROCESSOR_LOAD_TIME);

    it(
      "grayscale image",
      async () => {
        const image = await load_cached_image("skateboard");
        const { pixel_values, original_sizes, reshaped_input_sizes } = await processor(image);

        expect(pixel_values.dims).toEqual([1, 1, 28, 28]);
        expect(pixel_values.mean().item()).toBeCloseTo(0.08558923671585128, 6);

        expect(original_sizes).toEqual([[28, 28]]);
        expect(reshaped_input_sizes).toEqual([[28, 28]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });

  // MobileViTImageProcessor
  //  - tests converting RGB to BGR (do_flip_channel_order=true)
  describe("MobileViTImageProcessor (do_flip_channel_order=true)", () => {
    const model_id = "Xenova/mobilevitv2-1.0-imagenet1k-256";

    /** @type {MobileViTImageProcessor} */
    let processor;
    beforeAll(async () => {
      processor = await AutoImageProcessor.from_pretrained(model_id);
    }, MAX_PROCESSOR_LOAD_TIME);

    it(
      "RGB to BGR",
      async () => {
        const image = await load_cached_image("cats");
        const { pixel_values, original_sizes, reshaped_input_sizes } = await processor(image);
        const { data, dims } = pixel_values;

        expect(dims).toEqual([1, 3, 256, 256]);
        expect(pixel_values.mean().item()).toBeCloseTo(0.5215385556221008, 2);

        expect(original_sizes).toEqual([[480, 640]]);
        expect(reshaped_input_sizes).toEqual([[256, 256]]);

        // Ensure RGB to BGR conversion
        expect(data.slice(0, 3)).toBeCloseToNested([0.24313725531101227, 0.250980406999588, 0.364705890417099], 4);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
