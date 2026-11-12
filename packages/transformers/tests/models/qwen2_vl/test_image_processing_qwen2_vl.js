import { AutoImageProcessor, Qwen2VLImageProcessor, RawImage } from "../../../src/transformers.js";

import { load_cached_image } from "../../asset_cache.js";
import { MAX_PROCESSOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  // Qwen2VLImageProcessor
  // - custom image processing (min_pixels, max_pixels)
  // - multiple images (same and different sizes)
  describe("Qwen2VLImageProcessor", () => {
    const model_id = "hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration";

    /** @type {Qwen2VLImageProcessor} */
    let processor;
    beforeAll(async () => {
      processor = await AutoImageProcessor.from_pretrained(model_id);
    }, MAX_PROCESSOR_LOAD_TIME);

    it(
      "custom image processing",
      async () => {
        const image = await load_cached_image("white_image");
        const { pixel_values, image_grid_thw, original_sizes, reshaped_input_sizes } = await processor(image);

        expect(pixel_values.dims).toEqual([256, 1176]);
        expect(pixel_values.mean().item()).toBeCloseTo(2.050372362136841, 6);
        expect(image_grid_thw.tolist()).toEqual([[1n, 16n, 16n]]);

        expect(original_sizes).toEqual([[224, 224]]);
        expect(reshaped_input_sizes).toEqual([[224, 224]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "multiple images of same size",
      async () => {
        const image = await load_cached_image("white_image");
        const { pixel_values, image_grid_thw, original_sizes, reshaped_input_sizes } = await processor([image, image]);

        expect(pixel_values.dims).toEqual([512, 1176]);
        expect(image_grid_thw.tolist()).toEqual([
          [1n, 16n, 16n],
          [1n, 16n, 16n],
        ]);

        expect(original_sizes).toEqual([
          [224, 224],
          [224, 224],
        ]);
        expect(reshaped_input_sizes).toEqual([
          [224, 224],
          [224, 224],
        ]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "multiple images of different sizes",
      async () => {
        const img1 = new RawImage(new Uint8ClampedArray(480 * 640 * 3).fill(128), 640, 480, 3);
        const img2 = new RawImage(new Uint8ClampedArray(560 * 840 * 3).fill(128), 840, 560, 3);
        const { pixel_values, image_grid_thw, original_sizes, reshaped_input_sizes } = await processor([img1, img2]);

        const out1 = await processor(img1);
        const out2 = await processor(img2);

        expect(pixel_values.dims).toEqual([out1.pixel_values.dims[0] + out2.pixel_values.dims[0], 1176]);
        expect(image_grid_thw.tolist()).toEqual([out1.image_grid_thw.tolist()[0], out2.image_grid_thw.tolist()[0]]);
        expect(original_sizes).toEqual([out1.original_sizes[0], out2.original_sizes[0]]);
        expect(reshaped_input_sizes).toEqual([out1.reshaped_input_sizes[0], out2.reshaped_input_sizes[0]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
