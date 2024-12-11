import { AutoImageProcessor, Phi3VImageProcessor } from "../../../src/transformers.js";

import { load_cached_image } from "../../asset_cache.js";
import { MAX_PROCESSOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

const TARGET_IMAGE_SIZE = [3, 336, 336];

export default () => {
  // Phi3VImageProcessor
  // - custom image processing (patching)
  describe("Phi3VImageProcessor", () => {
    const model_id = "onnx-community/Phi-3.5-vision-instruct";

    /** @type {Record<string, import('../../../src/utils/image.js').RawImage>} */
    const images = {};
    /** @type {Phi3VImageProcessor} */
    let processor;
    beforeAll(async () => {
      processor = await AutoImageProcessor.from_pretrained(model_id);

      // Load images
      const gradient_image = await load_cached_image("gradient_1280x640");
      const white_image = await load_cached_image("white_image");

      images.gradient_image = gradient_image;
      images.white_image = white_image;
    }, MAX_PROCESSOR_LOAD_TIME);

    it(
      "square image (num_crops=4)",
      async () => {
        const num_crops = 4;
        const { pixel_values, image_sizes, num_img_tokens } = await processor(images.white_image, { num_crops });
        expect(pixel_values.dims).toEqual([1, 1 + num_crops, ...TARGET_IMAGE_SIZE]);
        expect(pixel_values.flatten(2).mean(2).tolist()).toBeCloseToNested([[2.050372362136841, 2.050372362136841, 2.050372362136841, 2.050372362136841, 2.050372362136841]], 1);
        expect(pixel_values.mean().item()).toBeCloseTo(2.050372362136841, 1);

        expect(image_sizes.tolist()).toEqual([[672n, 672n]]);
        expect(num_img_tokens).toEqual([757]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "non-square image (num_crops=4)",
      async () => {
        const num_crops = 4;
        const { pixel_values, image_sizes, num_img_tokens } = await processor(images.gradient_image, { num_crops });
        expect(pixel_values.dims).toEqual([1, 1 + num_crops, ...TARGET_IMAGE_SIZE]);

        // NOTE: We use a slighly different cropping strategy to the python implementation,
        // meaning the following tests would fail.
        // expect(pixel_values.flatten(2).mean(2).tolist()).toBeCloseToNested([[
        //   0.18679802119731903, -0.5585645437240601, 0.9321606755256653, 0.0, 0.0,
        // ]], 1);
        // expect(pixel_values.mean().item()).toBeCloseTo(0.11207880824804306, 6);

        expect(image_sizes.tolist()).toEqual([[336n, 672n]]);
        expect(num_img_tokens).toEqual([457]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "single image (num_crops=16)",
      async () => {
        const num_crops = 16;
        const { pixel_values, image_sizes, num_img_tokens } = await processor(images.gradient_image, { num_crops });
        expect(pixel_values.dims).toEqual([1, 1 + num_crops, 3, 336, 336]);
        expect(pixel_values.mean().item()).toBeCloseTo(0.4677375257015228, 1);

        expect(image_sizes.tolist()).toEqual([[1008n, 1680n]]);
        expect(num_img_tokens).toEqual([2353]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "multiple images (num_crops=4)",
      async () => {
        const num_crops = 4;
        const { pixel_values, image_sizes, num_img_tokens } = await processor([images.gradient_image, images.white_image], { num_crops });
        expect(pixel_values.dims).toEqual([2, 1 + num_crops, ...TARGET_IMAGE_SIZE]);
        expect(image_sizes.tolist()).toEqual([
          [336n, 672n],
          [672n, 672n],
        ]);
        expect(num_img_tokens).toEqual([457, 757]);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
