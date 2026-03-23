import { AutoImageProcessor, Idefics3ImageProcessor } from "../../../src/transformers.js";

import { load_cached_image } from "../../asset_cache.js";
import { MAX_PROCESSOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  // Idefics3ImageProcessor
  // - custom image processing (patching)
  describe("Idefics3ImageProcessor", () => {
    const model_id = "hf-internal-testing/tiny-random-Idefics3ForConditionalGeneration";

    /** @type {Record<string, import('../../../src/utils/image.js').RawImage>} */
    const images = {};
    /** @type {Idefics3ImageProcessor} */
    let processor;
    beforeAll(async () => {
      processor = await AutoImageProcessor.from_pretrained(model_id);

      // Load images
      const image = await load_cached_image("gradient_1280x640");
      const white_image = await load_cached_image("white_image");

      images.image = image;
      images.image_1 = await image.resize(1600, 1067);
      images.image_2 = await image.resize(224, 224);
      images.white_image = white_image;
      images.white_image_1 = await white_image.resize(1600, 1067);
      images.white_image_2 = await white_image.resize(224, 224);
    }, MAX_PROCESSOR_LOAD_TIME);

    it(
      "no image splitting",
      async () => {
        const { pixel_values, rows, cols } = await processor(images.image, { do_image_splitting: false, return_row_col_info: true });
        expect(pixel_values.dims).toEqual([1, 1, 3, 364, 364]);
        expect(pixel_values.mean().item()).toBeCloseTo(-0.001035306602716446, 2);
        expect(rows).toEqual([[0]]);
        expect(cols).toEqual([[0]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "batched no image splitting",
      async () => {
        const { pixel_values, pixel_attention_mask, rows, cols } = await processor([[images.white_image_1], [images.white_image_2], [images.white_image_1, images.white_image_2]], { do_image_splitting: false, return_row_col_info: true });
        expect(pixel_values.dims).toEqual([3, 2, 3, 364, 364]);
        expect(pixel_values.mean().item()).toBeCloseTo(2 / 3, 2);
        expect(pixel_attention_mask.dims).toEqual([3, 2, 364, 364]);
        expect(pixel_attention_mask.to("float32").mean().item()).toBeCloseTo(2 / 3, 3);
        expect(rows).toEqual([[0], [0], [0, 0]]);
        expect(cols).toEqual([[0], [0], [0, 0]]);

        // Test that the order of the pixel attention mask matches the python implementation
        expect(pixel_attention_mask.data.reduce((a, b, i) => a + i * b, 0)).toEqual(228217205216);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "correct patching",
      async () => {
        const { pixel_values, rows, cols } = await processor(images.image, { return_row_col_info: true });
        expect(pixel_values.dims).toEqual([1, 9, 3, 364, 364]);
        expect(pixel_values.flatten(2).mean(2).tolist()).toBeCloseToNested([[-0.7012196183204651, -0.30104631185531616, 0.09912905097007751, 0.49929487705230713, -0.5011996626853943, -0.10103467106819153, 0.2991456389427185, 0.6993265151977539, -0.0010353063698858023]], 1);
        expect(rows).toEqual([[2]]);
        expect(cols).toEqual([[4]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "unbatched, single image",
      async () => {
        const { pixel_values, rows, cols } = await processor(images.image_1, { return_row_col_info: true });
        expect(pixel_values.dims).toEqual([1, 13, 3, 364, 364]);

        expect(rows).toEqual([[3]]);
        expect(cols).toEqual([[4]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "unbatched, multiple images",
      async () => {
        const { pixel_values, rows, cols } = await processor([images.image_1, images.image_2], { return_row_col_info: true });
        expect(pixel_values.dims).toEqual([1, 30, 3, 364, 364]);

        expect(rows).toEqual([[3, 4]]);
        expect(cols).toEqual([[4, 4]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "batched, multiple images",
      async () => {
        const { pixel_values, rows, cols } = await processor([[images.image_1], [images.image_1, images.image_2]], { return_row_col_info: true });
        expect(pixel_values.dims).toEqual([2, 30, 3, 364, 364]);
        expect(rows).toEqual([[3], [3, 4]]);
        expect(cols).toEqual([[4], [4, 4]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
