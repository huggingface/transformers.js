import { AutoProcessor, Gemma4ImageProcessor } from "../../../src/transformers.js";

import { load_cached_image } from "../../asset_cache.js";
import { MAX_PROCESSOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  describe.skip("Gemma4ImageProcessor", () => {
    // Load image processor via processor (config is nested in processor_config.json)
    const model_id = "onnx-community/gemma-4-E2B-it-ONNX";

    /** @type {Gemma4ImageProcessor} */
    let image_processor;
    beforeAll(async () => {
      const processor = await AutoProcessor.from_pretrained(model_id);
      image_processor = processor.image_processor;
    }, MAX_PROCESSOR_LOAD_TIME);

    it(
      "cats (640x480)",
      async () => {
        const image = await load_cached_image("cats");
        const { pixel_values, image_position_ids, num_soft_tokens_per_image } = await image_processor(image);

        // max_patches = 280 * 9 = 2520, patch_dim = 16*16*3 = 768
        expect(pixel_values.dims).toEqual([1, 2520, 768]);
        expect(image_position_ids.dims).toEqual([1, 2520, 2]);
        expect(num_soft_tokens_per_image).toEqual([266]);
        expect(pixel_values.mean().item()).toBeCloseTo(0.4925, 2);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "tiger (612x408)",
      async () => {
        const image = await load_cached_image("tiger");
        const { pixel_values, num_soft_tokens_per_image } = await image_processor(image);

        expect(pixel_values.dims).toEqual([1, 2520, 768]);
        expect(num_soft_tokens_per_image).toEqual([260]);
        expect(pixel_values.mean().item()).toBeCloseTo(0.3586, 2);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "white image (224x224)",
      async () => {
        const image = await load_cached_image("white_image");
        const { pixel_values, num_soft_tokens_per_image } = await image_processor(image);

        expect(pixel_values.dims).toEqual([1, 2520, 768]);
        expect(num_soft_tokens_per_image).toEqual([256]);
        // White image: mostly 1.0 values with zero-padding → mean ~0.91
        expect(pixel_values.mean().item()).toBeCloseTo(0.9114, 2);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "gradient (1280x640)",
      async () => {
        const image = await load_cached_image("gradient_1280x640");
        const { pixel_values, num_soft_tokens_per_image } = await image_processor(image);

        expect(pixel_values.dims).toEqual([1, 2520, 768]);
        expect(num_soft_tokens_per_image).toEqual([253]);
        expect(pixel_values.mean().item()).toBeCloseTo(0.4514, 2);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "batched (cats + tiger)",
      async () => {
        const cats = await load_cached_image("cats");
        const tiger = await load_cached_image("tiger");
        const { pixel_values, image_position_ids, num_soft_tokens_per_image } = await image_processor([cats, tiger]);

        expect(pixel_values.dims).toEqual([2, 2520, 768]);
        expect(image_position_ids.dims).toEqual([2, 2520, 2]);
        expect(num_soft_tokens_per_image).toEqual([266, 260]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "position IDs are correct",
      async () => {
        const image = await load_cached_image("cats");
        const { image_position_ids, num_soft_tokens_per_image } = await image_processor(image);

        // First position should be [0, 0] (col=0, row=0)
        expect(Number(image_position_ids.data[0])).toEqual(0);
        expect(Number(image_position_ids.data[1])).toEqual(0);

        // Second position should be [1, 0] (col=1, row=0)
        expect(Number(image_position_ids.data[2])).toEqual(1);
        expect(Number(image_position_ids.data[3])).toEqual(0);

        // Padding positions should be -1
        const num_real_patches = num_soft_tokens_per_image[0] * 9; // 266 * 9 = 2394
        const first_pad_idx = num_real_patches * 2;
        expect(Number(image_position_ids.data[first_pad_idx])).toEqual(-1);
        expect(Number(image_position_ids.data[first_pad_idx + 1])).toEqual(-1);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
