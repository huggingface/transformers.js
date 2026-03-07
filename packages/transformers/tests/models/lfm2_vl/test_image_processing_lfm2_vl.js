import { AutoImageProcessor, Lfm2VlImageProcessor } from "../../../src/transformers.js";

import { load_cached_image } from "../../asset_cache.js";
import { MAX_PROCESSOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  describe("Lfm2VlImageProcessor", () => {
    const model_id = "onnx-community/LFM2-VL-450M-ONNX";

    /** @type {Record<string, import('../../../src/utils/image.js').RawImage>} */
    const images = {};
    /** @type {Lfm2VlImageProcessor} */
    let processor;
    beforeAll(async () => {
      processor = await AutoImageProcessor.from_pretrained(model_id);

      images.white_image = await load_cached_image("white_image");
      images.gradient_image = await load_cached_image("gradient_1280x640");
      images.cats = await load_cached_image("cats");
    }, MAX_PROCESSOR_LOAD_TIME);

    it("should be an instance of Lfm2VlImageProcessor", () => {
      expect(processor).toBeInstanceOf(Lfm2VlImageProcessor);
    });

    it(
      "small square image (single tile)",
      async () => {
        // White image: 224x224 -> smart_resize -> 256x256, single tile
        const { pixel_values, pixel_attention_mask, spatial_shapes } = await processor(images.white_image);

        // Shape: [1 tile, max_num_patches=1024, patch_dim=768]
        expect(pixel_values.dims).toEqual([1, 1024, 768]);
        expect(pixel_values.mean().item()).toBeCloseTo(0.25, 2);

        // Attention mask: 256 real patches out of 1024
        expect(pixel_attention_mask.dims).toEqual([1, 1024]);
        expect(pixel_attention_mask.sum().item()).toBe(256n);

        // Spatial shapes: 16x16 patches
        expect(spatial_shapes.dims).toEqual([1, 2]);
        expect(spatial_shapes.tolist()).toEqual([[16n, 16n]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "large non-square image (multi-tile with thumbnail)",
      async () => {
        // Gradient 1280x640 -> tiled 2x1 grid + thumbnail = 3 tiles
        const result = await processor(images.gradient_image, { return_row_col_info: true });
        const { pixel_values, pixel_attention_mask, spatial_shapes } = result;

        // Shape: [3 tiles, max_num_patches=1024, patch_dim=768]
        expect(pixel_values.dims).toEqual([3, 1024, 768]);
        expect(pixel_values.mean().item()).toBeCloseTo(-0.0009292, 3);

        // Attention mask per tile: tiles are fully packed (1024), thumbnail has 968
        expect(pixel_attention_mask.dims).toEqual([3, 1024]);
        expect(pixel_attention_mask.sum(1).tolist()).toEqual([1024n, 1024n, 968n]);

        // Spatial shapes: two 32x32 tiles + 22x44 thumbnail
        expect(spatial_shapes.tolist()).toEqual([
          [32n, 32n],
          [32n, 32n],
          [22n, 44n],
        ]);

        // Row/col info
        expect(result.image_rows).toEqual([1]);
        expect(result.image_cols).toEqual([2]);
        expect(result.image_sizes).toEqual([[352, 704]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "non-square image (single tile)",
      async () => {
        // Cats 640x480 -> smart_resize -> 416x576, single tile
        const { pixel_values, pixel_attention_mask, spatial_shapes } = await processor(images.cats);

        expect(pixel_values.dims).toEqual([1, 1024, 768]);
        expect(pixel_values.mean().item()).toBeCloseTo(0.0340382, 2);

        // 26*36 = 936 real patches
        expect(pixel_attention_mask.dims).toEqual([1, 1024]);
        expect(pixel_attention_mask.sum().item()).toBe(936n);

        expect(spatial_shapes.tolist()).toEqual([[26n, 36n]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "multiple images",
      async () => {
        const result = await processor([images.white_image, images.cats], { return_row_col_info: true });
        const { pixel_values, pixel_attention_mask, spatial_shapes } = result;

        // 2 images, each single tile
        expect(pixel_values.dims).toEqual([2, 1024, 768]);
        expect(pixel_attention_mask.dims).toEqual([2, 1024]);

        expect(spatial_shapes.tolist()).toEqual([
          [16n, 16n],
          [26n, 36n],
        ]);

        expect(result.image_rows).toEqual([1, 1]);
        expect(result.image_cols).toEqual([1, 1]);
        expect(result.image_sizes).toEqual([
          [256, 256],
          [416, 576],
        ]);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
