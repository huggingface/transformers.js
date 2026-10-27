import { AutoImageProcessor, Lfm3VlImageProcessor, RawImage } from "../../../src/transformers.js";

import { MAX_PROCESSOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  describe("Lfm3VlImageProcessor", () => {
    const model_id = "onnx-community/LFM3-VL-2B-ONNX";

    /** @type {Lfm3VlImageProcessor} */
    let processor;
    beforeAll(async () => {
      processor = await AutoImageProcessor.from_pretrained(model_id);
    }, MAX_PROCESSOR_LOAD_TIME);

    it("should be an instance of Lfm3VlImageProcessor", () => {
      expect(processor).toBeInstanceOf(Lfm3VlImageProcessor);
    });

    it(
      "small square image",
      async () => {
        const image = new RawImage(new Uint8Array(224 * 224 * 3).fill(255), 224, 224, 3);
        const { pixel_values, image_grid_thw, original_sizes, reshaped_input_sizes } = await processor(image);

        expect(pixel_values.dims).toEqual([256, 1536]);
        expect(pixel_values.mean().item()).toBeCloseTo(1, 2);
        expect(image_grid_thw.tolist()).toEqual([[1n, 16n, 16n]]);
        expect(original_sizes).toEqual([[224, 224]]);
        expect(reshaped_input_sizes).toEqual([[256, 256]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "multiple images with different spatial grids",
      async () => {
        const black = new RawImage(new Uint8Array(512 * 256 * 3), 512, 256, 3);
        const white = new RawImage(new Uint8Array(256 * 256 * 3).fill(255), 256, 256, 3);
        const { pixel_values, image_grid_thw } = await processor([black, white]);

        expect(pixel_values.dims).toEqual([768, 1536]);
        expect(image_grid_thw.tolist()).toEqual([
          [1n, 16n, 32n],
          [1n, 16n, 16n],
        ]);
        expect(pixel_values.data.slice(0, 512 * 1536).every((x) => x === -1)).toBe(true);
        expect(pixel_values.data.slice(512 * 1536).every((x) => x === 1)).toBe(true);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "spatial merge order and temporal patch duplication",
      async () => {
        const small_patch_processor = new Lfm3VlImageProcessor({ ...processor.config, do_resize: false, patch_size: 2 });
        const image = new RawImage(
          Uint8Array.from({ length: 8 * 8 * 3 }, (_, i) => i),
          8,
          8,
          3,
        );
        const { pixel_values, image_grid_thw } = await small_patch_processor(image);

        expect(pixel_values.dims).toEqual([16, 24]);
        expect(image_grid_thw.tolist()).toEqual([[1n, 4n, 4n]]);
        const first_patch = [0, 3, 24, 27, 0, 3, 24, 27, 1, 4, 25, 28, 1, 4, 25, 28, 2, 5, 26, 29, 2, 5, 26, 29];
        first_patch.forEach((value, i) => expect(pixel_values.data[i]).toBeCloseTo(value / 127.5 - 1, 6));
        const patch_starts = [0, 6, 48, 54, 12, 18, 60, 66, 96, 102, 144, 150, 108, 114, 156, 162];
        patch_starts.forEach((value, i) => expect(pixel_values.data[i * 24]).toBeCloseTo(value / 127.5 - 1, 6));
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
