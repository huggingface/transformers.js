import { AutoImageProcessor, SamImageProcessor } from "../../../src/transformers.js";

import { load_cached_image } from "../../asset_cache.js";
import { MAX_PROCESSOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  // SamImageProcessor
  //  - tests normal padding (do_pad=true, pad_size={"height":1024,"width":1024})
  //  - In addition to the image, pass in a list of points
  describe("SamImageProcessor", () => {
    const model_id = "Xenova/sam-vit-base";

    /** @type {SamImageProcessor} */
    let processor;
    beforeAll(async () => {
      processor = await AutoImageProcessor.from_pretrained(model_id);
    }, MAX_PROCESSOR_LOAD_TIME);

    it(
      "without input points",
      async () => {
        const image = await load_cached_image("pattern_3x3");
        const { pixel_values, original_sizes, reshaped_input_sizes } = await processor(image);
        expect(pixel_values.dims).toEqual([1, 3, 1024, 1024]);
        expect(pixel_values.mean().item()).toBeCloseTo(-0.4505715670146813, 6);

        expect(original_sizes).toEqual([[3, 3]]);
        expect(reshaped_input_sizes).toEqual([[1024, 1024]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "with input points",
      async () => {
        const image = await load_cached_image("pattern_3x3");
        const { original_sizes, reshaped_input_sizes, input_points } = await processor(image, {
          input_points: [[[1, 2]]],
        });

        expect(original_sizes).toEqual([[3, 3]]);
        expect(reshaped_input_sizes).toEqual([[1024, 1024]]);
        expect(input_points.tolist()).toBeCloseToNested([[[[341.3333, 682.6667]]]], 4);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "multiple points with labels",
      async () => {
        const image = await load_cached_image("pattern_3x3");
        const { original_sizes, reshaped_input_sizes, input_points, input_labels } = await processor(image, {
          input_points: [
            [
              [1, 2],
              [2, 1],
            ],
          ],
          input_labels: [[1, 0]],
        });

        expect(original_sizes).toEqual([[3, 3]]);
        expect(reshaped_input_sizes).toEqual([[1024, 1024]]);
        expect(input_points.tolist()).toBeCloseToNested(
          [
            [
              [
                [341.3333, 682.6667],
                [682.6667, 341.3333],
              ],
            ],
          ],
          4,
        );
        expect(input_labels.tolist()).toEqual([[[1n, 0n]]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "with input boxes",
      async () => {
        const image = await load_cached_image("pattern_3x3");
        const { original_sizes, reshaped_input_sizes, input_boxes } = await processor(image, {
          input_boxes: [[[0, 1, 2, 2]]],
        });

        expect(original_sizes).toEqual([[3, 3]]);
        expect(reshaped_input_sizes).toEqual([[1024, 1024]]);
        expect(input_boxes.tolist()).toBeCloseToNested([[[0, 341.3333, 682.6667, 682.6667]]], 4);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
