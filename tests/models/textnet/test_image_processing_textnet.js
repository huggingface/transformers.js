import { AutoImageProcessor, TextNetImageProcessor } from "../../../src/transformers.js";

import { load_cached_image } from "../../asset_cache.js";
import { MAX_PROCESSOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  describe("TextNetImageProcessor", () => {
    const model_id = "onnx-community/textnet-tiny";

    /** @type {TextNetImageProcessor} */
    let processor;
    beforeAll(async () => {
      processor = await AutoImageProcessor.from_pretrained(model_id);
    }, MAX_PROCESSOR_LOAD_TIME);

    it(
      "default",
      async () => {
        const image = await load_cached_image("receipt");
        const { pixel_values, original_sizes, reshaped_input_sizes } = await processor(image);

        expect(pixel_values.dims).toEqual([1, 3, 960, 640]);
        expect(pixel_values.mean().item()).toBeCloseTo(0.8106788992881775, 6);

        expect(original_sizes).toEqual([[864, 576]]);
        expect(reshaped_input_sizes).toEqual([[960, 640]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
