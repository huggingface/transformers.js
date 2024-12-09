import { AutoImageProcessor, JinaCLIPImageProcessor } from "../../../src/transformers.js";

import { load_cached_image } from "../../asset_cache.js";
import { MAX_PROCESSOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  // JinaCLIPImageProcessor
  // - custom config overrides
  describe("JinaCLIPImageProcessor", () => {
    const model_id = "jinaai/jina-clip-v2";

    /** @type {JinaCLIPImageProcessor} */
    let processor;
    beforeAll(async () => {
      processor = await AutoImageProcessor.from_pretrained(model_id);
    }, MAX_PROCESSOR_LOAD_TIME);

    it(
      "default",
      async () => {
        const image = await load_cached_image("tiger");
        const { pixel_values, original_sizes, reshaped_input_sizes } = await processor(image);

        expect(pixel_values.dims).toEqual([1, 3, 512, 512]);
        expect(pixel_values.mean().item()).toBeCloseTo(-0.06637834757566452, 3);

        expect(original_sizes).toEqual([[408, 612]]);
        expect(reshaped_input_sizes).toEqual([[512, 512]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
