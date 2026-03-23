import { AutoProcessor, PaliGemmaProcessor } from "../../../src/transformers.js";

import { load_cached_image } from "../../asset_cache.js";
import { MAX_PROCESSOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  const model_id = "hf-internal-testing/tiny-random-PaliGemmaForConditionalGeneration";

  describe("PaliGemmaProcessor", () => {
    /** @type {PaliGemmaProcessor} */
    let processor;
    let images = {};

    beforeAll(async () => {
      processor = await AutoProcessor.from_pretrained(model_id);
      images = {
        white_image: await load_cached_image("white_image"),
      };
    }, MAX_PROCESSOR_LOAD_TIME);

    it(
      "Image-only (default text)",
      async () => {
        const { input_ids, pixel_values } = await processor(images.white_image);
        expect(input_ids.dims).toEqual([1, 258]);
        expect(pixel_values.dims).toEqual([1, 3, 224, 224]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "Single image & text",
      async () => {
        const { input_ids, pixel_values } = await processor(images.white_image, "<image>What is on the flower?");
        expect(input_ids.dims).toEqual([1, 264]);
        expect(pixel_values.dims).toEqual([1, 3, 224, 224]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "Multiple images & text",
      async () => {
        const { input_ids, pixel_values } = await processor([images.white_image, images.white_image], "<image><image>Describe the images.");
        expect(input_ids.dims).toEqual([1, 518]);
        expect(pixel_values.dims).toEqual([2, 3, 224, 224]);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
