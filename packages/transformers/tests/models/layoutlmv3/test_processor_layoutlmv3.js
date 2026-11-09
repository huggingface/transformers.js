import { LayoutLMv3Processor } from "../../../src/transformers.js";

import { load_cached_image } from "../../asset_cache.js";
import { MAX_PROCESSOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  describe("LayoutLMv3Processor", () => {
    const model_id = "onnx-internal-testing/tiny-random-LayoutLMv3ForTokenClassification-ONNX";

    /** @type {LayoutLMv3Processor} */
    let processor;
    beforeAll(async () => {
      processor = await LayoutLMv3Processor.from_pretrained(model_id);
    }, MAX_PROCESSOR_LOAD_TIME);

    it(
      "default",
      async () => {
        const image = await load_cached_image("tiger");
        const { input_ids, attention_mask, bbox, pixel_values } = await processor(image, "Hello world", {
          boxes: [
            [100, 100, 200, 200],
            [300, 300, 400, 400],
          ],
        });

        expect(input_ids.dims).toEqual([1, 7]);
        expect(attention_mask.dims).toEqual([1, 7]);
        expect(bbox.dims).toEqual([1, 7, 4]);
        expect(pixel_values.dims).toEqual([1, 3, 224, 224]);
        expect(pixel_values.mean().item()).toBeCloseTo(-0.2270686775445938, 6);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it("throws when words are missing", async () => {
      const image = await load_cached_image("tiger");
      await expect(processor(image)).rejects.toThrow("LayoutLMv3Processor requires `text` and `boxes` because OCR is not supported in Transformers.js.");
    });
  });
};
