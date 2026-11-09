import { AutoImageProcessor, LayoutLMv3FeatureExtractor } from "../../../src/transformers.js";

import { load_cached_image } from "../../asset_cache.js";
import { MAX_PROCESSOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  describe("LayoutLMv3FeatureExtractor", () => {
    const model_id = "onnx-internal-testing/tiny-random-LayoutLMv3ForTokenClassification-ONNX";

    /** @type {LayoutLMv3FeatureExtractor} */
    let processor;
    beforeAll(async () => {
      processor = await AutoImageProcessor.from_pretrained(model_id);
    }, MAX_PROCESSOR_LOAD_TIME);

    it(
      "default",
      async () => {
        const image = await load_cached_image("tiger");
        const { pixel_values } = await processor(image);

        expect(pixel_values.dims).toEqual([1, 3, 224, 224]);
        expect(pixel_values.mean().item()).toBeCloseTo(-0.2270686775445938, 6);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
