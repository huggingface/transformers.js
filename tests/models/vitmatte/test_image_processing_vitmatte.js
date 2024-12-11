import { AutoImageProcessor, VitMatteImageProcessor } from "../../../src/transformers.js";

import { load_cached_image } from "../../asset_cache.js";
import { MAX_PROCESSOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  // VitMatteImageProcessor
  //  - tests custom overrides
  //  - tests multiple inputs
  //  - tests `size_divisibility` and no size (size_divisibility=32)
  //  - tests do_pad and `size_divisibility`
  describe("VitMatteImageProcessor", () => {
    const model_id = "Xenova/vitmatte-small-distinctions-646";

    /** @type {VitMatteImageProcessor} */
    let processor;
    beforeAll(async () => {
      processor = await AutoImageProcessor.from_pretrained(model_id);
    }, MAX_PROCESSOR_LOAD_TIME);

    it(
      "w/o resize",
      async () => {
        const image = await load_cached_image("vitmatte_image");
        const image2 = await load_cached_image("vitmatte_trimap");
        const { pixel_values, original_sizes, reshaped_input_sizes } = await processor(image, image2);
        const { data, dims } = pixel_values;

        expect(dims).toEqual([1, 4, 640, 960]);
        expect(pixel_values.mean().item()).toBeCloseTo(-0.4028555154800415);
        expect(data[0]).toBeCloseTo(-0.9921568632125854);
        expect(data[1]).toBeCloseTo(-0.9921568632125854);
        expect(data[5]).toBeCloseTo(-1.0);
        expect(data[640]).toBeCloseTo(-0.6784313917160034);
        expect(data[641]).toBeCloseTo(-0.6705882549285889);
        expect(data[640 * 960]).toBeCloseTo(-1.0);
        expect(data[640 * 960 + 1]).toBeCloseTo(-1.0);
        expect(data.at(-1)).toBeCloseTo(0.0);

        expect(original_sizes).toEqual([[640, 960]]);
        expect(reshaped_input_sizes).toEqual([[640, 960]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "w/ resize",
      async () => {
        const image = await load_cached_image("pattern_3x5");
        const image2 = await load_cached_image("pattern_3x5");
        const { pixel_values, original_sizes, reshaped_input_sizes } = await processor(image, image2);
        const { data, dims } = pixel_values;
        expect(dims).toEqual([1, 4, 32, 32]);
        expect(pixel_values.mean().item()).toBeCloseTo(-0.00867417361587286);
        expect(data[0]).toBeCloseTo(-0.9921568632125854);
        expect(data[1]).toBeCloseTo(-0.9686274528503418);
        expect(data[5]).toBeCloseTo(0.0);
        expect(data[32]).toBeCloseTo(-0.9215686321258545);
        expect(data[33]).toBeCloseTo(-0.8980392217636108);
        expect(data.at(-1)).toBeCloseTo(0.0);

        expect(original_sizes).toEqual([[5, 3]]);
        expect(reshaped_input_sizes).toEqual([[5, 3]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
