import { pipeline, ImageFeatureExtractionPipeline } from "../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../init.js";
import { load_cached_image } from "../asset_cache.js";

const PIPELINE_ID = "image-feature-extraction";

export default () => {
  describe("Image Feature Extraction", () => {
    const model_id = "hf-internal-testing/tiny-random-ViTMAEModel";
    /** @type {ImageFeatureExtractionPipeline} */
    let pipe;
    let images;
    beforeAll(async () => {
      pipe = await pipeline(PIPELINE_ID, model_id, DEFAULT_MODEL_OPTIONS);
      images = await Promise.all([load_cached_image("white_image"), load_cached_image("blue_image")]);
    }, MAX_MODEL_LOAD_TIME);

    it("should be an instance of ImageFeatureExtractionPipeline", () => {
      expect(pipe).toBeInstanceOf(ImageFeatureExtractionPipeline);
    });

    describe("batch_size=1", () => {
      it(
        "default",
        async () => {
          const output = await pipe(images[0]);
          expect(output.dims).toEqual([1, 91, 32]);
          expect(output.mean().item()).toBeCloseTo(-8.507473614471905e-10, 6);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    describe("batch_size>1", () => {
      it(
        "default",
        async () => {
          const output = await pipe(images);
          expect(output.dims).toEqual([images.length, 91, 32]);
          expect(output.mean().item()).toBeCloseTo(-5.997602414709036e-10, 6);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    afterAll(async () => {
      await pipe.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
