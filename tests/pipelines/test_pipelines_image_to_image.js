import { pipeline, ImageToImagePipeline } from "../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../init.js";
import { load_cached_image } from "../asset_cache.js";

const PIPELINE_ID = "image-to-image";

export default () => {
  describe("Image to Image", () => {
    const model_id = "hf-internal-testing/tiny-random-Swin2SRForImageSuperResolution";
    /** @type {ImageToImagePipeline} */
    let pipe;
    let images;
    beforeAll(async () => {
      pipe = await pipeline(PIPELINE_ID, model_id, DEFAULT_MODEL_OPTIONS);
      images = await Promise.all([load_cached_image("white_image"), load_cached_image("blue_image")]);
    }, MAX_MODEL_LOAD_TIME);

    it("should be an instance of ImageToImagePipeline", () => {
      expect(pipe).toBeInstanceOf(ImageToImagePipeline);
    });

    describe("batch_size=1", () => {
      it(
        "default",
        async () => {
          const output = await pipe(images[0]);
          expect(output.size).toEqual([64, 64]);
          expect(output.channels).toEqual(3);
          expect(output.data.reduce((a, b) => a + b, 0) / output.data.length).toBeCloseTo(110.107421875, 3);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    describe("batch_size>1", () => {
      it(
        "default",
        async () => {
          const output = await pipe(images);
          expect(output[0].size).toEqual([64, 64]);
          expect(output[0].channels).toEqual(3);
          expect(output[0].data.reduce((a, b) => a + b, 0) / output[0].data.length).toBeCloseTo(110.107421875, 3);
          expect(output[1].size).toEqual([64, 64]);
          expect(output[1].channels).toEqual(3);
          expect(output[1].data.reduce((a, b) => a + b, 0) / output[1].data.length).toBeCloseTo(110.60196940104167, 3);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    afterAll(async () => {
      await pipe.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
