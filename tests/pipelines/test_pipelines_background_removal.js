import { pipeline, BackgroundRemovalPipeline, RawImage } from "../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../init.js";
import { load_cached_image } from "../asset_cache.js";

const PIPELINE_ID = "background-removal";

export default () => {
  describe("Background Removal", () => {
    describe("Portrait Segmentation", () => {
      const model_id = "Xenova/modnet";
      /** @type {BackgroundRemovalPipeline} */
      let pipe;
      beforeAll(async () => {
        pipe = await pipeline(PIPELINE_ID, model_id, DEFAULT_MODEL_OPTIONS);
      }, MAX_MODEL_LOAD_TIME);

      it("should be an instance of BackgroundRemovalPipeline", () => {
        expect(pipe).toBeInstanceOf(BackgroundRemovalPipeline);
      });

      it(
        "single",
        async () => {
          const image = await load_cached_image("portrait_of_woman");

          const output = await pipe(image);
          expect(output).toHaveLength(1);
          expect(output[0]).toBeInstanceOf(RawImage);
          expect(output[0].width).toEqual(image.width);
          expect(output[0].height).toEqual(image.height);
          expect(output[0].channels).toEqual(4); // With alpha channel
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await pipe.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });

    describe("Selfie Segmentation", () => {
      const model_id = "onnx-community/mediapipe_selfie_segmentation";
      /** @type {BackgroundRemovalPipeline } */
      let pipe;
      beforeAll(async () => {
        pipe = await pipeline(PIPELINE_ID, model_id, DEFAULT_MODEL_OPTIONS);
      }, MAX_MODEL_LOAD_TIME);

      it(
        "single",
        async () => {
          const image = await load_cached_image("portrait_of_woman");

          const output = await pipe(image);
          expect(output).toHaveLength(1);
          expect(output[0]).toBeInstanceOf(RawImage);
          expect(output[0].width).toEqual(image.width);
          expect(output[0].height).toEqual(image.height);
          expect(output[0].channels).toEqual(4); // With alpha channel
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await pipe.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });
  });
};
