import { pipeline, DepthEstimationPipeline } from "../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../init.js";
import { load_cached_image } from "../asset_cache.js";

const PIPELINE_ID = "depth-estimation";

export default () => {
  describe("Depth Estimation", () => {
    const model_id = "hf-internal-testing/tiny-random-DPTForDepthEstimation";
    /** @type {DepthEstimationPipeline} */
    let pipe;
    let images;
    beforeAll(async () => {
      pipe = await pipeline(PIPELINE_ID, model_id, DEFAULT_MODEL_OPTIONS);
      images = await Promise.all([load_cached_image("white_image"), load_cached_image("blue_image")]);
    }, MAX_MODEL_LOAD_TIME);

    it("should be an instance of DepthEstimationPipeline", () => {
      expect(pipe).toBeInstanceOf(DepthEstimationPipeline);
    });

    describe("batch_size=1", () => {
      it(
        "default",
        async () => {
          const output = await pipe(images[0]);
          expect(output.predicted_depth.dims).toEqual([224, 224]);
          expect(output.predicted_depth.mean().item()).toBeCloseTo(0.000006106501587055391, 6);
          expect(output.depth.size).toEqual(images[0].size);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    describe("batch_size>1", () => {
      it(
        "default",
        async () => {
          const output = await pipe(images);
          expect(output).toHaveLength(images.length);
          expect(output[0].predicted_depth.dims).toEqual([224, 224]);
          expect(output[0].predicted_depth.mean().item()).toBeCloseTo(0.000006106501587055391, 6);
          expect(output[0].depth.size).toEqual(images[0].size);
          expect(output[1].predicted_depth.dims).toEqual([224, 224]);
          expect(output[1].predicted_depth.mean().item()).toBeCloseTo(0.0000014548650142387487, 6);
          expect(output[1].depth.size).toEqual(images[1].size);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    afterAll(async () => {
      await pipe.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
