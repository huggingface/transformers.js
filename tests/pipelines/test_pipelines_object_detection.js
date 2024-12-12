import { pipeline, ObjectDetectionPipeline } from "../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../init.js";
import { load_cached_image } from "../asset_cache.js";

const PIPELINE_ID = "object-detection";

export default () => {
  describe("Object Detection", () => {
    const model_id = "Xenova/yolos-tiny";
    /** @type {ObjectDetectionPipeline} */
    let pipe;
    beforeAll(async () => {
      pipe = await pipeline(PIPELINE_ID, model_id, DEFAULT_MODEL_OPTIONS);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "single + threshold",
      async () => {
        const image = await load_cached_image("cats");
        const output = await pipe(image, { threshold: 0.9 });

        const target = [
          {
            score: 0.9921281933784485,
            label: "remote",
            box: { xmin: 32, ymin: 78, xmax: 185, ymax: 117 },
          },
          {
            score: 0.9884883165359497,
            label: "remote",
            box: { xmin: 324, ymin: 82, xmax: 376, ymax: 191 },
          },
          {
            score: 0.9197800159454346,
            label: "cat",
            box: { xmin: 5, ymin: 56, xmax: 321, ymax: 469 },
          },
          {
            score: 0.9300552606582642,
            label: "cat",
            box: { xmin: 332, ymin: 25, xmax: 638, ymax: 369 },
          },
        ];
        expect(output).toBeCloseToNested(target, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await pipe.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
