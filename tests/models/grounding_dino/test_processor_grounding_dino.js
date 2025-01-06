import { AutoProcessor, full, GroundingDinoProcessor } from "../../../src/transformers.js";

import { load_cached_image } from "../../asset_cache.js";
import { MAX_PROCESSOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  const model_id = "hf-internal-testing/tiny-random-GroundingDinoForObjectDetection";

  describe("GroundingDinoProcessor", () => {
    /** @type {GroundingDinoProcessor} */
    let processor;
    let images = {};

    beforeAll(async () => {
      processor = await AutoProcessor.from_pretrained(model_id);
      images = {
        white_image: await load_cached_image("white_image"),
      };
    }, MAX_PROCESSOR_LOAD_TIME);

    it(
      "Single image & text",
      async () => {
        const { input_ids, pixel_values } = await processor(images.white_image, "a cat.");
        expect(input_ids.dims).toEqual([1, 5]);
        expect(pixel_values.dims).toEqual([1, 3, 800, 800]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "post_process_grounded_object_detection",
      async () => {
        const outputs = {
          logits: full([1, 900, 256], 0.5),
          pred_boxes: full([1, 900, 4], 0.5),
        };
        const inputs = {
          input_ids: full([1, 5], 1n),
        };

        const results = processor.post_process_grounded_object_detection(outputs, inputs.input_ids, {
          box_threshold: 0.3,
          text_threshold: 0.3,
          target_sizes: [[360, 240]],
        });
        const { scores, boxes, labels } = results[0];
        expect(scores).toHaveLength(900);
        expect(boxes).toHaveLength(900);
        expect(labels).toHaveLength(900);
        expect(boxes[0]).toEqual([60, 90, 180, 270]);
        expect(scores[0]).toBeCloseTo(0.622459352016449, 6);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
