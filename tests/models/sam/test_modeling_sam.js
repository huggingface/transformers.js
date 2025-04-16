import { SamProcessor, SamModel } from "../../../src/transformers.js";
import { load_cached_image } from "../../asset_cache.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  describe("SamModel", () => {
    const model_id = "Xenova/slimsam-77-uniform";

    /** @type {SamModel} */
    let model;
    /** @type {SamProcessor} */
    let processor;
    beforeAll(async () => {
      model = await SamModel.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      processor = await SamProcessor.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "w/ input_points",
      async () => {
        // Prepare image and input points
        const raw_image = await load_cached_image("corgi");
        const input_points = [[[340, 250]]];

        // Process inputs and perform mask generation
        const inputs = await processor(raw_image, { input_points });
        const { pred_masks, iou_scores } = await model(inputs);

        expect(pred_masks.dims).toEqual([1, 1, 3, 256, 256]);
        expect(pred_masks.mean().item()).toBeCloseTo(-5.769824981689453, 3);
        expect(iou_scores.dims).toEqual([1, 1, 3]);
        expect(iou_scores.tolist()).toBeCloseToNested([[[0.8583833575248718, 0.9773167967796326, 0.8511142730712891]]]);

        // Post-process masks
        const masks = await processor.post_process_masks(pred_masks, inputs.original_sizes, inputs.reshaped_input_sizes);
        expect(masks).toHaveLength(1);
        expect(masks[0].dims).toEqual([1, 3, 410, 614]);
        expect(masks[0].type).toEqual("bool");
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
