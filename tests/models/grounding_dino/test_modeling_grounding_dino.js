import { GroundingDinoProcessor, GroundingDinoForObjectDetection, RawImage } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  const text = "a cat."; // NB: text query needs to be lowercased + end with a dot

  // Empty white image
  const dims = [224, 224, 3];
  const image = new RawImage(new Uint8ClampedArray(dims[0] * dims[1] * dims[2]).fill(255), ...dims);

  describe("GroundingDinoForObjectDetection", () => {
    const model_id = "hf-internal-testing/tiny-random-GroundingDinoForObjectDetection";

    /** @type {GroundingDinoForObjectDetection} */
    let model;
    /** @type {GroundingDinoProcessor} */
    let processor;
    beforeAll(async () => {
      model = await GroundingDinoForObjectDetection.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      processor = await GroundingDinoProcessor.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "forward",
      async () => {
        const inputs = await processor(image, text);
        const { d_model, num_queries } = model.config;

        const { logits, pred_boxes } = await model(inputs);
        expect(logits.dims).toEqual([1, num_queries, d_model]);
        expect(pred_boxes.dims).toEqual([1, num_queries, 4]);
        expect(logits.max().item()).toBeCloseTo(56.237613677978516, 2);
        expect(logits.min().item()).toEqual(-Infinity);
        expect(pred_boxes.mean().item()).toEqual(0.2500016987323761);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
