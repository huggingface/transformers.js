import { PaliGemmaProcessor, PaliGemmaForConditionalGeneration, RawImage } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  const text = "<image>What is on the flower?";

  // Empty white image
  const dims = [224, 224, 3];
  const image = new RawImage(new Uint8ClampedArray(dims[0] * dims[1] * dims[2]).fill(255), ...dims);

  describe("PaliGemmaForConditionalGeneration", () => {
    const model_id = "hf-internal-testing/tiny-random-PaliGemmaForConditionalGeneration";

    /** @type {PaliGemmaForConditionalGeneration} */
    let model;
    /** @type {PaliGemmaProcessor} */
    let processor;
    beforeAll(async () => {
      model = await PaliGemmaForConditionalGeneration.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      processor = await PaliGemmaProcessor.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "forward",
      async () => {
        const inputs = await processor(image, text);

        const { logits } = await model(inputs);
        expect(logits.dims).toEqual([1, 264, 257216]);
        expect(logits.mean().item()).toBeCloseTo(-0.0023024685215204954, 6);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "batch_size=1",
      async () => {
        const inputs = await processor(image, text);
        const generate_ids = await model.generate({ ...inputs, max_new_tokens: 10 });

        const new_tokens = generate_ids.slice(null, [inputs.input_ids.dims.at(-1), null]);
        expect(new_tokens.tolist()).toEqual([[91711n, 24904n, 144054n, 124983n, 83862n, 124983n, 124983n, 124983n, 141236n, 124983n]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
