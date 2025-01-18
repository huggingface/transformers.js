import { Wav2Vec2Processor, MoonshineForConditionalGeneration, full, ones } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  describe("MoonshineForConditionalGeneration", () => {
    const model_id = "hf-internal-testing/tiny-random-MoonshineForConditionalGeneration";

    /** @type {MoonshineForConditionalGeneration} */
    let model;
    /** @type {Wav2Vec2Processor} */
    let processor;
    beforeAll(async () => {
      model = await MoonshineForConditionalGeneration.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      processor = await Wav2Vec2Processor.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    const input_values = new Float32Array(16000);

    it(
      "forward",
      async () => {
        const inputs = await processor(input_values);
        const { logits } = await model({
          ...inputs,
          decoder_input_ids: ones([1, 1]),
        });
        expect(logits.dims).toEqual([1, 1, 32768]);
        expect(logits.mean().item()).toBeCloseTo(0.016709428280591965, 6);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "batch_size=1",
      async () => {
        const inputs = await processor(input_values);
        const generate_ids = await model.generate({ ...inputs, max_new_tokens: 3 });

        const new_tokens = generate_ids;
        expect(new_tokens.tolist()).toEqual([[/* Decoder start token */ 1n, /* Generated */ 6891n, 21892n, 14850n]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
