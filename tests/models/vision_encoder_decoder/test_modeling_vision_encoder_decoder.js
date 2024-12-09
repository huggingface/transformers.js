import { GPT2Tokenizer, VisionEncoderDecoderModel, RawImage, full } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  describe("VisionEncoderDecoderModel", () => {
    const model_id = "hf-internal-testing/tiny-random-VisionEncoderDecoderModel-vit-gpt2";

    /** @type {VisionEncoderDecoderModel} */
    let model;
    /** @type {GPT2Tokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await VisionEncoderDecoderModel.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      tokenizer = await GPT2Tokenizer.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "batch_size=1",
      async () => {
        const outputs = await model.generate({
          pixel_values: full([1, 3, 30, 30], -1.0),
          max_length: 5,
        });
        expect(outputs.tolist()).toEqual([[0n, 400n, 400n, 400n, 400n]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    // TODO: Add back
    // it('batch_size>1', async () => {
    //     const outputs = await model.generate({
    //         pixel_values: cat([
    //             full([1, 3, 30, 30], -1.0),
    //             full([1, 3, 30, 30], 0.0),
    //         ]),
    //         max_length: 5,
    //     });
    //     expect(outputs.tolist()).toEqual([
    //         // Generation continues
    //         [0n, 400n, 400n, 400n, 400n],

    //         // Finishes early. 1023 is the padding token
    //         [0n, 0n, 1023n, 1023n, 1023n],
    //     ]);
    // }, MAX_TEST_EXECUTION_TIME);

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
