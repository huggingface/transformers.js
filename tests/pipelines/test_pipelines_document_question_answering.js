import { pipeline, DocumentQuestionAnsweringPipeline, RawImage } from "../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../init.js";

const PIPELINE_ID = "document-question-answering";

export default () => {
  describe("Document Question Answering", () => {
    const model_id = "hf-internal-testing/tiny-random-VisionEncoderDecoderModel-donutswin-mbart";

    /** @type {DocumentQuestionAnsweringPipeline} */
    let pipe;
    beforeAll(async () => {
      pipe = await pipeline(PIPELINE_ID, model_id, DEFAULT_MODEL_OPTIONS);
    }, MAX_MODEL_LOAD_TIME);

    it("should be an instance of DocumentQuestionAnsweringPipeline", () => {
      expect(pipe).toBeInstanceOf(DocumentQuestionAnsweringPipeline);
    });

    describe("batch_size=1", () => {
      it(
        "default",
        async () => {
          const dims = [64, 32, 3];
          const image = new RawImage(new Uint8ClampedArray(dims[0] * dims[1] * dims[2]).fill(255), ...dims);
          const question = "What is the invoice number?";
          const output = await pipe(image, question);

          const target = [{ answer: null }];
          expect(output).toEqual(target);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    afterAll(async () => {
      await pipe.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
