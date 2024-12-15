import { pipeline, SummarizationPipeline } from "../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../init.js";

const PIPELINE_ID = "summarization";

export default () => {
  describe("Summarization", () => {
    const model_id = "hf-internal-testing/tiny-random-T5ForConditionalGeneration";

    /** @type {SummarizationPipeline} */
    let pipe;
    beforeAll(async () => {
      pipe = await pipeline(PIPELINE_ID, model_id, DEFAULT_MODEL_OPTIONS);
    }, MAX_MODEL_LOAD_TIME);

    it("should be an instance of SummarizationPipeline", () => {
      expect(pipe).toBeInstanceOf(SummarizationPipeline);
    });

    describe("batch_size=1", () => {
      it(
        "default",
        async () => {
          const text = "This is a test.";
          const output = await pipe(text, {
            max_new_tokens: 5,
          });
          const target = [{ summary_text: "" }];
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
