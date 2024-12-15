import { pipeline, QuestionAnsweringPipeline } from "../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../init.js";

const PIPELINE_ID = "question-answering";

export default () => {
  describe("Question Answering", () => {
    const model_id = "hf-internal-testing/tiny-random-BertForQuestionAnswering";
    /** @type {QuestionAnsweringPipeline} */
    let pipe;
    beforeAll(async () => {
      pipe = await pipeline(PIPELINE_ID, model_id, DEFAULT_MODEL_OPTIONS);
    }, MAX_MODEL_LOAD_TIME);

    it("should be an instance of QuestionAnsweringPipeline", () => {
      expect(pipe).toBeInstanceOf(QuestionAnsweringPipeline);
    });

    describe("batch_size=1", () => {
      it(
        "default (top_k=1)",
        async () => {
          const output = await pipe("a", "b c");
          const target = { score: 0.11395696550607681, /* start: 0, end: 1, */ answer: "b" };
          expect(output).toBeCloseToNested(target, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it(
        "custom (top_k=3)",
        async () => {
          const output = await pipe("a", "b c", { top_k: 3 });
          const target = [
            { score: 0.11395696550607681, /* start: 0, end: 1, */ answer: "b" },
            { score: 0.11300431191921234, /* start: 2, end: 3, */ answer: "c" },
            { score: 0.10732574015855789, /* start: 0, end: 3, */ answer: "b c" },
          ];
          expect(output).toBeCloseToNested(target, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    afterAll(async () => {
      await pipe.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
