import { pipeline, TextClassificationPipeline } from "../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../init.js";

const PIPELINE_ID = "text-classification";

export default () => {
  describe("Text Classification", () => {
    const model_id = "hf-internal-testing/tiny-random-BertForSequenceClassification";

    /** @type {TextClassificationPipeline} */
    let pipe;
    beforeAll(async () => {
      pipe = await pipeline(PIPELINE_ID, model_id, DEFAULT_MODEL_OPTIONS);
    }, MAX_MODEL_LOAD_TIME);

    it("should be an instance of TextClassificationPipeline", () => {
      expect(pipe).toBeInstanceOf(TextClassificationPipeline);
    });

    describe("batch_size=1", () => {
      it(
        "default (top_k=1)",
        async () => {
          const output = await pipe("a");
          const target = [{ label: "LABEL_0", score: 0.5076976418495178 }];
          expect(output).toBeCloseToNested(target, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it(
        "custom (top_k=2)",
        async () => {
          const output = await pipe("a", { top_k: 2 });
          const target = [
            { label: "LABEL_0", score: 0.5076976418495178 },
            { label: "LABEL_1", score: 0.49230238795280457 },
          ];
          expect(output).toBeCloseToNested(target, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    describe("batch_size>1", () => {
      it(
        "default (top_k=1)",
        async () => {
          const output = await pipe(["a", "b c"]);
          const target = [
            { label: "LABEL_0", score: 0.5076976418495178 },
            { label: "LABEL_0", score: 0.5077522993087769 },
          ];
          expect(output).toBeCloseToNested(target, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it(
        "custom (top_k=2)",
        async () => {
          const output = await pipe(["a", "b c"], { top_k: 2 });
          const target = [
            [
              { label: "LABEL_0", score: 0.5076976418495178 },
              { label: "LABEL_1", score: 0.49230238795280457 },
            ],
            [
              { label: "LABEL_0", score: 0.5077522993087769 },
              { label: "LABEL_1", score: 0.49224773049354553 },
            ],
          ];
          expect(output).toBeCloseToNested(target, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "multi_label_classification",
        async () => {
          const problem_type = pipe.model.config.problem_type;
          pipe.model.config.problem_type = "multi_label_classification";

          const output = await pipe(["a", "b c"], { top_k: 2 });
          const target = [
            [
              { label: "LABEL_0", score: 0.5001373887062073 },
              { label: "LABEL_1", score: 0.49243971705436707 },
            ],
            [
              { label: "LABEL_0", score: 0.5001326203346252 },
              { label: "LABEL_1", score: 0.492380291223526 },
            ],
          ];
          expect(output).toBeCloseToNested(target, 5);

          // Reset problem type
          pipe.model.config.problem_type = problem_type;
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    afterAll(async () => {
      await pipe?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });

  describe("Text Classification (single logit)", () => {
    // A cross-encoder has one logit per input. Expected scores are from the Python pipeline
    // with cross-encoder/ms-marco-TinyBERT-L2-v2.
    const model_id = "Xenova/ms-marco-TinyBERT-L-2-v2";
    const texts = ["How many people live in Berlin? Berlin is well known for its museums.", "purple elephant"];

    /** @type {TextClassificationPipeline} */
    let pipe;
    beforeAll(async () => {
      pipe = await pipeline(PIPELINE_ID, model_id, DEFAULT_MODEL_OPTIONS);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "applies sigmoid",
      async () => {
        const output = await pipe(texts);
        const target = [
          { label: "LABEL_0", score: 0.00010145680425921455 },
          { label: "LABEL_0", score: 0.06936494261026382 },
        ];
        expect(output).toBeCloseToNested(target, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "regression",
      async () => {
        const problem_type = pipe.model.config.problem_type;
        pipe.model.config.problem_type = "regression";

        const output = await pipe(texts);
        const target = [
          { label: "LABEL_0", score: -9.195775985717773 },
          { label: "LABEL_0", score: -2.5964856147766113 },
        ];
        expect(output).toBeCloseToNested(target, 4);

        // Reset problem type
        pipe.model.config.problem_type = problem_type;
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await pipe?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
