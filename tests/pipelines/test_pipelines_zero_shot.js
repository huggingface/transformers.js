import { pipeline, ZeroShotClassificationPipeline } from "../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../init.js";

const PIPELINE_ID = "zero-shot-classification";

export default () => {
  describe("Zero-shot Classification", () => {
    const model_id = "hf-internal-testing/tiny-random-BertForSequenceClassification";
    /** @type {ZeroShotClassificationPipeline} */
    let pipe;

    beforeAll(async () => {
      pipe = await pipeline(PIPELINE_ID, model_id, {
        ...DEFAULT_MODEL_OPTIONS,

        // The model isn't designed for zero-shot classification, so we set the config
        config: {
          model_type: "bert",
          id2label: {
            0: "contradiction",
            1: "entailment",
          },
          label2id: {
            contradiction: 0,
            entailment: 1,
          },
        },
      });
    }, MAX_MODEL_LOAD_TIME);

    it("should be an instance of ZeroShotClassificationPipeline", () => {
      expect(pipe).toBeInstanceOf(ZeroShotClassificationPipeline);
    });
    const sequences_to_classify = ["one day I will see the world", "I love making pizza"];
    const candidate_labels = ["travel", "cooking", "dancing"];

    it(
      "Single sequence classification",
      async () => {
        const output = await pipe(sequences_to_classify[0], candidate_labels);
        const target = {
          sequence: "one day I will see the world",
          labels: ["dancing", "cooking", "travel"],
          scores: [0.3333353410546293, 0.3333348269618681, 0.3333298319835025],
        };
        expect(output).toBeCloseToNested(target, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "Batched classification",
      async () => {
        const output = await pipe(sequences_to_classify, candidate_labels);
        const target = [
          {
            sequence: "one day I will see the world",
            labels: ["dancing", "cooking", "travel"],
            scores: [0.3333353410546293, 0.3333348269618681, 0.3333298319835025],
          },
          {
            sequence: "I love making pizza",
            labels: ["dancing", "cooking", "travel"],
            scores: [0.3333347058960895, 0.3333337292465588, 0.3333315648573516],
          },
        ];
        expect(output).toBeCloseToNested(target, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "Batched + multilabel classification",
      async () => {
        const candidate_labels = ["travel", "cooking", "dancing"];

        const output = await pipe(sequences_to_classify, candidate_labels, { multi_label: true });
        const target = [
          {
            sequence: "one day I will see the world",
            labels: ["dancing", "cooking", "travel"],
            scores: [0.49231469615364476, 0.4923134953805702, 0.4923094795142658],
          },
          {
            sequence: "I love making pizza",
            labels: ["dancing", "cooking", "travel"],
            scores: [0.49230751217535645, 0.49230615475943956, 0.4923042569480609],
          },
        ];
        expect(output).toBeCloseToNested(target, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await pipe.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
