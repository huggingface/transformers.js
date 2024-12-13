import { pipeline, TokenClassificationPipeline } from "../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../init.js";

const PIPELINE_ID = "token-classification";

export default () => {
  describe("Token Classification", () => {
    const model_id = "hf-internal-testing/tiny-random-BertForTokenClassification";
    /** @type {TokenClassificationPipeline} */
    let pipe;
    beforeAll(async () => {
      pipe = await pipeline(PIPELINE_ID, model_id, DEFAULT_MODEL_OPTIONS);
    }, MAX_MODEL_LOAD_TIME);

    it("should be an instance of TokenClassificationPipeline", () => {
      expect(pipe).toBeInstanceOf(TokenClassificationPipeline);
    });

    describe("batch_size=1", () => {
      it(
        "default",
        async () => {
          const output = await pipe("1 2 3");

          // TODO: Add start/end to target
          const target = [
            {
              entity: "LABEL_0",
              score: 0.5292708,
              index: 1,
              word: "1",
              // 'start': 0, 'end': 1
            },
            {
              entity: "LABEL_0",
              score: 0.5353687,
              index: 2,
              word: "2",
              // 'start': 2, 'end': 3
            },
            {
              entity: "LABEL_1",
              score: 0.51381934,
              index: 3,
              word: "3",
              // 'start': 4, 'end': 5
            },
          ];
          expect(output).toBeCloseToNested(target, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it(
        "custom (ignore_labels set)",
        async () => {
          const output = await pipe("1 2 3", { ignore_labels: ["LABEL_0"] });
          const target = [
            {
              entity: "LABEL_1",
              score: 0.51381934,
              index: 3,
              word: "3",
              // 'start': 4, 'end': 5
            },
          ];
          expect(output).toBeCloseToNested(target, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    describe("batch_size>1", () => {
      it(
        "default",
        async () => {
          const output = await pipe(["1 2 3", "4 5"]);
          const target = [
            [
              {
                entity: "LABEL_0",
                score: 0.5292708,
                index: 1,
                word: "1",
                // 'start': 0, 'end': 1
              },
              {
                entity: "LABEL_0",
                score: 0.5353687,
                index: 2,
                word: "2",
                // 'start': 2, 'end': 3
              },
              {
                entity: "LABEL_1",
                score: 0.51381934,
                index: 3,
                word: "3",
                // 'start': 4, 'end': 5
              },
            ],
            [
              {
                entity: "LABEL_0",
                score: 0.5432807,
                index: 1,
                word: "4",
                // 'start': 0, 'end': 1
              },
              {
                entity: "LABEL_1",
                score: 0.5007693,
                index: 2,
                word: "5",
                // 'start': 2, 'end': 3
              },
            ],
          ];
          expect(output).toBeCloseToNested(target, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it(
        "custom (ignore_labels set)",
        async () => {
          const output = await pipe(["1 2 3", "4 5"], { ignore_labels: ["LABEL_0"] });
          const target = [
            [
              {
                entity: "LABEL_1",
                score: 0.51381934,
                index: 3,
                word: "3",
                // 'start': 4, 'end': 5
              },
            ],
            [
              {
                entity: "LABEL_1",
                score: 0.5007693,
                index: 2,
                word: "5",
                // 'start': 2, 'end': 3
              },
            ],
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
