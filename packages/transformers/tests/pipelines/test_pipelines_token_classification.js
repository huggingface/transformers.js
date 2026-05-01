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
              start: 0,
              end: 1,
            },
            {
              entity: "LABEL_0",
              score: 0.5353687,
              index: 2,
              word: "2",
              start: 2,
              end: 3,
            },
            {
              entity: "LABEL_1",
              score: 0.51381934,
              index: 3,
              word: "3",
              start: 4,
              end: 5,
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
              start: 4,
              end: 5,
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
                start: 0,
                end: 1,
              },
              {
                entity: "LABEL_0",
                score: 0.5353687,
                index: 2,
                word: "2",
                start: 2,
                end: 3,
              },
              {
                entity: "LABEL_1",
                score: 0.51381934,
                index: 3,
                word: "3",
                start: 4,
                end: 5,
              },
            ],
            [
              {
                entity: "LABEL_0",
                score: 0.5432807,
                index: 1,
                word: "4",
                start: 0,
                end: 1,
              },
              {
                entity: "LABEL_1",
                score: 0.5007693,
                index: 2,
                word: "5",
                start: 2,
                end: 3,
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
                start: 4,
                end: 5,
              },
            ],
            [
              {
                entity: "LABEL_1",
                score: 0.5007693,
                index: 2,
                word: "5",
                start: 2,
                end: 3,
              },
            ],
          ];
          expect(output).toBeCloseToNested(target, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    afterAll(async () => {
      await pipe?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });

  describe("Token Classification w/ aggregation_strategy", () => {
    const model_id = "hf-internal-testing/tiny-random-BertForTokenClassification";
    const inputs = ["1 2 3", "4 5"];

    describe("BIO (B-PER / I-PER)", () => {
      /** @type {TokenClassificationPipeline} */
      let pipe;
      beforeAll(async () => {
        pipe = await pipeline(PIPELINE_ID, model_id, {
          ...DEFAULT_MODEL_OPTIONS,
          config: {
            model_type: "bert",
            id2label: { 0: "B-PER", 1: "I-PER" },
            label2id: { "B-PER": 0, "I-PER": 1 },
          },
        });
      }, MAX_MODEL_LOAD_TIME);

      it(
        "aggregation_strategy='simple' merges adjacent same-tag tokens",
        async () => {
          const output = await pipe(inputs, { aggregation_strategy: "simple" });
          const target = [
            [
              { entity_group: "PER", score: 0.5292708, word: "1", start: 0, end: 1 },
              { entity_group: "PER", score: 0.524594, word: "2 3", start: 2, end: 5 },
            ],
            [{ entity_group: "PER", score: 0.52202505, word: "4 5", start: 0, end: 3 }],
          ];
          expect(output).toBeCloseToNested(target, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await pipe?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });

    describe("BIO with O (O / B-PER)", () => {
      /** @type {TokenClassificationPipeline} */
      let pipe;
      beforeAll(async () => {
        pipe = await pipeline(PIPELINE_ID, model_id, {
          ...DEFAULT_MODEL_OPTIONS,
          config: {
            model_type: "bert",
            id2label: { 0: "O", 1: "B-PER" },
            label2id: { O: 0, "B-PER": 1 },
          },
        });
      }, MAX_MODEL_LOAD_TIME);

      it(
        "aggregation_strategy='simple' drops O-labeled tokens",
        async () => {
          const output = await pipe(inputs, { aggregation_strategy: "simple" });
          const target = [
            [{ entity_group: "PER", score: 0.51381934, word: "3", start: 4, end: 5 }],
            [{ entity_group: "PER", score: 0.5007693, word: "5", start: 2, end: 3 }],
          ];
          expect(output).toBeCloseToNested(target, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "aggregation_strategy='simple' with ignore_labels=[] keeps O as a group",
        async () => {
          const output = await pipe(inputs, { aggregation_strategy: "simple", ignore_labels: [] });
          const target = [
            [
              { entity_group: "O", score: 0.5323198, word: "1 2", start: 0, end: 3 },
              { entity_group: "PER", score: 0.51381934, word: "3", start: 4, end: 5 },
            ],
            [
              { entity_group: "O", score: 0.5432808, word: "4", start: 0, end: 1 },
              { entity_group: "PER", score: 0.5007693, word: "5", start: 2, end: 3 },
            ],
          ];
          expect(output).toBeCloseToNested(target, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await pipe?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });

    describe("BIOES (E-PER / B-PER)", () => {
      /** @type {TokenClassificationPipeline} */
      let pipe;
      beforeAll(async () => {
        pipe = await pipeline(PIPELINE_ID, model_id, {
          ...DEFAULT_MODEL_OPTIONS,
          config: {
            model_type: "bert",
            id2label: { 0: "E-PER", 1: "B-PER" },
            label2id: { "E-PER": 0, "B-PER": 1 },
          },
        });
      }, MAX_MODEL_LOAD_TIME);

      it(
        "aggregation_strategy='simple' treats E- as continuation, B- opens a new group, E- terminates",
        async () => {
          const output = await pipe(inputs, { aggregation_strategy: "simple" });
          // Labels for `1 2 3`: [E-PER, E-PER, B-PER].
          //   E-PER at the start opens a group (best-effort recovery for a malformed sequence),
          //   the second E-PER continues + closes it, then B-PER opens a fresh group.
          // Labels for `4 5`: [E-PER, B-PER].
          const target = [
            [
              { entity_group: "PER", score: 0.5323198, word: "1 2", start: 0, end: 3 },
              { entity_group: "PER", score: 0.51381934, word: "3", start: 4, end: 5 },
            ],
            [
              { entity_group: "PER", score: 0.5432808, word: "4", start: 0, end: 1 },
              { entity_group: "PER", score: 0.5007693, word: "5", start: 2, end: 3 },
            ],
          ];
          expect(output).toBeCloseToNested(target, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await pipe?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });

    describe("BIOES terminator (E- closes a group even when tag matches)", () => {
      /** @type {TokenClassificationPipeline} */
      let pipe;
      beforeAll(async () => {
        // With this remap, the tiny model assigns `I-PER` to tokens "1" and "2"
        // and `E-PER` to token "3". The E- must close the group, so a second
        // sentence of `I-PER, E-PER` (tokens "4 5") forms its own group.
        pipe = await pipeline(PIPELINE_ID, model_id, {
          ...DEFAULT_MODEL_OPTIONS,
          config: {
            model_type: "bert",
            id2label: { 0: "I-PER", 1: "E-PER" },
            label2id: { "I-PER": 0, "E-PER": 1 },
          },
        });
      }, MAX_MODEL_LOAD_TIME);

      it(
        "aggregation_strategy='simple' folds I-* / E-* into one group per terminator",
        async () => {
          const output = await pipe(inputs, { aggregation_strategy: "simple" });
          const target = [
            [{ entity_group: "PER", score: 0.52614963, word: "1 2 3", start: 0, end: 5 }],
            [{ entity_group: "PER", score: 0.522025, word: "4 5", start: 0, end: 3 }],
          ];
          expect(output).toBeCloseToNested(target, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await pipe?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });

    describe("BIOES singleton (S-)", () => {
      /** @type {TokenClassificationPipeline} */
      let pipe;
      beforeAll(async () => {
        // `S-PER` on every token — each one must be its own group.
        pipe = await pipeline(PIPELINE_ID, model_id, {
          ...DEFAULT_MODEL_OPTIONS,
          config: {
            model_type: "bert",
            id2label: { 0: "S-PER", 1: "S-PER" },
            label2id: { "S-PER": 0 },
          },
        });
      }, MAX_MODEL_LOAD_TIME);

      it(
        "aggregation_strategy='simple' keeps each S- token as its own group",
        async () => {
          const output = await pipe(inputs, { aggregation_strategy: "simple" });
          const target = [
            [
              { entity_group: "PER", score: 0.5292708, word: "1", start: 0, end: 1 },
              { entity_group: "PER", score: 0.5353687, word: "2", start: 2, end: 3 },
              { entity_group: "PER", score: 0.51381934, word: "3", start: 4, end: 5 },
            ],
            [
              { entity_group: "PER", score: 0.5432808, word: "4", start: 0, end: 1 },
              { entity_group: "PER", score: 0.5007693, word: "5", start: 2, end: 3 },
            ],
          ];
          expect(output).toBeCloseToNested(target, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await pipe?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });

    it(
      "throws on an unsupported aggregation_strategy",
      async () => {
        const pipe = await pipeline(PIPELINE_ID, model_id, DEFAULT_MODEL_OPTIONS);
        try {
          // @ts-expect-error intentionally bad
          await expect(pipe("1 2 3", { aggregation_strategy: "first" })).rejects.toThrow(/aggregation_strategy/);
        } finally {
          await pipe.dispose();
        }
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
