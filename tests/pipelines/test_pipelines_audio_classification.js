import { pipeline, AudioClassificationPipeline } from "../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../init.js";

const PIPELINE_ID = "audio-classification";

export default () => {
  describe("Audio Classification", () => {
    const model_id = "hf-internal-testing/tiny-random-unispeech";
    const audios = [new Float32Array(16000).fill(0), Float32Array.from({ length: 16000 }, (_, i) => i)];

    /** @type {AudioClassificationPipeline} */
    let pipe;
    beforeAll(async () => {
      pipe = await pipeline(PIPELINE_ID, model_id, DEFAULT_MODEL_OPTIONS);
    }, MAX_MODEL_LOAD_TIME);

    it("should be an instance of AudioClassificationPipeline", () => {
      expect(pipe).toBeInstanceOf(AudioClassificationPipeline);
    });

    describe("batch_size=1", () => {
      it(
        "default (top_k=5)",
        async () => {
          const output = await pipe(audios[0]);
          const target = [
            { score: 0.5043687224388123, label: "LABEL_0" },
            { score: 0.4956313371658325, label: "LABEL_1" },
          ];
          expect(output).toBeCloseToNested(target, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it(
        "custom (top_k=1)",
        async () => {
          const output = await pipe(audios[0], { top_k: 1 });
          const target = [{ score: 0.5043687224388123, label: "LABEL_0" }];
          expect(output).toBeCloseToNested(target, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    describe("batch_size>1", () => {
      it(
        "default (top_k=5)",
        async () => {
          const output = await pipe(audios);
          const target = [
            [
              { score: 0.5043687224388123, label: "LABEL_0" },
              { score: 0.4956313371658325, label: "LABEL_1" },
            ],
            [
              { score: 0.5187293887138367, label: "LABEL_0" },
              { score: 0.4812707006931305, label: "LABEL_1" },
            ],
          ];
          expect(output).toBeCloseToNested(target, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it(
        "custom (top_k=1)",
        async () => {
          const output = await pipe(audios, { top_k: 1 });
          const target = [[{ score: 0.5043687224388123, label: "LABEL_0" }], [{ score: 0.5187293887138367, label: "LABEL_0" }]];
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
