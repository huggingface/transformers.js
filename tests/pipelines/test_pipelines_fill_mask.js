import { pipeline, FillMaskPipeline } from "../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../init.js";

const PIPELINE_ID = "fill-mask";

export default () => {
  describe("Fill Mask", () => {
    const model_id = "hf-internal-testing/tiny-random-BertForMaskedLM";

    /** @type {FillMaskPipeline} */
    let pipe;
    beforeAll(async () => {
      pipe = await pipeline(PIPELINE_ID, model_id, DEFAULT_MODEL_OPTIONS);
    }, MAX_MODEL_LOAD_TIME);

    it("should be an instance of FillMaskPipeline", () => {
      expect(pipe).toBeInstanceOf(FillMaskPipeline);
    });

    describe("batch_size=1", () => {
      it(
        "default (top_k=5)",
        async () => {
          const output = await pipe("a [MASK] c");
          const target = [
            { score: 0.0013377574505284429, token: 854, token_str: "##ο", sequence: "aο c" },
            { score: 0.001248967950232327, token: 962, token_str: "##ち", sequence: "aち c" },
            { score: 0.0012304208939895034, token: 933, token_str: "##ع", sequence: "aع c" },
            { score: 0.0012301815440878272, token: 313, token_str: "ფ", sequence: "a ფ c" },
            { score: 0.001222139224410057, token: 624, token_str: "未", sequence: "a 未 c" },
          ];
          expect(output).toBeCloseToNested(target, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it(
        "custom (top_k=2)",
        async () => {
          const output = await pipe("a [MASK] c", { top_k: 2 });
          const target = [
            { score: 0.0013377574505284429, token: 854, token_str: "##ο", sequence: "aο c" },
            { score: 0.001248967950232327, token: 962, token_str: "##ち", sequence: "aち c" },
          ];
          expect(output).toBeCloseToNested(target, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    describe("batch_size>1", () => {
      it(
        "default (top_k=5)",
        async () => {
          const output = await pipe(["a [MASK] c", "a b [MASK] c"]);
          const target = [
            [
              { score: 0.0013377574505284429, token: 854, token_str: "##ο", sequence: "aο c" },
              { score: 0.001248967950232327, token: 962, token_str: "##ち", sequence: "aち c" },
              { score: 0.0012304208939895034, token: 933, token_str: "##ع", sequence: "aع c" },
              { score: 0.0012301815440878272, token: 313, token_str: "ფ", sequence: "a ფ c" },
              { score: 0.001222139224410057, token: 624, token_str: "未", sequence: "a 未 c" },
            ],
            [
              { score: 0.0013287801994010806, token: 962, token_str: "##ち", sequence: "a bち c" },
              { score: 0.0012486606137827039, token: 823, token_str: "##ن", sequence: "a bن c" },
              { score: 0.0012320734094828367, token: 1032, token_str: "##ც", sequence: "a bც c" },
              { score: 0.0012295148335397243, token: 854, token_str: "##ο", sequence: "a bο c" },
              { score: 0.0012277684872969985, token: 624, token_str: "未", sequence: "a b 未 c" },
            ],
          ];
          expect(output).toBeCloseToNested(target, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it(
        "custom (top_k=2)",
        async () => {
          const output = await pipe(["a [MASK] c", "a b [MASK] c"], { top_k: 2 });
          const target = [
            [
              { score: 0.0013377574505284429, token: 854, token_str: "##ο", sequence: "aο c" },
              { score: 0.001248967950232327, token: 962, token_str: "##ち", sequence: "aち c" },
            ],
            [
              { score: 0.0013287801994010806, token: 962, token_str: "##ち", sequence: "a bち c" },
              { score: 0.0012486606137827039, token: 823, token_str: "##ن", sequence: "a bن c" },
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
