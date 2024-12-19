import { pipeline, FillMaskPipeline } from "../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../init.js";

const PIPELINE_ID = "fill-mask";

export default () => {
  describe("Fill Mask", () => {
    describe("Standard", () => {
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

    describe("Custom tokenizer", () => {
      const model_id = "hf-internal-testing/tiny-random-ModernBertForMaskedLM";

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
            const output = await pipe("The capital of France is [MASK].");
            const target = [
              { score: 0.2106737643480301, sequence: "The capital of France isả.", token: 35165, token_str: "ả" },
              { score: 0.18418768048286438, sequence: "The capital of France isDispatch.", token: 48010, token_str: "Dispatch" },
              { score: 0.16561225056648254, sequence: "The capital of France is Ther.", token: 20763, token_str: " Ther" },
              { score: 0.07070659101009369, sequence: "The capital of France isschild.", token: 50040, token_str: "schild" },
              { score: 0.029540402814745903, sequence: "The capital of France isbles.", token: 9143, token_str: "bles" },
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
                { score: 0.06699250638484955, sequence: "a oocytes c", token: 36805, token_str: " oocytes" },
                { score: 0.05928678810596466, sequence: "ancia c", token: 19003, token_str: "ncia" },
                { score: 0.057058464735746384, sequence: "aả c", token: 35165, token_str: "ả" },
                { score: 0.04978331923484802, sequence: "amq c", token: 37365, token_str: "mq" },
                { score: 0.04839889705181122, sequence: "a1371 c", token: 26088, token_str: "1371" },
              ],
              [
                { score: 0.06364646553993225, sequence: "a b oocytes c", token: 36805, token_str: " oocytes" },
                { score: 0.03993292525410652, sequence: "a bectin c", token: 41105, token_str: "ectin" },
                { score: 0.03932870551943779, sequence: "a bả c", token: 35165, token_str: "ả" },
                { score: 0.037771403789520264, sequence: "a boplastic c", token: 21945, token_str: "oplastic" },
                { score: 0.03748754784464836, sequence: "a b Ther c", token: 20763, token_str: " Ther" },
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
  });
};
