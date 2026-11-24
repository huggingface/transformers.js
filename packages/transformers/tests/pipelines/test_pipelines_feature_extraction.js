import { pipeline, FeatureExtractionPipeline, Tensor } from "../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../init.js";

const PIPELINE_ID = "feature-extraction";

export default () => {
  describe("Feature Extraction", () => {
    const model_id = "hf-internal-testing/tiny-random-BertModel";

    const texts = ["This is a simple test.", "Hello world"];

    /** @type {FeatureExtractionPipeline} */
    let pipe;
    beforeAll(async () => {
      pipe = await pipeline(PIPELINE_ID, model_id, DEFAULT_MODEL_OPTIONS);
    }, MAX_MODEL_LOAD_TIME);

    it("should be an instance of FeatureExtractionPipeline ", () => {
      expect(pipe).toBeInstanceOf(FeatureExtractionPipeline);
    });

    describe("batch_size=1", () => {
      it(
        "default",
        async () => {
          const output = await pipe(texts[0]);
          expect(output.dims).toEqual([1, 20, 32]);
          expect(output.type).toEqual("float32");
          expect(output.mean().item()).toBeCloseTo(-1.538501215314625e-9, 6);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it(
        "w/ cls pooling",
        async () => {
          const output = await pipe(texts[0], { pooling: "cls" });
          expect(output.dims).toEqual([1, 32]);
          expect(output.type).toEqual("float32");
          expect(output.mean().item()).toBeCloseTo(2.491287887096405e-8, 6);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it(
        "w/ mean pooling & normalization",
        async () => {
          const output = await pipe(texts[0], { pooling: "mean", normalize: true });
          expect(output.dims).toEqual([1, 32]);
          expect(output.type).toEqual("float32");
          expect(output.mean().item()).toBeCloseTo(-2.0245352061465383e-9, 6);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it(
        "w/ mean pooling & binary quantization",
        async () => {
          const output = await pipe(texts[0], { pooling: "mean", quantize: true, precision: "binary" });
          expect(output.dims).toEqual([1, 32 / 8]);
          expect(output.type).toEqual("int8");
          expect(output.mean().item()).toEqual(-15);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it("w/ cls pooling & ubinary quantization", async () => {
        const output = await pipe(texts[0], { pooling: "cls", quantize: true, precision: "ubinary" });
        expect(output.dims).toEqual([1, 32 / 8]);
        expect(output.type).toEqual("uint8");
        expect(output.mean().item()).toEqual(140);
      });
    });

    describe("batch_size>1", () => {
      it(
        "default",
        async () => {
          const output = await pipe(texts);
          expect(output.dims).toEqual([texts.length, 20, 32]);
          expect(output.type).toEqual("float32");
          expect(output.mean().item()).toBeCloseTo(2.345950544935249e-9, 6);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it(
        "w/ cls pooling",
        async () => {
          const output = await pipe(texts, { pooling: "cls" });
          expect(output.dims).toEqual([texts.length, 32]);
          expect(output.type).toEqual("float32");
          expect(output.mean().item()).toBeCloseTo(1.6298145055770874e-8, 6);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it(
        "w/ mean pooling & normalization",
        async () => {
          const output = await pipe(texts, { pooling: "mean", normalize: true });
          expect(output.dims).toEqual([texts.length, 32]);
          expect(output.type).toEqual("float32");
          expect(output.mean().item()).toBeCloseTo(-1.538609240014921e-10, 6);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it("w/ mean pooling & binary quantization", async () => {
        const output = await pipe(texts, { pooling: "mean", quantize: true, precision: "binary" });
        expect(output.dims).toEqual([texts.length, 32 / 8]);
        expect(output.type).toEqual("int8");
        expect(output.mean().item()).toEqual(-14);
      });
      it("w/ cls pooling & ubinary quantization", async () => {
        const output = await pipe(texts, { pooling: "cls", quantize: true, precision: "ubinary" });
        expect(output.dims).toEqual([texts.length, 32 / 8]);
        expect(output.type).toEqual("uint8");
        expect(output.mean().item()).toEqual(140);
      });
    });

    afterAll(async () => {
      await pipe?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });

  describe("Feature Extraction last-token pooling", () => {
    it.each(["last_token", "eos"])("%s skips padding at the end of each sequence", async (pooling) => {
      const attention_mask = new Tensor("int64", [1n, 1n, 0n, 1n, 1n, 1n, 0n, 1n, 1n], [3, 3]);
      const last_hidden_state = new Tensor("float32", [1, 2, 3, 4, 90, 91, 5, 6, 7, 8, 9, 10, 92, 93, 11, 12, 13, 14], [3, 3, 2]);
      const extractor = new FeatureExtractionPipeline({
        task: PIPELINE_ID,
        tokenizer: () => ({ attention_mask }),
        model: async () => ({ last_hidden_state }),
      });

      const output = await extractor(["shorter", "longest input", "left padded"], { pooling });

      expect(output.tolist()).toEqual([
        [3, 4],
        [9, 10],
        [13, 14],
      ]);
    });
  });
};
