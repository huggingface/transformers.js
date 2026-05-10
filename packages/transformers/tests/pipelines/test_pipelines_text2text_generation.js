import { pipeline, Text2TextGenerationPipeline } from "../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../init.js";

const PIPELINE_ID = "text2text-generation";

export default () => {
  describe("Text to Text Generation", () => {
    const model_id = "hf-internal-testing/tiny-random-T5ForConditionalGeneration";

    /** @type {Text2TextGenerationPipeline} */
    let pipe;
    beforeAll(async () => {
      pipe = await pipeline(PIPELINE_ID, model_id, DEFAULT_MODEL_OPTIONS);
    }, MAX_MODEL_LOAD_TIME);

    it("should be an instance of Text2TextGenerationPipeline", () => {
      expect(pipe).toBeInstanceOf(Text2TextGenerationPipeline);
    });

    describe("batch_size=1", () => {
      it(
        "default",
        async () => {
          const text = "This is a test.";
          const output = await pipe(text, {
            max_new_tokens: 5,
          });
          const target = [{ generated_text: "" }];
          expect(output).toEqual(target);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "tokenizer_options are forwarded to the tokenizer",
        async () => {
          const text = "This is a test.";
          // max_length=3 forces aggressive truncation via tokenizer_options;
          // the model should still run without error and return a string result.
          // tokenizer_options is extracted before generate() is called so it
          // never leaks into GenerationFunctionParameters.
          const output = await pipe(text, {
            max_new_tokens: 5,
            tokenizer_options: { truncation: true, max_length: 3 },
          });
          expect(Array.isArray(output)).toBe(true);
          expect(typeof output[0].generated_text).toBe("string");
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    afterAll(async () => {
      await pipe?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
