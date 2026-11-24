import { pipeline, Text2TextGenerationPipeline } from "../../src/transformers.js";
import { jest } from "@jest/globals";

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
    });

    describe("tokenizer options", () => {
      const model_max_length = 4;
      const long_text = "This is a test."; // More than `model_max_length` tokens
      const short_text = "Hi"; // Fewer than `model_max_length` tokens

      let tokenizer_spy, generate_spy;
      beforeEach(() => {
        jest.spyOn(pipe.tokenizer, "model_max_length", "get").mockReturnValue(model_max_length);
        tokenizer_spy = jest.spyOn(pipe.tokenizer, "_call");
        generate_spy = jest.spyOn(pipe.model, "generate");
      });
      afterEach(() => {
        jest.restoreAllMocks();
      });

      it(
        "defaults to padding and truncation",
        async () => {
          await pipe(long_text, { max_new_tokens: 1 });
          const [, tokenizer_options] = tokenizer_spy.mock.calls.at(-1);
          expect(tokenizer_options).toEqual({ padding: true, truncation: true });
          const [{ input_ids }] = generate_spy.mock.calls.at(-1);
          expect(input_ids.dims).toEqual([1, model_max_length]);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "truncation=false keeps inputs longer than model_max_length",
        async () => {
          const num_tokens = pipe.tokenizer.encode(long_text).length;
          expect(num_tokens).toBeGreaterThan(model_max_length);

          await pipe(long_text, { max_new_tokens: 1, truncation: false });
          const [{ input_ids }] = generate_spy.mock.calls.at(-1);
          expect(input_ids.dims).toEqual([1, num_tokens]);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "padding='max_length' pads inputs to model_max_length",
        async () => {
          await pipe(short_text, { max_new_tokens: 1, padding: "max_length" });
          const [, tokenizer_options] = tokenizer_spy.mock.calls.at(-1);
          expect(tokenizer_options).toEqual({ padding: "max_length", truncation: true });
          const [{ input_ids }] = generate_spy.mock.calls.at(-1);
          expect(input_ids.dims).toEqual([1, model_max_length]);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "does not pass tokenizer options to model.generate",
        async () => {
          await pipe(long_text, { max_new_tokens: 1, padding: false, truncation: false });
          const [generate_kwargs] = generate_spy.mock.calls.at(-1);
          expect(generate_kwargs).not.toHaveProperty("padding");
          expect(generate_kwargs).not.toHaveProperty("truncation");
          expect(generate_kwargs).toHaveProperty("max_new_tokens", 1);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    afterAll(async () => {
      await pipe?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
