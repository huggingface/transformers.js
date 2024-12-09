import { WhisperTokenizer, WhisperForConditionalGeneration, full } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  describe("WhisperForConditionalGeneration", () => {
    const model_id = "Xenova/tiny-random-WhisperForConditionalGeneration";

    /** @type {WhisperForConditionalGeneration} */
    let model;
    /** @type {WhisperTokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await WhisperForConditionalGeneration.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      tokenizer = await WhisperTokenizer.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    describe("prefix tokens", () => {
      const input_features = full([1, 80, 3000], 0.0);

      describe("English-only", () => {
        it(
          "default",
          async () => {
            const outputs = await model.generate({
              input_features,
              is_multilingual: false,
              max_new_tokens: 1,
            });

            expect(outputs.tolist()).toEqual([[/* Prefix */ 50258n, 50363n, /* Generated */ 45084n]]);
          },
          MAX_TEST_EXECUTION_TIME,
        );

        it(
          "return_timestamps=true",
          async () => {
            const outputs = await model.generate({
              input_features,
              is_multilingual: false,
              max_new_tokens: 1,
              return_timestamps: true,
            });

            expect(outputs.tolist()).toEqual([[/* Prefix */ 50258n, /* Generated */ 50366n]]);
          },
          MAX_TEST_EXECUTION_TIME,
        );
      });

      describe("multilingual", () => {
        it(
          "language unset; task unset",
          async () => {
            // language defaults to 'en'
            // task defaults to 'transcribe'

            const outputs = await model.generate({
              input_features,
              max_new_tokens: 1,
            });

            expect(outputs.tolist()).toEqual([[/* Prefix */ 50258n, 50259n, 50359n, 50363n, /* Generated */ 45084n]]);
          },
          MAX_TEST_EXECUTION_TIME,
        );

        it(
          "language set; task unset",
          async () => {
            // task defaults to 'transcribe'
            const outputs = await model.generate({
              input_features,
              max_new_tokens: 1,
              language: "af",
            });

            expect(outputs.tolist()).toEqual([[/* Prefix */ 50258n, 50327n, 50359n, 50363n, /* Generated */ 45084n]]);
          },
          MAX_TEST_EXECUTION_TIME,
        );

        it(
          "language set; task set",
          async () => {
            const outputs = await model.generate({
              input_features,
              max_new_tokens: 1,
              language: "zh",
              task: "translate",
            });

            expect(outputs.tolist()).toEqual([[/* Prefix */ 50258n, 50260n, 50358n, 50363n, /* Generated */ 45084n]]);
          },
          MAX_TEST_EXECUTION_TIME,
        );

        it(
          "return_timestamps=true",
          async () => {
            const outputs = await model.generate({
              input_features,
              max_new_tokens: 1,
              language: "en",
              task: "transcribe",
              return_timestamps: true,
            });

            expect(outputs.tolist()).toEqual([[/* Prefix */ 50258n, 50259n, 50359n, /* Generated */ 50400n]]);
          },
          MAX_TEST_EXECUTION_TIME,
        );
      });
    });

    describe("decoder_start_ids", () => {
      const input_features = full([1, 80, 3000], 0.0);

      it(
        "broadcast inputs",
        async () => {
          const { decoder_start_token_id, lang_to_id, task_to_id, no_timestamps_token_id } = model.generation_config;

          const outputs = await model.generate({
            input_features, // batch size 1
            max_new_tokens: 1,
            decoder_input_ids: [
              // batch size 2
              // <|startoftranscript|> <|lang_id|> <|task|> [<|notimestamps|>]
              [decoder_start_token_id, lang_to_id["<|en|>"], task_to_id["translate"], no_timestamps_token_id],
              [decoder_start_token_id, lang_to_id["<|fr|>"], task_to_id["transcribe"], no_timestamps_token_id],
            ],
          });
          expect(outputs.tolist()).toEqual([
            [/* Prefix */ 50258n, 50259n, 50358n, 50363n, /* Generated */ 45084n],
            [/* Prefix */ 50258n, 50265n, 50359n, 50363n, /* Generated */ 45084n],
          ]);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
