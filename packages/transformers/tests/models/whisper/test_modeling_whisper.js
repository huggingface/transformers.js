import { WhisperTokenizer, WhisperForConditionalGeneration, full, Tensor } from "../../../src/transformers.js";
import { jest } from "@jest/globals";

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

            expect(outputs.tolist()).toEqual([[/* Prefix */ 50258n, /* Generated */ 51682n]]);
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

            expect(outputs.tolist()).toEqual([[/* Prefix */ 50258n, 50259n, 50359n, /* Generated */ 51812n]]);
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

    describe("seek loop", () => {
      const NO_TIMESTAMPS = 50363;
      const TIMESTAMP_BEGIN = NO_TIMESTAMPS + 1;
      const EOS = 50257;
      const INIT = [50258, 50260, 50360];

      /**
       * Run the seek loop with generation mocked at the PreTrainedModel level.
       * This exercises Whisper's real segmentation logic without loading audio or running a model.
       * @param {number[]} tokens Tokens returned by each mocked decoding pass.
       * @param {number} num_frames Number of real (unpadded) input frames.
       */
      const runSeek = async (tokens, num_frames = 3000) => {
        const parent = Object.getPrototypeOf(WhisperForConditionalGeneration.prototype);
        let passes = 0;
        let segmentInputDims;
        const generate = jest.spyOn(parent, "generate").mockImplementation(async ({ decoder_input_ids, inputs }) => {
          ++passes;
          if (passes > 50) {
            throw new Error("seek loop did not terminate after 50 passes");
          }
          segmentInputDims = inputs.dims;
          const ids = [...decoder_input_ids, ...tokens, EOS].map(BigInt);
          return new Tensor("int64", BigInt64Array.from(ids), [1, ids.length]);
        });

        try {
          const whisper = Object.create(WhisperForConditionalGeneration.prototype);
          whisper.config = { max_source_positions: 1500 };
          const output = await whisper._generate_with_seek({
            inputs: new Tensor("float32", new Float32Array(80 * 3000), [1, 80, 3000]),
            generation_config: {
              no_timestamps_token_id: NO_TIMESTAMPS,
              eos_token_id: EOS,
              return_token_timestamps: false,
              num_frames,
            },
            logits_processor: [],
            init_tokens: INIT,
            kwargs: {},
          });
          return { passes, output, segmentInputDims };
        } finally {
          generate.mockRestore();
        }
      };

      it("does not decode padded input when num_frames is zero", async () => {
        const { passes, output } = await runSeek([], 0);

        expect(passes).toBe(0);
        expect(output.tolist()).toEqual([[...INIT.map(BigInt), BigInt(EOS)]]);
      });

      it("uses the real frame count for padded short audio", async () => {
        const { passes, segmentInputDims } = await runSeek([TIMESTAMP_BEGIN, 1654, TIMESTAMP_BEGIN + 250, TIMESTAMP_BEGIN + 1200], 500);

        expect(passes).toBe(1);
        expect(segmentInputDims).toEqual([1, 80, 3000]);
      });

      it("terminates when a complete segment has zero offset", async () => {
        const { passes } = await runSeek([TIMESTAMP_BEGIN, 1654, TIMESTAMP_BEGIN, TIMESTAMP_BEGIN + 1200]);

        expect(passes).toBe(1);
      });

      it("continues through ordinary multi-segment output", async () => {
        const { passes } = await runSeek([TIMESTAMP_BEGIN, 1654, TIMESTAMP_BEGIN + 500, TIMESTAMP_BEGIN + 1200]);

        expect(passes).toBe(3);
      });
    });

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
