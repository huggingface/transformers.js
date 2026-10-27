import {
  // Pipelines
  pipeline,
  TextGenerationPipeline,

  // Logits processors
  WhisperTimeStampLogitsProcessor,

  // Other
  Tensor,
} from "../../src/transformers.js";

import { init } from "../init.js";
init();

const MAX_MODEL_LOAD_TIME = 10_000; // 10 seconds
const MAX_TEST_EXECUTION_TIME = 10_000; // 10 seconds
const MAX_MODEL_DISPOSE_TIME = 1_000; // 1 second

const DEFAULT_MODEL_OPTIONS = {
  dtype: "fp32",
};

describe("Logits Processors", () => {
  describe("text-generation", () => {
    const model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM";

    /** @type {TextGenerationPipeline} */
    let pipe;
    beforeAll(async () => {
      pipe = await pipeline("text-generation", model_id, {
        // TODO move to config
        ...DEFAULT_MODEL_OPTIONS,
      });
    }, MAX_MODEL_LOAD_TIME);

    describe("bad_word_ids", () => {
      it(
        "basic",
        async () => {
          const text_input = "hello";

          const generated_text_target = "\uff0d Giuseppeitte natoud";
          const text_target = [{ generated_text: text_input + generated_text_target }];

          const output = await pipe(text_input, {
            max_new_tokens: 5,
            bad_words_ids: [
              // default: [1n, 22172n, 18547n, 8143n, 22202n, 9456n, 17213n]
              [18547],

              // block #1: [1n, 22172n, 31583n, 18824n, 16621n, 8136n, 16012n]
              [18824, 16621],
            ],
          });
          expect(output).toEqual(text_target);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "many bad words",
        async () => {
          const text_input = "hello";

          const generated_text_target = "erdingsdelete войsequ族";
          const text_target = [{ generated_text: text_input + generated_text_target }];

          // Construct long list of bad words
          const bad_words_ids = [];
          // default: [1n, 22172n, 18547n, 8143n, 22202n, 9456n, 17213n]
          for (let i = 0; i < 100000; ++i) {
            bad_words_ids.push([i * 2]); // block all even numbers
          }
          // block #1: [1n, 22172n, 18547n, 8143n, 30327n, 624n, 2806n, 2004n]
          bad_words_ids.push([8143, 30327]);

          // block #2: [1n, 22172n, 18547n, 8143n, 29485n, 3799n, 29331n]
          bad_words_ids.push([18547, 8143, 29485]);

          // block #3: [1n, 22172n, 18547n, 8143n, 7587n, 6831n, 30999n]
          const output = await pipe(text_input, { max_new_tokens: 5, bad_words_ids });
          expect(output).toEqual(text_target);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "different lengths",
        async () => {
          const text_input = "this is a test";

          const generated_text_target = "кт México constructed lake års";
          const text_target = [{ generated_text: text_input + generated_text_target }];

          const output = await pipe(text_input, {
            max_new_tokens: 5,
            bad_words_ids: [
              // default: [1n, 445n, 338n, 263n, 1243n, 3931n, 14756n, 7811n, 21645n, 31252n]
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3931], // should never trigger (longer than input sequence)

              // block #1: [1n, 445n, 338n, 263n, 1243n, 3931n, 14756n, 7811n, 21645n, 31252n]
              [3931, 14756, 7811],

              // result: [1n, 445n, 338n, 263n, 1243n, 3931n, 14756n, 13319n, 19437n, 21948n]
            ],
          });
          expect(output).toEqual(text_target);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    afterAll(async () => {
      await pipe?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
});

describe("WhisperTimeStampLogitsProcessor", () => {
  // Small synthetic vocabulary: [0..8] text tokens, 9 = <|endoftext|>, 10 = <|notimestamps|>, [11..18] timestamp tokens
  const eos_token_id = 9;
  const no_timestamps_token_id = 10;
  const timestamp_begin = no_timestamps_token_id + 1;
  const vocab_size = 19;
  const init_tokens = [1, 2, 3]; // begin_index = 3

  const create_logits = () => new Tensor("float32", new Float32Array(vocab_size).fill(0), [1, vocab_size]);
  const finite_ids = (data) => Array.from(data, (x, i) => (Number.isFinite(x) ? i : -1)).filter((i) => i >= 0);

  it("should apply `max_initial_timestamp_index` at the first decoding step", () => {
    const max_initial_timestamp_index = 2;
    const processor = new WhisperTimeStampLogitsProcessor({ eos_token_id, no_timestamps_token_id, max_initial_timestamp_index }, init_tokens);

    const logits = processor([init_tokens.map(BigInt)], create_logits());

    // Only the first `max_initial_timestamp_index + 1` timestamp tokens may be sampled at the first step
    const last_allowed = timestamp_begin + max_initial_timestamp_index;
    expect(finite_ids(logits.data)).toEqual([timestamp_begin, timestamp_begin + 1, last_allowed]);
  });

  it("should not restrict the initial timestamp when `max_initial_timestamp_index` is unset", () => {
    const processor = new WhisperTimeStampLogitsProcessor({ eos_token_id, no_timestamps_token_id, max_initial_timestamp_index: null }, init_tokens);

    const logits = processor([init_tokens.map(BigInt)], create_logits());

    // All timestamp tokens remain available (text tokens are still suppressed at the first step)
    const all_timestamps = Array.from({ length: vocab_size - timestamp_begin }, (_, i) => timestamp_begin + i);
    expect(finite_ids(logits.data)).toEqual(all_timestamps);
  });

  it("should only apply `max_initial_timestamp_index` at the first decoding step", () => {
    const max_initial_timestamp_index = 2;
    const processor = new WhisperTimeStampLogitsProcessor({ eos_token_id, no_timestamps_token_id, max_initial_timestamp_index }, init_tokens);

    // One text token has already been sampled after the prompt
    const logits = processor([[...init_tokens, 5].map(BigInt)], create_logits());

    const last_allowed = timestamp_begin + max_initial_timestamp_index;
    for (let i = last_allowed + 1; i < vocab_size; ++i) {
      expect(logits.data[i]).toBe(0);
    }
  });
});
