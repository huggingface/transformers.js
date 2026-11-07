import {
  // Pipelines
  pipeline,
  TextGenerationPipeline,
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
  // Small synthetic vocabulary: [0..99] text tokens, 100 = <|notimestamps|>, [101..199] timestamps
  const no_timestamps_token_id = 100;
  const timestamp_begin = no_timestamps_token_id + 1;
  const vocab_size = 200;
  const init_tokens = [1, 2, 3];

  const create_processor = async (config) => {
    const { WhisperTimeStampLogitsProcessor } = await import("../../src/generation/logits_process.js");
    return new WhisperTimeStampLogitsProcessor(
      { eos_token_id: 2, no_timestamps_token_id, ...config },
      init_tokens,
    );
  };

  const run = async (config, input_ids) => {
    const processor = await create_processor(config);
    const { Tensor } = await import("../../src/utils/tensor.js");
    const logits = new Tensor("float32", new Float32Array(vocab_size), [1, vocab_size]);
    processor._call([input_ids], logits);
    return logits.data;
  };

  const allowed_timestamps = (data) =>
    Array.from(data)
      .map((x, i) => (Number.isFinite(x) ? i : -1))
      .filter((i) => i >= timestamp_begin);

  it("limits the first timestamp by max_initial_timestamp_index", async () => {
    // First decoding step: no token has been generated yet, so the processor masks
    // text tokens and must also cap the timestamps it allows.
    const data = await run({ max_initial_timestamp_index: 1 }, [1n, 2n, 3n]);

    expect(allowed_timestamps(data)).toEqual([timestamp_begin, timestamp_begin + 1]);
  });

  it("keeps every timestamp when max_initial_timestamp_index is null", async () => {
    const data = await run({ max_initial_timestamp_index: null }, [1n, 2n, 3n]);

    expect(allowed_timestamps(data)).toHaveLength(vocab_size - timestamp_begin);
  });

  it("keeps every timestamp when the config omits max_initial_timestamp_index", async () => {
    // Python reads the option with `getattr(..., None)`, so a missing key means "no cap".
    // Guarding with `!== null` instead of `!= null` would compute `subarray(NaN)` and mask
    // every logit in the row, including the text tokens.
    const data = await run({}, [1n, 2n, 3n]);

    expect(allowed_timestamps(data)).toHaveLength(vocab_size - timestamp_begin);
  });

  it("does not cap later decoding steps", async () => {
    const data = await run(
      { max_initial_timestamp_index: 1 },
      [1n, 2n, 3n, BigInt(timestamp_begin + 50), 5n],
    );

    // Past the first step the cap must not apply: a timestamp far beyond
    // `timestamp_begin + max_initial_timestamp_index` stays available.
    expect(data[timestamp_begin + 60]).not.toBe(-Infinity);
  });

  it("defaults to no restriction", async () => {
    const { WhisperGenerationConfig } = await import(
      "../../src/models/whisper/generation_whisper.js"
    );

    expect(new WhisperGenerationConfig({}).max_initial_timestamp_index).toBeNull();
  });
});
