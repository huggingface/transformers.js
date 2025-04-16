import {
  // Pipelines
  pipeline,
  TextGenerationPipeline,
} from "../../src/transformers.js";

import { init } from "../init.js";
import { compare } from "../test_utils.js";
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

          const generated_text_target = " Bert explicit wed digasset";
          const text_target = [{ generated_text: text_input + generated_text_target }];

          const output = await pipe(text_input, {
            max_new_tokens: 5,
            bad_words_ids: [
              // default: [22172n, 18547n, 8136n, 16012n, 28064n, 11361n]
              [18547],

              // block #1: [22172n, 16662n, 6261n, 18916n, 29109n, 799n]
              [6261, 18916],
            ],
          });
          compare(output, text_target);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "many bad words",
        async () => {
          const text_input = "hello";

          const generated_text_target = "erdingsdeletearus)?nor";
          const text_target = [{ generated_text: text_input + generated_text_target }];

          // Construct long list of bad words
          const bad_words_ids = [];
          // default:  [22172n, 18547n, 8136n, 16012n, 28064n, 11361n]
          for (let i = 0; i < 100000; ++i) {
            bad_words_ids.push([i * 2]); // block all even numbers
          }
          // block #1: [22172n, 18547n, 8143n, 30327n, 20061n, 18193n]
          bad_words_ids.push([8143, 30327]);

          // block #2: [22172n, 18547n, 8143n, 29485n, 3799n, 29331n]
          bad_words_ids.push([18547, 8143, 29485]);

          // block #3: [22172n, 18547n, 8143n, 26465n, 6877n, 15459n]
          const output = await pipe(text_input, { max_new_tokens: 5, bad_words_ids });
          compare(output, text_target);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "different lengths",
        async () => {
          const text_input = "this is a test";

          const generated_text_target = "кт México constructed lake user";
          const text_target = [{ generated_text: text_input + generated_text_target }];

          const output = await pipe(text_input, {
            max_new_tokens: 5,
            bad_words_ids: [
              // default: [445n, 338n, 263n, 1243n, 3931n, 14756n, 7811n, 21645n, 16426n]
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3931], // should never trigger (longer than input sequence)

              // block #1: [445n, 338n, 263n, 1243n, 3931n, 14756n, 7811n, 21645n, 16426n]
              [3931, 14756, 7811],

              // result: [445n, 338n, 263n, 1243n, 3931n, 14756n, 13319n, 19437n, 1404n]
            ],
          });
          compare(output, text_target);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    afterAll(async () => {
      await pipe?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
});
