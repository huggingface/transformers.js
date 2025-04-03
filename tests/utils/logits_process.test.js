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
    });

    describe("good_words_ids", () => {
      it(
        "generates nothing given empty good_words_ids",
        async () => {
          const text_input = "hello";
          const generated_text_target = "";
          const text_target = [{ generated_text: text_input + generated_text_target }];
          const output = await pipe(text_input, {
            max_new_tokens: 5,
            good_words_ids: [
              [],
            ],
          });
          compare(output, text_target);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "passes basic test",
        async () => {
          const text_input = "hello";
          // Default output tokens for this input: 22172,18547,8136,18547,8136
          // Default output text for this input: helloerdingsAndroid Load Между ligger
          const generated_text_target = "Android helloAndroid hello hello";
          const text_target = [{ generated_text: text_input + generated_text_target }];
          const output = await pipe(text_input, {
            max_new_tokens: 5,
            good_words_ids: [
              [22172, 8136], // hello, Android
            ],
          });
          compare(output, text_target);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "passes test with many good words",
        async () => {
          const text_input = "hello";
          const generated_text_target = "erdingsAndroidierraég migli";
          const text_target = [{ generated_text: text_input + generated_text_target }];
          const good_words_ids = [];
          for (let i = 0; i < 100000; ++i) {
            good_words_ids.push([i * 2 + 1]); // allow all odd numbers
          }
          good_words_ids.push([22172, 8136]);
          const output = await pipe(text_input, { max_new_tokens: 5, good_words_ids });
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
