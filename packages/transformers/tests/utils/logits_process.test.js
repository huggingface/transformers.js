import {
  // Pipelines
  pipeline,
  TextGenerationPipeline,

  // Logits processors and warpers
  LogitsProcessor,
  LogitsProcessorList,
  TopKLogitsWarper,
  TopPLogitsWarper,
  Tensor,
  random,
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

    describe("top_p", () => {
      it(
        "a very small top_p is the same as greedy search",
        async () => {
          const text_input = "hello";
          const greedy = await pipe(text_input, { max_new_tokens: 5 });

          // Only the most likely token is left to sample from. `top_k: 0` disables top-k.
          const output = await pipe(text_input, { max_new_tokens: 5, do_sample: true, top_k: 0, top_p: 1e-6 });
          expect(output).toEqual(greedy);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    describe("sampling with top_k and top_p", () => {
      // Replaces the scores of every step by the given probabilities (token id -> probability).
      class FixedScores extends LogitsProcessor {
        constructor(probs) {
          super();
          this.probs = probs;
        }
        _call(input_ids, logits) {
          logits.data.fill(-Infinity);
          for (const [token_id, prob] of Object.entries(this.probs)) {
            logits.data[token_id] = Math.log(prob);
          }
          return logits;
        }
      }

      // Raises the score of one token, like a caller that wants to promote it.
      class Boost extends LogitsProcessor {
        constructor(token_id) {
          super();
          this.token_id = token_id;
        }
        _call(input_ids, logits) {
          logits.data[this.token_id] += 1000;
          return logits;
        }
      }

      /**
       * Samples `max_new_tokens` tokens with the given processor and returns the distinct sampled token ids.
       */
      const sampled_tokens = async (processor, options, max_new_tokens = 60) => {
        const logits_processor = new LogitsProcessorList();
        logits_processor.push(processor);

        random.seed(0);
        const input_ids = new Tensor("int64", [1n, 22172n], [1, 2]);
        const output = await pipe.model.generate({
          input_ids,
          max_new_tokens,
          do_sample: true,
          logits_processor,
          ...options,
        });
        const new_tokens = Array.from(output.data.slice(input_ids.dims[1]), Number);
        return [...new Set(new_tokens)].sort((a, b) => a - b);
      };

      it(
        "samples from all tokens that are tied at the top-k cutoff",
        async () => {
          // top-k keeps all four tokens (three are tied for second place), and top-p needs all of them.
          const processor = new FixedScores({ 10: 0.4, 11: 0.2, 12: 0.2, 13: 0.2 });
          expect(await sampled_tokens(processor, { top_k: 2, top_p: 0.99 })).toEqual([10, 11, 12, 13]);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "keeps only as many tokens tied at the top-p cutoff as needed",
        async () => {
          const processor = new FixedScores({ 10: 0.7, 11: 0.1, 12: 0.1, 13: 0.1 });
          expect(await sampled_tokens(processor, { top_k: 0, top_p: 0.75 })).toEqual([10, 11]);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "applies top-p to the tokens that top-k keeps",
        async () => {
          const processor = new FixedScores({ 10: 0.4, 11: 0.3, 12: 0.2, 13: 0.1 });

          // Over the whole vocabulary, three tokens are needed to reach 0.75.
          expect(await sampled_tokens(processor, { top_k: 0, top_p: 0.75 })).toEqual([10, 11, 12]);

          // top-k keeps 0.4, 0.3 and 0.2, which are renormalized to 4/9, 3/9 and 2/9. Two tokens reach 0.75.
          expect(await sampled_tokens(processor, { top_k: 3, top_p: 0.75 })).toEqual([10, 11]);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "top-k alone samples from exactly top_k tokens",
        async () => {
          const processor = new FixedScores({ 10: 0.4, 11: 0.3, 12: 0.2, 13: 0.1 });
          expect(await sampled_tokens(processor, { top_k: 2 })).toEqual([10, 11]);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "custom logits processors run before the cutoff",
        async () => {
          // The boosted token is not among the most likely tokens of the model.
          const token_id = 77;
          for (const options of [{ top_k: 1 }, { top_k: 1, top_p: 0.9 }, { top_k: 0, top_p: 0.5 }]) {
            expect(await sampled_tokens(new Boost(token_id), options, 5)).toEqual([token_id]);
          }
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "a top_p that filters nothing does not change the sampled tokens",
        async () => {
          const generate = async (options) => {
            random.seed(0);
            const input_ids = new Tensor("int64", [1n, 22172n], [1, 2]);
            const output = await pipe.model.generate({ input_ids, max_new_tokens: 10, do_sample: true, ...options });
            return Array.from(output.data, Number);
          };
          expect(await generate({ top_k: 50, top_p: 1 - 1e-9 })).toEqual(await generate({ top_k: 50 }));
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    afterAll(async () => {
      await pipe?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });

  describe("warpers", () => {
    // Two sequences with the probabilities of four tokens each
    const PROBS = [
      [0.1, 0.4, 0.2, 0.3],
      [0.7, 0.15, 0.1, 0.05],
    ];
    const INPUT_IDS = [[0n], [0n]];
    const make_logits = () => new Tensor("float32", PROBS.flat().map(Math.log), [2, 4]);

    /** Applies the warper and returns which tokens are kept. */
    const kept = (warper, logits = make_logits()) => {
      const output = warper(INPUT_IDS, logits);
      const original = make_logits().data;
      return [0, 1].map((i) =>
        Array.from(output.data.subarray(4 * i, 4 * (i + 1)), (x, j) => {
          if (x !== -Infinity) expect(x).toEqual(original[4 * i + j]); // kept scores are not modified
          return x !== -Infinity;
        }),
      );
    };

    describe("TopPLogitsWarper", () => {
      // Expected values are from `TopPLogitsWarper` in the Python library.
      it("keeps the most likely tokens whose probabilities add up to top_p", () => {
        expect(kept(new TopPLogitsWarper(0.5))).toEqual([
          [false, true, false, true],
          [true, false, false, false],
        ]);
        expect(kept(new TopPLogitsWarper(0.75))).toEqual([
          [false, true, true, true],
          [true, true, false, false],
        ]);
      });

      it("keeps a single token for a very small top_p", () => {
        expect(kept(new TopPLogitsWarper(1e-6))).toEqual([
          [false, true, false, false],
          [true, false, false, false],
        ]);
      });

      it("supports min_tokens_to_keep", () => {
        expect(kept(new TopPLogitsWarper(1e-6, { min_tokens_to_keep: 2 }))).toEqual([
          [false, true, false, true],
          [true, true, false, false],
        ]);
      });

      it("supports filter_value", () => {
        const output = new TopPLogitsWarper(0.5, { filter_value: -100 })(INPUT_IDS, make_logits());
        expect(Array.from(output.data.subarray(0, 4), (x) => x === -100)).toEqual([true, false, true, false]);
      });

      it("keeps only as many tokens with the same score as needed", () => {
        const logits = new Tensor("float32", [0.7, 0.1, 0.1, 0.1].map(Math.log), [1, 4]);
        const output = new TopPLogitsWarper(0.75)([[0n]], logits);
        expect(output.data[0]).not.toEqual(-Infinity);
        expect(output.data.filter((x) => x !== -Infinity).length).toEqual(2);
      });

      it("ignores tokens that are already filtered", () => {
        const logits = make_logits();
        logits.data[1] = -Infinity; // the remaining probabilities are 1/6, 2/6 and 3/6
        expect(kept(new TopPLogitsWarper(0.6), logits)[0]).toEqual([false, false, true, true]);
      });
    });

    describe("TopKLogitsWarper", () => {
      it("keeps the top_k most likely tokens", () => {
        expect(kept(new TopKLogitsWarper(2))[0]).toEqual([false, true, false, true]);
        expect(kept(new TopKLogitsWarper(1))[0]).toEqual([false, true, false, false]);
      });

      it("keeps tokens with the same score as the last one", () => {
        const logits = new Tensor("float32", [0.7, 0.1, 0.1, 0.1].map(Math.log), [1, 4]);
        const output = new TopKLogitsWarper(2)([[0n]], logits);
        expect(Array.from(output.data, (x) => x !== -Infinity)).toEqual([true, true, true, true]);
      });

      it("keeps everything if top_k is larger than the vocabulary", () => {
        expect(kept(new TopKLogitsWarper(10))).toEqual([
          [true, true, true, true],
          [true, true, true, true],
        ]);
      });

      it("can be combined with top_p", () => {
        // top-k keeps the probabilities 0.4, 0.3 and 0.2, which are renormalized to 4/9, 3/9 and 2/9
        const logits = new TopKLogitsWarper(3)(INPUT_IDS, make_logits());
        expect(kept(new TopPLogitsWarper(0.7), logits)[0]).toEqual([false, true, false, true]);
      });
    });
  });
});
