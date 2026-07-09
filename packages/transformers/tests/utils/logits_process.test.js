import {
  // Pipelines
  pipeline,
  TextGenerationPipeline,
  SchemaConstrainedLogitsProcessor,
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

describe("SchemaConstrainedLogitsProcessor", () => {
  const tokenizer = {
    get_vocab() {
      return { "{": 0, "}": 1, foo: 2, bar: 3 };
    },
    decode(ids) {
      return ["{", "}", "foo", "bar"][ids[0]];
    },
    eos_token_id: null,
    bos_token_id: null,
    all_special_ids: [],
  };

  it("applies llguidance masks and commits sampled tokens", async () => {
    const committed = [];
    const runtime = {
      createTokenizer(config) {
        expect(config.tokens.length).toEqual(4);
        return { config };
      },
      createInterpreter({ tokenizer: llguidanceTokenizer, response_format }) {
        expect(llguidanceTokenizer.config.tokens.length).toEqual(4);
        expect(response_format).toEqual({ type: "json_object" });
        return {
          computeMask() {
            return { mask: [1, 0, 1, 0] };
          },
          commitToken(token_id) {
            committed.push(token_id);
          },
        };
      },
    };

    const processor = await SchemaConstrainedLogitsProcessor.fromResponseFormat({ type: "json_object" }, tokenizer, runtime);
    const logits = new Tensor("float32", new Float32Array([1, 2, 3, 4]), [1, 4]);

    processor([[0n]], logits);
    processor.onTokenSampled(2);

    expect(Array.from(logits.data)).toEqual([1, -Infinity, 3, -Infinity]);
    expect(committed).toEqual([2]);
  });

  it("passes callable tokenizers directly to llguidance", async () => {
    const callableTokenizer = Object.assign(() => {}, tokenizer);
    const runtime = {
      acceptsTokenizerObjects: true,
      createTokenizer() {
        throw new Error("createTokenizer should not be called");
      },
      createInterpreter({ tokenizer: llguidanceTokenizer }) {
        expect(llguidanceTokenizer).toBe(callableTokenizer);
        expect(typeof llguidanceTokenizer).toEqual("function");
        expect(llguidanceTokenizer.get_vocab()).toEqual(tokenizer.get_vocab());
        return {
          computeMask() {
            return { mask: [1, 1, 1, 1] };
          },
          commitToken() {},
        };
      },
    };

    await expect(SchemaConstrainedLogitsProcessor.fromResponseFormat({ type: "json_object" }, callableTokenizer, runtime)).resolves.toBeInstanceOf(SchemaConstrainedLogitsProcessor);
  });

  it("stops after llguidance reaches a stop state", async () => {
    const runtime = {
      createTokenizer(config) {
        return { config };
      },
      createInterpreter() {
        return {
          computeMask() {
            throw new Error("computeMask should not be called after stop");
          },
          commitToken() {
            return { stop: true };
          },
        };
      },
    };

    const processor = await SchemaConstrainedLogitsProcessor.fromResponseFormat({ type: "json_object" }, { ...tokenizer, eos_token_id: 1 }, runtime);
    processor.onTokenSampled(2);

    const logits = new Tensor("float32", new Float32Array([1, 2, 3, 4]), [1, 4]);
    processor([[0n]], logits);

    expect(Array.from(logits.data)).toEqual([1, 2, 3, 4]);
    expect(processor.shouldStop([[0n]])).toEqual([true]);
  });

  it("stops when llguidance reports compute after stop", async () => {
    const runtime = {
      createTokenizer(config) {
        return { config };
      },
      createInterpreter() {
        return {
          computeMask() {
            throw new Error("computeMask failed: compute_mask() called after stop");
          },
          commitToken() {},
        };
      },
    };

    const processor = await SchemaConstrainedLogitsProcessor.fromResponseFormat({ type: "json_object" }, { ...tokenizer, eos_token_id: 1 }, runtime);
    const logits = new Tensor("float32", new Float32Array([1, 2, 3, 4]), [1, 4]);
    processor([[0n]], logits);

    expect(Array.from(logits.data)).toEqual([1, 2, 3, 4]);
    expect(processor.shouldStop([[0n]])).toEqual([true]);
  });

  it("validates response_format before loading llguidance", async () => {
    await expect(SchemaConstrainedLogitsProcessor.fromResponseFormat({ type: "text" }, tokenizer)).rejects.toThrow("Unsupported `response_format.type`");
  });
});
