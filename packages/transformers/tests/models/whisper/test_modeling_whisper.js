import { jest } from "@jest/globals";

import { WhisperTokenizer, WhisperForConditionalGeneration, PreTrainedModel, Tensor, full } from "../../../src/transformers.js";

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

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });

  describe("WhisperForConditionalGeneration seek loop", () => {
    // <|startoftranscript|> <|en|> <|transcribe|>; each seek pass then generates two text tokens and <|endoftext|>
    const init_tokens = [50258, 50259, 50359];
    const eos_token_id = 50257;

    // Helper function to create a model with no weights: `super.generate` and the
    // DTW step are stubbed, so only the seek loop itself runs
    function createModel() {
      const model = Object.create(WhisperForConditionalGeneration.prototype);
      model.config = { max_source_positions: 1500 };
      model.configs = { generation_config: { eos_token_id, no_timestamps_token_id: 50363, alignment_heads: [[0, 0]] } };
      model._extract_token_timestamps = (outputs) => new Tensor("float32", new Float32Array(outputs.sequences.dims[1]), outputs.sequences.dims);
      return model;
    }

    // Helper function to stub `super.generate`, returning a new KV cache for every seek pass.
    // A supplied cache is updated in place and handed back as the same object, so mirror that here.
    function mockSuperGenerate(generated = [100n, 200n]) {
      const caches = [];
      jest.spyOn(PreTrainedModel.prototype, "generate").mockImplementation(async (params) => {
        const tokens = [...init_tokens.map(BigInt), ...generated, BigInt(eos_token_id)];
        const sequences = new Tensor("int64", tokens, [1, tokens.length]);
        if (!params.generation_config.return_dict_in_generate) return sequences;
        const past_key_values = params.past_key_values ?? { dispose: jest.fn(async () => {}) };
        caches.push(past_key_values);
        return { sequences, past_key_values };
      });
      return caches;
    }

    // Two 30-second segments of silence: 2 x 3000 mel frames
    const inputs = full([1, 80, 6000], 0.0);

    afterEach(() => {
      jest.restoreAllMocks();
    });

    it(
      "disposes the KV cache of every seek pass with token timestamps",
      async () => {
        const caches = mockSuperGenerate();
        const model = createModel();

        const output = await model.generate({ inputs, decoder_input_ids: init_tokens, return_timestamps: true, return_token_timestamps: true });

        expect(output.sequences.dims).toEqual([1, init_tokens.length + 5]);
        expect(caches).toHaveLength(2);
        for (const cache of caches) {
          expect(cache.dispose).toHaveBeenCalledTimes(1);
        }
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "disposes the KV cache of a seek pass that generates no tokens",
      async () => {
        const caches = mockSuperGenerate([]);
        const model = createModel();

        await model.generate({ inputs, decoder_input_ids: init_tokens, return_timestamps: true, return_token_timestamps: true });

        expect(caches).toHaveLength(1);
        expect(caches[0].dispose).toHaveBeenCalledTimes(1);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "does not dispose a caller-supplied KV cache",
      async () => {
        const caches = mockSuperGenerate();
        const supplied_past_key_values = { dispose: jest.fn(async () => {}) };
        const model = createModel();

        await model.generate({ inputs, decoder_input_ids: init_tokens, return_timestamps: true, return_token_timestamps: true, past_key_values: supplied_past_key_values });

        expect(caches).toEqual([supplied_past_key_values, supplied_past_key_values]);
        expect(supplied_past_key_values.dispose).not.toHaveBeenCalled();
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "works when each seek pass returns only the sequences",
      async () => {
        const caches = mockSuperGenerate();
        const model = createModel();

        const output = await model.generate({ inputs, decoder_input_ids: init_tokens, return_timestamps: true });

        expect(output.dims).toEqual([1, init_tokens.length + 5]);
        expect(caches).toHaveLength(0);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
