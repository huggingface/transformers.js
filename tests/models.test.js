/*
 * Test that models loaded outside of the `pipeline` function work correctly (e.g., `AutoModel.from_pretrained(...)`);
 */

import { AutoTokenizer, AutoModel, BertModel, GPT2Model, T5ForConditionalGeneration, BertTokenizer, GPT2Tokenizer, T5Tokenizer } from "../src/transformers.js";
import { init, MAX_TEST_EXECUTION_TIME, DEFAULT_MODEL_OPTIONS } from "./init.js";
import { compare, collect_and_execute_tests } from "./test_utils.js";

// Initialise the testing environment
init();

describe("Loading different architecture types", () => {
  // List all models which will be tested
  const models_to_test = [
    // [name, modelClass, tokenizerClass]
    ["hf-internal-testing/tiny-random-BertForMaskedLM", BertModel, BertTokenizer], // Encoder-only
    ["hf-internal-testing/tiny-random-GPT2LMHeadModel", GPT2Model, GPT2Tokenizer], // Decoder-only
    ["hf-internal-testing/tiny-random-T5ForConditionalGeneration", T5ForConditionalGeneration, T5Tokenizer], // Encoder-decoder
  ];

  const texts = ["Once upon a time", "I like to eat apples"];

  for (const [model_id, modelClass, tokenizerClass] of models_to_test) {
    // Test that both the auto model and the specific model work
    const tokenizers = [AutoTokenizer, tokenizerClass];
    const models = [AutoModel, modelClass];

    for (let i = 0; i < tokenizers.length; ++i) {
      const tokenizerClassToTest = tokenizers[i];
      const modelClassToTest = models[i];

      it(
        `${model_id} (${modelClassToTest.name})`,
        async () => {
          // Load model and tokenizer
          const tokenizer = await tokenizerClassToTest.from_pretrained(model_id);
          const model = await modelClassToTest.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);

          const tests = [
            texts[0], // single
            texts, // batched
          ];
          for (const test of tests) {
            const inputs = await tokenizer(test, { truncation: true, padding: true });
            if (model.config.is_encoder_decoder) {
              inputs.decoder_input_ids = inputs.input_ids;
            }
            const output = await model(inputs);

            if (output.logits) {
              // Ensure correct shapes
              const expected_shape = [...inputs.input_ids.dims, model.config.vocab_size];
              const actual_shape = output.logits.dims;
              compare(expected_shape, actual_shape);
            } else if (output.last_hidden_state) {
              const expected_shape = [...inputs.input_ids.dims, model.config.d_model];
              const actual_shape = output.last_hidden_state.dims;
              compare(expected_shape, actual_shape);
            } else {
              console.warn("Unexpected output", output);
              throw new Error("Unexpected output");
            }
          }
          await model.dispose();
        },
        MAX_TEST_EXECUTION_TIME,
      );
    }
  }
});

await collect_and_execute_tests("Model-specific tests", "modeling");
