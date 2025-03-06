/*
 * Test that models loaded outside of the `pipeline` function work correctly (e.g., `AutoModel.from_pretrained(...)`);
 */

import { AutoTokenizer, AutoProcessor, BertForMaskedLM, GPT2LMHeadModel, T5ForConditionalGeneration, BertTokenizer, GPT2Tokenizer, T5Tokenizer, LlamaTokenizer, LlamaForCausalLM, WhisperForConditionalGeneration, WhisperProcessor, AutoModelForMaskedLM, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForSpeechSeq2Seq } from "../src/transformers.js";
import { init, MAX_TEST_EXECUTION_TIME, DEFAULT_MODEL_OPTIONS } from "./init.js";
import { compare, collect_and_execute_tests } from "./test_utils.js";

// Initialise the testing environment
init();

describe("Loading different architecture types", () => {
  // List all models which will be tested
  const models_to_test = [
    // [name, [AutoModelClass, ModelClass], [AutoProcessorClass, ProcessorClass], [modelOptions?], [modality?]]
    ["hf-internal-testing/tiny-random-BertForMaskedLM", [AutoModelForMaskedLM, BertForMaskedLM], [AutoTokenizer, BertTokenizer]], // Encoder-only
    ["hf-internal-testing/tiny-random-GPT2LMHeadModel", [AutoModelForCausalLM, GPT2LMHeadModel], [AutoTokenizer, GPT2Tokenizer]], // Decoder-only
    ["hf-internal-testing/tiny-random-T5ForConditionalGeneration", [AutoModelForSeq2SeqLM, T5ForConditionalGeneration], [AutoTokenizer, T5Tokenizer]], // Encoder-decoder
    ["onnx-internal-testing/tiny-random-LlamaForCausalLM-ONNX_external", [AutoModelForCausalLM, LlamaForCausalLM], [AutoTokenizer, LlamaTokenizer]], // Decoder-only w/ external data
    ["onnx-internal-testing/tiny-random-WhisperForConditionalGeneration-ONNX_external", [AutoModelForSpeechSeq2Seq, WhisperForConditionalGeneration], [AutoProcessor, WhisperProcessor], {}], // Encoder-decoder-only w/ external data
  ];

  const texts = ["Once upon a time", "I like to eat apples"];

  for (const [model_id, models, processors, modelOptions] of models_to_test) {
    // Test that both the auto model and the specific model work
    for (let i = 0; i < processors.length; ++i) {
      const processorClassToTest = processors[i];
      const modelClassToTest = models[i];

      it(
        `${model_id} (${modelClassToTest.name})`,
        async () => {
          // Load model and processor
          const processor = await processorClassToTest.from_pretrained(model_id);
          const model = await modelClassToTest.from_pretrained(model_id, modelOptions ?? DEFAULT_MODEL_OPTIONS);

          const tests = [
            texts[0], // single
            texts, // batched
          ];

          const { model_type } = model.config;
          const tokenizer = model_type === "whisper" ? processor.tokenizer : processor;
          const feature_extractor = model_type === "whisper" ? processor.feature_extractor : null;

          for (const test of tests) {
            const inputs = await tokenizer(test, { truncation: true, padding: true });
            if (model.config.is_encoder_decoder) {
              inputs.decoder_input_ids = inputs.input_ids;
            }
            if (feature_extractor) {
              Object.assign(inputs, await feature_extractor(new Float32Array(16000)));
            }

            const output = await model(inputs);

            if (output.logits) {
              // Ensure correct shapes
              const input_ids = inputs.input_ids ?? inputs.decoder_input_ids;
              const expected_shape = [...input_ids.dims, model.config.vocab_size];
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
