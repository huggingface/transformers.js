/*
 * Test that models loaded outside of the `pipeline` function work correctly (e.g., `AutoModel.from_pretrained(...)`);
 */

import { AutoTokenizer, AutoModel, AutoProcessor, BertModel, GPT2Model, T5ForConditionalGeneration, CLIPTextModelWithProjection, CLIPVisionModelWithProjection, BertTokenizer, GPT2Tokenizer, T5Tokenizer, RawImage } from "../src/transformers.js";

import { init, MAX_TEST_EXECUTION_TIME } from "./init.js";

import { compare } from "./test_utils.js";

// Initialise the testing environment
init();

describe("Models", () => {
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
            const model = await modelClassToTest.from_pretrained(model_id);

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

  describe("Running specific models", () => {
    const models_to_test = ["hf-internal-testing/tiny-random-CLIPModel"];
    it(
      `CLIP (text)`,
      async () => {
        const model_id = models_to_test[0];

        // Load tokenizer and text model
        const tokenizer = await AutoTokenizer.from_pretrained(model_id);
        const text_model = await CLIPTextModelWithProjection.from_pretrained(model_id, { revision: "refs/pr/5" });

        // Run tokenization
        const texts = ["a photo of a car", "a photo of a football match"];
        const text_inputs = tokenizer(texts, { padding: true, truncation: true });

        // Compute embeddings
        const { text_embeds } = await text_model(text_inputs);

        // Ensure correct shapes
        const expected_shape = [texts.length, text_model.config.projection_dim];
        const actual_shape = text_embeds.dims;
        compare(expected_shape, actual_shape);

        await text_model.dispose();
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      `CLIP (vision)`,
      async () => {
        const model_id = models_to_test[0];

        // Load processor and vision model
        const processor = await AutoProcessor.from_pretrained(model_id);
        const vision_model = await CLIPVisionModelWithProjection.from_pretrained(model_id, { revision: "refs/pr/5" });

        // Read image and run processor
        const image = await RawImage.read("https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/football-match.jpg");
        const image_inputs = await processor(image);

        // Compute embeddings
        const { image_embeds } = await vision_model(image_inputs);

        // Ensure correct shapes
        const expected_shape = [1, vision_model.config.projection_dim];
        const actual_shape = image_embeds.dims;
        compare(expected_shape, actual_shape);

        await vision_model.dispose();
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
});
