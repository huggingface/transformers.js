import { AutoTokenizer, AutoProcessor, load_image, CLIPVisionModelWithProjection, CLIPTextModelWithProjection } from "../../../src/transformers.js";

import { MAX_TEST_EXECUTION_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  const models_to_test = ["hf-internal-testing/tiny-random-CLIPModel"];
  it(
    `CLIPTextModelWithProjection`,
    async () => {
      const model_id = models_to_test[0];

      // Load tokenizer and text model
      const tokenizer = await AutoTokenizer.from_pretrained(model_id);
      const text_model = await CLIPTextModelWithProjection.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);

      // Run tokenization
      const texts = ["a photo of a car", "a photo of a football match"];
      const text_inputs = tokenizer(texts, { padding: true, truncation: true });

      // Compute embeddings
      const { text_embeds } = await text_model(text_inputs);

      // Ensure correct shapes
      const expected_shape = [texts.length, text_model.config.projection_dim];
      const actual_shape = text_embeds.dims;
      expect(expected_shape).toEqual(actual_shape);

      await text_model.dispose();
    },
    MAX_TEST_EXECUTION_TIME,
  );

  it(
    `CLIPVisionModelWithProjection`,
    async () => {
      const model_id = models_to_test[0];

      // Load processor and vision model
      const processor = await AutoProcessor.from_pretrained(model_id);
      const vision_model = await CLIPVisionModelWithProjection.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);

      // Read image and run processor
      const image = await load_image("https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/football-match.jpg");
      const image_inputs = await processor(image);

      // Compute embeddings
      const { image_embeds } = await vision_model(image_inputs);

      // Ensure correct shapes
      const expected_shape = [1, vision_model.config.projection_dim];
      const actual_shape = image_embeds.dims;
      expect(expected_shape).toEqual(actual_shape);

      await vision_model.dispose();
    },
    MAX_TEST_EXECUTION_TIME,
  );
};
