import { AutoProcessor, Gemma4Processor } from "../../../src/transformers.js";

import { load_cached_image, load_cached_audio } from "../../asset_cache.js";
import { MAX_PROCESSOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  describe.skip("Gemma4Processor", () => {
    const model_id = "onnx-community/gemma-4-E2B-it-ONNX";

    /** @type {Gemma4Processor} */
    let processor;

    beforeAll(async () => {
      processor = await AutoProcessor.from_pretrained(model_id);
    }, MAX_PROCESSOR_LOAD_TIME);

    // ===== IMAGE TESTS =====

    it(
      "single image (cats, 640x480)",
      async () => {
        const image = await load_cached_image("cats");
        const messages = [{ role: "user", content: [{ type: "image" }, { type: "text", text: "Describe." }] }];
        const text = processor.apply_chat_template(messages, { add_generation_prompt: true });
        const { input_ids, pixel_values, image_position_ids, num_soft_tokens_per_image } = await processor(text, image, null, { add_special_tokens: false });

        // Python ref: pv_shape=(1, 2520, 768), num_soft=[266]
        expect(pixel_values.dims).toEqual([1, 2520, 768]);
        expect(image_position_ids.dims).toEqual([1, 2520, 2]);
        expect(num_soft_tokens_per_image).toEqual([266]);

        // input_ids should contain expanded image tokens
        expect(input_ids.dims[0]).toEqual(1);
        expect(input_ids.dims[1]).toBeGreaterThan(266);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "single image (tiger, 612x408)",
      async () => {
        const image = await load_cached_image("tiger");
        const messages = [{ role: "user", content: [{ type: "image" }, { type: "text", text: "Describe." }] }];
        const text = processor.apply_chat_template(messages, { add_generation_prompt: true });
        const { pixel_values, num_soft_tokens_per_image } = await processor(text, image, null, { add_special_tokens: false });

        // Python ref: pv_shape=(1, 2520, 768), num_soft=[260]
        expect(pixel_values.dims).toEqual([1, 2520, 768]);
        expect(num_soft_tokens_per_image).toEqual([260]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "single image (white, 224x224)",
      async () => {
        const image = await load_cached_image("white_image");
        const messages = [{ role: "user", content: [{ type: "image" }, { type: "text", text: "Describe." }] }];
        const text = processor.apply_chat_template(messages, { add_generation_prompt: true });
        const { pixel_values, num_soft_tokens_per_image } = await processor(text, image, null, { add_special_tokens: false });

        // Python ref: pv_shape=(1, 2520, 768), num_soft=[256]
        expect(pixel_values.dims).toEqual([1, 2520, 768]);
        expect(num_soft_tokens_per_image).toEqual([256]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "single image (gradient, 1280x640)",
      async () => {
        const image = await load_cached_image("gradient_1280x640");
        const messages = [{ role: "user", content: [{ type: "image" }, { type: "text", text: "Describe." }] }];
        const text = processor.apply_chat_template(messages, { add_generation_prompt: true });
        const { pixel_values, num_soft_tokens_per_image } = await processor(text, image, null, { add_special_tokens: false });

        // Python ref: pv_shape=(1, 2520, 768), num_soft=[253]
        expect(pixel_values.dims).toEqual([1, 2520, 768]);
        expect(num_soft_tokens_per_image).toEqual([253]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "batched images (cats + tiger)",
      async () => {
        const cats = await load_cached_image("cats");
        const tiger = await load_cached_image("tiger");
        const messages = [
          {
            role: "user",
            content: [{ type: "image" }, { type: "image" }, { type: "text", text: "Compare." }],
          },
        ];
        const text = processor.apply_chat_template(messages, { add_generation_prompt: true });
        const { pixel_values, num_soft_tokens_per_image } = await processor(text, [cats, tiger], null, { add_special_tokens: false });

        // Python ref: pv_shape=(2, 2520, 768), num_soft=[266, 260]
        expect(pixel_values.dims).toEqual([2, 2520, 768]);
        expect(num_soft_tokens_per_image).toEqual([266, 260]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    // ===== AUDIO TOKEN COUNT TESTS =====

    it("_compute_audio_num_tokens", () => {
      // Python reference values
      expect(processor._compute_audio_num_tokens(208000, 16000)).toEqual(325);
      expect(processor._compute_audio_num_tokens(16000, 16000)).toEqual(25);
      expect(processor._compute_audio_num_tokens(800, 16000)).toEqual(1);
      expect(processor._compute_audio_num_tokens(48000, 16000)).toEqual(75);
      expect(processor._compute_audio_num_tokens(160000, 16000)).toEqual(250);
      expect(processor._compute_audio_num_tokens(480000, 16000)).toEqual(750);
      expect(processor._compute_audio_num_tokens(100, 16000)).toEqual(0);
    });

    // ===== AUDIO + TEXT TESTS =====

    it(
      "audio + text",
      async () => {
        const audio = await load_cached_audio("mlk");
        const messages = [{ role: "user", content: [{ type: "audio" }, { type: "text", text: "Transcribe." }] }];
        const text = processor.apply_chat_template(messages, { add_generation_prompt: true });

        const { input_ids, input_features, input_features_mask } = await processor(text, null, audio, { add_special_tokens: false });

        expect(input_features.dims).toEqual([1, 1299, 128]);
        expect(input_features_mask.dims).toEqual([1, 1299]);

        // 325 audio tokens should be expanded in input_ids
        expect(input_ids.dims[1]).toBeGreaterThan(325);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    // ===== IMAGE + AUDIO + TEXT =====

    it(
      "image + audio + text",
      async () => {
        const image = await load_cached_image("cats");
        const audio = await load_cached_audio("mlk");
        const messages = [
          {
            role: "user",
            content: [{ type: "image" }, { type: "audio" }, { type: "text", text: "Describe and transcribe." }],
          },
        ];
        const text = processor.apply_chat_template(messages, { add_generation_prompt: true });

        const { input_ids, pixel_values, image_position_ids, num_soft_tokens_per_image, input_features } = await processor(text, image, audio, { add_special_tokens: false });

        // Image outputs
        expect(pixel_values.dims).toEqual([1, 2520, 768]);
        expect(image_position_ids.dims).toEqual([1, 2520, 2]);
        expect(num_soft_tokens_per_image).toEqual([266]);

        // Audio outputs
        expect(input_features.dims).toEqual([1, 1299, 128]);

        // Combined input_ids should have both expanded image (266) and audio (325) tokens
        expect(input_ids.dims[1]).toBeGreaterThan(266 + 325);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    // ===== TEXT-ONLY =====

    it("text only", async () => {
      const messages = [{ role: "user", content: [{ type: "text", text: "Hello" }] }];
      const text = processor.apply_chat_template(messages, { add_generation_prompt: true });
      const { input_ids } = await processor(text, null, null, { add_special_tokens: false });

      // Python ref: ids_shape=(1, 10)
      expect(input_ids.dims).toEqual([1, 10]);
    });
  });
};
