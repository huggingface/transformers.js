import { AutoProcessor, GraniteSpeechProcessor } from "../../../src/transformers.js";

import { load_cached_audio } from "../../asset_cache.js";
import { MAX_PROCESSOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  describe("GraniteSpeechProcessor", () => {
    const model_id = "onnx-community/granite-4.0-1b-speech-ONNX";

    /** @type {GraniteSpeechProcessor} */
    let processor;
    beforeAll(async () => {
      processor = await AutoProcessor.from_pretrained(model_id);
    }, MAX_PROCESSOR_LOAD_TIME);

    it(
      "text + audio",
      async () => {
        const audio = await load_cached_audio("mlk");

        const messages = [{ role: "user", content: "<|audio|>can you transcribe the speech into a written format?" }];
        const text = processor.tokenizer.apply_chat_template(messages, {
          add_generation_prompt: true,
          tokenize: false,
        });

        const { input_ids, input_features } = await processor(text, audio);

        // input_ids: 151 tokens (text + expanded audio tokens)
        expect(input_ids.dims).toEqual([1, 151]);

        // input_features: [1, 650, 160]
        expect(input_features.dims).toEqual([1, 650, 160]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "tokens and features stay aligned for boundary audio lengths",
      async () => {
        const boundary_lengths = [4960, 9760, 139360]; // For L = 160 * (30k + 1)
        const { projector_window_size, projector_downsample_rate } = processor.feature_extractor.config;
        const effective_window_size = Math.floor(projector_window_size / projector_downsample_rate);

        for (const L of boundary_lengths) {
          const audio = new Float32Array(L);
          const { input_features } = await processor.feature_extractor(audio);

          const predicted_tokens = processor._get_num_audio_features(L);
          const time_steps = input_features.dims[1];
          const projector_tokens = Math.ceil(time_steps / projector_window_size) * effective_window_size;

          expect(projector_tokens).toEqual(predicted_tokens);
        }
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
