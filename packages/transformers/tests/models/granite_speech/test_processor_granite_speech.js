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
  });
};
