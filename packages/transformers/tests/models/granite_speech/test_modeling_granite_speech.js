import { GPT2Tokenizer, GraniteSpeechForConditionalGeneration } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  describe("GraniteSpeechForConditionalGeneration", () => {
    const model_id = "onnx-internal-testing/tiny-random-GraniteSpeechForConditionalGeneration";

    /** @type {GraniteSpeechForConditionalGeneration} */
    let model;
    /** @type {GPT2Tokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await GraniteSpeechForConditionalGeneration.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      tokenizer = await GPT2Tokenizer.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "text-only generation",
      async () => {
        const messages = [{ role: "user", content: "What is the capital of France?" }];
        const text = tokenizer.apply_chat_template(messages, {
          add_generation_prompt: true,
          tokenize: false,
        });

        const inputs = tokenizer(text, { add_special_tokens: false });
        const outputs = await model.generate({
          ...inputs,
          max_new_tokens: 10,
        });

        expect(outputs.tolist()).toEqual([[6584n, 25n, 3639n, 374n, 279n, 6864n, 315n, 9822n, 30n, 198n, 36660n, 3931n, 2891n, 25n, 84130n, 50307n, 50306n, 77183n, 9714n, 7869n, 2959n, 36251n, 63351n, 16719n]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
