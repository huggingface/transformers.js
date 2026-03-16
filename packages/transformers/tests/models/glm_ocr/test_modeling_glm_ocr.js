import { GlmOcrForConditionalGeneration, Glm46VProcessor, RawImage } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  const CONVERSATION_WITH_IMAGE = [
    {
      role: "user",
      content: [{ type: "image" }, { type: "text", text: "Describe this image." }],
    },
  ];

  // Empty white image
  const dims = [224, 224, 3];
  const image = new RawImage(new Uint8ClampedArray(dims[0] * dims[1] * dims[2]).fill(255), ...dims);

  describe("GlmOcrForConditionalGeneration", () => {
    const model_id = "onnx-internal-testing/tiny-random-GlmOcrForConditionalGeneration";

    /** @type {GlmOcrForConditionalGeneration} */
    let model;
    /** @type {Glm46VProcessor} */
    let processor;
    beforeAll(async () => {
      model = await GlmOcrForConditionalGeneration.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      processor = await Glm46VProcessor.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "forward",
      async () => {
        const text = processor.apply_chat_template(CONVERSATION_WITH_IMAGE, {
          add_generation_prompt: true,
        });
        const inputs = await processor(text, image);
        const { logits } = await model(inputs);
        expect(logits.dims).toEqual([1, 76, 59392]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "text + image (batch_size=1)",
      async () => {
        const text = processor.apply_chat_template(CONVERSATION_WITH_IMAGE, {
          add_generation_prompt: true,
        });
        const inputs = await processor(text, image);
        const generate_ids = await model.generate({
          ...inputs,
          max_new_tokens: 10,
        });

        const new_tokens = generate_ids.slice(null, [inputs.input_ids.dims.at(-1), null]);
        expect(new_tokens.tolist()).toEqual([[3875n, 22214n, 21946n, 27197n, 15231n, 15231n, 15231n, 15231n, 15231n, 15231n]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
