import { Qwen2VLProcessor, Qwen2VLForConditionalGeneration, RawImage } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  const CONVERSATION = [
    {
      role: "user",
      content: [{ type: "text", text: "Hello" }],
    },
  ];

  // Example adapted from https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
  const CONVERSATION_WITH_IMAGE = [
    {
      role: "user",
      content: [{ type: "image" }, { type: "text", text: "Describe this image." }],
    },
  ];
  // Empty white image
  const dims = [224, 224, 3];
  const image = new RawImage(new Uint8ClampedArray(dims[0] * dims[1] * dims[2]).fill(255), ...dims);

  describe("Qwen2VLForConditionalGeneration", () => {
    const model_id = "hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration";

    /** @type {Qwen2VLForConditionalGeneration} */
    let model;
    /** @type {Qwen2VLProcessor} */
    let processor;
    beforeAll(async () => {
      model = await Qwen2VLForConditionalGeneration.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      processor = await Qwen2VLProcessor.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "forward",
      async () => {
        const text = processor.apply_chat_template(CONVERSATION_WITH_IMAGE, {
          add_generation_prompt: true,
        });
        const inputs = await processor(text, image);
        const { logits } = await model(inputs);
        expect(logits.dims).toEqual([1, 89, 152064]);
        expect(logits.mean().item()).toBeCloseTo(-0.0011299321195110679, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "text-only (batch_size=1)",
      async () => {
        const text = processor.apply_chat_template(CONVERSATION, {
          add_generation_prompt: true,
        });
        const inputs = await processor(text);
        const generate_ids = await model.generate({
          ...inputs,
          max_new_tokens: 10,
        });

        const new_tokens = generate_ids.slice(null, [inputs.input_ids.dims.at(-1), null]);
        expect(new_tokens.tolist()).toEqual([[24284n, 63986n, 108860n, 84530n, 8889n, 23262n, 128276n, 64948n, 136757n, 138348n]]);
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
        expect(new_tokens.tolist()).toEqual([[24284n, 35302n, 60575n, 38679n, 113390n, 115118n, 137596n, 38241n, 96726n, 142301n]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
