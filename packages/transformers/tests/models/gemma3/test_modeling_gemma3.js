import { Gemma3ForConditionalGeneration, Gemma3ForCausalLM, AutoProcessor, AutoTokenizer, RawImage } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  const CONVERSATION = [
    {
      role: "user",
      content: [{ type: "text", text: "Hello" }],
    },
  ];

  const CONVERSATION_WITH_IMAGE = [
    {
      role: "user",
      content: [{ type: "image" }, { type: "text", text: "Describe this image." }],
    },
  ];

  // Empty white image
  const dims = [224, 224, 3];
  const image = new RawImage(new Uint8ClampedArray(dims[0] * dims[1] * dims[2]).fill(255), ...dims);

  describe("Gemma3ForConditionalGeneration", () => {
    const model_id = "onnx-internal-testing/tiny-random-Gemma3ForConditionalGeneration";

    /** @type {Gemma3ForConditionalGeneration} */
    let model;
    /** @type {AutoProcessor} */
    let processor;
    beforeAll(async () => {
      model = await Gemma3ForConditionalGeneration.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      processor = await AutoProcessor.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "text-only forward",
      async () => {
        const text = processor.apply_chat_template(CONVERSATION, { add_generation_prompt: true });
        const inputs = await processor(text);
        const { logits } = await model(inputs);
        expect(logits.dims).toEqual([1, 11, 262208]);
        expect(logits.mean().item()).toBeCloseTo(-0.004435515962541103, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "text + image forward",
      async () => {
        const text = processor.apply_chat_template(CONVERSATION_WITH_IMAGE, { add_generation_prompt: true });
        const inputs = await processor(text, image);
        const { logits } = await model(inputs);
        expect(logits.dims).toEqual([1, 21, 262208]);
        expect(logits.mean().item()).toBeCloseTo(-0.0029795959126204252, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "text-only (batch_size=1)",
      async () => {
        const text = processor.apply_chat_template(CONVERSATION, { add_generation_prompt: true });
        const inputs = await processor(text);
        const generate_ids = await model.generate({
          ...inputs,
          max_new_tokens: 10,
          do_sample: false,
        });
        const new_tokens = generate_ids.slice(null, [inputs.input_ids.dims.at(-1), null]);
        expect(new_tokens.tolist()).toEqual([[107n, 107n, 107n, 107n, 107n, 107n, 107n, 107n, 107n, 107n]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "text + image (batch_size=1)",
      async () => {
        const text = processor.apply_chat_template(CONVERSATION_WITH_IMAGE, { add_generation_prompt: true });
        const inputs = await processor(text, image);
        const generate_ids = await model.generate({
          ...inputs,
          max_new_tokens: 10,
          do_sample: false,
        });
        const new_tokens = generate_ids.slice(null, [inputs.input_ids.dims.at(-1), null]);
        expect(new_tokens.tolist()).toEqual([[107n, 107n, 107n, 107n, 107n, 107n, 107n, 107n, 107n, 107n]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });

  describe("Gemma3ForCausalLM", () => {
    const model_id = "onnx-internal-testing/tiny-random-Gemma3ForCausalLM";

    /** @type {Gemma3ForCausalLM} */
    let model;
    /** @type {AutoTokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await Gemma3ForCausalLM.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      tokenizer = await AutoTokenizer.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "batch_size=1",
      async () => {
        const inputs = tokenizer("hello");
        const outputs = await model.generate({
          ...inputs,
          max_new_tokens: 5,
          do_sample: false,
        });
        const new_tokens = outputs.slice(null, [inputs.input_ids.dims.at(-1), null]);
        expect(new_tokens.tolist()).toEqual([[23391n, 23391n, 23391n, 23391n, 23391n]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
