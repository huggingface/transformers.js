import { Qwen3_5ForConditionalGeneration, AutoProcessor, RawImage } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  // Empty white image
  const dims = [224, 224, 3];
  const image = new RawImage(new Uint8ClampedArray(dims[0] * dims[1] * dims[2]).fill(255), ...dims);

  // TODO: Enable when "Updating cos_cache and sin_cache in RotaryEmbedding" is supported on CPU Execution Provider.
  // Currently only works on WebGPU.
  describe.skip("Qwen3_5ForConditionalGeneration", () => {
    const model_id = "onnx-internal-testing/tiny-random-Qwen3_5ForConditionalGeneration";

    /** @type {Qwen3_5ForConditionalGeneration} */
    let model;
    /** @type {AutoProcessor} */
    let processor;
    beforeAll(async () => {
      model = await Qwen3_5ForConditionalGeneration.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      processor = await AutoProcessor.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "generate w/ past_key_values",
      async () => {
        const conversation = [
          {
            role: "user",
            content: [{ type: "image" }, { type: "text", text: "What animal is this? Respond in one word." }],
          },
        ];

        const text = processor.apply_chat_template(conversation, {
          add_generation_prompt: true,
        });
        const inputs = await processor(text, image);

        const { sequences, past_key_values } = await model.generate({
          ...inputs,
          max_new_tokens: 8,
          return_dict_in_generate: true,
          do_sample: false,
        });

        const first_response = processor.batch_decode(sequences.slice(null, [inputs.input_ids.dims.at(-1), null]), { skip_special_tokens: true })[0];

        const continued_conversation = [
          ...conversation,
          {
            role: "assistant",
            content: [{ type: "text", text: first_response }],
          },
          {
            role: "user",
            content: [{ type: "text", text: "What color is this animal?" }],
          },
        ];

        const continued_text = processor.apply_chat_template(continued_conversation, {
          add_generation_prompt: true,
        });

        const no_pkv_inputs = await processor(continued_text, image);
        const no_pkv_outputs = await model.generate({
          ...no_pkv_inputs,
          max_new_tokens: 8,
          do_sample: false,
        });

        const pkv_inputs = await processor(continued_text);
        const pkv_outputs = await model.generate({
          ...pkv_inputs,
          image_grid_thw: inputs.image_grid_thw,
          past_key_values,
          max_new_tokens: 8,
          do_sample: false,
        });

        const no_pkv_new_tokens = no_pkv_outputs.slice(null, [no_pkv_inputs.input_ids.dims.at(-1), null]);
        const pkv_new_tokens = pkv_outputs.slice(null, [pkv_inputs.input_ids.dims.at(-1), null]);

        expect(no_pkv_new_tokens.dims).toEqual([1, 8]);
        expect(pkv_new_tokens.dims).toEqual([1, 8]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
