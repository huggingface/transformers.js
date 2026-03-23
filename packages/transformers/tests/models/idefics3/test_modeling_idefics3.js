import { Idefics3Processor, Idefics3ForConditionalGeneration, RawImage } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  const conversation = [
    {
      role: "user",
      content: [{ type: "image" }, { type: "text", text: "Can you describe this image?" }],
    },
  ];

  // Empty white and black images
  const white_image_dims = [224, 224, 3];
  const white_image = new RawImage(new Uint8ClampedArray(white_image_dims[0] * white_image_dims[1] * white_image_dims[2]).fill(255), ...white_image_dims);
  const black_image_dims = [720, 360, 3];
  const black_image = new RawImage(new Uint8ClampedArray(black_image_dims[0] * black_image_dims[1] * black_image_dims[2]).fill(0), ...black_image_dims);

  describe("Idefics3ForConditionalGeneration", () => {
    const model_id = "hf-internal-testing/tiny-random-Idefics3ForConditionalGeneration";

    /** @type {Idefics3ForConditionalGeneration} */
    let model;
    /** @type {Idefics3Processor} */
    let processor;
    /** @type {string} */
    let text;
    beforeAll(async () => {
      model = await Idefics3ForConditionalGeneration.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      processor = await Idefics3Processor.from_pretrained(model_id);

      text = processor.apply_chat_template(conversation, {
        add_generation_prompt: true,
      });
    }, MAX_MODEL_LOAD_TIME);

    it(
      "forward w/ image splitting (default)",
      async () => {
        const inputs = await processor(text, white_image, {
          do_image_splitting: true,
        });

        const { logits } = await model(inputs);
        expect(logits.dims).toEqual([1, 3041, 128259]);
        expect(logits.mean().item()).toBeCloseTo(-0.0002692154666874558, 6);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "forward w/o image splitting",
      async () => {
        const inputs = await processor(text, white_image, {
          do_image_splitting: false,
        });

        const { logits } = await model(inputs);
        expect(logits.dims).toEqual([1, 189, 128259]);
        expect(logits.mean().item()).toBeCloseTo(-0.00019743280427064747, 6);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "batch_size=1 w/ image splitting",
      async () => {
        const inputs = await processor(text, white_image, {
          do_image_splitting: true,
        });
        const generate_ids = await model.generate({
          ...inputs,
          max_new_tokens: 10,

          // To obtain unique output tokens, deterministically
          repetition_penalty: 2.0,
        });
        expect(generate_ids.dims).toEqual([1, 3051]);

        const new_tokens = generate_ids.slice(null, [inputs.input_ids.dims.at(-1), null]);
        expect(new_tokens.tolist()).toEqual([[64531n, 121777n, 70370n, 105334n, 12720n, 113356n, 47739n, 59240n, 102001n, 60344n]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "batch_size=1 w/o image splitting",
      async () => {
        const inputs = await processor(text, white_image, {
          do_image_splitting: false,
        });
        const generate_ids = await model.generate({
          ...inputs,
          max_new_tokens: 10,

          // To obtain unique output tokens, deterministically
          repetition_penalty: 2.0,
        });
        expect(generate_ids.dims).toEqual([1, 199]);

        const new_tokens = generate_ids.slice(null, [inputs.input_ids.dims.at(-1), null]);
        expect(new_tokens.tolist()).toEqual([[64531n, 121777n, 70370n, 105334n, 12720n, 113356n, 47739n, 59240n, 59697n, 65246n]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "batch_size=1 multi-image w/o image splitting",
      async () => {
        const multi_image_conversation = [
          {
            role: "user",
            content: [{ type: "image" }, { type: "image" }, { type: "text", text: "Can you describe these images?" }],
          },
        ];

        const multi_image_text = processor.apply_chat_template(multi_image_conversation, {
          add_generation_prompt: true,
        });
        const inputs = await processor(multi_image_text, [white_image, black_image], {
          do_image_splitting: false,
        });
        const generate_ids = await model.generate({
          ...inputs,
          max_new_tokens: 10,

          // To obtain unique output tokens, deterministically
          repetition_penalty: 2.0,
        });
        expect(generate_ids.dims).toEqual([1, 374]);

        const new_tokens = generate_ids.slice(null, [inputs.input_ids.dims.at(-1), null]);
        expect(new_tokens.tolist()).toEqual([[73189n, 99346n, 113252n, 51743n, 33499n, 66430n, 78739n, 89539n, 121023n, 14474n]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
