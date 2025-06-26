import { LlavaForConditionalGeneration, RawImage, LlavaProcessor } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  const prompts = [
    // Example adapted from https://huggingface.co/docs/transformers/model_doc/llava#transformers.LlavaForConditionalGeneration.forward.example
    "USER: <image>\nWhat's the content of the image? ASSISTANT:",
    "<image>Hi",
  ];

  // Empty white image
  const dims = [224, 224, 3];
  const image = new RawImage(new Uint8ClampedArray(dims[0] * dims[1] * dims[2]).fill(255), ...dims);

  describe("LlavaForConditionalGeneration", () => {
    const model_id = "Xenova/tiny-random-LlavaForConditionalGeneration";

    /** @type {LlavaForConditionalGeneration} */
    let model;
    /** @type {LlavaProcessor} */
    let processor;
    beforeAll(async () => {
      model = await LlavaForConditionalGeneration.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      processor = await LlavaProcessor.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "forward",
      async () => {
        const inputs = await processor(image, prompts[0]);
        const { logits } = await model(inputs);
        expect(logits.dims).toEqual([1, 246, 32002]);
        expect(logits.mean().item()).toBeCloseTo(-0.0005688573000952601, 8);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "batch_size=1",
      async () => {
        const inputs = await processor(image, prompts[0]);
        const generate_ids = await model.generate({ ...inputs, max_new_tokens: 10 });
        expect(generate_ids.dims).toEqual([1, 256]);
        const new_ids = generate_ids.slice(null, [inputs.input_ids.dims[1], null]);
        expect(new_ids.tolist()).toEqual([[21557n, 16781n, 27238n, 8279n, 20454n, 11927n, 12462n, 12306n, 2414n, 7561n]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "batch_size>1",
      async () => {
        const inputs = await processor([image, image], prompts, {
          padding: true,
        });
        const generate_ids = await model.generate({ ...inputs, max_new_tokens: 10 });
        const new_ids = generate_ids.slice(null, [inputs.input_ids.dims[1], null]);
        expect(new_ids.tolist()).toEqual([
          [21557n, 16781n, 27238n, 8279n, 20454n, 11927n, 12462n, 12306n, 2414n, 7561n],
          [1217n, 22958n, 22913n, 10381n, 148n, 31410n, 31736n, 7358n, 9150n, 28635n],
        ]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "generate w/ past_key_values",
      async () => {
        // Empty white image
        const dims = [224, 224, 3];
        const image = new RawImage(new Uint8ClampedArray(dims[0] * dims[1] * dims[2]).fill(255), ...dims);
        const inputs = await processor(image, prompts[0]);

        // Generate first sequence w/o PKV
        // NOTE: `return_dict_in_generate=true` is required to get PKV
        const { past_key_values, sequences } = await model.generate({
          ...inputs,
          max_new_tokens: 5,
          do_sample: false,
          return_dict_in_generate: true,
        });

        // Run w/o PKV
        const generated_ids = await model.generate({
          ...inputs,
          max_new_tokens: 8,
          do_sample: false,
        });

        // Run w/ PKV
        const generated_ids_pkv = await model.generate({
          input_ids: sequences,
          past_key_values,
          max_new_tokens: 3,
          do_sample: false,
        });

        const result = generated_ids.slice(null, [inputs.input_ids.dims[1], null]).tolist();
        const result_pkv = generated_ids_pkv.slice(null, [inputs.input_ids.dims[1], null]).tolist();

        // Ensure output is the same and correct
        const target = [[21557n, 16781n, 27238n, 8279n, 20454n, 11927n, 12462n, 12306n]];
        expect(result).toEqual(target);
        expect(result_pkv).toEqual(target);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
