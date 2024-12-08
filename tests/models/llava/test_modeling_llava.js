import { LlamaTokenizer, CLIPImageProcessor, LlavaForConditionalGeneration, RawImage } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  const prompts = [
    // Example adapted from https://huggingface.co/docs/transformers/model_doc/llava#transformers.LlavaForConditionalGeneration.forward.example
    "<image>\nUSER: What's the content of the image?\nASSISTANT:",
    "<image>Hi",
  ];

  // Empty white image
  const dims = [224, 224, 3];
  const image = new RawImage(new Uint8ClampedArray(dims[0] * dims[1] * dims[2]).fill(255), ...dims);

  describe("LlavaForConditionalGeneration", () => {
    const model_id = "Xenova/tiny-random-LlavaForConditionalGeneration";

    /** @type {LlavaForConditionalGeneration} */
    let model;
    /** @type {LlamaTokenizer} */
    let tokenizer;
    /** @type {CLIPImageProcessor} */
    let processor;
    beforeAll(async () => {
      model = await LlavaForConditionalGeneration.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      tokenizer = await LlamaTokenizer.from_pretrained(model_id);
      processor = await CLIPImageProcessor.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "forward",
      async () => {
        const text_inputs = tokenizer(prompts[0]);
        const vision_inputs = await processor(image);
        const inputs = { ...text_inputs, ...vision_inputs };

        const { logits } = await model(inputs);
        expect(logits.dims).toEqual([1, 244, 32002]);
        expect(logits.mean().item()).toBeCloseTo(-0.0005755752790719271, 8);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "batch_size=1",
      async () => {
        const text_inputs = tokenizer(prompts[0]);
        const vision_inputs = await processor(image);
        const inputs = { ...text_inputs, ...vision_inputs };

        const generate_ids = await model.generate({ ...inputs, max_new_tokens: 10 });
        expect(generate_ids.tolist()).toEqual([[1n, 32000n, 29871n, 13n, 11889n, 29901n, 1724n, 29915n, 29879n, 278n, 2793n, 310n, 278n, 1967n, 29973n, 13n, 22933n, 9047n, 13566n, 29901n, 21557n, 16781n, 27238n, 8279n, 20454n, 11927n, 12462n, 12306n, 2414n, 7561n]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "batch_size>1",
      async () => {
        const text_inputs = tokenizer(prompts, { padding: true });
        const vision_inputs = await processor([image, image]);
        const inputs = { ...text_inputs, ...vision_inputs };

        const generate_ids = await model.generate({ ...inputs, max_new_tokens: 10 });
        expect(generate_ids.tolist()).toEqual([
          [1n, 32000n, 29871n, 13n, 11889n, 29901n, 1724n, 29915n, 29879n, 278n, 2793n, 310n, 278n, 1967n, 29973n, 13n, 22933n, 9047n, 13566n, 29901n, 21557n, 16781n, 27238n, 8279n, 20454n, 11927n, 12462n, 12306n, 2414n, 7561n],
          [0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 1n, 32000n, 6324n, 1217n, 22958n, 22913n, 10381n, 148n, 31410n, 31736n, 7358n, 9150n, 28635n],
        ]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
