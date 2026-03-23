import { AutoProcessor, VLChatProcessor } from "../../../src/transformers.js";

import { MAX_PROCESSOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  describe("VLChatProcessor", () => {
    const model_id = "onnx-community/Janus-1.3B-ONNX";

    /** @type {VLChatProcessor} */
    let processor;
    beforeAll(async () => {
      processor = await AutoProcessor.from_pretrained(model_id);
    }, MAX_PROCESSOR_LOAD_TIME);

    it(
      "Image and text",
      async () => {
        // Prepare inputs
        const conversation = [
          {
            role: "User",
            content: "<image_placeholder>\nConvert the formula into latex code.",
            images: ["https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/quadratic_formula.png"],
          },
        ];

        const { input_ids, attention_mask, images_seq_mask, images_emb_mask, pixel_values, original_sizes, reshaped_input_sizes } = await processor(conversation);
        const num_tokens = 631;
        const { num_image_tokens } = processor.config; // 576
        const { image_size } = processor.image_processor.config; // 384

        expect(input_ids.dims).toEqual([1, num_tokens]);
        expect(attention_mask.dims).toEqual([1, num_tokens]);
        expect(images_seq_mask.dims).toEqual([1, num_tokens]);
        expect(images_seq_mask.to("float32").mean().item()).toBeCloseTo(num_image_tokens / num_tokens, 6);
        expect(images_emb_mask.dims).toEqual([1, 1, num_image_tokens]);
        expect(images_emb_mask.to("float32").mean().item()).toBeCloseTo(1);
        expect(pixel_values.dims).toEqual([1, 1, 3, image_size, image_size]);
        expect(pixel_values.mean().item()).toBeCloseTo(0.5999642610549927, 6);

        expect(original_sizes).toEqual([[206, 767]]);
        expect(reshaped_input_sizes).toEqual([[103, image_size]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
