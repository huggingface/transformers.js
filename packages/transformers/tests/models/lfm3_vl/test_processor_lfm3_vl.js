import { AutoProcessor, Lfm3VlProcessor, RawImage } from "../../../src/transformers.js";

import { MAX_PROCESSOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  describe("Lfm3VlProcessor", () => {
    const model_id = "onnx-community/LFM3-VL-2B-ONNX";

    /** @type {Lfm3VlProcessor} */
    let processor;
    beforeAll(async () => {
      processor = await AutoProcessor.from_pretrained(model_id);
    }, MAX_PROCESSOR_LOAD_TIME);

    it("should be an instance of Lfm3VlProcessor", () => {
      expect(processor).toBeInstanceOf(Lfm3VlProcessor);
    });

    it(
      "image and text",
      async () => {
        const image = new RawImage(new Uint8Array(224 * 224 * 3).fill(255), 224, 224, 3);
        const messages = [{ role: "user", content: [{ type: "image" }, { type: "text", text: "Describe this image." }] }];
        const text = processor.apply_chat_template(messages, { add_generation_prompt: true });
        const { input_ids, attention_mask, pixel_values, image_grid_thw } = await processor(image, text, { add_special_tokens: false });

        expect(input_ids.data.filter((id) => id === 125030n)).toHaveLength(64);
        expect(attention_mask.dims).toEqual(input_ids.dims);
        expect(pixel_values.dims).toEqual([256, 1536]);
        expect(image_grid_thw.tolist()).toEqual([[1n, 16n, 16n]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "expands image placeholders in batch order",
      async () => {
        const black = new RawImage(new Uint8Array(512 * 256 * 3), 512, 256, 3);
        const white = new RawImage(new Uint8Array(224 * 224 * 3).fill(255), 224, 224, 3);
        const { input_ids, attention_mask } = await processor([black, white], ["<|image|> Hello", "<|image|>"], { padding: true, add_special_tokens: false });

        expect(input_ids.tolist().map((ids) => ids.filter((id) => id === 125030n).length)).toEqual([128, 64]);
        expect(attention_mask.tolist()[1].filter((value) => value === 1n)).toHaveLength(64);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "supports text-only inputs and rejects mismatched image placeholders",
      async () => {
        expect((await processor(null, "Hello")).input_ids.tolist()).toEqual(processor.tokenizer("Hello").input_ids.tolist());
        await expect(processor(null, "<|image|>")).rejects.toThrow("received 0 images");
        const image = new RawImage(new Uint8Array(256 * 256 * 3), 256, 256, 3);
        await expect(processor(image, "Hello")).rejects.toThrow("Found 0");
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
