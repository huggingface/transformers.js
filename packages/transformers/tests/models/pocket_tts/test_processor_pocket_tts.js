import { jest } from "@jest/globals";
import { AutoProcessor, PocketTTSProcessor, Tensor } from "../../../src/transformers.js";
import { get_files } from "../../../src/utils/model_registry/get_files.js";
import { MAX_PROCESSOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  describe("PocketTTSProcessor", () => {
    const model_id = "onnx-community/phonon-31m-en-ONNX";

    /** @type {PocketTTSProcessor} */
    let processor;
    beforeAll(async () => {
      processor = await AutoProcessor.from_pretrained(model_id);
    }, MAX_PROCESSOR_LOAD_TIME);
    afterEach(() => jest.restoreAllMocks());

    const render = (content) => processor.apply_chat_template([{ role: "user", content }], { add_generation_prompt: true });
    const token_ids = (inputs) => inputs.input_ids.tolist()[0].map(Number);

    it(
      "is listed with the model files and loads the official tokenizer, chat template and config",
      async () => {
        const files = await get_files(model_id, { dtype: "q4", device: "cpu" });
        expect(files).toEqual(expect.arrayContaining(["onnx/model_q4.onnx", "onnx/mimi_decoder_q4.onnx", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"]));

        const loaded = await PocketTTSProcessor.from_pretrained(model_id);
        for (const instance of [processor, loaded]) {
          expect(instance).toBeInstanceOf(PocketTTSProcessor);
          expect(instance.config).toMatchObject({
            processor_class: "PocketTTSProcessor",
            feature_extractor: { sampling_rate: 24000 },
            short_text_max_words: 4,
            frames_after_eos_short_text: 3,
            frames_after_eos: 1,
          });
          expect(instance.apply_chat_template([{ role: "user", content: "Hello." }], { add_generation_prompt: true })).toBe("Hello.<|audio_bos|>");
        }
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it("resolves punctuation IDs once per instance", () => {
      const { tokenizer } = processor;
      const convert = jest.spyOn(tokenizer, "convert_tokens_to_ids");
      const copy = new PocketTTSProcessor(processor.config, { tokenizer }, processor.chat_template);
      expect(convert.mock.calls).toEqual([[[".", "!", "...", "?"]], [[",", ";", ":"]]]);
      expect(copy.config).toBe(processor.config);
      expect(copy.chat_template).toBe(processor.chat_template);

      const [period, comma, semicolon, colon] = tokenizer.convert_tokens_to_ids([".", ",", ";", ":"]);
      expect(copy.sentence_end_ids).toContain(period);
      // The vocabulary has no ";" and ":" pieces, and missing IDs are left out.
      expect([semicolon, colon]).toEqual([undefined, undefined]);
      expect(copy.clause_end_ids).toEqual(new Set([comma]));

      convert.mockClear();
      expect(copy.split_text("hello. goodbye.", 5)).toEqual(["Hello.", "goodbye."]);
      expect(convert).not.toHaveBeenCalledWith([".", "!", "...", "?"]);
      expect(convert).not.toHaveBeenCalledWith([",", ";", ":"]);
    });

    // Token counts: "Hello." is 2, "goodbye." 4, "Price 3.5." 7 and "next." 2.
    it.each([
      ["hello. goodbye.", 5, ["Hello.", "goodbye."]],
      ["hello. goodbye.", 6, ["Hello. goodbye."]],
      ["hello <|audio_bos|> there.", 8, ["Hello there."]],
      ["hello <|audio_prompt|> there.", 8, ["Hello there."]],
      ["hello <|audio_latent|> there.", 8, ["Hello there."]],
      ["<|audio_prompt|>hello. <|audio_latent|>goodbye.<|audio_bos|>", 5, ["Hello.", "goodbye."]],
      ["Hi.Bye.", 50, ["Hi. Bye."]],
      ["Wait... what? Yes! Go.", 4, ["Wait. what?", "Yes! Go."]],
      ["one, two, three, four, five, six.", 6, ["One, two, three,", "four, five, six."]],
      ["a; b: c, d", 3, ["A,", "b,", "c,", "d."]],
      ["price 3.5. next.", 8, ["Price 3.5.", "next."]],
      ["price ١.٥. next.", 8, ["Price ١.٥.", "next."]],
      ["version 2.0 is out. pi is 3.14159.", 8, ["Version 2.0 is out.", "pi is 3.14159."]],
      ["it’s “Hello” — see you 🚀…", 100, ['It\'s "Hello" see you.']],
      ["hi 漢. bye €.", 8, ["Hi 漢.", "bye €."]],
      ["a".repeat(80), 10, ["A" + "a".repeat(79) + "."]],
    ])("splits %j with a budget of %i tokens", (text, budget, expected) => {
      expect(processor.split_text(text, budget)).toEqual(expected);
    });
    it.each(["<|audio_bos|>", " <|audio_prompt|> <|audio_latent|> "])("rejects text containing only audio markers: %s", (text) => {
      expect(() => processor.split_text(text)).toThrow("expects a non-empty text string");
    });
    it("bounds decoding work for long text", () => {
      const text = "Hello. ".repeat(400);
      const decode = jest.spyOn(processor.tokenizer, "decode");
      expect(processor.split_text(text)).toHaveLength(16);
      const num_tokens = processor.tokenizer.tokenize(text).length;
      expect(num_tokens).toBe(800);
      expect(decode.mock.calls.reduce((sum, [ids]) => sum + ids.length, 0)).toBeLessThan(12 * num_tokens);
    });

    it("expands the voice placeholder without adding special tokens", async () => {
      const { tokenizer } = processor;
      const [placeholder, bos] = tokenizer.convert_tokens_to_ids(["<|audio_prompt|>", "<|audio_bos|>"]);
      const voice = new Tensor("float32", [1, 2, 3, 4], [2, 2]);
      const inputs = await processor(render([{ type: "audio" }, { type: "text", text: "Hello." }]), { audio_prompt_latents: [voice] });
      expect(token_ids(inputs)).toEqual([placeholder, placeholder, ...tokenizer.encode("Hello.", { add_special_tokens: false }), bos]);
      expect(inputs.attention_mask.tolist()).toEqual([[1n, 1n, 1n, 1n, 1n]]);
      expect(inputs.audio_prompt_latents.dims).toEqual([1, 2, 2]);
      expect(inputs.audio_prompt_latents.tolist()).toEqual([
        [
          [1, 2],
          [3, 4],
        ],
      ]);
      expect(inputs.audio_prompt_latents_mask.tolist()).toEqual([[1n, 1n]]);
    });
    it.each([
      ["One two three four.", 3n],
      ["One two three four five.", 1n],
    ])("keeps the configured EOS tail for %j", async (text, frames_after_eos) => {
      const inputs = await processor(render(text));
      expect(inputs.frames_after_eos.tolist()).toEqual([frames_after_eos]);
      expect(inputs.audio_prompt_latents).toBeUndefined();
    });
    it("round-trips characters outside the vocabulary", async () => {
      const { tokenizer } = processor;
      const text = "Hi 漢 € there.";
      expect(processor.split_text(text)).toEqual([text]);
      const ids = token_ids(await processor(render(text)));
      expect(ids).not.toContain(tokenizer.convert_tokens_to_ids("<unk>"));
      expect(tokenizer.decode(ids, { skip_special_tokens: true })).toBe(text);
    });
    it("rejects unrendered prompts and mismatched voices", async () => {
      await expect(processor("Hello.")).rejects.toThrow("apply_chat_template");
      await expect(processor("<|audio_prompt|>Hello.<|audio_bos|>")).rejects.toThrow("per voice placeholder");
    });
  });
};
