import { jest } from "@jest/globals";
import { mkdtemp, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { AutoProcessor, PocketTTSProcessor, Tensor } from "../../../src/transformers.js";
import { get_files } from "../../../src/utils/model_registry/get_files.js";
import { create_tokenizer, AUDIO_TOKEN_IDS, CHAT_TEMPLATE, TOKENIZER_JSON } from "../gradium_phonon_tts/tokenizer_fixture.js";

export default () => {
  describe("PocketTTSProcessor", () => {
    let processor;
    beforeEach(() => {
      processor = new PocketTTSProcessor({}, { tokenizer: create_tokenizer() }, CHAT_TEMPLATE);
    });
    afterEach(() => jest.restoreAllMocks());

    it.each([AutoProcessor, PocketTTSProcessor])("loads standard metadata with %p", async (ProcessorClass) => {
      const modelPath = await mkdtemp(join(tmpdir(), "pocket-tts-"));
      try {
        const config = { processor_class: "PocketTTSProcessor", feature_extractor: { sampling_rate: 24000 }, frames_after_eos_short_text: 3 };
        await Promise.all(Object.entries({
          "preprocessor_config.json": config,
          "tokenizer_config.json": { tokenizer_class: "PreTrainedTokenizer", unk_token: "<unk>", chat_template: CHAT_TEMPLATE },
          "tokenizer.json": TOKENIZER_JSON,
        }).map(([name, data]) => writeFile(join(modelPath, name), JSON.stringify(data))));
        const files = await get_files(modelPath, {
          config: { model_type: "gradium_phonon_tts", architectures: ["GradiumPhononTTSForConditionalGeneration"] },
          dtype: "q4", device: "cpu",
        });
        expect(files).toEqual([
          "config.json", "onnx/model_q4.onnx", "onnx/mimi_decoder_q4.onnx", "generation_config.json",
          "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json",
        ]);
        const loaded = await ProcessorClass.from_pretrained(modelPath, { local_files_only: true });
        expect(loaded).toBeInstanceOf(PocketTTSProcessor);
        expect(loaded.config).toEqual(config);
        const prompt = loaded.apply_chat_template([{ role: "user", content: "Hello." }], { add_generation_prompt: true });
        expect(prompt).toBe("Hello.<|audio_bos|>");
        const inputs = await loaded(prompt);
        expect(inputs.frames_after_eos.tolist()).toEqual([3n]);
        Object.values(inputs).forEach((tensor) => tensor.dispose());
      } finally {
        await rm(modelPath, { recursive: true, force: true });
      }
    });

    it.each([
      ["hello. goodbye.", 8, ["Hello.", "goodbye."]],
      ["Hi.Bye.", 50, ["Hi. Bye."]],
      ["one,two,three,four,five,six.", 17, ["One, two, three,", "four, five, six."]],
      ["price 3.5. next.", 10, ["Price 3.5.", "next."]],
      ["price ١.٥. next.", 10, ["Price ١.٥.", "next."]],
      ["it’s “Hello” — see you 🚀…", 100, ['It\'s "Hello" see you.']],
      ["hi 漢", 50, ["Hi 漢."]],
      ["a".repeat(80), 10, ["A" + "a".repeat(79) + "."]],
    ])("splits %s like Python", (text, budget, expected) => {
      expect(processor.split_text(text, budget)).toEqual(expected);
    });
    it("bounds decoding work for long text", () => {
      const decode = jest.spyOn(processor.tokenizer, "decode");
      processor.split_text("Hello. ".repeat(400));
      expect(decode.mock.calls.reduce((sum, [ids]) => sum + ids.length, 0)).toBeLessThan(400 * 7 * 12);
    });
    it("returns upstream model inputs with expanded voice placeholders and an EOS tail", async () => {
      const voice = new Tensor("float32", [1, 2, 3, 4], [2, 2]);
      const prompt = processor.apply_chat_template([{ role: "user", content: [{ type: "audio" }, { type: "text", text: "Hello." }] }], { add_generation_prompt: true });
      const inputs = await processor(prompt, { audio_prompt_latents: [voice] });
      expect(Array.from(inputs.input_ids.data).filter((id) => id === BigInt(AUDIO_TOKEN_IDS[1]))).toHaveLength(2);
      expect(inputs.input_ids.data.at(-1)).toBe(BigInt(AUDIO_TOKEN_IDS[0]));
      expect(inputs.audio_prompt_latents.dims).toEqual([1, 2, 2]);
      expect(inputs.audio_prompt_latents_mask.tolist()).toEqual([[1n, 1n]]);
      expect(inputs.frames_after_eos.tolist()).toEqual([4n]);
      Object.values(inputs).forEach((tensor) => tensor.dispose());
      expect(Array.from(voice.data)).toEqual([1, 2, 3, 4]);
      voice.dispose();
    });
    it("rejects unrendered prompts and mismatched voices", async () => {
      await expect(processor("Hello.")).rejects.toThrow("apply_chat_template");
      await expect(processor("<|audio_prompt|>Hello.<|audio_bos|>")).rejects.toThrow("per voice placeholder");
    });
  });
};
