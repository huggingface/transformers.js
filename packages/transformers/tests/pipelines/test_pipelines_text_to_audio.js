import { jest } from "@jest/globals";
import { env, pipeline, AutoProcessor, TextToAudioPipeline, ModelRegistry, Tensor, full } from "../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, MAX_TEST_TIME, DEFAULT_MODEL_OPTIONS } from "../init.js";

const PIPELINE_ID = "text-to-audio";

export default () => {
  describe("Text to Audio", () => {
    const model_id = "Xenova/tiny-random-vits";

    /** @type {TextToAudioPipeline} */
    let pipe;
    beforeAll(async () => {
      pipe = await pipeline(PIPELINE_ID, model_id, DEFAULT_MODEL_OPTIONS);
    }, MAX_MODEL_LOAD_TIME);

    it("should be an instance of TextToAudioPipeline", () => {
      expect(pipe).toBeInstanceOf(TextToAudioPipeline);
    });

    it(
      "default",
      async () => {
        const output = await pipe("hello");
        expect(output.audio).toHaveLength(6400);
        // NOTE: The mean value is not deterministic, so we just check the first few digits
        expect(output.audio.reduce((a, b) => a + b, 0) / output.audio.length).toBeCloseTo(-0.0125, 2);
        expect(output.sampling_rate).toEqual(16000);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await pipe?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });

  it.each(["vits", "musicgen", "supertonic"])("dispatches %s by model type when a processor is present", async (model_type) => {
    const pipe = new TextToAudioPipeline({ model: { config: { model_type } }, processor: {} });
    const method = model_type === "supertonic" ? "_call_supertonic" : "_call_text_to_waveform";
    const call = jest.spyOn(pipe, method).mockResolvedValue("audio");
    expect(await pipe("Hello.", {})).toBe("audio");
    expect(call).toHaveBeenCalledTimes(1);
  });

  describe("Text to Audio (SpeechT5)", () => {
    it(
      "includes the processor in the pipeline files",
      async () => {
        const files = await ModelRegistry.get_pipeline_files(PIPELINE_ID, "Xenova/speecht5_tts", DEFAULT_MODEL_OPTIONS);
        expect(files).toContain("preprocessor_config.json");
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it("returns the vocoder's 1D waveform as audio", async () => {
      const waveform = new Tensor(
        "float32",
        Float32Array.from({ length: 256 }, (_, i) => i / 256),
        [256],
      );
      const model = {
        config: { model_type: "speecht5" },
        generate_speech: async () => ({ waveform }),
      };
      const tokenizer = () => ({ input_ids: new Tensor("int64", new BigInt64Array([4n, 5n, 2n]), [1, 3]) });
      const processor = { feature_extractor: { config: { sampling_rate: 16000 } } };
      const pipe = new TextToAudioPipeline({ task: PIPELINE_ID, model, tokenizer, processor, vocoder: {} });

      const output = await pipe("hello", { speaker_embeddings: new Float32Array(512) });
      expect(output.audio).toEqual(waveform.data);
      expect(output.sampling_rate).toEqual(16000);
    });
  });

  describe("Text to Audio (Phonon)", () => {
    const model_id = "onnx-community/phonon-31m-en-ONNX";
    const voice_url = `https://huggingface.co/${model_id}/resolve/main/voices/default.bin`;

    describe("voices and chunks", () => {
      // Three voice frames of two channels, stored channel-first as in the `voices/*.bin` files.
      const voice_data = () => new Float32Array([1, 2, 3, 10, 20, 30]);
      let processor, generate, pipe;
      beforeAll(async () => {
        // The official tokenizer and chat template have to load before `env.fetch` only serves the voice.
        processor = await AutoProcessor.from_pretrained(model_id);
      }, MAX_MODEL_LOAD_TIME);
      beforeEach(() => {
        jest.spyOn(env, "fetch").mockImplementation(async () => new Response(voice_data().buffer));
        // The model double returns one sample value per call: 1 for the first chunk, 2 for the second, ...
        generate = jest.fn(async () => ({ audio_values: full([1, 1, 4], generate.mock.calls.length) }));
        const model = { config: { model_type: "gradium_phonon_tts", audio_prompt_dim: 2 }, generate };
        pipe = new TextToAudioPipeline({ task: PIPELINE_ID, model, processor, tokenizer: processor.tokenizer });
      });
      afterEach(() => jest.restoreAllMocks());
      const decode = ({ input_ids }) => processor.tokenizer.decode(input_ids.tolist()[0].map(Number), { skip_special_tokens: true });

      it.each([
        ["a URL string", () => voice_url],
        ["a URL", () => new URL(voice_url)],
        ["a Float32Array", voice_data],
        ["a flat tensor", () => new Tensor("float32", voice_data(), [6])],
        ["a [1, audio_prompt_dim, frames] tensor", () => new Tensor("float32", voice_data(), [1, 2, 3])],
      ])("passes a voice given as %s to the model frame by frame", async (_name, speaker_embeddings) => {
        const output = await pipe("Hello.", { speaker_embeddings: speaker_embeddings(), temperature: 0, max_new_tokens: 7 });
        expect(generate).toHaveBeenCalledTimes(1);
        const [[inputs]] = generate.mock.calls;
        expect(inputs.audio_prompt_latents.dims).toEqual([1, 3, 2]);
        expect([...inputs.audio_prompt_latents.data]).toEqual([1, 10, 2, 20, 3, 30]);
        const placeholder = BigInt(processor.tokenizer.convert_tokens_to_ids("<|audio_prompt|>"));
        expect(inputs.input_ids.data.filter((id) => id === placeholder)).toHaveLength(3);
        expect(inputs).toMatchObject({ temperature: 0, max_new_tokens: 7 });
        expect([...output.audio]).toEqual([1, 1, 1, 1]);
        expect(output.sampling_rate).toBe(24000);
      });

      it("strips audio markers from the text", async () => {
        await pipe("Hello <|audio_bos|> there.", { speaker_embeddings: voice_data() });
        expect(generate.mock.calls.map(([inputs]) => decode(inputs))).toEqual(["Hello there."]);
      });

      it("generates sentence chunks and batch items in order", async () => {
        // Each sentence has 36 tokens, so the pair exceeds the 50-token chunk budget.
        const first = "The quick brown fox jumps over the lazy dog near the quiet river bank on a warm summer evening while the birds sing.";
        const second = "Meanwhile, the old gray cat sleeps in the sun by the kitchen window and dreams of catching a fish in the pond.";
        const output = await pipe([`${first} ${second}`, "Bye."], { speaker_embeddings: voice_data() });
        expect(generate.mock.calls.map(([inputs]) => decode(inputs))).toEqual([first, second, "Bye."]);
        expect(output.map(({ audio }) => [...audio])).toEqual([
          [1, 1, 1, 1, 2, 2, 2, 2],
          [3, 3, 3, 3],
        ]);
      });

      it("reports missing and malformed voices before generation", async () => {
        await expect(pipe("Hello.")).rejects.toThrow("Speaker embeddings must be provided");
        await expect(pipe("Hello.", { speaker_embeddings: new Float32Array(3) })).rejects.toThrow("[1, 2, frames]");
        // A [frames, audio_prompt_dim] tensor would otherwise be read as channel-first data.
        const frames_first = new Tensor("float32", voice_data(), [3, 2]);
        await expect(pipe("Hello.", { speaker_embeddings: frames_first })).rejects.toThrow("use PocketTTSProcessor and model.generate()");
        env.fetch.mockResolvedValueOnce(new Response("", { status: 404 }));
        await expect(pipe("Hello.", { speaker_embeddings: voice_url })).rejects.toThrow("404");
        expect(generate).not.toHaveBeenCalled();
      });
    });

    it(
      "generates audible speech with the published q4 graphs",
      async () => {
        const pipe = await pipeline("text-to-speech", model_id, { dtype: "q4" });
        const warn = jest.spyOn(console, "warn");
        try {
          const { audio, sampling_rate } = await pipe("Hello from Phonon.", { speaker_embeddings: voice_url, temperature: 0 });
          expect(sampling_rate).toBe(24000);
          expect(audio.length).toBeGreaterThan(0);
          const rms = Math.sqrt(audio.reduce((sum, sample) => sum + sample * sample, 0) / audio.length);
          expect(rms).toBeGreaterThan(0.03);
          expect(warn).not.toHaveBeenCalled();
        } finally {
          warn.mockRestore();
          await pipe.dispose();
        }
      },
      MAX_TEST_TIME,
    );
  });
};
