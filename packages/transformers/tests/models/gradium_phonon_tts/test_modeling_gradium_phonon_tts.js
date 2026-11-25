import { jest } from "@jest/globals";
import { Tensor as ORTTensor } from "onnxruntime-node";
import { env, pipeline, PretrainedConfig, Tensor, TextToAudioPipeline, GradiumPhononTTSForConditionalGeneration, PocketTTSProcessor } from "../../../src/transformers.js";
import { get_model_files } from "../../../src/utils/model_registry/get_model_files.js";
import { getCacheNames } from "../../../src/configs.js";
import { random } from "../../../src/utils/random.js";
import { create_tokenizer, AUDIO_TOKEN_IDS, CHAT_TEMPLATE } from "./tokenizer_fixture.js";
import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

const config = () =>
  new PretrainedConfig({
    model_type: "gradium_phonon_tts",
    audio_prompt_dim: 2,
    latent_dim: 2,
    frame_rate: 12.5,
    eos_threshold: 4,
    num_speakers: 1,
    duration_delta: 0,
    audio_bos_token_id: AUDIO_TOKEN_IDS[0],
    audio_prompt_token_id: AUDIO_TOKEN_IDS[1],
    audio_latent_token_id: AUDIO_TOKEN_IDS[2],
    num_hidden_layers: 1,
    num_attention_heads: 1,
    num_key_value_heads: 1,
    hidden_size: 2,
    head_dim: 2,
    "transformers.js_config": { dtype: "fp32" },
  });
const voice_url = "https://example.com/voice.bin";
const samples_per_frame = 1920;

const gpu_tensor = (dims) => ORTTensor.fromGpuBuffer({}, { dataType: "float32", dims, dispose: jest.fn() });

function create_pipeline(block_size = 4, eos = () => 5, distinguish_chunks = false) {
  const noise = [],
    outputs = [];
  let decoded_chunks = 0;
  const batch = "batch_size";
  const generator_inputs = [{ name: "input_ids", type: "int64", shape: [batch, "text_length"] }, { name: "audio_prompt_latents", type: "float32", shape: [batch, "voice_length", 2] }, { name: "attention_mask", type: "int64", shape: [batch, "total_sequence_length"] }, { name: "position_ids", type: "int64", shape: [batch, "new_sequence_length"] }, { name: "past_key_values.audio_latents", type: "float32", shape: [batch, "past_frames", 2] }, { name: "noise", type: "float32", shape: [batch, block_size, 2] }, ...["key", "value"].map((kind) => ({ name: `past_key_values.0.${kind}`, type: "float32", shape: [batch, 1, "past_sequence_length", 2] }))];
  const flow = {
    inputMetadata: generator_inputs,
    inputNames: generator_inputs.map(({ name }) => name),
    outputNames: ["audio_latents", "present.audio_latents", "eos_logits", "present.0.key", "present.0.value"],
    run: jest.fn(async (inputs) => {
      const start = inputs["past_key_values.audio_latents"].dims[1],
        frames = start + block_size;
      for (const { name, type, shape } of generator_inputs) {
        expect(inputs[name].type).toBe(type);
        expect(inputs[name].dims).toHaveLength(shape.length);
      }
      const sequence_length = inputs["past_key_values.0.key"].dims[2] + inputs.input_ids.dims[1] + inputs.audio_prompt_latents.dims[1] + block_size;
      noise.push(Array.from(inputs.noise.data));
      const result = {
        ...Object.fromEntries(["key", "value"].map((kind) => [`present.0.${kind}`, gpu_tensor([1, 1, sequence_length, 2])])),
        "present.audio_latents": gpu_tensor([1, frames, 2]),
        audio_latents: new ORTTensor("float32", new Float32Array(frames * 2).fill(0.5), [1, frames, 2]),
        eos_logits: new ORTTensor(
          "float32",
          Array.from({ length: block_size }, (_, i) => eos(start + i)),
          [1, block_size],
        ),
      };
      outputs.push({ ...result });
      return result;
    }),
    release: jest.fn(),
  };
  const decoder = {
    inputNames: ["audio_latents", "num_frames"],
    outputNames: ["audio_values", "audio_lengths"],
    run: jest.fn(async ({ audio_latents, num_frames }) => {
      expect(audio_latents.type).toBe("float32");
      expect(audio_latents.dims).toHaveLength(3);
      expect(num_frames.type).toBe("int64");
      expect(num_frames.dims).toEqual([1]);
      expect(num_frames.data[0]).toBeLessThanOrEqual(BigInt(audio_latents.dims[1]));
      const samples = Number(num_frames.data[0]) * samples_per_frame;
      return {
        audio_values: new ORTTensor("float32", new Float32Array(samples).fill(distinguish_chunks ? ++decoded_chunks * 0.25 : 0.25), [1, 1, samples]),
        audio_lengths: new ORTTensor("int64", [BigInt(samples)], [1]),
      };
    }),
    release: jest.fn(),
  };
  const model = new GradiumPhononTTSForConditionalGeneration(config(), { model: flow, mimi_decoder: decoder }, {});
  const processor = new PocketTTSProcessor(
    {
      feature_extractor: { sampling_rate: 24000, frame_size: samples_per_frame },
      fade_in_duration: 0,
      short_text_max_words: 4,
      frames_after_eos_short_text: 3,
      frames_after_eos: 1,
    },
    { tokenizer: create_tokenizer() },
    CHAT_TEMPLATE,
  );
  const pipe = new TextToAudioPipeline({ model, processor, tokenizer: processor.tokenizer });
  return { pipe, model, processor, flow, decoder, noise, outputs };
}

export default () => {
  describe("GradiumPhononTTSForConditionalGeneration", () => {
    it.each(["fp32", "q4", "q4f16"])("lists both generation and decoder graphs for %s", async (dtype) => {
      const suffix = dtype === "fp32" ? "" : `_${dtype}`;
      expect(await get_model_files("unused", { config: config(), dtype })).toEqual(["config.json", `onnx/model${suffix}.onnx`, `onnx/mimi_decoder${suffix}.onnx`, "generation_config.json"]);
    });
    it("pins feedback and decoder input latents to the GPU", () => {
      expect([...getCacheNames(config(), { prefix: "present", session_name: "model" })]).toEqual(["present.0.key", "present.0.value", "present.audio_latents", "audio_latents"]);
    });

    describe("generation", () => {
      let fixture, voice, owned, fetch;
      beforeEach(() => {
        fixture = create_pipeline();
        voice = new Tensor("float32", [0.5, 0.25], [1, 2, 1]);
        owned = [voice];
        fetch = jest.spyOn(env, "fetch").mockImplementation(async () => new Response(new Float32Array([0.5, 0.25]).buffer));
      });
      afterEach(async () => {
        owned.forEach((tensor) => tensor.dispose());
        await fixture.pipe.dispose();
        expect(fixture.flow.release).toHaveBeenCalledTimes(1);
        expect(fixture.decoder.release).toHaveBeenCalledTimes(1);
        jest.restoreAllMocks();
        random.seed();
      });
      const speak = (options = {}) => fixture.pipe("Hello.", { speaker_embeddings: voice, temperature: 0, ...options });
      const generate = async (options = {}) => {
        const prompt = fixture.processor.apply_chat_template([{ role: "user", content: [{ type: "audio" }, { type: "text", text: "Hello." }] }], { add_generation_prompt: true });
        const row = voice.view(2, 1).transpose(1, 0);
        const inputs = await fixture.processor(prompt, { audio_prompt_latents: [row] });
        owned.push(row, ...Object.values(inputs));
        const result = await fixture.model.generate({ ...inputs, ...options });
        owned.push(result.audio_values);
        return result;
      };

      it("generates decoded audio and releases intermediate state", async () => {
        const output = await generate({ temperature: 0 });
        expect(output.audio_values.dims).toEqual([1, 1, 4 * samples_per_frame]);
        expect(output.audio_values.data.every((x) => x === 0.25)).toBe(true);
        expect(fixture.decoder.run).toHaveBeenCalledTimes(1);
        expect(fixture.decoder.run.mock.calls[0][0].audio_latents).toBe(fixture.outputs[0].audio_latents);
        for (const name of ["audio_latents", "present.audio_latents", "eos_logits", "present.0.key", "present.0.value"]) {
          expect(() => fixture.outputs[0][name].data).toThrow("disposed");
        }
        expect(Array.from(voice.data)).toEqual([0.5, 0.25]);
      });
      it("packs unmasked text and uses KV positions for continuation", async () => {
        const warn = jest.spyOn(console, "warn");
        const [bos, prompt] = AUDIO_TOKEN_IDS;
        const input_ids = new Tensor("int64", [prompt, 42, 43, 44, bos, 3], [1, 6]);
        const attention_mask = new Tensor("int64", [1, 1, 0, 1, 1, 0], [1, 6]);
        const audio_prompt_latents = new Tensor("float32", [0.5, 0.25], [1, 1, 2]);
        owned.push(input_ids, attention_mask, audio_prompt_latents);
        // Four logical prompt positions (one voice placeholder, two text tokens, BOS).
        await generate({ input_ids, attention_mask, audio_prompt_latents, max_length: 10, min_frames_before_eos: 6 });
        expect(fixture.flow.run).toHaveBeenCalledTimes(2);
        const [[prefill], [continuation]] = fixture.flow.run.mock.calls;
        expect(prefill.input_ids.dims).toEqual([1, 2]);
        expect([...prefill.input_ids.data]).toEqual([42n, 44n]);
        expect(prefill.audio_prompt_latents).toBe(audio_prompt_latents.ort_tensor);
        expect(warn).not.toHaveBeenCalledWith(expect.stringContaining("Too many inputs"));
        expect(prefill.position_ids.dims).toEqual([1, 7]);
        expect([...prefill.position_ids.data]).toEqual([0n, 1n, 2n, 3n, 4n, 5n, 6n]);
        expect(prefill.attention_mask.dims).toEqual([1, 7]);
        expect([...prefill.attention_mask.data]).toEqual(Array(7).fill(1n));
        expect(continuation.position_ids.dims).toEqual([1, 4]);
        expect([...continuation.position_ids.data]).toEqual([7n, 8n, 9n, 10n]);
        expect(continuation.attention_mask.dims).toEqual([1, 11]);
        expect([...continuation.attention_mask.data]).toEqual(Array(11).fill(1n));
        expect(fixture.decoder.run.mock.calls[0][0].num_frames.data[0]).toBe(6n);
        expect(fixture.decoder.run.mock.calls[0][0].audio_latents).toBe(fixture.outputs[1].audio_latents);
      });
      const [bos, placeholder, latent] = AUDIO_TOKEN_IDS;
      it.each([
        ["placeholders without a voice", [placeholder, 42, bos], null, "voice frame count"],
        ["wrong placeholder count", [placeholder, placeholder, 42, bos], [1, 125, 2], "voice frame count"],
        ["missing final BOS", [bos, 42], null, "last unmasked token"],
        ["duplicate BOS", [bos, 42, bos], null, "exactly one unmasked audio_bos_token_id"],
        ["placeholder after text", [42, placeholder, bos], [1, 1, 2], "before the first text token"],
        ["audio latent marker", [42, latent, bos], null, "must not contain audio_latent_token_id"],
        ["rank-2 voice", [placeholder, 42, bos], [1, 2], "float32 tensor of shape"],
        ["wrong voice batch size", [placeholder, 42, bos], [2, 1, 2], "float32 tensor of shape"],
        ["wrong voice width", [placeholder, 42, bos], [1, 1, 3], "float32 tensor of shape"],
        ["wrong voice dtype", [placeholder, 42, bos], [1, 1, 2], "float32 tensor of shape", "int64"],
      ])("rejects %s before inference", async (_name, ids, shape, message, dtype = "float32") => {
        const input_ids = new Tensor("int64", ids, [1, ids.length]);
        const audio_prompt_latents = shape ? new Tensor(dtype, Array(shape.reduce((a, b) => a * b, 1)).fill(0), shape) : undefined;
        owned.push(input_ids, ...(audio_prompt_latents ? [audio_prompt_latents] : []));
        await expect(fixture.model.generate({ input_ids, audio_prompt_latents })).rejects.toThrow(message);
        expect(fixture.flow.run).not.toHaveBeenCalled();
      });
      it("generates without a voice when the prompt has no placeholders", async () => {
        const input_ids = new Tensor("int64", [42, bos], [1, 2]);
        owned.push(input_ids);
        const { audio_values } = await fixture.model.generate({ input_ids, max_new_tokens: 1 });
        owned.push(audio_values);
        expect(audio_values.dims).toEqual([1, 1, samples_per_frame]);
        expect(fixture.flow.run.mock.calls[0][0].audio_prompt_latents.dims).toEqual([1, 0, 2]);
      });
      it("rejects a batch of prompts before inference", async () => {
        const input_ids = new Tensor("int64", [42, bos, 43, bos], [2, 2]);
        owned.push(input_ids);
        await expect(fixture.model.generate({ input_ids })).rejects.toThrow("accepts one prompt per call");
        expect(fixture.flow.run).not.toHaveBeenCalled();
      });
      it("rejects a mismatched prompt mask before inference", async () => {
        const attention_mask = new Tensor("int64", [1], [1, 1]);
        owned.push(attention_mask);
        await expect(generate({ attention_mask })).rejects.toThrow("attention_mask must match the shape of input_ids");
        expect(fixture.flow.run).not.toHaveBeenCalled();
      });
      it("warns and truncates at the default safety limit if EOS never fires", async () => {
        const warn = jest.spyOn(console, "warn").mockImplementation(() => {});
        const input_ids = new Tensor("int64", [42, bos], [1, 2]);
        owned.push(input_ids);
        const { audio_values } = await fixture.model.generate({ input_ids, temperature: 0, eos_threshold: 6 });
        owned.push(audio_values);
        expect(warn).toHaveBeenCalledWith(expect.stringContaining("Phonon reached its safety frame limit"));
        expect(audio_values.dims).toEqual([1, 1, 30 * samples_per_frame]);
      });
      it("rejects padded voice prompts before inference", async () => {
        const audio_prompt_latents_mask = new Tensor("int64", [0], [1, 1]);
        owned.push(audio_prompt_latents_mask);
        await expect(generate({ audio_prompt_latents_mask })).rejects.toThrow("unpadded voice prompt");
        expect(fixture.flow.run).not.toHaveBeenCalled();
      });
      it.each([1, 4, 8])("retains the EOS frame and tail across blocks of %i", async (size) => {
        await fixture.pipe.dispose();
        fixture = create_pipeline(size, (frame) => (frame === size - 1 ? 5 : -5));
        const output = await generate({ frames_after_eos: 3 });
        expect(output.audio_values.dims).toEqual([1, 1, (size + 3) * samples_per_frame]);
        expect(fixture.decoder.run).toHaveBeenCalledTimes(1);
        expect(fixture.decoder.run.mock.calls[0][0].num_frames.data[0]).toBe(BigInt(size + 3));
        expect(fixture.flow.run).toHaveBeenCalledTimes(Math.ceil((size + 3) / size));
        const second = fixture.flow.run.mock.calls[1][0];
        const first = fixture.outputs[0];
        expect(second.input_ids.dims).toEqual([1, 0]);
        expect(second.audio_prompt_latents.dims).toEqual([1, 0, 2]);
        expect(second["past_key_values.audio_latents"]).toBe(first["present.audio_latents"]);
        expect(second["past_key_values.0.key"]).toBe(first["present.0.key"]);
      });
      it("honors generation settings and per-call overrides", async () => {
        fixture.model.configs.generation_config = { eos_threshold: 10, min_frames_before_eos: 5, temperature: 0 };
        const output = await generate({ eos_threshold: 4 });
        expect(output.audio_values.dims).toEqual([1, 1, 9 * samples_per_frame]);
        expect(fixture.noise.flat().every((x) => x === 0)).toBe(true);
        expect((await generate({ max_new_tokens: 2 })).audio_values.dims).toEqual([1, 1, 2 * samples_per_frame]);
      });
      it("scales noise by the square root of temperature", async () => {
        random.seed(123);
        await generate({ temperature: 0.3 });
        const baseline = fixture.noise.at(-1);
        random.seed(123);
        await generate({ temperature: 1.2 });
        fixture.noise.at(-1).forEach((value, i) => expect(value).toBeCloseTo(baseline[i] * 2));
        await generate({ do_sample: false });
        expect(fixture.noise.at(-1).every((x) => x === 0)).toBe(true);
      });
      it.each([{ max_new_tokens: 0 }, { frames_after_eos: -1 }, { temperature: -1 }])("rejects invalid generation options %j", async (options) => {
        await expect(generate(options)).rejects.toThrow();
        expect(fixture.flow.run).not.toHaveBeenCalled();
      });
      it("rejects solver settings absent from the export", async () => {
        await expect(generate({ num_flow_steps: 4 })).rejects.toThrow("require re-export");
        expect(fixture.flow.run).not.toHaveBeenCalled();
      });
      it.each(["flow", "decoder"])("releases state when %s inference fails", async (failure) => {
        fixture.model.configs.generation_config = { min_frames_before_eos: 5 };
        jest.spyOn(console, "error").mockImplementation(() => {});
        if (failure === "flow") fixture.flow.run.mockImplementationOnce(fixture.flow.run.getMockImplementation()).mockRejectedValueOnce(new Error("failed"));
        else fixture.decoder.run.mockRejectedValueOnce(new Error("failed"));
        await expect(speak()).rejects.toThrow("failed");
        for (const tensor of Object.values(fixture.outputs[0])) {
          expect(() => tensor.data).toThrow("disposed");
        }
        expect(Array.from(voice.data)).toEqual([0.5, 0.25]);
      });
      it.each([voice_url, new URL(voice_url), new Float32Array([0.5, 0.25])])("accepts voice input %p", async (speaker_embeddings) => {
        const output = await speak({ speaker_embeddings });
        expect(output.audio).toHaveLength(4 * samples_per_frame);
        expect(output.audio.every((x) => x === 0.25)).toBe(true);
        expect(output.sampling_rate).toBe(24000);
      });
      it.each([[[1, 2, 3]], [[6]]])("converts multi-frame channel-first voice data from shape %j", async (shape) => {
        const speaker_embeddings = new Tensor("float32", [1, 2, 3, 10, 20, 30], shape);
        owned.push(speaker_embeddings);
        await speak({ speaker_embeddings });
        const feed = fixture.flow.run.mock.calls[0][0].audio_prompt_latents;
        expect(feed.dims).toEqual([1, 3, 2]);
        expect([...feed.data]).toEqual([1, 10, 2, 20, 3, 30]);
      });
      it("preserves sentence chunk and batch order and starts each prefill fresh", async () => {
        await fixture.pipe.dispose();
        fixture = create_pipeline(4, () => 5, true);
        const first = "Hello there, everyone in this very long room.";
        const second = "See you soon.";
        const output = await fixture.pipe([`${first} ${second}`, "Bye."], { speaker_embeddings: voice });
        const first_samples = 2 * samples_per_frame;
        expect(output.map((x) => x.audio.length)).toEqual([first_samples + 4 * samples_per_frame, 4 * samples_per_frame]);
        expect(output[0].audio.subarray(0, first_samples).every((x) => x === 0.25)).toBe(true);
        expect(output[0].audio.subarray(first_samples).every((x) => x === 0.5)).toBe(true);
        expect(output[1].audio.every((x) => x === 0.75)).toBe(true);
        const prefills = fixture.flow.run.mock.calls.map(([inputs]) => {
          expect(inputs["past_key_values.audio_latents"].dims).toEqual([1, 0, 2]);
          return fixture.processor.tokenizer.decode([...inputs.input_ids.data].map(Number));
        });
        expect(prefills).toEqual([first, second, "Bye."]);
      });
      it("reports missing and malformed voices before inference", async () => {
        await expect(fixture.pipe("Hello.")).rejects.toThrow("Speaker embeddings must be provided");
        await expect(speak({ speaker_embeddings: new Float32Array(3) })).rejects.toThrow("[1, 2, frames]");
        const rank_two = new Tensor("float32", [1, 2, 3, 4, 5, 6], [3, 2]);
        owned.push(rank_two);
        await expect(speak({ speaker_embeddings: rank_two })).rejects.toThrow("use PocketTTSProcessor and model.generate()");
        fetch.mockResolvedValueOnce(new Response("", { status: 404 }));
        await expect(speak({ speaker_embeddings: voice_url })).rejects.toThrow("404");
        expect(fixture.flow.run).not.toHaveBeenCalled();
      });
    });

    describe("onnx-community/phonon-31m-en-ONNX", () => {
      let tts;
      const speaker_embeddings = "https://huggingface.co/onnx-community/phonon-31m-en-ONNX/resolve/main/voices/default.bin";
      beforeAll(async () => {
        tts = await pipeline("text-to-speech", "onnx-community/phonon-31m-en-ONNX", { dtype: "q4" });
      }, MAX_MODEL_LOAD_TIME);
      afterAll(async () => {
        await tts?.dispose();
      });
      it(
        "produces audible speech and stops naturally before the safety limit",
        async () => {
          const warn = jest.spyOn(console, "warn");
          try {
            // Seven text tokens give ceil((7 / 3 + 2) * 12.5) = 55 frames.
            const { audio, sampling_rate } = await tts("Hello from Phonon.", { speaker_embeddings, temperature: 0 });
            expect(sampling_rate).toBe(24000);
            expect(audio.length).toBeGreaterThan(0);
            expect(audio.length).toBeLessThan(55 * samples_per_frame);
            expect(audio.length % samples_per_frame).toBe(0);
            expect(audio.every(Number.isFinite)).toBe(true);
            const rms = Math.sqrt(audio.reduce((sum, sample) => sum + sample * sample, 0) / audio.length);
            expect(rms).toBeGreaterThan(0.03);
            expect(warn).not.toHaveBeenCalled();
          } finally {
            warn.mockRestore();
          }
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it.each([
        [6, 1],
        [11, 2],
      ])(
        "retains %i frames after %i real generator calls",
        async (frames, blocks) => {
          const model = tts.model;
          expect(model).toBeInstanceOf(GradiumPhononTTSForConditionalGeneration);
          expect(Object.keys(model.sessions)).toEqual(["model", "mimi_decoder"]);
          const { model: flow, mimi_decoder } = model.sessions;
          const generator = jest.spyOn(flow, "run");
          const run_decoder = mimi_decoder.run.bind(mimi_decoder);
          const decoder = jest.spyOn(mimi_decoder, "run").mockImplementation(async (inputs) => {
            const result = await run_decoder(inputs);
            expect(result.audio_lengths.dims).toEqual([1]);
            expect(result.audio_lengths.data[0]).toBe(BigInt(frames * samples_per_frame));
            expect(result.audio_values.dims).toEqual([1, 1, frames * samples_per_frame]);
            return result;
          });
          const generation_config = model.configs.generation_config;
          // Make continuation independent of the real graph's EOS timing.
          model.configs.generation_config = { ...generation_config, min_frames_before_eos: frames };
          try {
            const speak = () => tts("Hello from Phonon.", { speaker_embeddings, temperature: 0, max_new_tokens: frames });
            const first = await speak();
            expect(generator).toHaveBeenCalledTimes(blocks);
            expect(decoder).toHaveBeenCalledTimes(1);
            expect(first.audio).toHaveLength(frames * samples_per_frame);
            expect(first.sampling_rate).toBe(24000);
            expect(first.audio.every(Number.isFinite)).toBe(true);
            expect((await speak()).audio).toEqual(first.audio);
          } finally {
            model.configs.generation_config = generation_config;
            decoder.mockRestore();
            generator.mockRestore();
          }
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });
  });
};
