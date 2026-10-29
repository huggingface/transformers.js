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

const gpu_tensor = (dims) => ORTTensor.fromGpuBuffer({}, { dataType: "float32", dims, dispose: jest.fn() });

function create_pipeline(block_size = 4, eos = () => 5) {
  const noise = [],
    outputs = [];
  const flow = {
    inputMetadata: [{ name: "noise", shape: [1, block_size, 2] }, { name: "past_key_values.audio_latents", type: "float32", shape: [1, "past_frames", 2] }, ...["key", "value"].map((kind) => ({ name: `past_key_values.0.${kind}`, type: "float32", shape: [1, 1, "past_length", 2] }))],
    inputNames: ["input_ids", "audio_prompt_latents", "past_key_values.audio_latents", "noise", "attention_mask", "audio_prompt_latents_mask", "past_key_values.0.key", "past_key_values.0.value"],
    outputNames: ["audio_latents", "present.audio_latents", "eos_logits", "present.0.key", "present.0.value"],
    run: jest.fn(async (inputs) => {
      const start = inputs["past_key_values.audio_latents"].dims[1],
        frames = start + block_size;
      noise.push(Array.from(inputs.noise.data));
      const result = {
        ...Object.fromEntries(["key", "value"].map((kind) => [`present.0.${kind}`, gpu_tensor([1])])),
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
    run: jest.fn(async ({ num_frames }) => {
      const samples = Number(num_frames.data[0]) * 2;
      return { audio_values: new ORTTensor("float32", new Float32Array(samples).fill(0.25), [1, 1, samples]) };
    }),
    release: jest.fn(),
  };
  const model = new GradiumPhononTTSForConditionalGeneration(config(), { model: flow, mimi_decoder: decoder }, {});
  const processor = new PocketTTSProcessor(
    {
      feature_extractor: { sampling_rate: 24000, frame_size: 2 },
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
        expect(output.audio_values.dims).toEqual([1, 1, 8]);
        expect(Array.from(output.audio_values.data)).toEqual(Array(8).fill(0.25));
        expect(fixture.decoder.run).toHaveBeenCalledTimes(1);
        expect(fixture.decoder.run.mock.calls[0][0].audio_latents).toBe(fixture.outputs[0].audio_latents);
        for (const name of ["audio_latents", "present.audio_latents", "present.0.key", "present.0.value"]) {
          expect(() => fixture.outputs[0][name].data).toThrow("disposed");
        }
        expect(Array.from(voice.data)).toEqual([0.5, 0.25]);
      });
      it.each([1, 4, 8])("retains the EOS frame and tail across blocks of %i", async (size) => {
        await fixture.pipe.dispose();
        fixture = create_pipeline(size, (frame) => (frame === size - 1 ? 5 : -5));
        const output = await generate({ frames_after_eos: 3 });
        expect(output.audio_values.dims).toEqual([1, 1, (size + 3) * 2]);
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
        expect(output.audio_values.dims).toEqual([1, 1, 18]);
        expect(fixture.noise.flat().every((x) => x === 0)).toBe(true);
        expect((await generate({ max_new_tokens: 2 })).audio_values.dims).toEqual([1, 1, 4]);
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
        for (const [name, tensor] of Object.entries(fixture.outputs[0])) {
          if (name !== "eos_logits") expect(() => tensor.data).toThrow("disposed");
        }
        expect(Array.from(voice.data)).toEqual([0.5, 0.25]);
      });
      it.each([voice_url, new URL(voice_url), new Float32Array([0.5, 0.25])])("accepts voice input %p", async (speaker_embeddings) => {
        const output = await speak({ speaker_embeddings });
        expect(output.audio).toEqual(new Float32Array(8).fill(0.25));
        expect(output.sampling_rate).toBe(24000);
      });
      it("starts fresh for each sentence chunk and batched input", async () => {
        const text = "Hello. ".repeat(10);
        const output = await fixture.pipe([text, "Bye."], { speaker_embeddings: voice });
        expect(output.map((x) => x.audio.length)).toEqual([12, 8]);
        for (const [inputs] of fixture.flow.run.mock.calls) expect(inputs["past_key_values.audio_latents"].dims).toEqual([1, 0, 2]);
      });
      it("reports missing and malformed voices before inference", async () => {
        await expect(fixture.pipe("Hello.")).rejects.toThrow("Speaker embeddings must be provided");
        await expect(speak({ speaker_embeddings: new Float32Array(3) })).rejects.toThrow("[1, 2, frames]");
        fetch.mockResolvedValueOnce(new Response("", { status: 404 }));
        await expect(speak({ speaker_embeddings: voice_url })).rejects.toThrow("404");
        expect(fixture.flow.run).not.toHaveBeenCalled();
      });
    });

    describe("onnx-community/phonon-31m-en-ONNX", () => {
      let tts;
      beforeAll(async () => {
        tts = await pipeline("text-to-speech", "onnx-community/phonon-31m-en-ONNX", { dtype: "q4" });
      }, MAX_MODEL_LOAD_TIME);
      afterAll(async () => {
        await tts?.dispose();
      });
      it(
        "produces deterministic finite audio with both sessions owned by the model",
        async () => {
          const speaker_embeddings = new Float32Array(512 * 125);
          const options = { speaker_embeddings, temperature: 0, max_new_tokens: 6 };
          const first = await tts("Hello from Phonon.", options);
          expect(tts.model).toBeInstanceOf(GradiumPhononTTSForConditionalGeneration);
          expect(Object.keys(tts.model.sessions)).toEqual(["model", "mimi_decoder"]);
          expect(first.sampling_rate).toBe(24000);
          expect(first.audio.length).toBeGreaterThan(0);
          expect(first.audio.every(Number.isFinite)).toBe(true);
          expect((await tts("Hello from Phonon.", options)).audio).toEqual(first.audio);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });
  });
};
