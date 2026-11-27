import { jest } from "@jest/globals";
import { Tensor as ORTTensor } from "onnxruntime-node";
import { AutoProcessor, PretrainedConfig, Tensor, GradiumPhononTTSForConditionalGeneration } from "../../../src/transformers.js";
import { get_model_files } from "../../../src/utils/model_registry/get_model_files.js";
import { getCacheNames } from "../../../src/configs.js";
import { random } from "../../../src/utils/random.js";
import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

// Audio marker IDs of the checkpoint. Any other ID is a text token.
const BOS = 4;
const PLACEHOLDER = 5;
const LATENT = 6;
const SAMPLES_PER_FRAME = 1920;
const GENERATION_CONFIG = { do_sample: true, temperature: 0.3, num_flow_steps: 1, eos_threshold: 4, min_frames_before_eos: 0 };

const config = () =>
  new PretrainedConfig({
    model_type: "gradium_phonon_tts",
    audio_prompt_dim: 2,
    latent_dim: 2,
    frame_rate: 12.5,
    audio_bos_token_id: BOS,
    audio_prompt_token_id: PLACEHOLDER,
    audio_latent_token_id: LATENT,
    num_hidden_layers: 1,
    num_attention_heads: 1,
    num_key_value_heads: 1,
    hidden_size: 2,
    head_dim: 2,
    "transformers.js_config": { dtype: "fp32" },
  });

const ids = (data, dims = [1, data.length]) => new Tensor("int64", data, dims);
const voice = (frames = 1) =>
  new Tensor(
    "float32",
    Float32Array.from({ length: frames * 2 }, (_, i) => 0.5 + i),
    [1, frames, 2],
  );
const gpu_tensor = (dims) => ORTTensor.fromGpuBuffer({}, { dataType: "float32", dims, dispose: jest.fn() });

/**
 * A model on generator and decoder doubles that declare the published graphs' inputs, in their order.
 * Every frame has the EOS logit `eos(frame)`, and `outputs` collects what each generator call returned.
 */
function create_model({ block_size = 4, eos = () => 5, generation_config = GENERATION_CONFIG } = {}) {
  const batch = "batch_size";
  const inputMetadata = [
    { name: "input_ids", type: "int64", shape: [batch, "text_length"] },
    { name: "audio_prompt_latents", type: "float32", shape: [batch, "voice_length", 2] },
    { name: "attention_mask", type: "int64", shape: [batch, "total_sequence_length"] },
    { name: "position_ids", type: "int64", shape: [batch, "new_sequence_length"] },
    { name: "past_key_values.audio_latents", type: "float32", shape: [batch, "past_frames", 2] },
    { name: "noise", type: "float32", shape: [batch, block_size, 2] },
    { name: "past_key_values.0.key", type: "float32", shape: [batch, 1, "past_sequence_length", 2] },
    { name: "past_key_values.0.value", type: "float32", shape: [batch, 1, "past_sequence_length", 2] },
  ];
  const outputs = [];
  const generator = {
    inputMetadata,
    inputNames: inputMetadata.map(({ name }) => name),
    outputNames: ["audio_latents", "present.audio_latents", "eos_logits", "present.0.key", "present.0.value"],
    run: jest.fn(async (feeds) => {
      const start = feeds["past_key_values.audio_latents"].dims[1];
      const frames = start + block_size;
      const sequence_length = feeds["past_key_values.0.key"].dims[2] + feeds.audio_prompt_latents.dims[1] + feeds.input_ids.dims[1] + block_size;
      const result = {
        audio_latents: new ORTTensor("float32", new Float32Array(frames * 2), [1, frames, 2]),
        "present.audio_latents": gpu_tensor([1, frames, 2]),
        eos_logits: new ORTTensor(
          "float32",
          Float32Array.from({ length: block_size }, (_, i) => eos(start + i)),
          [1, block_size],
        ),
        "present.0.key": gpu_tensor([1, 1, sequence_length, 2]),
        "present.0.value": gpu_tensor([1, 1, sequence_length, 2]),
      };
      outputs.push({ ...result });
      return result;
    }),
    release: jest.fn(),
  };
  const decoder = {
    inputNames: ["audio_latents", "num_frames"],
    outputNames: ["audio_values", "audio_lengths"],
    run: jest.fn(async ({ num_frames }) => {
      const samples = Number(num_frames.data[0]) * SAMPLES_PER_FRAME;
      return {
        audio_values: new ORTTensor("float32", new Float32Array(samples).fill(0.25), [1, 1, samples]),
        audio_lengths: new ORTTensor("int64", [BigInt(samples)], [1]),
      };
    }),
    release: jest.fn(),
  };
  const model = new GradiumPhononTTSForConditionalGeneration(config(), { model: generator, mimi_decoder: decoder }, { generation_config });
  return { model, generator, decoder, outputs };
}

/** The number of frames the model sent to the decoder in its last call. */
const decoded_frames = (decoder) => Number(decoder.run.mock.calls.at(-1)[0].num_frames.data[0]);

export default () => {
  describe("GradiumPhononTTSForConditionalGeneration", () => {
    afterEach(() => {
      jest.restoreAllMocks();
      random.seed();
    });

    it.each(["fp32", "q4", "q4f16"])("lists both generation and decoder graphs for %s", async (dtype) => {
      const suffix = dtype === "fp32" ? "" : `_${dtype}`;
      expect(await get_model_files("unused", { config: config(), dtype })).toEqual(["config.json", `onnx/model${suffix}.onnx`, `onnx/mimi_decoder${suffix}.onnx`, "generation_config.json"]);
    });
    it("pins feedback and decoder input latents to the GPU", () => {
      expect([...getCacheNames(config(), { prefix: "present", session_name: "model" })]).toEqual(["present.0.key", "present.0.value", "present.audio_latents", "audio_latents"]);
    });

    describe("generator and decoder feeds", () => {
      it("packs the unmasked prompt and continues from the KV length", async () => {
        const { model, generator, decoder, outputs } = create_model();
        const audio_prompt_latents = voice();
        // Four unmasked positions: one voice placeholder, two text tokens and the audio BOS.
        const { audio_values } = await model.generate({
          input_ids: ids([PLACEHOLDER, 42, 43, 44, BOS, 3]),
          attention_mask: ids([1, 1, 0, 1, 1, 0]),
          audio_prompt_latents,
          audio_prompt_latents_mask: ids([1]),
          max_length: 10,
          min_frames_before_eos: 6,
        });
        expect(generator.run).toHaveBeenCalledTimes(2);
        const [[prefill], [continuation]] = generator.run.mock.calls;
        for (const feeds of [prefill, continuation]) {
          expect(Object.keys(feeds).sort()).toEqual([...generator.inputNames].sort());
        }
        expect([...prefill.input_ids.data]).toEqual([42n, 44n]);
        expect(prefill.audio_prompt_latents).toBe(audio_prompt_latents.ort_tensor);
        // One voice frame, two text tokens and a block of four frames.
        expect([...prefill.position_ids.data]).toEqual([0n, 1n, 2n, 3n, 4n, 5n, 6n]);
        expect([...prefill.attention_mask.data]).toEqual(Array(7).fill(1n));

        // The latent cache, which the graph lists first, holds four frames; the KV cache holds seven positions.
        expect(continuation.input_ids.dims).toEqual([1, 0]);
        expect(continuation.audio_prompt_latents.dims).toEqual([1, 0, 2]);
        expect(continuation["past_key_values.audio_latents"]).toBe(outputs[0]["present.audio_latents"]);
        expect(continuation["past_key_values.0.key"]).toBe(outputs[0]["present.0.key"]);
        expect([...continuation.position_ids.data]).toEqual([7n, 8n, 9n, 10n]);
        expect([...continuation.attention_mask.data]).toEqual(Array(11).fill(1n));

        // max_length leaves six frames after the four prompt positions.
        const [[decoded]] = decoder.run.mock.calls;
        expect(decoded.audio_latents).toBe(outputs[1].audio_latents);
        expect(decoded.num_frames.type).toBe("int64");
        expect(decoded.num_frames.dims).toEqual([1]);
        expect(decoded.num_frames.data[0]).toBe(6n);
        expect(audio_values.dims).toEqual([1, 1, 6 * SAMPLES_PER_FRAME]);
        expect(audio_values.data.every((x) => x === 0.25)).toBe(true);
      });

      it.each([2, 8])("keeps the EOS tail across a block boundary with blocks of %i frames", async (block_size) => {
        // EOS fires on the last frame of the first block.
        const { model, generator, decoder } = create_model({ block_size, eos: (frame) => (frame === block_size - 1 ? 5 : -5) });
        await model.generate({ input_ids: ids([42, BOS]), frames_after_eos: 3 });
        expect(generator.run).toHaveBeenCalledTimes(Math.ceil((block_size + 3) / block_size));
        expect(generator.run.mock.calls[0][0].noise.dims).toEqual([1, block_size, 2]);
        expect(decoder.run).toHaveBeenCalledTimes(1);
        expect(decoded_frames(decoder)).toBe(block_size + 3);
      });

      it.each(["frames", 0])("rejects an export with the noise block size %p", (block_size) => {
        expect(() => create_model({ block_size })).toThrow("Incompatible Phonon ONNX export");
      });

      it("generates without a voice when the prompt has no placeholders", async () => {
        const { model, generator } = create_model();
        const { audio_values } = await model.generate({ input_ids: ids([42, BOS]), max_new_tokens: 1 });
        expect(audio_values.dims).toEqual([1, 1, SAMPLES_PER_FRAME]);
        expect(generator.run.mock.calls[0][0].audio_prompt_latents.dims).toEqual([1, 0, 2]);
      });

      it("warns and truncates at the default safety limit if EOS never fires", async () => {
        const warn = jest.spyOn(console, "warn").mockImplementation(() => {});
        const { model, decoder } = create_model({ eos: () => -5 });
        // One text token gives ceil((1 / 3 + 2) * 12.5) = 30 frames.
        await model.generate({ input_ids: ids([42, BOS]) });
        expect(warn).toHaveBeenCalledWith(expect.stringContaining("Phonon reached its safety frame limit"));
        expect(decoded_frames(decoder)).toBe(30);
      });

      it.each([
        ["a batch of prompts", { input_ids: ids([42, BOS, 43, BOS], [2, 2]) }, "accepts one prompt per call"],
        ["a prompt mask of another shape", { attention_mask: ids([1]) }, "attention_mask must match the shape of input_ids"],
        ["placeholders without a voice", { audio_prompt_latents: undefined }, "voice frame count"],
        ["a placeholder count that differs from the voice", { input_ids: ids([PLACEHOLDER, PLACEHOLDER, 42, BOS]) }, "voice frame count"],
        ["a placeholder after text", { input_ids: ids([42, PLACEHOLDER, BOS]) }, "before the first text token"],
        ["a missing final BOS", { input_ids: ids([PLACEHOLDER, BOS, 42]) }, "last unmasked token"],
        ["a duplicate BOS", { input_ids: ids([PLACEHOLDER, BOS, 42, BOS]) }, "exactly one unmasked audio_bos_token_id"],
        ["an audio latent marker", { input_ids: ids([PLACEHOLDER, 42, LATENT, BOS]) }, "must not contain audio_latent_token_id"],
        ["a padded voice", { audio_prompt_latents_mask: ids([0]) }, "unpadded voice prompt"],
        ["a voice mask of another length", { audio_prompt_latents_mask: ids([1, 1]) }, "unpadded voice prompt"],
        ["an empty frame budget", { max_new_tokens: 0 }, "frame budget must be a positive integer"],
        ["a negative EOS tail", { frames_after_eos: -1 }, "frames_after_eos a non-negative integer"],
        ["a negative temperature", { temperature: -1 }, "temperature must be a finite non-negative number"],
        ["solver steps absent from the export", { num_flow_steps: 4 }, "require re-export"],
      ])("rejects %s before inference", async (_name, options, message) => {
        const { model, generator } = create_model();
        await expect(model.generate({ input_ids: ids([PLACEHOLDER, 42, BOS]), audio_prompt_latents: voice(), ...options })).rejects.toThrow(message);
        expect(generator.run).not.toHaveBeenCalled();
      });
    });

    describe("generation settings", () => {
      it("applies the checkpoint config, then generation_config, then direct options", async () => {
        // Every frame has an EOS logit of 5, and one frame is kept after EOS.
        const { model, decoder } = create_model({ generation_config: { ...GENERATION_CONFIG, eos_threshold: 10, min_frames_before_eos: 5 } });
        const frames = async (options) => {
          await model.generate({ input_ids: ids([42, BOS]), max_new_tokens: 12, ...options });
          return decoded_frames(decoder);
        };
        expect(await frames()).toBe(12);
        expect(await frames({ generation_config: { eos_threshold: 4 } })).toBe(7);
        expect(await frames({ generation_config: { eos_threshold: 4, min_frames_before_eos: 9 }, min_frames_before_eos: 2 })).toBe(4);
        expect(await frames({ eos_threshold: 4, min_frames_before_eos: 0 })).toBe(2);
        // Direct options that are not set leave the merged config in place.
        expect(await frames({ generation_config: { eos_threshold: 4 }, eos_threshold: undefined, min_frames_before_eos: null })).toBe(7);
      });

      it("keeps `temperature: 0` and `do_sample: false` overrides, and scales noise by the square root of temperature", async () => {
        const { model, generator } = create_model();
        const noise = async (options) => {
          await model.generate({ input_ids: ids([42, BOS]), ...options });
          return [...generator.run.mock.calls.at(-1)[0].noise.data];
        };
        const is_silent = (values) => values.every((x) => x === 0);
        expect(is_silent(await noise())).toBe(false);
        expect(is_silent(await noise({ temperature: 0 }))).toBe(true);
        expect(is_silent(await noise({ do_sample: false }))).toBe(true);
        expect(is_silent(await noise({ generation_config: { temperature: 0 } }))).toBe(true);
        expect(is_silent(await noise({ generation_config: { do_sample: false } }))).toBe(true);
        expect(is_silent(await noise({ generation_config: { do_sample: false }, do_sample: true }))).toBe(false);

        random.seed(123);
        const baseline = await noise({ temperature: 0.3 });
        random.seed(123);
        const scaled = await noise({ temperature: 1.2 });
        scaled.forEach((value, i) => expect(value).toBeCloseTo(baseline[i] * 2));
      });

      it("reports settings that neither the checkpoint nor the call provides", async () => {
        // A repo without generation_config.json loads as an empty config.
        const { model, generator } = create_model({ generation_config: {} });
        const input_ids = ids([42, BOS]);
        await expect(model.generate({ input_ids })).rejects.toThrow("Missing Phonon generation settings: do_sample, temperature, num_flow_steps, eos_threshold, min_frames_before_eos. Set them in the model's generation_config.json or pass them to generate().");
        const generation_config = { do_sample: false, temperature: 0, num_flow_steps: 1 };
        await expect(model.generate({ input_ids, generation_config, eos_threshold: 4 })).rejects.toThrow("Missing Phonon generation settings: min_frames_before_eos.");
        expect(generator.run).not.toHaveBeenCalled();

        const { audio_values } = await model.generate({ input_ids, generation_config, eos_threshold: 4, min_frames_before_eos: 0 });
        expect(audio_values.dims).toEqual([1, 1, 2 * SAMPLES_PER_FRAME]);
      });
    });

    describe("cleanup", () => {
      const expect_released = (outputs) => {
        for (const tensor of Object.values(outputs)) {
          expect(() => tensor.data).toThrow("disposed");
        }
      };

      it("releases intermediate state and leaves the caller's voice intact", async () => {
        const { model, outputs } = create_model();
        const audio_prompt_latents = voice(2);
        const { audio_values } = await model.generate({ input_ids: ids([PLACEHOLDER, PLACEHOLDER, 42, BOS]), audio_prompt_latents, min_frames_before_eos: 5 });
        expect(audio_values.dims).toEqual([1, 1, 7 * SAMPLES_PER_FRAME]);
        expect(outputs).toHaveLength(2);
        outputs.forEach(expect_released);
        expect([...audio_prompt_latents.data]).toEqual([0.5, 1.5, 2.5, 3.5]);
      });

      it.each(["generator", "decoder"])("releases state when %s inference fails", async (failure) => {
        jest.spyOn(console, "error").mockImplementation(() => {});
        const { model, generator, decoder, outputs } = create_model();
        // Fail the second generator call, so there is state from the first one to release.
        if (failure === "generator") generator.run.mockImplementationOnce(generator.run.getMockImplementation()).mockRejectedValueOnce(new Error("failed"));
        else decoder.run.mockRejectedValueOnce(new Error("failed"));
        const audio_prompt_latents = voice();
        await expect(model.generate({ input_ids: ids([PLACEHOLDER, 42, BOS]), audio_prompt_latents, min_frames_before_eos: 5 })).rejects.toThrow("failed");
        expect(outputs).toHaveLength(failure === "generator" ? 1 : 2);
        outputs.forEach(expect_released);
        expect([...audio_prompt_latents.data]).toEqual([0.5, 1.5]);
      });
    });

    describe("onnx-community/phonon-31m-en-ONNX (q4)", () => {
      const model_id = "onnx-community/phonon-31m-en-ONNX";
      let model, inputs;
      beforeAll(async () => {
        const processor = await AutoProcessor.from_pretrained(model_id);
        model = await GradiumPhononTTSForConditionalGeneration.from_pretrained(model_id, { dtype: "q4" });
        // Voice files hold flat float32 data in [audio_prompt_dim, frames] order.
        const response = await fetch(`https://huggingface.co/${model_id}/resolve/main/voices/default.bin`);
        const data = new Float32Array(await response.arrayBuffer());
        const voice = new Tensor("float32", data, [data.length]).view(model.config.audio_prompt_dim, -1).transpose(1, 0);
        const prompt = processor.apply_chat_template([{ role: "user", content: [{ type: "audio" }, { type: "text", text: "Hello from Phonon." }] }], { add_generation_prompt: true });
        inputs = await processor(prompt, { audio_prompt_latents: [voice] });
      }, MAX_MODEL_LOAD_TIME);
      afterAll(async () => {
        await model?.dispose();
      });

      it.each([
        [6, 1],
        [11, 2],
      ])(
        "feeds the graph contract for %i frames over %i generator calls",
        async (frames, calls) => {
          const { model: generator, mimi_decoder } = model.sessions;
          const generate = jest.spyOn(generator, "run");
          const decode = jest.spyOn(mimi_decoder, "run");
          // Make continuation independent of the real graph's EOS timing.
          const run = () => model.generate({ ...inputs, temperature: 0, max_new_tokens: frames, min_frames_before_eos: frames });
          const { audio_values } = await run();

          expect(generate).toHaveBeenCalledTimes(calls);
          const block_size = generate.mock.calls[0][0].noise.dims[1];
          let past_length = 0;
          for (const [i, [feeds]] of generate.mock.calls.entries()) {
            expect(Object.keys(feeds).sort()).toEqual([...generator.inputNames].sort());
            // The prefill adds the voice frames and text tokens, which replace the placeholders and drop the BOS.
            const new_length = block_size + (i === 0 ? inputs.input_ids.dims[1] - 1 : 0);
            expect(feeds.position_ids.dims).toEqual([1, new_length]);
            expect([...feeds.position_ids.data]).toEqual(Array.from({ length: new_length }, (_, j) => BigInt(past_length + j)));
            expect(feeds["past_key_values.0.key"].dims[2]).toBe(past_length);
            expect(feeds["past_key_values.audio_latents"].dims[1]).toBe(i * block_size);
            past_length += new_length;
            expect(feeds.attention_mask.dims).toEqual([1, past_length]);
            expect(feeds.attention_mask.data.every((x) => x === 1n)).toBe(true);
          }

          expect(decode).toHaveBeenCalledTimes(1);
          const { num_frames } = decode.mock.calls[0][0];
          expect(num_frames.type).toBe("int64");
          expect(num_frames.dims).toEqual([1]);
          expect(num_frames.data[0]).toBe(BigInt(frames));
          expect(audio_values.dims).toEqual([1, 1, frames * SAMPLES_PER_FRAME]);
          expect(audio_values.data.every(Number.isFinite)).toBe(true);
          expect((await run()).audio_values.data).toEqual(audio_values.data);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "stops at a natural end of speech with the checkpoint's generation config",
        async () => {
          const warn = jest.spyOn(console, "warn");
          // Seven text tokens give a safety limit of ceil((7 / 3 + 2) * 12.5) = 55 frames.
          const { audio_values } = await model.generate({ ...inputs, temperature: 0 });
          const audio = audio_values.data;
          expect(audio.length).toBeGreaterThan(0);
          expect(audio.length).toBeLessThan(55 * SAMPLES_PER_FRAME);
          expect(audio.length % SAMPLES_PER_FRAME).toBe(0);
          expect(audio.every(Number.isFinite)).toBe(true);
          const rms = Math.sqrt(audio.reduce((sum, sample) => sum + sample * sample, 0) / audio.length);
          expect(rms).toBeGreaterThan(0.03);
          expect(warn).not.toHaveBeenCalled();
        },
        MAX_TEST_EXECUTION_TIME,
      );

      // generate() does not check these: the graph declares them, so ONNX Runtime does.
      it.each([
        ["width", "float32", (frames, dim) => [1, frames, dim - 1]],
        ["data type", "float64", (frames, dim) => [1, frames, dim]],
        ["batch size", "float32", (frames, dim) => [2, frames, dim]],
      ])(
        "leaves a voice of the wrong %s to ONNX Runtime",
        async (_name, type, shape) => {
          jest.spyOn(console, "error").mockImplementation(() => {});
          const generate = jest.spyOn(model.sessions.model, "run");
          const [, frames, dim] = inputs.audio_prompt_latents.dims;
          const dims = shape(frames, dim);
          const audio_prompt_latents = new Tensor(type, Array(dims.reduce((a, b) => a * b)).fill(0), dims);
          await expect(model.generate({ ...inputs, audio_prompt_latents, audio_prompt_latents_mask: undefined })).rejects.toThrow();
          expect(generate).toHaveBeenCalledTimes(1);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });
  });
};
