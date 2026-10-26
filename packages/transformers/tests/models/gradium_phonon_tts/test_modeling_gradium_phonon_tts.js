import { jest } from "@jest/globals";
import { Tensor as ORTTensor } from "onnxruntime-node";
import { env, pipeline, ModelRegistry, PretrainedConfig, Tensor, TextToAudioPipeline, GradiumPhononTTSForConditionalGeneration, PocketTTSProcessor } from "../../../src/transformers.js";
import { get_model_files } from "../../../src/utils/model_registry/get_model_files.js";
import { getCacheNames } from "../../../src/configs.js";
import { selectDtype } from "../../../src/utils/dtypes.js";
import { create_tokenizer } from "./tokenizer_fixture.js";
import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME } from "../../init.js";

const MODEL_ID = "onnx-community/phonon-31m-en-ONNX";
// Voices are app-level: flat float32 files in the model repo, read here as an app would.
const hub_voice_url = (id = "default") => `https://huggingface.co/${MODEL_ID}/resolve/main/voices/${id}.bin`;
const hub_voice = async (id) => {
  const data = new Float32Array(await (await fetch(hub_voice_url(id))).arrayBuffer());
  return new Tensor("float32", data, [1, 512, data.length / 512]);
};

const dtype_config = { dtype: "fp32" };
const phonon = {
  voice_dim: 2,
  latent_dim: 2,
  sample_rate: 24000,
  frame_rate: 12.5,
  temperature: 0.3,
  eos_threshold: 4,
  conditions: { num_speakers: { default: "2" }, duration_delta: { default: 0.5 } },
  flow_lm: { num_layers: 1, num_heads: 1, head_dim: 2 },
  mimi_decoder: { num_layers: 1, num_heads: 1, head_dim: 2, conv_states: { "past_conv.test": [1, 2, 1] } },
};
const config = () => new PretrainedConfig({ model_type: "gradium_phonon_tts", phonon, "transformers.js_config": dtype_config });
const url = "https://example.com/voice.bin";
const processor_config = { processor_class: "PocketTTSProcessor", short_text_max_words: 4, frames_after_eos_short_text: 3, frames_after_eos: 1 };
const create_processor = (config = processor_config) => new PocketTTSProcessor(config, { tokenizer: create_tokenizer() });
// Final per-chunk model inputs, decoded back to text.
const decoded = async (processor, text, max_tokens) => (await processor(text, { max_tokens })).map(({ input_ids }) => processor.tokenizer.decode(input_ids));
// Phonon voice files are raw float32 data in [1, voice_dim, frames] order.
const voice_file = (values = [0.25, -0.5]) => new Response(new Float32Array(values).buffer);

function fake_pipeline(eos = 5) {
  const steps = [];
  const offsets = [];
  const state = () => ({ "present.0.key": new ORTTensor("float32", [0], [1]), "present.0.value": new ORTTensor("float32", [0], [1]) });
  const flow = {
    inputNames: ["text_ids", "voice_latents", "latent", "noise", "num_speakers", "duration_delta", "past_key_values.0.key", "past_key_values.0.value"],
    run: jest.fn(async (inputs) => {
      if (inputs.latent.dims[1]) {
        steps.push({ noise: Array.from(inputs.noise.data), num_speakers: inputs.num_speakers.data[0], duration_delta: inputs.duration_delta.data[0] });
      }
      return { ...state(), latent_out: new ORTTensor("float32", [0, 0], [1, 1, 2]), eos_logit: new ORTTensor("float32", [eos], [1]) };
    }),
    release: jest.fn(),
  };
  const decoder = {
    inputNames: ["latent", "past_offset", "past_conv.test", "past_key_values.0.key", "past_key_values.0.value"],
    // One decoder call per chunk: two samples per latent frame.
    run: jest.fn(async (inputs) => (offsets.push(inputs.past_offset.data[0]), {
      ...state(),
      audio: new ORTTensor("float32", Array.from({ length: inputs.latent.dims[1] }, () => [0.25, -0.5]).flat(), [1, 2 * inputs.latent.dims[1]]),
      present_offset: new ORTTensor("int64", [1n], [1]),
      "present_conv.test": new ORTTensor("float32", [0], [1]),
    })),
    release: jest.fn(),
  };
  const model = new GradiumPhononTTSForConditionalGeneration(config(), { flow_lm: flow, mimi_decoder: decoder });
  const fetch = jest.spyOn(env, "fetch").mockImplementation(async () => voice_file());
  const processor = create_processor();
  const pipe = new TextToAudioPipeline({ model, tokenizer: processor.tokenizer, processor });
  // Voice-conditioning runs are the only flow runs with voice frames.
  const voice_runs = () => flow.run.mock.calls.map(([inputs], i) => ({ inputs, i })).filter(({ inputs }) => inputs.voice_latents.dims[2] > 0);
  const prefix = async (n = 0) => (await flow.run.mock.results[voice_runs()[n].i].value)["present.0.key"];
  // Latent frames of each decoder call.
  const decoded_frames = () => decoder.run.mock.calls.map(([inputs]) => inputs.latent.dims[1]);
  return { pipe, model, flow, decoder, fetch, steps, offsets, voice_runs, prefix, decoded_frames };
}

// Python PocketTTSProcessor.split_text -> apply_chat_template -> encode, using
// the same offline fixture's normalizer and Rust tokenizers backend. These are
// the final per-chunk model inputs, not slices of the whole-text encoding.
const OFFLINE_PARITY_CASES = [
  {
    text: "hello. goodbye.",
    budget: 8,
    ids: [
      [1, 42, 71, 78, 78, 81, 16],
      [1, 41, 81, 81, 70, 68, 91, 71, 16],
    ],
  },
  {
    text: "one, two, three, four, five, six.",
    budget: 17,
    ids: [
      [1, 49, 80, 71, 14, 1, 86, 89, 81, 14, 1, 86, 74, 84, 71, 71, 14],
      [1, 40, 81, 87, 84, 14, 1, 72, 75, 88, 71, 14, 1, 85, 75, 90, 16],
    ],
  },
  { text: "it’s “Hello” — see you 🚀…", budget: 100, ids: [[1, 43, 86, 9, 85, 1, 4, 42, 71, 78, 78, 81, 4, 1, 85, 71, 71, 1, 91, 81, 87, 16]] },
  {
    text: "price 3.5. next.",
    budget: 10,
    ids: [
      [1, 50, 84, 75, 69, 71, 1, 21, 16, 23, 16],
      [1, 48, 71, 90, 86, 16],
    ],
  },
];

export default () => {
  describe("GradiumPhononTTSForConditionalGeneration", () => {
    afterEach(() => jest.restoreAllMocks());

    // The Hub aliases use variant suffixes, not decoder precision:
    // mimi_decoder_q4 is fp32 and mimi_decoder_q4f16 is fp16.
    // Without a dtype, the configured fp32 wins over wasm's q8 default.
    it.each([
      [{ dtype: "fp32" }, ""],
      [{ dtype: "q4" }, "_q4"],
      [{ dtype: "q4f16" }, "_q4f16"],
      [{ device: "wasm" }, ""],
    ])("lists both graphs for %j", async (options, suffix) => {
      expect(await get_model_files("unused", { config: config(), ...options })).toEqual(["config.json", `onnx/flow_lm${suffix}.onnx`, `onnx/mimi_decoder${suffix}.onnx`]);
      for (const dtype of options.dtype ? [] : [undefined, "auto"]) {
        expect(selectDtype(dtype ?? dtype_config.dtype, "flow_lm", "wasm", { configDtype: dtype_config.dtype })).toBe("fp32");
      }
    });

    // On WebGPU, the flow model's KV cache (the only state fed back between calls) stays on the GPU.
    it("uses the flow model's KV cache as the standard cache names", () => {
      expect([...getCacheNames(config(), { prefix: "present", session_name: "flow_lm" })]).toEqual(["present.0.key", "present.0.value"]);
    });

    describe("text preparation", () => {
      it.each(OFFLINE_PARITY_CASES)("matches Python's per-chunk model inputs: $text", async ({ text, budget, ids }) => {
        expect((await create_processor()(text, { max_tokens: budget })).map((chunk) => chunk.input_ids)).toEqual(ids);
      });
      it("groups short sentences and keeps unsplittable clauses intact", async () => {
        const processor = create_processor();
        expect(await decoded(processor, "Hi. Bye. Yes. No.", 10)).toEqual(["Hi. Bye.", "Yes. No."]);
        expect(processor.split_text("a".repeat(80), 10)).toHaveLength(1);
      });
      it("joins decoded pieces with spaces like Python when punctuation has no following whitespace", async () => {
        const processor = create_processor();
        expect(processor.split_text("Hi.Bye.")).toEqual(["Hi. Bye."]);
        expect(await decoded(processor, "one,two,three,four,five,six.", 17)).toEqual(["One, two, three,", "Four, five, six."]);
      });
      it("keeps bounded decimal lookaround correct for byte-fallback digits", async () => {
        expect(await decoded(create_processor(), "price ١.٥. next.", 10)).toEqual(["Price ١.٥.", "Next."]);
      });
      it.each([50, 1_000_000])("bounds decoding work linearly for long paragraphs (budget %i)", (budget) => {
        const work = (sentences) => {
          const processor = create_processor();
          const text = "Hello. ".repeat(sentences);
          const tokens = processor.tokenizer.encode(text, { add_special_tokens: false });
          const decode = jest.spyOn(processor.tokenizer, "decode");
          processor.split_text(text, budget);
          const decoded_tokens = decode.mock.calls.reduce((total, [ids]) => total + ids.length, 0);
          expect(decoded_tokens).toBeLessThan(tokens.length * 12);
          return decoded_tokens;
        };
        expect(work(400)).toBeLessThan(work(100) * 5);
      });
      it("applies byte fallback using only public tokenizer methods", async () => {
        const processor = create_processor();
        const encode = jest.spyOn(processor.tokenizer, "encode");
        expect(await decoded(processor, "hi 漢")).toEqual(["Hi 漢."]);
        expect(encode).not.toHaveBeenCalled();
      });
      it("takes the EOS tail per chunk from the processor config", async () => {
        const processor = create_processor({ short_text_max_words: 2, frames_after_eos_short_text: 7, frames_after_eos: 5 });
        const chunks = await processor("Hi there. This one is longer.", { max_tokens: 5 });
        expect(chunks.map((chunk) => chunk.frames_after_eos)).toEqual([7, 5]);
      });
    });

    describe("generation", () => {
      it("decodes each chunk in one decoder call and frees each call's prefix", async () => {
        const { pipe, fetch, steps, offsets, voice_runs, prefix, decoded_frames } = fake_pipeline();
        try {
          const result = await pipe("Hello.", { speaker_embeddings: url, temperature: 0 });
          // EOS on the first frame, then a 3-frame tail for short texts: four latents, decoded at once.
          expect(decoded_frames()).toEqual([4]);
          expect(Array.from(result.audio)).toEqual([0.25, -0.5, 0.25, -0.5, 0.25, -0.5, 0.25, -0.5]);
          expect(steps).toHaveLength(4);
          expect(steps.every(({ noise }) => noise.every((value) => value === 0))).toBe(true);
          // The decoder starts every chunk from an empty streaming state.
          expect(offsets).toEqual([0n]);
          await pipe("Hello.", { speaker_embeddings: url });
          // Nothing is cached: every call loads and conditions its voice.
          expect(fetch).toHaveBeenCalledTimes(2);
          expect(voice_runs()).toHaveLength(2);
          for (const n of [0, 1]) await expect(prefix(n).then((p) => p.data)).rejects.toThrow("disposed");
        } finally {
          await pipe.dispose();
        }
      });
      it("requires speaker embeddings", async () => {
        const { pipe, flow } = fake_pipeline();
        try {
          await expect(pipe("Hello.")).rejects.toThrow("Speaker embeddings must be provided for Phonon models.");
          expect(flow.run).not.toHaveBeenCalled();
        } finally {
          await pipe.dispose();
        }
      });
      it("yields one waveform per chunk from generate_speech", async () => {
        const { pipe, model, decoded_frames } = fake_pipeline();
        const voice = new Tensor("float32", [0.5, 0.25], [1, 2, 1]);
        try {
          const chunks = await pipe.processor("Hello. ".repeat(10));
          expect(chunks.length).toBeGreaterThan(1);
          const parts = [];
          for await (const pcm of model.generate_speech({ chunks, speaker_embeddings: voice, temperature: 0 })) parts.push(pcm);
          expect(parts).toHaveLength(chunks.length);
          expect(decoded_frames()).toHaveLength(chunks.length);
          expect(parts.map((pcm) => pcm.length)).toEqual(decoded_frames().map((frames) => 2 * frames));
        } finally {
          voice.dispose();
          await pipe.dispose();
        }
      });
      it("takes conditioning defaults from the config and forwards overrides", async () => {
        const { pipe, steps } = fake_pipeline();
        try {
          await pipe("Hello.", { speaker_embeddings: url });
          expect(steps[0]).toMatchObject({ num_speakers: 2n, duration_delta: 0.5 });
          await pipe("Hello.", { speaker_embeddings: url, num_speakers: 0, duration_delta: -1 });
          expect(steps.at(-1)).toMatchObject({ num_speakers: 0n, duration_delta: -1 });
        } finally {
          await pipe.dispose();
        }
      });
      it("shares a tensor prefix across chunks, then reconditions in-place changes on the next call", async () => {
        const { pipe, voice_runs, prefix } = fake_pipeline();
        const voice = new Tensor("float32", [0.5, 0.25], [1, 2, 1]);
        const text = "Hello. ".repeat(10);
        try {
          expect((await pipe.processor(text)).length).toBeGreaterThan(1);
          await pipe(text, { speaker_embeddings: voice, temperature: 0 });
          expect(voice_runs()).toHaveLength(1);
          expect(() => voice_runs()[0].inputs.voice_latents.data).not.toThrow();
          await expect(prefix().then((p) => p.data)).rejects.toThrow("disposed");
          voice.data[0] = 0.75;
          await pipe("Another sentence.", { speaker_embeddings: voice });
          expect(voice_runs()).toHaveLength(2);
        } finally {
          await pipe.dispose();
          expect(Array.from(voice.data)).toEqual([0.75, 0.25]);
          voice.dispose();
        }
      });
      it.each(["break", "throw"])("frees per-call tensor prefixes when the consumer exits a stream with %s", async (exit) => {
        const { pipe, model, prefix } = fake_pipeline();
        const voice = new Tensor("float32", [0.5, 0.25], [1, 2, 1]);
        const chunks = await pipe.processor("Hello.");
        const consume = async () => {
          for await (const pcm of model.generate_speech({ chunks, speaker_embeddings: voice })) {
            if (exit === "break") break;
            throw new Error("playback failed");
          }
        };
        try {
          if (exit === "break") await consume();
          else await expect(consume()).rejects.toThrow("playback failed");
          await expect(prefix().then((p) => p.data)).rejects.toThrow("disposed");
          expect(Array.from(voice.data)).toEqual([0.5, 0.25]);
        } finally {
          voice.dispose();
          await pipe.dispose();
        }
      });
      it("reshapes flat voice data from URLs and arrays", async () => {
        const { pipe, fetch, voice_runs } = fake_pipeline();
        fetch.mockImplementation(async () => voice_file([1, 2, 3, 4]));
        try {
          for (const speaker_embeddings of [url, new URL(url), new Float32Array([1, 2, 3, 4])]) {
            expect((await pipe("Hello.", { speaker_embeddings })).audio.length).toBeGreaterThan(0);
          }
          expect(fetch).toHaveBeenCalledTimes(2);
          for (const { inputs } of voice_runs()) {
            expect(inputs.voice_latents.dims).toEqual([1, 2, 2]);
            expect(Array.from(inputs.voice_latents.data)).toEqual([1, 2, 3, 4]);
          }
        } finally { await pipe.dispose(); }
      });
      it("reports failed voice downloads before running the model", async () => {
        const { pipe, flow, fetch } = fake_pipeline();
        fetch.mockResolvedValueOnce(new Response("not found", { status: 404, statusText: "Not Found" }));
        try {
          await expect(pipe("Hello.", { speaker_embeddings: url })).rejects.toThrow("Unable to load speaker embeddings: 404 Not Found.");
          expect(flow.run).not.toHaveBeenCalled();
          await pipe("Try again.", { speaker_embeddings: url });
          expect(fetch).toHaveBeenCalledTimes(2);
        } finally { await pipe.dispose(); }
      });
      it("uses the shorter EOS tail for more than four words", async () => {
        const { pipe, decoded_frames } = fake_pipeline();
        try {
          await pipe("One two three four five.", { speaker_embeddings: url });
          expect(decoded_frames()).toEqual([2]);
        } finally {
          await pipe.dispose();
        }
      });
      it("stops at the frame budget and warns when EOS never arrives", async () => {
        const { pipe, decoded_frames } = fake_pipeline(-5);
        const warn = jest.spyOn(console, "warn").mockImplementation(() => {});
        try {
          const ids = pipe.tokenizer.encode("Hi.", { add_special_tokens: false });
          await pipe("Hi.", { speaker_embeddings: url, temperature: 0 });
          expect(decoded_frames()).toEqual([Math.ceil((ids.length / 3 + 2) * 12.5)]);
          expect(warn).toHaveBeenCalledWith(expect.stringContaining("truncated"));
        } finally {
          await pipe.dispose();
        }
      });
      it("rejects voice data that does not fit [1, voice_dim, frames]", async () => {
        const { pipe } = fake_pipeline();
        try {
          for (const speaker_embeddings of [new Float32Array(3), new Float32Array(0), new Tensor("int64", [1n, 2n], [2])]) {
            await expect(pipe("Hello.", { speaker_embeddings })).rejects.toThrow("[1, 2, frames]");
          }
          await expect(pipe("Hello.", { speaker_embeddings: {} })).rejects.toThrow("must be a `Tensor`");
        } finally {
          await pipe.dispose();
        }
      });
    });
    describe(`${MODEL_ID} (q4)`, () => {
      let tts;
      beforeAll(async () => {
        tts = await pipeline("text-to-speech", MODEL_ID, { dtype: "q4" });
      }, MAX_MODEL_LOAD_TIME);
      afterAll(async () => { await tts?.dispose(); }, MAX_MODEL_DISPOSE_TIME);

      it("loads the native architecture and the published graph/state interfaces", () => {
        expect(tts.model).toBeInstanceOf(GradiumPhononTTSForConditionalGeneration);
        expect(tts.processor).toBeInstanceOf(PocketTTSProcessor);
        expect(tts.model.config.model_type).toBe("gradium_phonon_tts");
        const { flow_lm, mimi_decoder } = tts.model.sessions;
        expect(flow_lm.inputNames).toEqual(expect.arrayContaining(["text_ids", "voice_latents", "latent", "noise", "num_speakers", "duration_delta"]));
        expect(flow_lm.outputNames).toEqual(expect.arrayContaining(["latent_out", "eos_logit"]));
        expect(mimi_decoder.inputNames).toEqual(expect.arrayContaining(["latent", "past_offset", ...Object.keys(tts.model.config.phonon.mimi_decoder.conv_states)]));
        expect(mimi_decoder.outputNames).toEqual(expect.arrayContaining(["audio"]));
        // The flow model's KV cache is the state fed back between calls.
        const cache = [...getCacheNames(tts.model.config, { prefix: "present", session_name: "flow_lm" })];
        expect(cache).toEqual(flow_lm.outputNames.filter((name) => name.startsWith("present.")));
        for (const name of cache) expect(flow_lm.inputNames).toContain(name.replace("present.", "past_key_values."));
      });

      it("discovers all complete mixed-precision variants", async () => {
        expect(await ModelRegistry.get_available_dtypes(MODEL_ID)).toEqual(["fp32", "q4", "q4f16"]);
      }, MAX_TEST_EXECUTION_TIME);

      it("generates finite 24 kHz audio per chunk, stops on EOS and repeats deterministically", async () => {
        const text = "Hello from Phonon. This is a second sentence.";
        const { frame_rate, samples_per_frame } = tts.model.config.phonon;
        const speaker_embeddings = await hub_voice();
        try {
          const chunks = await tts.processor(text, { max_tokens: 10 });
          expect(chunks.length).toBeGreaterThan(1);
          const parts = [];
          for await (const pcm of tts.model.generate_speech({ chunks, speaker_embeddings, temperature: 0 })) parts.push(pcm);
          expect(parts).toHaveLength(chunks.length);
          for (const [i, pcm] of parts.entries()) {
            const max_frames = Math.ceil((chunks[i].input_ids.length / 3 + 2) * frame_rate);
            expect(pcm.length % samples_per_frame).toBe(0);
            expect(pcm.length / samples_per_frame).toBeGreaterThan(1);
            expect(pcm.length / samples_per_frame).toBeLessThan(max_frames);
            expect(pcm.every(Number.isFinite)).toBe(true);
            expect(pcm.some((sample) => Math.abs(sample) > 0.01)).toBe(true);
          }
          const output = await tts("Hello from Phonon.", { speaker_embeddings, temperature: 0 });
          expect(output.sampling_rate).toBe(24000);
          expect(output.audio.length % samples_per_frame).toBe(0);
          expect((await tts("Hello from Phonon.", { speaker_embeddings, temperature: 0 })).audio).toEqual(output.audio);
        } finally {
          speaker_embeddings.dispose();
        }
      }, MAX_TEST_EXECUTION_TIME);

      it("takes voices as URLs, arrays and tensors, and recovers from an early generator exit", async () => {
        const url = hub_voice_url("LMDIYE0Oa377Qh6Z");
        const voice = await hub_voice("LMDIYE0Oa377Qh6Z");
        try {
          expect(voice.dims).toEqual([1, 512, 125]);
          for await (const pcm of tts.model.generate_speech({ chunks: await tts.processor("Hello."), speaker_embeddings: voice, temperature: 0 })) {
            expect(pcm.length).toBeGreaterThan(0);
            break;
          }
          const outputs = [];
          for (const speaker_embeddings of [url, new URL(url), voice, new Float32Array(voice.data)]) {
            outputs.push((await tts("Hello.", { speaker_embeddings, temperature: 0 })).audio);
          }
          for (const audio of outputs.slice(1)) expect(audio).toEqual(outputs[0]);
          const batch = await tts(["Hello.", "Goodbye."], { speaker_embeddings: voice, temperature: 0 });
          expect(batch).toHaveLength(2);
          expect(batch.every(({ audio }) => audio.length > 0 && audio.every(Number.isFinite))).toBe(true);
        } finally {
          voice.dispose();
        }
      }, MAX_TEST_EXECUTION_TIME);
    });
  });
};
