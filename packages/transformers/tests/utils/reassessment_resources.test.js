import { jest } from "@jest/globals";
import { PreTrainedTokenizer } from "../../src/tokenization_utils.js";
import { TextGenerationPipeline } from "../../src/pipelines/text-generation.js";
import { Pipeline, prepareAudios } from "../../src/pipelines/_base.js";
import { Tensor } from "../../src/utils/tensor.js";
import { env } from "../../src/env.js";
import { ModelFileNotFoundError } from "../../src/utils/hub/utils.js";
import { ModelRegistry } from "../../src/utils/model_registry/ModelRegistry.js";
import { AssetLoadingContext } from "../../src/utils/hub/AssetLoadingContext.js";
import { PretrainedConfig } from "../../src/configs.js";
import { pipeline } from "../../src/pipelines.js";
import { SUPPORTED_TASKS } from "../../src/pipelines/index.js";
import { mkdtemp, mkdir, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

function localTokenizer(response_template, prefix = "<assistant>") {
  const vocab = Object.fromEntries(Array.from({ length: 128 }, (_, i) => [String.fromCharCode(i), i]));
  return new PreTrainedTokenizer(
    {
      model: { type: "BPE", vocab, merges: [] },
      pre_tokenizer: { type: "Split", pattern: { String: "" }, behavior: "Isolated", invert: false },
      decoder: { type: "Fuse" },
      post_processor: null,
      normalizer: null,
      added_tokens: [],
    },
    {
      clean_up_tokenization_spaces: true,
      pad_token: "\u0000",
      chat_template: "{% for message in messages %}{{ message.role }}:{{ message.content }}{% endfor %}" + prefix,
      response_template,
    },
  );
}

const argumentsTemplate = { start_anchor: "<assistant>", fields: { tool_calls: { open: "<tool>", close: "</tool>", content: "json" } } };
const output = '<tool>{"arguments":"hello ."}</tool>';

function pipeFor(tokenizer, suffix) {
  return new TextGenerationPipeline({
    task: "text-generation",
    tokenizer,
    model: {
      _prepare_generation_config: () => ({ eos_token_id: [] }),
      generate: async ({ input_ids }) => {
        const rows = input_ids.tolist().map((row) => [...row, ...tokenizer.encode(suffix).map(BigInt)]);
        return new Tensor("int64", BigInt64Array.from(rows.flat()), [rows.length, rows[0].length]);
      },
    },
  });
}

describe("local parsed decoding", () => {
  it("keeps structured strings lossless for text, IDs, meaningful prefixes and bigint batches", () => {
    const tokenizer = localTokenizer(argumentsTemplate);
    expect(tokenizer.decode(tokenizer.encode("hello ."))).toBe("hello.");
    const expected = { tool_calls: { arguments: "hello ." } };
    expect(tokenizer.parse_response(output, { prefix: "<assistant>" })).toEqual(expected);
    expect(tokenizer.parse_response(tokenizer.encode(output), { prefix: tokenizer.encode("<assistant>") })).toEqual(expected);
    const prefix = '<assistant><tool>{"arguments":"hello .';
    expect(tokenizer.parse_response(tokenizer.encode('"}</tool>'), { prefix: tokenizer.encode(prefix) })).toEqual(expected);
    expect(tokenizer.parse_response([tokenizer.encode(output).map(BigInt)], { prefix: [tokenizer.encode("<assistant>").map(BigInt)] })).toEqual([expected]);
  });

  it("uses lossless parsed pipeline decoding, including prefix text", async () => {
    const tokenizer = localTokenizer(argumentsTemplate, '<assistant><tool>{"arguments":"hello .');
    const pipe = pipeFor(tokenizer, '"}</tool>');
    const result = await pipe([{ role: "user", content: "test" }]);
    expect(result[0].generated_text.at(-1)).toEqual({ role: "assistant", content: "", tool_calls: { arguments: "hello ." } });
  });

  it.each([undefined, "custom"])("returns reusable batched histories and preserves explicit role %j", async (role) => {
    const template = { ...(role === undefined ? {} : { defaults: { role } }), start_anchor: "<assistant>", fields: { content: { open: "<final>", close: "</final>" } } };
    const pipe = pipeFor(localTokenizer(template), "<final>hello</final>");
    const batch = [[{ role: "user", content: "a" }], [{ role: "user", content: "b" }]];
    const result = await pipe(batch);
    for (const row of result) {
      expect(row[0].generated_text.at(-1)).toEqual({ role: role ?? "assistant", content: "hello" });
      await expect(pipe(row[0].generated_text)).resolves.toBeDefined();
    }
  });

  it("keeps ordinary unparsed pipeline cleanup", async () => {
    const pipe = pipeFor(localTokenizer(undefined), "hello .");
    expect(await pipe("prompt", { return_full_text: false })).toEqual([{ generated_text: "hello." }]);
  });
});

describe("audio fail-fast capability", () => {
  const original = globalThis.AudioContext;
  afterEach(() => {
    if (original === undefined) delete globalThis.AudioContext;
    else globalThis.AudioContext = original;
  });
  it.each(["https://example.test/audio.wav", "not a valid URL"])("does not fetch or buffer without AudioContext: %s", async (url) => {
    delete globalThis.AudioContext;
    const arrayBuffer = jest.fn();
    const fetch = jest.fn(async () => ({ arrayBuffer }));
    const pipe = new Pipeline({ task: "test", model: null, sessionEnv: { fetch } });
    await expect(prepareAudios(pipe, url, 16000)).rejects.toThrow("AudioContext");
    expect(fetch).not.toHaveBeenCalled();
    expect(arrayBuffer).not.toHaveBeenCalled();
  });
  it("accepts typed arrays without AudioContext", async () => {
    delete globalThis.AudioContext;
    const pipe = new Pipeline({ task: "test", model: null });
    for (const input of [new Float32Array([1]), new Float64Array([1])]) {
      expect(await prepareAudios(pipe, input, 16000)).toEqual([new Float32Array(input)]);
    }
  });
  it("uses the scoped fetch when decoding is supported", async () => {
    globalThis.AudioContext = class {
      async decodeAudioData() {
        return { getChannelData: () => new Float32Array([0.5]), numberOfChannels: 1 };
      }
      async close() {}
    };
    const fetch = jest.fn(async () => new Response(new Uint8Array([1, 2])));
    const pipe = new Pipeline({ task: "test", model: null, sessionEnv: { fetch } });
    expect(await prepareAudios(pipe, "https://example.test/audio.wav", 16000)).toEqual([new Float32Array([0.5])]);
    expect(fetch).toHaveBeenCalledTimes(1);
  });
});

describe("required offline tokenizers", () => {
  it("rejects a tokenizer-less text pipeline before loading its model", async () => {
    const modelLoad = jest.spyOn(SUPPORTED_TASKS["text-generation"].model, "from_pretrained").mockResolvedValue({});
    try {
      await expect(
        pipeline("text-generation", "offline/missing-text", {
          config: new PretrainedConfig({ model_type: "gpt2" }),
          dtype: "fp32",
          env: { allowRemoteModels: false },
        }),
      ).rejects.toBeInstanceOf(ModelFileNotFoundError);
      expect(modelLoad).not.toHaveBeenCalled();
    } finally {
      modelLoad.mockRestore();
    }
  });
  it("loads cached metadata and tokenizer data with remote access disabled", async () => {
    const saved = { useCustomCache: env.useCustomCache, customCache: env.customCache };
    const fixture = localTokenizer(argumentsTemplate);
    const entries = new Map();
    const cache = { match: async (key) => (entries.has(key) ? new Response(entries.get(key)) : undefined), put: async () => {} };
    const options = { env: { allowRemoteModels: false, allowLocalModels: true } };
    const context = new AssetLoadingContext(options);
    for (const [name, value] of [
      ["tokenizer_config.json", fixture.config],
      ["tokenizer.json", fixture._tokenizerJSON],
    ]) {
      entries.set(context.resolve("offline/cached", name, cache).proposedCacheKey, JSON.stringify(value));
    }
    Object.assign(env, { useCustomCache: true, customCache: cache });
    try {
      const loaded = await PreTrainedTokenizer.from_pretrained("offline/cached", options);
      expect(loaded.decode(loaded.encode("hello ."))).toBe("hello.");
    } finally {
      Object.assign(env, saved);
    }
  });
  it.each(["global", "session", "local_files_only"])("reports missing files and loads cached/local files in %s mode", async (mode) => {
    const directory = await mkdtemp(join(tmpdir(), "reassessment-tokenizer-"));
    const original = env.allowRemoteModels;
    const fetch = jest.fn(() => {
      throw new Error("Unexpected network access");
    });
    const options = { env: { fetch, useFSCache: false, useBrowserCache: false, useCustomCache: false } };
    if (mode === "global") env.allowRemoteModels = false;
    if (mode === "session") options.env.allowRemoteModels = false;
    if (mode === "local_files_only") options.local_files_only = true;
    try {
      await expect(PreTrainedTokenizer.from_pretrained(directory, options)).rejects.toBeInstanceOf(ModelFileNotFoundError);
      await expect(
        pipeline("text-generation", directory, {
          ...options,
          config: new PretrainedConfig({ model_type: "gpt2" }),
          dtype: "fp32",
        }),
      ).rejects.toBeInstanceOf(ModelFileNotFoundError);
      // Discovery remains legitimately empty for tasks which do not require a tokenizer.
      expect(await ModelRegistry.get_tokenizer_files(directory, options)).toEqual([]);
      const visionFiles = await ModelRegistry.get_pipeline_files("image-classification", directory, {
        ...options,
        config: new PretrainedConfig({ model_type: "vit" }),
        dtype: "fp32",
      });
      expect(visionFiles).toContain("onnx/model.onnx");
      expect(visionFiles).not.toContain("tokenizer.json");
      const fixture = localTokenizer(argumentsTemplate);
      // Use a distinct identity: metadata memoization intentionally retains an earlier absence.
      const present = join(directory, "present");
      await mkdir(present);
      await writeFile(join(present, "tokenizer.json"), JSON.stringify(fixture._tokenizerJSON));
      await writeFile(join(present, "tokenizer_config.json"), JSON.stringify(fixture.config));
      const loaded = await PreTrainedTokenizer.from_pretrained(present, options);
      expect(loaded.decode(loaded.encode("hello ."))).toBe("hello.");
      expect(fetch).not.toHaveBeenCalled();
    } finally {
      env.allowRemoteModels = original;
      await rm(directory, { recursive: true, force: true });
    }
  });
});
