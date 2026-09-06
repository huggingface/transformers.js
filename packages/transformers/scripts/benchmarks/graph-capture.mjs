/**
 * Compare decode capture off/on using a graph-ready ORT GenAI Phi-4-mini export.
 * Node >=24, a Dawn Node binding with shader-f16/subgroups, and local model files
 * are required. No native ORT/GenAI inference library is used.
 *
 * node scripts/benchmarks/graph-capture.mjs --model PATH --webgpu-module PATH_TO_DAWN_NODE
 *   [--prompt-tokens 128] [--new-tokens 128] [--context 2048] [--warmup 2] [--runs 5]
 *   [--output results.json]
 */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { createRequire, register } from "node:module";
import { open, readFile, writeFile, mkdtemp, mkdir, copyFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";

const { values } = parseArgs({
  options: {
    model: { type: "string" },
    "webgpu-module": { type: "string" },
    output: { type: "string", default: "graph-capture-results.json" },
    "prompt-tokens": { type: "string", default: "128" },
    "new-tokens": { type: "string", default: "128" },
    context: { type: "string", default: "2048" },
    warmup: { type: "string", default: "2" },
    runs: { type: "string", default: "5" },
  },
});
if (!values.model || !values["webgpu-module"]) throw new Error("Required: --model <GenAI Phi-4-mini export> --webgpu-module <dawn.node or package name>");
if (Number(process.versions.node.split(".")[0]) < 24) throw new Error("Use Node.js 24 or newer for ORT's Asyncify runtime.");
const numbers = Object.fromEntries(
  ["prompt-tokens", "new-tokens", "context", "warmup", "runs"].map((name) => {
    const value = Number(values[name]);
    if (!Number.isSafeInteger(value) || value < (name === "warmup" ? 0 : 1)) throw new Error(`Invalid --${name}`);
    return [name, value];
  }),
);
if (numbers["new-tokens"] < 2 || numbers.context < numbers["prompt-tokens"] + numbers["new-tokens"])
  throw new Error("Need at least two new tokens and sufficient --context.");

const require = createRequire(import.meta.url);
const bindingPath = values["webgpu-module"];
const binding = bindingPath.endsWith(".node") ? require(path.resolve(bindingPath)) : await import(bindingPath);
Object.assign(globalThis, binding.globals);
Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu: binding.create(["backend=d3d12"]) } });
const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
if (!adapter?.features.has("shader-f16") || !adapter.features.has("subgroups"))
  throw new Error("The Dawn binding must provide shader-f16 and subgroups. On Windows use a Dawn build with DXC enabled.");
const adapterInfo = adapter.info;
register("./ort-web-loader.mjs", import.meta.url);
const ort = await import("onnxruntime-web/webgpu");
const ortDist = path.dirname(require.resolve("onnxruntime-web"));
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = {
  mjs: pathToFileURL(path.join(ortDist, "ort-wasm-simd-threaded.asyncify.mjs")).href,
  wasm: pathToFileURL(path.join(ortDist, "ort-wasm-simd-threaded.asyncify.wasm")).href,
};
// Transformers.js resolves local model paths in Node; the Web bundle takes bytes.
const originalCreate = ort.InferenceSession.create.bind(ort.InferenceSession);
ort.InferenceSession.create = async (model, options) => originalCreate(typeof model === "string" ? new Uint8Array(await readFile(model)) : model, options);
const { AutoModelForCausalLM, AutoTokenizer, Tensor, env } = await import("../../src/transformers.js");
env.useWasmCache = false;

// Read >2 GiB files in chunks (fs.readFile has a smaller single-read limit).
async function readLargeFile(filename) {
  const handle = await open(filename, "r");
  try {
    const data = new Uint8Array((await handle.stat()).size);
    for (let offset = 0; offset < data.length; ) {
      const { bytesRead } = await handle.read(data, offset, Math.min(256 * 1024 * 1024, data.length - offset), offset);
      if (!bytesRead) throw new Error(`Unexpected EOF: ${filename}`);
      offset += bytesRead;
    }
    return data;
  } finally {
    await handle.close();
  }
}
const modelSource = path.resolve(values.model);
const genai = JSON.parse(await readFile(path.join(modelSource, "genai_config.json"), "utf8"));
if (genai.model.type !== "phi3") throw new Error("This benchmark stages a Phi-4-mini (phi3) GenAI export.");
const decoder = genai.model.decoder;
const graph = await readFile(path.join(modelSource, decoder.filename));
// The graph-ready GenAI Phi-4 export uses a single model.onnx.data sidecar.
const weightName = `${decoder.filename}.data`;
console.log("Reading and hashing model assets (excluded from timing)...");
const weights = await readLargeFile(path.join(modelSource, weightName));
const hash = (data) => {
  const digest = createHash("sha256");
  for (let offset = 0; offset < data.length; offset += 256 * 1024 * 1024) digest.update(data.subarray(offset, offset + 256 * 1024 * 1024));
  return digest.digest("hex");
};
const staging = await mkdtemp(path.join(tmpdir(), "transformers-phi4-capture-"));
await mkdir(path.join(staging, "onnx"));
await writeFile(path.join(staging, "onnx", "model_q4f16.onnx"), graph);
for (const name of ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"]) {
  try {
    await copyFile(path.join(modelSource, name), path.join(staging, name));
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }
}
const config = {
  architectures: ["Phi3ForCausalLM"],
  model_type: "phi3",
  hidden_size: decoder.hidden_size,
  num_hidden_layers: decoder.num_hidden_layers,
  num_attention_heads: decoder.num_attention_heads,
  num_key_value_heads: decoder.num_key_value_heads,
  head_dim: decoder.head_size,
  vocab_size: genai.model.vocab_size,
  bos_token_id: genai.model.bos_token_id,
  eos_token_id: genai.model.eos_token_id,
  pad_token_id: genai.model.pad_token_id,
  max_position_embeddings: genai.model.context_length,
  "transformers.js_config": { use_static_cache: true, max_cache_length: numbers.context },
};
await writeFile(path.join(staging, "config.json"), JSON.stringify(config));
await writeFile(
  path.join(staging, "generation_config.json"),
  JSON.stringify({ eos_token_id: genai.model.eos_token_id, pad_token_id: genai.model.pad_token_id }),
);
const tokenizer = await AutoTokenizer.from_pretrained(staging, { local_files_only: true });
const prefix = tokenizer.encode("<|user|>Summarize this information: ", { add_special_tokens: false });
const suffix = tokenizer.encode("<|end|><|assistant|>", { add_special_tokens: false });
const body = tokenizer.encode(
  "The city library offers books, computer access, quiet study rooms, and weekly science workshops. ".repeat(numbers["prompt-tokens"]),
  { add_special_tokens: false },
);
const prompt =
  numbers["prompt-tokens"] > prefix.length + suffix.length
    ? [...prefix, ...body.slice(0, numbers["prompt-tokens"] - prefix.length - suffix.length), ...suffix]
    : body.slice(0, numbers["prompt-tokens"]);
assert.equal(prompt.length, numbers["prompt-tokens"]);
const inputs = {
  input_ids: new Tensor("int64", prompt.map(BigInt), [1, prompt.length]),
  attention_mask: new Tensor("int64", Array(prompt.length).fill(1n), [1, prompt.length]),
};
const report = {
  date: new Date().toISOString(),
  node: process.version,
  ort: ort.env.versions.web,
  webgpu_module: bindingPath,
  adapter: { vendor: adapterInfo.vendor, architecture: adapterInfo.architecture, device: adapterInfo.device, description: adapterInfo.description },
  model: modelSource,
  graph_sha256: hash(graph),
  weights_sha256: hash(weights),
  weights_bytes: weights.byteLength,
  workload: numbers,
  prompt_ids: prompt,
  trials: {},
};
let expected;
try {
  for (const capture of [false, true]) {
    const createStart = performance.now();
    const model = await AutoModelForCausalLM.from_pretrained(staging, {
      device: "webgpu",
      dtype: "q4f16",
      local_files_only: true,
      session_options: {
        enableGraphCapture: capture,
        executionProviders: [{ name: "webgpu", validationMode: "basic" }],
        externalData: [{ path: weightName, data: weights }],
      },
    });
    const modelLoadMs = performance.now() - createStart;
    const trials = [];
    try {
      for (let run = -numbers.warmup; run < numbers.runs; ++run) {
        const times = [];
        let firstPut = true;
        const start = performance.now();
        const output = await model.generate({
          ...inputs,
          max_new_tokens: numbers["new-tokens"],
          eos_token_id: null,
          do_sample: false,
          streamer: {
            put() {
              if (firstPut) firstPut = false;
              else times.push(performance.now());
            },
            end() {},
          },
        });
        const ids = Array.from(output.data, Number);
        if (expected) assert.deepEqual(ids, expected, "Capture on/off must generate identical token IDs");
        else expected = ids;
        assert.equal(times.length, numbers["new-tokens"]);
        const decodeMs = times.at(-1) - times[0];
        const trial = {
          ttft_ms: times[0] - start,
          decode_ms: decodeMs,
          decode_tokens_per_second: ((times.length - 1) * 1000) / decodeMs,
          total_ms: times.at(-1) - start,
        };
        console.log(
          `capture=${capture} ${run < 0 ? "warmup" : "trial"} ${run}: ${trial.decode_tokens_per_second.toFixed(2)} decode tok/s, TTFT ${trial.ttft_ms.toFixed(2)} ms`,
        );
        if (run >= 0) trials.push(trial);
      }
    } finally {
      await model.dispose();
    }
    report.trials[capture ? "capture_on" : "capture_off"] = { model_load_ms: modelLoadMs, runs: trials };
  }
  const median = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
  const off = median(report.trials.capture_off.runs.map((x) => x.decode_tokens_per_second));
  const on = median(report.trials.capture_on.runs.map((x) => x.decode_tokens_per_second));
  report.summary = {
    capture_off_decode_tps: off,
    capture_on_decode_tps: on,
    speedup: on / off,
    improvement_percent: (on / off - 1) * 100,
    identical_tokens: true,
  };
  report.generated_ids = expected.slice(prompt.length);
  report.generated_text = tokenizer.decode(report.generated_ids, { skip_special_tokens: true });
  await writeFile(path.resolve(values.output), JSON.stringify(report, null, 2) + "\n");
  console.log(JSON.stringify(report.summary, null, 2));
  console.log(`Results: ${path.resolve(values.output)}`);
} finally {
  (await ort.env.webgpu.device)?.destroy();
  delete globalThis.navigator;
}
