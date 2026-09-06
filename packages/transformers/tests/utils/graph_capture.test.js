import { jest } from "@jest/globals";
import { Tensor } from "onnxruntime-common";
import { DecodeGraphCaptureSession } from "../../src/backends/utils/graph-capture.js";
import { getStaticCacheInfo } from "../../src/utils/static-cache.js";

const cacheNames = ["past_key_values.0.key", "past_key_values.0.value"];
const savedUsage = globalThis.GPUBufferUsage;
const savedMode = globalThis.GPUMapMode;
globalThis.GPUBufferUsage = { STORAGE: 128, COPY_SRC: 4, COPY_DST: 8, MAP_READ: 1 };
globalThis.GPUMapMode = { READ: 1 };
afterAll(() => {
  if (savedUsage === undefined) delete globalThis.GPUBufferUsage;
  else globalThis.GPUBufferUsage = savedUsage;
  if (savedMode === undefined) delete globalThis.GPUMapMode;
  else globalThis.GPUMapMode = savedMode;
});

function makeDevice() {
  return {
    createBuffer: jest.fn(({ size, usage }) => ({
      size,
      usage,
      bytes: new Uint8Array(size),
      destroy: jest.fn(),
      mapAsync: jest.fn(async () => {}),
      getMappedRange() {
        return this.bytes.buffer;
      },
    })),
    queue: {
      writeBuffer: jest.fn((buffer, offset, data) => {
        expect(offset % 4).toBe(0);
        expect(data.byteLength % 4).toBe(0);
        buffer.bytes.set(data, offset);
      }),
      submit: jest.fn((commands) => commands.forEach((command) => command.forEach((copy) => copy()))),
    },
    createCommandEncoder() {
      const copies = [];
      return {
        copyBufferToBuffer(source, sourceOffset, target, targetOffset, size) {
          expect(size % 4).toBe(0);
          copies.push(() => target.bytes.set(source.bytes.subarray(sourceOffset, sourceOffset + size), targetOffset));
        },
        finish: () => copies,
      };
    },
  };
}
function data(tensor) {
  if (tensor.location === "cpu") return tensor.data;
  const Type = tensor.type === "int64" ? BigInt64Array : Float32Array;
  return new Type(tensor.gpuBuffer.bytes.buffer, 0, tensor.size);
}
function makeSession(device = makeDevice()) {
  const capturedBuffers = new Map();
  const session = {
    inputNames: ["input_ids", "attention_mask", ...cacheNames],
    outputNames: ["logits", "present.0.key", "present.0.value"],
    inputMetadata: [{ name: "input_ids", isTensor: true, type: "int64", shape: ["batch_size", "sequence_length"] }, { name: "attention_mask", isTensor: true, type: "int64", shape: ["batch_size", "total_sequence_length"] }, ...cacheNames.map((name) => ({ name, isTensor: true, type: "float32", shape: ["batch_size", 2, "past_sequence_length", 2] }))],
    outputMetadata: [{ name: "logits", isTensor: true, type: "float32", shape: ["batch_size", 1, 4] }, ...cacheNames.map((name) => ({ name: name.replace("past_key_values", "present"), isTensor: true, type: "float32", shape: ["batch_size", 2, "total_sequence_length", 2] }))],
    // A deterministic shared-KV decoder: append token values to the cache and
    // derive logits from its active prefix. Wrong masks or strides change results.
    run: jest.fn(async (feeds, fetches) => {
      await Promise.resolve();
      const captured = !fetches;
      if (captured) {
        fetches = {};
        for (const name of ["logits", "present.0.key", "present.0.value"]) {
          const dims = name === "logits" ? [1, 1, 4] : feeds[name.replace("present", "past_key_values")].dims;
          const size = dims.reduce((a, b) => a * b, 1);
          if (!capturedBuffers.has(name)) capturedBuffers.set(name, device.createBuffer({ size: Math.ceil((size * 4) / 16) * 16, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE }));
          const buffer = capturedBuffers.get(name);
          fetches[name] = Tensor.fromGpuBuffer(buffer, { dataType: "float32", dims, download: async () => new Float32Array(buffer.bytes.buffer, 0, size).slice(), dispose: jest.fn() });
        }
      }
      const ids = Array.from(data(feeds.input_ids), Number);
      const length = Number(Array.from(data(feeds.attention_mask)).reduce((a, b) => a + b, 0n));
      const offset = length - ids.length;
      for (const name of cacheNames) {
        const cache = fetches[name.replace("past_key_values", "present")];
        if (!captured) expect(cache).toBe(feeds[name]);
        else data(cache).set(data(feeds[name]));
        const values = data(cache);
        for (let head = 0; head < 2; ++head)
          for (let i = 0; i < ids.length; ++i) {
            values[head * cache.dims[2] * 2 + (offset + i) * 2] = ids[i];
            values[head * cache.dims[2] * 2 + (offset + i) * 2 + 1] = head;
          }
      }
      const active = data(fetches["present.0.key"]).filter((_, i) => i < length * 2 && i % 2 === 0);
      const sum = active.reduce((a, b) => a + b, 0);
      const values = [sum, length, ids.at(-1), -sum];
      const logits = fetches.logits ?? new Tensor("float32", values, [1, 1, 4]);
      if (fetches.logits) data(logits).set(values);
      return { ...fetches, logits };
    }),
    release: jest.fn(async () => {}),
  };
  return session;
}
function feeds(ids, previous = null) {
  const past = previous?.["present.0.key"];
  const length = (past ? (getStaticCacheInfo(past)?.length ?? past.dims[2]) : 0) + ids.length;
  return {
    input_ids: new Tensor("int64", ids.map(BigInt), [1, ids.length]),
    attention_mask: new Tensor("int64", Array(length).fill(1n), [1, length]),
    ...Object.fromEntries(cacheNames.map((name) => [name, previous?.[name.replace("past_key_values", "present")] ?? new Tensor("float32", [], [1, 2, 0, 2])])),
  };
}

describe("decode-only graph capture", () => {
  let device, prefill, state;
  beforeEach(() => {
    device = makeDevice();
    prefill = makeSession();
    state = new DecodeGraphCaptureSession(prefill, device, { cacheNames, maxCacheLength: 8, enableGraphCapture: true });
  });
  afterEach(async () => state.dispose());

  it.each([[[1]], [[1, 2, 3]]])("skips capture for a prompt of %j, including a one-token prompt", async (ids) => {
    const output = await state.run(feeds(ids));
    expect(prefill.run.mock.calls.filter(([, , options]) => options.extra.gpu_graph_id === "-1")).toHaveLength(1);
    expect(prefill.run.mock.calls.every(([, , options]) => options.extra.gpu_graph_id === "-1")).toBe(true);
    expect(state.decodeInputs.size).toBe(0);
    expect(getStaticCacheInfo(output["present.0.key"]).length).toBe(ids.length);
    expect(output["present.0.key"].dims).toEqual([1, 2, 8, 2]);
  });

  it("captures only decode and shares the same KV input/output buffers with prefill", async () => {
    let output = await state.run(feeds([1, 2]));
    const first = output.logits;
    output = await state.run(feeds([3], output));
    const second = output.logits;
    output = await state.run(feeds([4], output));
    expect(state.decodeInputs.get("input_ids").dims).toEqual([1, 1]);
    expect(state.decodeInputs.get("attention_mask").dims).toEqual([1, 8]);
    expect(prefill.run.mock.calls.filter(([, , options]) => options.extra.gpu_graph_id === "-1")).toHaveLength(1);
    expect(prefill.run.mock.calls.filter(([, , options]) => options.extra.gpu_graph_id === "0")).toHaveLength(2);
    expect(prefill.run.mock.calls[1][0].input_ids).toBe(prefill.run.mock.calls[2][0].input_ids);
    expect(prefill.run.mock.calls[1][0][cacheNames[0]]).toBe(prefill.run.mock.calls[0][0][cacheNames[0]]);
    expect(Array.from(first.data)).toEqual([3, 2, 2, -3]);
    expect(Array.from(second.data)).toEqual([6, 3, 3, -6]);
    expect(Array.from(output.logits.data)).toEqual([10, 4, 4, -10]);
    expect(state.decodeOutputs.get("logits").location).toBe("gpu-buffer");
  });

  it("bypasses an existing captured decoder for later prefills and clears trailing mask values", async () => {
    let old = await state.run(feeds([1, 2, 3, 4]));
    old = await state.run(feeds([5], old));
    let output = await state.run(feeds([7]));
    expect(() => getStaticCacheInfo(old["present.0.key"])).toThrow("no longer valid");
    output = await state.run(feeds([2], output));
    expect(Array.from(output.logits.data)).toEqual([9, 2, 2, -9]);
    expect(Array.from(data(state.decodeInputs.get("attention_mask")))).toEqual([1n, 1n, 0n, 0n, 0n, 0n, 0n, 0n]);
    expect(prefill.run.mock.calls.filter(([, , options]) => options.extra.gpu_graph_id === "-1")).toHaveLength(2);
    expect(prefill.run.mock.calls.filter(([, , options]) => options.extra.gpu_graph_id === "0")).toHaveLength(2);
    expect(state.decodeInputs.get("input_ids").dims).toEqual([1, 1]);
  });

  it("runs multi-token cache continuation as prefill after capture", async () => {
    let output = await state.run(feeds([1]));
    output = await state.run(feeds([2], output));
    output = await state.run(feeds([3, 4], output));
    expect(Array.from(output.logits.data)).toEqual([10, 4, 4, -10]);
    expect(prefill.run.mock.calls.filter(([, , options]) => options.extra.gpu_graph_id === "-1")).toHaveLength(2);
    expect(prefill.run.mock.calls.filter(([, , options]) => options.extra.gpu_graph_id === "0")).toHaveLength(1);
    output = await state.run(feeds([5], output));
    expect(Array.from(output.logits.data)).toEqual([15, 5, 5, -15]);
  });

  it("supports an identical static-KV path with capture disabled", async () => {
    state.enableGraphCapture = false;
    let output = await state.run(feeds([1, 2]));
    output = await state.run(feeds([3], output));
    expect(Array.from(output.logits.data)).toEqual([6, 3, 3, -6]);
    expect(prefill.run.mock.calls.filter(([, , options]) => options.extra.gpu_graph_id === "-1")).toHaveLength(2);
    expect(prefill.run.mock.calls.every(([, , options]) => options.extra.gpu_graph_id === "-1")).toBe(true);
  });

  it("imports compact external KV data using each head's stride", async () => {
    const old = Object.fromEntries(cacheNames.map((name) => [name.replace("past_key_values", "present"), new Tensor("float32", [1, 0, 2, 0, 1, 1, 2, 1], [1, 2, 2, 2])]));
    const output = await state.run(feeds([3], old));
    expect(Array.from(output.logits.data)).toEqual([6, 3, 3, -6]);
    const values = data(state.cache.get(cacheNames[0]));
    expect(Array.from(values.slice(16, 22))).toEqual([1, 1, 2, 1, 3, 1]);
  });

  it("does not let disposing returned KV views destroy captured buffers", async () => {
    const output = await state.run(feeds([1]));
    const buffer = output["present.0.key"].gpuBuffer;
    output["present.0.key"].dispose();
    expect(buffer.destroy).not.toHaveBeenCalled();
    await state.dispose();
    await state.dispose();
    expect(buffer.destroy).toHaveBeenCalledTimes(1);
  });

  it("checks capacity before changing any buffers, then permits a valid call", async () => {
    const output = await state.run(feeds([1, 2]));
    const writes = device.queue.writeBuffer.mock.calls.length;
    await expect(state.run(feeds([3, 4, 5, 6, 7, 8, 9], output))).rejects.toThrow("capacity");
    expect(device.queue.writeBuffer).toHaveBeenCalledTimes(writes);
    expect(Array.from((await state.run(feeds([3], output))).logits.data)).toEqual([6, 3, 3, -6]);
  });

  it("rejects batches and mismatched attention-mask lengths", async () => {
    const input = feeds([1, 2]);
    input.input_ids = new Tensor("int64", [1n, 2n], [2, 1]);
    await expect(state.run(input)).rejects.toThrow("batch size 1");
    const badMask = feeds([1, 2]);
    badMask.attention_mask = new Tensor("int64", [1n], [1, 1]);
    await expect(state.run(badMask)).rejects.toThrow("attention_mask length");
    expect(prefill.run).not.toHaveBeenCalled();
  });

  it("frees partial decode allocations and retries without disabling capture", async () => {
    const output = await state.run(feeds([1, 2]));
    device.createBuffer.mockImplementationOnce(() => {
      throw new Error("out of memory");
    });
    await expect(state.run(feeds([3], output))).rejects.toThrow("out of memory");
    expect(state.decodeInputs.size).toBe(0);
    expect(Array.from((await state.run(feeds([3], output))).logits.data)).toEqual([6, 3, 3, -6]);
    expect(prefill.run.mock.calls.at(-1)[2].extra.gpu_graph_id).toBe("0");
  });

  it("rejects use of disposed buffers", async () => {
    const output = await state.run(feeds([1]));
    const buffer = output["present.0.key"].gpuBuffer;
    await state.dispose();
    expect(buffer.destroy).toHaveBeenCalledTimes(1);
    await expect(state.run(feeds([1]))).rejects.toThrow("disposed");
  });
});

const runtime = { Tensor, env: { versions: { web: "test" }, wasm: { proxy: false, wasmPaths: "test/" }, webgpu: {} }, InferenceSession: { create: jest.fn() } };
jest.unstable_mockModule("onnxruntime-node", () => runtime);
jest.unstable_mockModule("onnxruntime-web/webgpu", () => runtime);
jest.unstable_mockModule("../../src/utils/model-loader.js", () => ({ getCoreModelFile: async () => new Uint8Array(), getModelDataFiles: async () => [] }));
const { createInferenceSession, runInferenceSession } = await import("../../src/backends/onnx.js");
const { constructSessions, sessionRun } = await import("../../src/models/session.js");
const { Tensor: TransformersTensor } = await import("../../src/utils/tensor.js");
const { DynamicCache } = await import("../../src/cache_utils.js");
const staticConfig = { use_static_cache: true, cache_names: cacheNames, max_cache_length: 8 };
const modelConfig = { model_type: "phi3", num_hidden_layers: 1, num_attention_heads: 2, num_key_value_heads: 2, hidden_size: 4, vocab_size: 4 };

describe("decode capture integration", () => {
  let device, prefill;
  beforeEach(() => {
    device = makeDevice();
    prefill = makeSession();
    runtime.env.webgpu.device = Promise.resolve(device);
    runtime.env.wasm.proxy = false;
    runtime.env.versions.web = "test";
    runtime.InferenceSession.create.mockReset().mockResolvedValue(prefill);
  });

  it("enables C++ capture while retaining ordinary JS bindings and skips prefill", async () => {
    const release = prefill.release;
    const options = { executionProviders: ["webgpu"], enableGraphCapture: true };
    const session = await createInferenceSession(new Uint8Array(), options, staticConfig);
    expect(runtime.InferenceSession.create.mock.calls[0][1]).toMatchObject({ enableGraphCapture: false, extra: { "ep.webgpuexecutionprovider.enableGraphCapture": "1" } });
    let output = await runInferenceSession(session, feeds([1, 2]));
    output = await runInferenceSession(session, feeds([3], output));
    expect(runtime.InferenceSession.create).toHaveBeenCalledTimes(1);
    expect(prefill.run.mock.calls.map(([, , options]) => options.extra.gpu_graph_id)).toEqual(["-1", "0"]);
    expect(options.enableGraphCapture).toBe(true);
    await session.release();
    expect(release).toHaveBeenCalledTimes(1);
  });
  it("preserves logical KV lengths through Transformers.js tensor wrapping and cache updates", async () => {
    const session = await createInferenceSession(new Uint8Array(), { executionProviders: ["webgpu"], enableGraphCapture: true }, staticConfig);
    const input = Object.fromEntries(Object.entries(feeds([1, 2])).map(([name, tensor]) => [name, new TransformersTensor(tensor)]));
    const output = await sessionRun(session, input);
    const cache = new DynamicCache({ [cacheNames[0]]: output["present.0.key"], [cacheNames[1]]: output["present.0.value"] });
    expect(cache.get_seq_length()).toBe(2);
    expect(output["present.0.key"].dims[2]).toBe(8);
    expect(output.logits.tolist()).toEqual([[[3, 2, 2, -3]]]);
    await cache.dispose();
    await session.release();
  });

  it("never enables capture on encoder/component sessions", async () => {
    await constructSessions("test/model", { model: "model" }, { device: "webgpu", dtype: "fp32", config: modelConfig, session_options: { enableGraphCapture: true } });
    expect(runtime.InferenceSession.create.mock.calls[0][1].enableGraphCapture).toBe(false);
  });

  it("automatically selects static KV handling for a decoder with capture enabled", async () => {
    const sessions = await constructSessions("test/model", { model: "model" }, { device: "webgpu", dtype: "fp32", config: { ...modelConfig, "transformers.js_config": { max_cache_length: 8 } }, session_options: { enableGraphCapture: true } }, { model: true });
    expect(sessions.model.config).toMatchObject(staticConfig);
    await sessions.model.release();
  });

  it("recovers the global inference queue after an invalid call", async () => {
    const session = await createInferenceSession(new Uint8Array(), { executionProviders: ["webgpu"], enableGraphCapture: true }, staticConfig);
    await expect(runInferenceSession(session, feeds(Array(9).fill(1)))).rejects.toThrow("capacity");
    await expect(runInferenceSession(session, feeds([1]))).resolves.toHaveProperty("logits");
    await session.release();
  });

  it("releases buffers before rejecting a queued run after disposal", async () => {
    const session = await createInferenceSession(new Uint8Array(), { executionProviders: ["webgpu"], enableGraphCapture: true }, staticConfig);
    await runInferenceSession(session, feeds([1]));
    const released = session.release();
    await expect(runInferenceSession(session, feeds([2]))).rejects.toThrow("disposed");
    await released;
  });

  it.each(["cpu", "wasm"])("rejects a static cache with %s", async (provider) => {
    await expect(createInferenceSession(new Uint8Array(), { executionProviders: [provider], enableGraphCapture: true }, staticConfig)).rejects.toThrow("ONNX Runtime Web");
  });

  it("rejects proxy mode and native ORT", async () => {
    const options = { executionProviders: ["webgpu"], enableGraphCapture: true };
    runtime.env.wasm.proxy = true;
    await expect(createInferenceSession(new Uint8Array(), options, staticConfig)).rejects.toThrow("proxy");
    runtime.env.wasm.proxy = false;
    delete runtime.env.versions.web;
    await expect(createInferenceSession(new Uint8Array(), options, staticConfig)).rejects.toThrow("ONNX Runtime Web");
  });
});
