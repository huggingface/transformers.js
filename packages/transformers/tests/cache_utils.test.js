import { StaticCache, Tensor, AutoModelForCausalLM, AutoTokenizer } from "../src/transformers.js";
import { presentNameToPastName } from "../src/cache_utils.js";
import { runDecoderSession } from "../src/models/modeling_utils.js";
import { sessionRun } from "../src/models/session.js";
import { createGpuBufferTensor } from "../src/backends/onnx.js";
// In Node, `src/backends/onnx.js` uses `onnxruntime-node`'s `env` and Tensor class, not the
// copy of `onnxruntime-common`, so the fake WebGPU device must be injected there, and
// fake ORT output tensors must be instances of its Tensor class.
import * as ONNX_NODE from "onnxruntime-node";

import { init, MAX_MODEL_LOAD_TIME, MAX_MODEL_DISPOSE_TIME, MAX_TEST_EXECUTION_TIME, DEFAULT_MODEL_OPTIONS } from "./init.js";

// Initialise the testing environment
init();

class FakeGpuBuffer {
  destroyed = false;
  constructor(descriptor) {
    this.size = descriptor.size;
    this.usage = descriptor.usage;
  }
  destroy() {
    this.destroyed = true;
  }
}

class FakeGpuDevice {
  /** @type {FakeGpuBuffer[]} */
  buffers = [];
  #calls = 0;
  #failOnCall;

  /**
   * @param {Object} [options]
   * @param {number} [options.failOnCall]
   */
  constructor({ failOnCall = Infinity } = {}) {
    this.#failOnCall = failOnCall;
  }

  createBuffer(descriptor) {
    if (++this.#calls === this.#failOnCall) {
      throw new Error("FakeGpuDevice: createBuffer failed.");
    }
    const buffer = new FakeGpuBuffer(descriptor);
    this.buffers.push(buffer);
    return buffer;
  }
}

// GPUBufferUsage flag values.
const GPU_BUFFER_USAGE = Object.freeze({ COPY_SRC: 0x0004, COPY_DST: 0x0008, STORAGE: 0x0080 });

const NUM_KV_HEADS = 8;
const HEAD_DIM = 64;

/**
 * Build a fake decoder session whose metadata mirrors a GQA-based WebGPU export.
 *
 * @param {Object} [options]
 * @param {number} [options.numLayers] Number of decoder layers.
 * @param {string|number} [options.presentSeqDim] Sequence-axis dim of the `present.*` outputs.
 * @param {string} [options.device] The session's device.
 */
function makeGqaSession({ numLayers = 2, presentSeqDim = "total_sequence_length", device = "webgpu" } = {}) {
  const inputMetadata = [{ name: "input_ids", isTensor: true, type: "int64", shape: ["batch_size", "sequence_length"] }];
  const outputMetadata = [{ name: "logits", isTensor: true, type: "float32", shape: ["batch_size", "sequence_length", 32064] }];
  for (let i = 0; i < numLayers; ++i) {
    for (const kv of ["key", "value"]) {
      inputMetadata.push({
        name: `past_key_values.${i}.${kv}`,
        isTensor: true,
        type: "float32",
        shape: ["batch_size", NUM_KV_HEADS, "past_sequence_length", HEAD_DIM],
      });
      outputMetadata.push({
        name: `present.${i}.${kv}`,
        isTensor: true,
        type: "float32",
        shape: ["batch_size", NUM_KV_HEADS, presentSeqDim, HEAD_DIM],
      });
    }
  }
  return { config: { device }, inputMetadata, outputMetadata };
}

/** Collect the cache input names of a fake session. */
function cacheInputNames(session) {
  return new Set(session.inputMetadata.map((m) => m.name).filter((name) => name.startsWith("past_")));
}

describe("StaticCache", () => {
  describe("state machine (no allocation)", () => {
    it("constructs with a positive integer max_cache_len", () => {
      const cache = new StaticCache({ max_cache_len: 4096 });
      expect(cache.max_cache_len).toBe(4096);
      expect(cache.allocated).toBe(false);
      expect(cache.get_seq_length()).toBe(0);
    });

    it("rejects invalid max_cache_len values", () => {
      for (const max_cache_len of [0, -1, 1.5, NaN, Infinity, "4096", null, undefined]) {
        expect(() => new StaticCache({ max_cache_len })).toThrow(/max_cache_len/);
      }
    });

    it("checks capacity without mutating state, and only _commit advances the length", () => {
      const cache = new StaticCache({ max_cache_len: 8 });

      expect(() => cache._checkCapacity(8)).not.toThrow();
      expect(() => cache._checkCapacity(9)).toThrow(/capacity exceeded/);
      expect(cache.get_seq_length()).toBe(0);

      cache._commit(5);
      expect(cache.get_seq_length()).toBe(5);
      expect(() => cache._checkCapacity(3)).not.toThrow();
      expect(() => cache._checkCapacity(4)).toThrow(/capacity exceeded/);
      // A failed check must leave the logical length untouched.
      expect(cache.get_seq_length()).toBe(5);

      cache._commit(3);
      expect(cache.get_seq_length()).toBe(8);
      expect(() => cache._checkCapacity(1)).toThrow(/capacity exceeded/);
    });

    it("update() is a no-op", () => {
      const cache = new StaticCache({ max_cache_len: 8 });
      cache._commit(2);
      expect(() => cache.update({})).not.toThrow();
      expect(cache.get_seq_length()).toBe(2);
      expect(Object.keys(cache)).toHaveLength(0);
    });

    it("cannot be acquired by two generations at once, nor disposed while in use", async () => {
      const cache = new StaticCache({ max_cache_len: 8 });
      cache._acquire();
      expect(() => cache._acquire()).toThrow(/already in use/);
      await expect(cache.dispose()).rejects.toThrow(/in use/);
      cache._release();
      // Released caches can be reused by a later generation, and disposed.
      expect(() => cache._acquire()).not.toThrow();
      cache._release();
      await expect(cache.dispose()).resolves.toBeUndefined();
    });

    it("rejects any use after dispose, and dispose is idempotent", async () => {
      const cache = new StaticCache({ max_cache_len: 8 });
      await cache.dispose();
      await expect(cache.dispose()).resolves.toBeUndefined();

      expect(() => cache._acquire()).toThrow(/disposed/);
      expect(() => cache._checkCapacity(1)).toThrow(/disposed/);
      await expect(cache._allocate({}, new Set(), {})).rejects.toThrow(/disposed/);
    });
  });

  describe("allocation (mocked WebGPU device)", () => {
    const ORT_ENV = ONNX_NODE.env;
    /** @type {FakeGpuDevice} */
    let device;

    beforeAll(() => {
      // `createGpuBufferTensor` reads the global GPUBufferUsage flags, which don't exist in Node.
      globalThis.GPUBufferUsage ??= GPU_BUFFER_USAGE;
    });

    beforeEach(() => {
      device = new FakeGpuDevice();
      ORT_ENV.webgpu.device = device;
    });

    afterEach(() => {
      delete ORT_ENV.webgpu.device;
    });

    it("allocates one fixed-size GPU entry per cache input and binds present outputs in place", async () => {
      const cache = new StaticCache({ max_cache_len: 128 });
      const session = makeGqaSession();
      await cache._allocate(session, cacheInputNames(session), { batch_size: 1 });

      expect(cache.allocated).toBe(true);
      expect(Object.keys(cache).sort()).toEqual(["past_key_values.0.key", "past_key_values.0.value", "past_key_values.1.key", "past_key_values.1.value"]);

      // The symbolic sequence axis is pinned to max_cache_len; other symbols come from `symbols`.
      const entry = cache["past_key_values.0.key"];
      expect(entry.dims).toEqual([1, NUM_KV_HEADS, 128, HEAD_DIM]);
      expect(entry.location).toBe("gpu-buffer");

      // One zero-copy buffer per entry, sized for float32 and 16-byte aligned.
      expect(device.buffers).toHaveLength(4);
      const expectedBytes = 1 * NUM_KV_HEADS * 128 * HEAD_DIM * 4;
      for (const buffer of device.buffers) {
        expect(buffer.size).toBe(expectedBytes);
        expect(buffer.usage).toBe(GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_SRC | GPU_BUFFER_USAGE.COPY_DST);
        expect(buffer.destroyed).toBe(false);
      }

      // Each `present.*` output is fetched into the matching `past.*` tensor (same wrapper),
      // while non-cache outputs stay ORT-allocated (null).
      const fetches = cache._getFetches();
      expect(fetches["present.0.key"]).toBe(cache["past_key_values.0.key"]);
      expect(fetches["present.1.value"]).toBe(cache["past_key_values.1.value"]);
      expect(fetches["logits"]).toBeNull();

      await cache.dispose();
    });

    it("allocates non-attention states (hybrid caches) from fully-resolved dims", async () => {
      const cache = new StaticCache({ max_cache_len: 128 });
      const session = makeGqaSession({ numLayers: 1 });
      // Model a conv state entry: fixed dims apart from the batch axis.
      session.inputMetadata.push({ name: "past_conv.0", isTensor: true, type: "float32", shape: ["batch_size", 512, 3] });
      session.outputMetadata.push({ name: "present_conv.0", isTensor: true, type: "float32", shape: ["batch_size", 512, 3] });
      await cache._allocate(session, cacheInputNames(session), { batch_size: 1 });

      expect(cache["past_conv.0"].dims).toEqual([1, 512, 3]);
      expect(cache._getFetches()["present_conv.0"]).toBe(cache["past_conv.0"]);

      await cache.dispose();
    });

    it("destroys a fresh buffer if wrapping it into a tensor fails", async () => {
      const original = ONNX_NODE.Tensor.fromGpuBuffer;
      ONNX_NODE.Tensor.fromGpuBuffer = () => {
        throw new Error("fromGpuBuffer failed.");
      };
      try {
        await expect(createGpuBufferTensor("float32", [1, 2, 2])).rejects.toThrow(/fromGpuBuffer failed/);
      } finally {
        ONNX_NODE.Tensor.fromGpuBuffer = original;
      }
      expect(device.buffers).toHaveLength(1);
      expect(device.buffers[0].destroyed).toBe(true);
    });

    it("rolls back a mid-allocation failure and can retry cleanly", async () => {
      const cache = new StaticCache({ max_cache_len: 128 });
      const session = makeGqaSession();

      // Fail on the 3rd of 4 buffer allocations.
      device = new FakeGpuDevice({ failOnCall: 3 });
      ORT_ENV.webgpu.device = device;
      await expect(cache._allocate(session, cacheInputNames(session), { batch_size: 1 })).rejects.toThrow(/createBuffer failed/);

      // The two buffers created before the failure must be destroyed, and no partial
      // state may remain on the cache.
      expect(device.buffers).toHaveLength(2);
      expect(device.buffers.every((buffer) => buffer.destroyed)).toBe(true);
      expect(cache.allocated).toBe(false);
      expect(Object.keys(cache)).toHaveLength(0);
      expect(cache._getFetches()).toBeNull();

      // A retry on a healthy device starts from a clean slate and succeeds.
      device = new FakeGpuDevice();
      ORT_ENV.webgpu.device = device;
      await expect(cache._allocate(session, cacheInputNames(session), { batch_size: 1 })).resolves.toBeUndefined();
      expect(cache.allocated).toBe(true);
      expect(device.buffers).toHaveLength(4);

      await cache.dispose();
    });

    it("throws on an unresolvable symbolic dimension instead of guessing, and cleans up", async () => {
      const cache = new StaticCache({ max_cache_len: 128 });
      const session = makeGqaSession({ numLayers: 1 });
      // Only the second-to-last (sequence) axis of an attention entry may be inferred;
      // an unknown symbol elsewhere must be reported, not silently defaulted.
      session.inputMetadata.find((m) => m.name === "past_key_values.0.value").shape = ["batch_size", NUM_KV_HEADS, "past_sequence_length", "head_dim"];

      await expect(cache._allocate(session, cacheInputNames(session), { batch_size: 1 })).rejects.toThrow(/past_key_values\.0\.value.*head_dim|head_dim.*past_key_values\.0\.value/s);

      // The entry allocated before the failure (past_key_values.0.key) must be destroyed.
      expect(device.buffers).toHaveLength(1);
      expect(device.buffers[0].destroyed).toBe(true);
      expect(cache.allocated).toBe(false);
      expect(Object.keys(cache)).toHaveLength(0);
    });

    it("rejects concatenation-style exports that cannot share past/present buffers", async () => {
      const cache = new StaticCache({ max_cache_len: 128 });
      // Non-GQA (MHA) exports declare the present sequence axis as an expression over
      // the past length, so the output grows every step.
      const session = makeGqaSession({ presentSeqDim: "past_sequence_length + 1" });

      await expect(cache._allocate(session, cacheInputNames(session), { batch_size: 1 })).rejects.toThrow(/past_present_share_buffer/);

      // All buffers were created before the fetches check, so all must be destroyed.
      expect(device.buffers).toHaveLength(4);
      expect(device.buffers.every((buffer) => buffer.destroyed)).toBe(true);
      expect(cache.allocated).toBe(false);
    });

    it("rejects a fixed present sequence length that differs from max_cache_len", async () => {
      const cache = new StaticCache({ max_cache_len: 128 });
      const session = makeGqaSession({ presentSeqDim: 64 });
      await expect(cache._allocate(session, cacheInputNames(session), { batch_size: 1 })).rejects.toThrow(/does not match the allocated cache length/);
      expect(cache.allocated).toBe(false);

      // A fixed size that matches the allocated length is share-buffer compatible.
      const matching = makeGqaSession({ presentSeqDim: 128 });
      await cache._allocate(matching, cacheInputNames(matching), { batch_size: 1 });
      expect(cache.allocated).toBe(true);
      await cache.dispose();
    });

    it("rejects sessions that are not on the webgpu device", async () => {
      const cache = new StaticCache({ max_cache_len: 128 });
      const session = makeGqaSession({ device: "wasm" });
      await expect(cache._allocate(session, cacheInputNames(session), { batch_size: 1 })).rejects.toThrow(/webgpu/);
      expect(device.buffers).toHaveLength(0);
    });

    it("rejects sessions without cache inputs", async () => {
      const cache = new StaticCache({ max_cache_len: 128 });
      const session = {
        config: { device: "webgpu" },
        inputMetadata: [{ name: "input_ids", isTensor: true, type: "int64", shape: ["batch_size", "sequence_length"] }],
        outputMetadata: [{ name: "logits", isTensor: true, type: "float32", shape: ["batch_size", "sequence_length", 32064] }],
      };
      await expect(cache._allocate(session, new Set(), {})).rejects.toThrow(/no cache inputs/);
    });

    it("dispose() destroys all buffers and fully resets the cache", async () => {
      const cache = new StaticCache({ max_cache_len: 128 });
      const session = makeGqaSession();
      await cache._allocate(session, cacheInputNames(session), { batch_size: 1 });
      cache._commit(100);

      await cache.dispose();

      expect(device.buffers).toHaveLength(4);
      expect(device.buffers.every((buffer) => buffer.destroyed)).toBe(true);
      expect(Object.keys(cache)).toHaveLength(0);
      expect(cache.allocated).toBe(false);
      expect(cache.get_seq_length()).toBe(0);
    });
  });

  describe("generate() integration (tiny model, CPU)", () => {
    const model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM";

    /** @type {import("../src/transformers.js").PreTrainedModel} */
    let model;
    let tokenizer;
    let inputs;
    /** @type {number} Prompt length in tokens. */
    let n;

    beforeAll(async () => {
      model = await AutoModelForCausalLM.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      tokenizer = await AutoTokenizer.from_pretrained(model_id);
      inputs = tokenizer("hello");
      n = inputs.input_ids.dims.at(-1);
    }, MAX_MODEL_LOAD_TIME);

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);

    it(
      "capacity precheck is exact and fires before any inference runs",
      async () => {
        const max_new_tokens = 4;

        const tooSmall = new StaticCache({ max_cache_len: n + max_new_tokens - 2 });
        await expect(model.generate({ ...inputs, past_key_values: tooSmall, max_new_tokens, do_sample: false })).rejects.toThrow(/exceeds the StaticCache capacity/);

        expect(tooSmall.allocated).toBe(false);
        expect(() => tooSmall._acquire()).not.toThrow();
        tooSmall._release();
        await tooSmall.dispose();

        // Exactly `max_length - 1`.
        const exact = new StaticCache({ max_cache_len: n + max_new_tokens - 1 });
        await expect(model.generate({ ...inputs, past_key_values: exact, max_new_tokens, do_sample: false })).rejects.toThrow(/webgpu/);
        await exact.dispose();
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "fails with a clear error on non-WebGPU sessions and is released for retry",
      async () => {
        const cache = new StaticCache({ max_cache_len: 64 });
        await expect(model.generate({ ...inputs, past_key_values: cache, max_new_tokens: 2, do_sample: false })).rejects.toThrow(/webgpu/);
        expect(cache.allocated).toBe(false);

        // The `finally` in generate() must release the cache even on failure.
        await expect(model.generate({ ...inputs, past_key_values: cache, max_new_tokens: 2, do_sample: false })).rejects.toThrow(/webgpu/);
        await cache.dispose();
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "cannot be used by a second generation while one is in flight",
      async () => {
        const cache = new StaticCache({ max_cache_len: 64 });
        cache._acquire(); // Simulate a generation in flight.
        await expect(model.generate({ ...inputs, past_key_values: cache, max_new_tokens: 2, do_sample: false })).rejects.toThrow(/already in use/);
        cache._release();
        await cache.dispose();
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
});

describe("sessionRun with pre-allocated outputs (fetches)", () => {
  /** Create a fake InferenceSession whose `run` is provided by the test. */
  function makeFakeSession(run) {
    return {
      inputNames: ["input_ids"],
      outputNames: ["logits", "present.0.key"],
      config: {},
      run,
    };
  }

  /** Create a Tensor wrapper backed by onnxruntime-node's Tensor class. */
  function makeCacheTensor() {
    return new Tensor(new ONNX_NODE.Tensor("float32", new Float32Array(8), [1, 2, 2, 2]));
  }

  const INPUTS = () => ({ input_ids: new Tensor("int64", new BigInt64Array([1n]), [1, 1]) });

  it("returns the caller's exact wrapper for outputs written in place", async () => {
    const present = makeCacheTensor();
    let receivedFetches;
    const session = makeFakeSession(async (feeds, fetches) => {
      receivedFetches = fetches;
      // ORT returns pre-allocated outputs as the exact tensor objects passed in `fetches`.
      return {
        logits: new ONNX_NODE.Tensor("float32", new Float32Array(2), [1, 2]),
        "present.0.key": fetches["present.0.key"],
      };
    });

    const outputs = await sessionRun(session, INPUTS(), { logits: null, "present.0.key": present });

    // The fetches are unwrapped before being handed to ORT: raw ort tensors and nulls.
    expect(receivedFetches["present.0.key"]).toBe(present.ort_tensor);
    expect(receivedFetches["logits"]).toBeNull();

    expect(outputs["present.0.key"]).toBe(present);
    expect(outputs["logits"]).toBeInstanceOf(Tensor);
    expect(outputs["logits"]).not.toBe(present);
  });

  it("wraps an output freshly if ORT did not write it into the requested tensor", async () => {
    const present = makeCacheTensor();
    const substitute = new ONNX_NODE.Tensor("float32", new Float32Array(8), [1, 2, 2, 2]);
    const session = makeFakeSession(async () => ({
      logits: new ONNX_NODE.Tensor("float32", new Float32Array(2), [1, 2]),
      "present.0.key": substitute,
    }));

    const outputs = await sessionRun(session, INPUTS(), { logits: null, "present.0.key": present });

    expect(outputs["present.0.key"]).not.toBe(present);
    expect(outputs["present.0.key"]).toBeInstanceOf(Tensor);
    expect(outputs["present.0.key"].ort_tensor).toBe(substitute);
  });
});

describe("runDecoderSession StaticCache integrity guard (mocked WebGPU device)", () => {
  const ORT_ENV = ONNX_NODE.env;
  /** @type {FakeGpuDevice} */
  let device;

  beforeAll(() => {
    globalThis.GPUBufferUsage ??= GPU_BUFFER_USAGE;
  });

  beforeEach(() => {
    device = new FakeGpuDevice();
    ORT_ENV.webgpu.device = device;
  });

  afterEach(() => {
    delete ORT_ENV.webgpu.device;
  });

  async function makeAllocatedCache() {
    const cache = new StaticCache({ max_cache_len: 128 });
    const session = makeGqaSession();
    await cache._allocate(session, cacheInputNames(session), { batch_size: 1 });
    return cache;
  }

  /**
   * Create a fake decode session. `substituteName` selects a fetched output that ORT
   * "fails" to write in place, returning a freshly allocated tensor for it instead.
   */
  function makeDecodeSession({ substituteName = null } = {}) {
    return {
      inputNames: ["input_ids"],
      config: { device: "webgpu" },
      async run(feeds, fetches) {
        const outputs = { logits: new ONNX_NODE.Tensor("float32", new Float32Array(4), [1, 1, 4]) };
        for (const [name, tensor] of Object.entries(fetches)) {
          if (tensor === null) continue;
          outputs[name] = name === substituteName ? new ONNX_NODE.Tensor("float32", new Float32Array(tensor.dims.reduce((a, b) => a * b, 1)), tensor.dims) : tensor;
        }
        return outputs;
      },
    };
  }

  const inputIds = (numTokens) => ({ input_ids: new Tensor("int64", new BigInt64Array(numTokens).fill(1n), [1, numTokens]) });

  it("commits the step when all cache outputs were written in place", async () => {
    const cache = await makeAllocatedCache();
    const outputs = await runDecoderSession(makeDecodeSession(), inputIds(3), cache);
    expect(cache.get_seq_length()).toBe(3);
    expect(outputs["present.0.key"]).toBe(cache["past_key_values.0.key"]);
    await cache.dispose();
  });

  it("fails loudly without committing if an output was not written into the cache tensor", async () => {
    const cache = await makeAllocatedCache();

    // ORT contractually writes fetched outputs in place; if a backend ever substituted
    // its own tensor, the new KV data would not be in the cache's buffers. Committing
    // would silently corrupt the cache, so the step must fail before `_commit`.
    const broken = makeDecodeSession({ substituteName: "present.1.value" });
    await expect(runDecoderSession(broken, inputIds(3), cache)).rejects.toThrow(/present\.1\.value.*not written into the pre-allocated cache tensor/);
    expect(cache.get_seq_length()).toBe(0);

    // The cache state is untouched, so a healthy session can still use it.
    await runDecoderSession(makeDecodeSession(), inputIds(2), cache);
    expect(cache.get_seq_length()).toBe(2);
    await cache.dispose();
  });
});

describe("long-generation resource behaviour (mocked WebGPU device)", () => {
  const ORT_ENV = ONNX_NODE.env;
  /** @type {FakeGpuDevice} */
  let device;

  beforeAll(() => {
    globalThis.GPUBufferUsage ??= GPU_BUFFER_USAGE;
  });

  beforeEach(() => {
    device = new FakeGpuDevice();
    ORT_ENV.webgpu.device = device;
  });

  afterEach(() => {
    delete ORT_ENV.webgpu.device;
  });

  it("holds one wrapper per entry, releases every transient output, and allocates zero buffers across hundreds of steps", async () => {
    const NUM_STEPS = 300;
    const cache = new StaticCache({ max_cache_len: NUM_STEPS });
    const session = makeGqaSession();
    await cache._allocate(session, cacheInputNames(session), { batch_size: 1 });
    expect(device.buffers).toHaveLength(4);

    // A fake decode session faithful to ORT's `run(feeds, fetches)` contract: fetched
    // outputs are returned as the exact ort tensors that were passed in, while `logits`
    // is a freshly ORT-allocated GPU tensor on every step, whose release we can observe
    // through its `dispose` callback.
    let transientDisposed = 0;
    const fakeSession = {
      inputNames: ["input_ids"],
      outputNames: [...Object.keys(cache._getFetches())],
      config: { device: "webgpu" },
      async run(feeds, fetches) {
        const logitsBuffer = { destroyed: false, destroy() {} };
        const logits = /** @type {any} */ (ONNX_NODE.Tensor).fromGpuBuffer(logitsBuffer, {
          dataType: "float32",
          dims: [1, 1, 4],
          dispose: () => {
            transientDisposed += 1;
          },
        });
        const outputs = { logits };
        for (const [name, tensor] of Object.entries(fetches)) {
          if (tensor !== null) outputs[name] = tensor;
        }
        return outputs;
      },
    };

    const fetches = cache._getFetches();
    const presentNames = Object.keys(fetches).filter((name) => fetches[name] !== null);
    /** @type {Map<string, Set<Tensor>>} Every distinct wrapper observed per present output. */
    const seenWrappers = new Map(presentNames.map((name) => [name, new Set()]));

    for (let step = 0; step < NUM_STEPS; ++step) {
      // One decode step, as performed by `runDecoderSession`.
      cache._checkCapacity(1);
      const outputs = await sessionRun(fakeSession, { input_ids: new Tensor("int64", new BigInt64Array([1n]), [1, 1]) }, fetches);
      cache._commit(1);

      for (const name of presentNames) {
        seenWrappers.get(name).add(outputs[name]);
      }

      // Apply `generate()`'s identity-based disposal (it runs once, on the final outputs;
      // here deliberately on every step) to verify it never touches the cache's own
      // tensors while transient outputs stay freeable.
      const cachedTensors = new Set(Object.values(cache));
      for (const tensor of Object.values(outputs)) {
        if (tensor.location === "gpu-buffer" && !cachedTensors.has(tensor)) {
          tensor.dispose();
        }
      }
    }

    expect(cache.get_seq_length()).toBe(NUM_STEPS);

    // Exactly one wrapper per cache entry across all steps: the cache's own. No per-step
    // alias wrappers accumulate for the GC.
    for (const name of presentNames) {
      const wrappers = seenWrappers.get(name);
      expect(wrappers.size).toBe(1);
      expect(wrappers.has(cache[presentNameToPastName(name)])).toBe(true);
    }

    // Every transient (non-cache) output was released, once per step...
    expect(transientDisposed).toBe(NUM_STEPS);

    for (const name of presentNames) {
      expect(cache[presentNameToPastName(name)].location).toBe("gpu-buffer");
    }
    expect(device.buffers).toHaveLength(4);
    expect(device.buffers.every((buffer) => !buffer.destroyed)).toBe(true);

    await cache.dispose();
    expect(device.buffers.every((buffer) => buffer.destroyed)).toBe(true);
  });
});
