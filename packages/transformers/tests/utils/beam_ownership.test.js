import { env } from "../../src/env.js";
import { Tensor, index_select_async } from "../../src/utils/tensor.js";
import { getTensorConstructor } from "../../src/backends/onnx.js";
import { PreTrainedModel } from "../../src/models/modeling_utils.js";
import { GenerationConfig } from "../../src/generation/configuration_utils.js";
import { DynamicCache } from "../../src/cache_utils.js";

// Mock only the GPU allocation boundary; execute the real generation/cache code.
describe("Beam resource ownership", () => {
  let allocated, originalDevice, originalUsage;
  beforeEach(() => {
    allocated = [];
    originalDevice = env.backends.onnx.webgpu.device;
    originalUsage = globalThis.GPUBufferUsage;
    globalThis.GPUBufferUsage = { COPY_SRC: 1, COPY_DST: 2, STORAGE: 4, MAP_READ: 8 };
    env.backends.onnx.webgpu.device = {
      limits: { maxBufferSize: 1e9 },
      createBuffer(options) {
        const buffer = {
          ...options,
          destroyed: false,
          destroy() {
            this.destroyed = true;
          },
        };
        allocated.push(buffer);
        return buffer;
      },
      createCommandEncoder() {
        return {
          copyBufferToBuffer() {},
          finish() {
            return {};
          },
        };
      },
      queue: { submit() {} },
    };
  });
  afterEach(() => {
    env.backends.onnx.webgpu.device = originalDevice;
    if (originalUsage === undefined) delete globalThis.GPUBufferUsage;
    else globalThis.GPUBufferUsage = originalUsage;
  });

  const gpu = (dims, options = {}, usage = 1) => new Tensor(getTensorConstructor().fromGpuBuffer({ usage }, { dataType: "float32", dims, ...options }));
  const modelWith = (extra = {}) =>
    Object.assign(Object.create(PreTrainedModel.prototype), {
      config: { is_encoder_decoder: false },
      _validate_model_class() {},
      _prepare_generation_config(config, kwargs) {
        return new GenerationConfig(kwargs);
      },
      _prepare_model_inputs() {
        const ids = new Tensor("int64", [1n], [1, 1]);
        return { inputs_tensor: ids, model_input_name: "input_ids", model_inputs: { input_ids: ids, attention_mask: new Tensor("int64", [1n], [1, 1]), ...extra } };
      },
      prepare_inputs_for_generation(ids, inputs) {
        return inputs;
      },
      async forward() {
        return { logits: new Tensor("float32", [0, 1, 2, 3, 0, 1, 2, 3], [2, 1, 4]) };
      },
    });

  it.each([false, true])("releases expanded inputs on forward failure=%s without consuming caller input", async (fail) => {
    const positions = gpu([1, 1], { dataType: "int64" });
    const model = modelWith({ position_ids: positions });
    if (fail)
      model.forward = async () => {
        throw new Error("forward failed");
      };
    const result = model.generate({ num_beams: 2, max_new_tokens: 1, eos_token_id: null });
    if (fail) await expect(result).rejects.toThrow("forward failed");
    else expect((await result).tolist()).toEqual([[1n, 3n]]);
    expect(allocated.length).toBe(1);
    expect(allocated.every((buffer) => buffer.destroyed)).toBe(true);
    expect(positions.location).toBe("gpu-buffer");
    positions.dispose();
  });

  it("rolls back partial cache copies without consuming source tensors", async () => {
    const first = gpu([2, 4]);
    const second = gpu(
      [2, 4],
      {
        download: async () => {
          throw new Error("readback failed");
        },
      },
      0,
    );
    const cache = new DynamicCache({ first, second });
    await expect(PreTrainedModel.prototype._reorder_cache.call({ _get_cache_indices: (_key, indices) => indices }, cache, [1, 0])).rejects.toThrow("readback failed");
    expect(allocated.length).toBe(1);
    expect(allocated.every((buffer) => buffer.destroyed)).toBe(true);
    expect(first.location).toBe("gpu-buffer");
    expect(second.location).toBe("gpu-buffer");
    await cache.dispose();
  });

  it.each(["dispose", "update"])("releases downloaded GPU cache resources through %s", async (action) => {
    let destroyed = false;
    const odd = gpu([1, 3], {
      dataType: "float16",
      download: async () => new Uint16Array(3),
      dispose: () => {
        destroyed = true;
      },
    });
    const cache = new DynamicCache({ odd });
    await index_select_async(odd, [0, 0]);
    expect(odd.location).toBe("cpu");
    expect(destroyed).toBe(false);
    if (action === "dispose") await cache.dispose();
    else cache.update({ odd: new Tensor("float16", new Uint16Array(3), [1, 3]) });
    expect(destroyed).toBe(true);
  });

  it("releases previously downloaded cache sources after successful reordering", async () => {
    let destroyed = false;
    const source = gpu([1, 4], {
      download: async () => new Float32Array([1, 2, 3, 4]),
      dispose: () => {
        destroyed = true;
      },
    });
    await source.ort_tensor.getData(false);
    const reordered = await PreTrainedModel.prototype._reorder_cache.call({ _get_cache_indices: (_key, indices) => indices }, new DynamicCache({ source }), [0, 0]);
    expect(Array.from(reordered.source.data)).toEqual([1, 2, 3, 4, 1, 2, 3, 4]);
    expect(destroyed).toBe(true);
    await reordered.dispose();
  });

  it("releases the active cache when a later forward fails and preserves a supplied cache", async () => {
    const source = gpu([1, 1, 1, 4]);
    const caller = new DynamicCache({ "past_key_values.0.key": source });
    const model = modelWith({ past_key_values: caller });
    let calls = 0;
    let outputDestroyed = false;
    model.forward = async () => {
      if (++calls === 2) throw new Error("second forward failed");
      return {
        logits: new Tensor("float32", [0, 1, 2, 3, 0, 1, 2, 3], [2, 1, 4]),
        "present.0.key": gpu([2, 1, 1, 4], {
          dispose: () => {
            outputDestroyed = true;
          },
        }),
      };
    };
    await expect(model.generate({ num_beams: 2, max_new_tokens: 3, eos_token_id: null })).rejects.toThrow("second forward failed");
    expect(calls).toBe(2);
    expect(allocated.length).toBe(2);
    expect(allocated.every((buffer) => buffer.destroyed)).toBe(true);
    expect(outputDestroyed).toBe(true);
    expect(source.location).toBe("gpu-buffer");
    await caller.dispose();
  });

  it("rejects decoder-only multimodal beams before inference", async () => {
    const model = modelWith({ pixel_values: new Tensor("float32", [1], [1, 1]) });
    model.forward = async () => {
      throw new Error("must not run");
    };
    await expect(model.generate({ num_beams: 2, max_new_tokens: 1 })).rejects.toThrow("decoder-only multimodal inputs");
  });

  it("rejects beam attentions used for token timestamps before inference", async () => {
    const model = modelWith();
    model.forward = async () => {
      throw new Error("must not run");
    };
    await expect(model.generate({ num_beams: 2, output_attentions: true, max_new_tokens: 1 })).rejects.toThrow("output_attentions or token timestamps");
  });
});
