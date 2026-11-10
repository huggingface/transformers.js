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

  it.each([false, true])("releases the internally encoded source after beam expansion (failure=%s)", async (fail) => {
    const model = modelWith();
    let destroyed = 0;
    const encoded = gpu([1, 1, 4], { dispose: () => ++destroyed });
    model.config.is_encoder_decoder = true;
    model._prepare_encoder_decoder_kwargs_for_generation = async ({ model_inputs }) => ({ ...model_inputs, encoder_outputs: encoded });
    if (fail)
      model.forward = async () => {
        throw Error("injected encoder continuation failure");
      };
    const result = model.generate({ num_beams: 2, max_new_tokens: 1, decoder_start_token_id: 1 });
    if (fail) await expect(result).rejects.toThrow("injected encoder continuation failure");
    else await result;
    expect(destroyed).toBe(1);
    expect(allocated.every((buffer) => buffer.destroyed)).toBe(true);
  });

  it.each(["beam validation", "processor preparation"])("releases encoded output after %s fails", async (failure) => {
    const model = modelWith();
    let destroyed = 0;
    const encoded = gpu([1, 1, 4], { dispose: () => ++destroyed });
    model.config.is_encoder_decoder = true;
    model._prepare_encoder_decoder_kwargs_for_generation = async ({ model_inputs }) => ({ ...model_inputs, encoder_outputs: encoded });
    if (failure === "processor preparation")
      model._get_logits_processor = () => {
        throw Error("injected processor failure");
      };
    await expect(model.generate({ num_beams: 2, num_return_sequences: 3, max_new_tokens: 1, decoder_start_token_id: 1 })).rejects.toThrow(failure === "beam validation" ? "must be <= num_beams" : "injected processor failure");
    expect(destroyed).toBe(1);
  });

  it("releases unused GPU step outputs before the next forward", async () => {
    const model = modelWith();
    let calls = 0,
      destroyed = 0;
    model.forward = async () => {
      expect(destroyed).toBe(calls);
      ++calls;
      return { logits: new Tensor("float32", [0, 1, 2, 3, 0, 1, 2, 3], [2, 1, 4]), unused: gpu([2, 1, 4], { dispose: () => ++destroyed }) };
    };
    await model.generate({ num_beams: 2, max_new_tokens: 3 });
    expect(calls).toBe(3);
    expect(destroyed).toBe(3);
  });

  it("rejects decoder-only multimodal beams before inference", async () => {
    const model = modelWith({ pixel_values: new Tensor("float32", [1], [1, 1]) });
    model.forward = async () => {
      throw new Error("must not run");
    };
    await expect(model.generate({ num_beams: 2, max_new_tokens: 1 })).rejects.toThrow("decoder-only multimodal inputs");
  });

  it.each([false, true])("retains public attention buffers and releases them on error=%s", async (fail) => {
    const model = modelWith();
    let destroyed = 0;
    let calls = 0;
    model.forward = async () => {
      if (++calls === 2 && fail) throw new Error("injected");
      return { logits: new Tensor("float32", [0, 1, 2, 3, 0, 1, 2, 3], [2, 1, 4]), "cross_attentions.0": gpu([2, 1, 1, 4], { dispose: () => ++destroyed }) };
    };
    const promise = model.generate({ num_beams: 2, output_attentions: true, return_dict_in_generate: true, max_new_tokens: 2 });
    if (fail) {
      await expect(promise).rejects.toThrow("injected");
      expect(destroyed).toBe(1);
    } else {
      const result = await promise;
      expect(result.cross_attentions.length).toBe(2);
      expect(destroyed).toBe(0);
      expect(result.beam_indices.tolist()[0].slice(0, 2)).toEqual([0n, 0n]);
      for (const step of result.cross_attentions) for (const tensor of step) tensor.dispose();
      expect(destroyed).toBe(2);
    }
  });
});
