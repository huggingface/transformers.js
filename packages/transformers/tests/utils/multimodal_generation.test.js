import { UltravoxModel } from "../../src/models/ultravox/modeling_ultravox.js";
import { SmolVLMForConditionalGeneration } from "../../src/models/smolvlm/modeling_smolvlm.js";
import { jest } from "@jest/globals";
import { Tensor } from "../../src/utils/tensor.js";
import { GenerationConfig } from "../../src/generation/configuration_utils.js";
import { LlavaForConditionalGeneration } from "../../src/models/llava/modeling_llava.js";
import { default_merge_input_ids_with_image_features } from "../../src/models/modeling_utils.js";
import { DynamicCache } from "../../src/cache_utils.js";

const tensor = (type, data, dims) => new Tensor(type, data, dims);
const ids = () => tensor("int64", [0n, 9n, 1n, 9n, 9n, 2n], [2, 3]);
function synthetic(extra = {}) {
  const forwards = [];
  const allocated = [];
  const own = (t) => {
    t.dispose = jest.fn(t.dispose.bind(t));
    allocated.push(t);
    return t;
  };
  const model = Object.assign(Object.create(LlavaForConditionalGeneration.prototype), {
    config: { model_type: "llava", image_token_index: 9, is_encoder_decoder: false },
    _validate_model_class() {},
    _prepare_generation_config: (_config, kwargs) => new GenerationConfig(kwargs),
    _prepare_model_inputs() {
      const input_ids = ids();
      return { model_input_name: "input_ids", inputs_tensor: input_ids, model_inputs: { input_ids, attention_mask: tensor("int64", [0n, 1n, 1n, 1n, 1n, 1n], [2, 3]), pixel_values: tensor("float32", [2, 4, 6], [3, 1]), ...extra } };
    },
    prepare_inputs_for_generation(_ids, inputs) {
      return inputs;
    },
    encode_text: jest.fn(async ({ input_ids }) => own(tensor("float32", new Float32Array(input_ids.size * 2), [...input_ids.dims, 2]))),
    encode_image: jest.fn(async () => own(tensor("float16", new Uint16Array([0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700]), [3, 2]))),
    async forward(inputs) {
      forwards.push({ embeds: inputs.inputs_embeds?.tolist(), mask: inputs.attention_mask.tolist(), positions: inputs.position_ids?.tolist(), deltas: inputs.rope_deltas?.tolist(), ids: inputs.input_ids.tolist() });
      const batch = inputs.input_ids.dims[0];
      const logits = tensor(
        "float32",
        Array.from({ length: batch * 10 }, (_, i) => i % 10),
        [batch, 1, 10],
      );
      const length = inputs.attention_mask.dims[1];
      return { logits, "present.0.key": tensor("float32", new Float32Array(batch * length), [batch, 1, length, 1]), "present.0.value": tensor("float32", new Float32Array(batch * length), [batch, 1, length, 1]) };
    },
  });
  return { model, forwards, allocated };
}

describe("Prepared multimodal generation", () => {
  it("shares preparation with audio without treating raw audio as beam rows", async () => {
    const { model, forwards } = synthetic({ pixel_values: null, audio_values: tensor("float32", [2, 4, 6], [3, 1]) });
    Object.setPrototypeOf(model, UltravoxModel.prototype);
    model.config = { model_type: "ultravox", audio_token_id: 9, is_encoder_decoder: false };
    model.encode_audio = jest.fn(async () => tensor("float32", [2, 3, 4, 5], [2, 1, 2]));
    const prepare = model._prepare_model_inputs.bind(model);
    model._prepare_model_inputs = () => {
      const result = prepare();
      result.model_inputs.input_ids.data[4] = 0n;
      return result;
    };
    await model.generate({ num_beams: 4, max_new_tokens: 2 });
    expect(model.encode_audio).toHaveBeenCalledTimes(1);
    expect(forwards[0].embeds[4][0]).toEqual([4, 5]);
    expect(forwards[1].embeds).toBeUndefined();
  });
  it("rejects ragged raw audio before scattering features into the wrong sample", async () => {
    const { model } = synthetic({ pixel_values: null, audio_values: tensor("float32", [2, 4, 6], [3, 1]) });
    Object.setPrototypeOf(model, UltravoxModel.prototype);
    model.config = { model_type: "ultravox", audio_token_id: 9, is_encoder_decoder: false };
    model.encode_audio = model.encode_image;
    await expect(model.generate({ num_beams: 2, max_new_tokens: 1 })).rejects.toThrow("variable-length padded audio is unsupported");
  });
  it.each(["pixel_values_videos", "video_grid_thw"])("guards unsupported %s before encoder allocation", async (key) => {
    const { model } = synthetic();
    await expect(model.generate({ num_beams: 2, max_new_tokens: 1, [key]: tensor("int64", [1n], [1, 1]) })).rejects.toThrow("decoder-only multimodal inputs");
    expect(model.encode_text).not.toHaveBeenCalled();
    expect(model.encode_image).not.toHaveBeenCalled();
  });
  it("opts SmolVLM images in explicitly", () => {
    const model = Object.assign(Object.create(SmolVLMForConditionalGeneration.prototype), { config: { model_type: "smolvlm" } });
    expect(model._multimodal_generation_config().modality_input_names).toEqual(["pixel_values"]);
    model.config.model_type = "custom";
    expect(model._multimodal_generation_config()).toBeNull();
  });
  it("keeps original token history lengths separate from an expanded embedding prefix", async () => {
    const { model, forwards } = synthetic();
    model._merge_input_ids_with_image_features = () => ({
      inputs_embeds: tensor("float32", new Float32Array(20), [2, 5, 2]),
      attention_mask: tensor("int64", [0n, 1n, 1n, 1n, 1n, 1n, 1n, 1n, 1n, 1n], [2, 5]),
    });
    const result = await model.generate({ num_beams: 2, max_new_tokens: 2 });
    expect(result.dims).toEqual([2, 5]);
    expect(forwards[0].embeds[0]).toHaveLength(5);
    expect(forwards[1].mask[0]).toEqual([0n, 1n, 1n, 1n, 1n, 1n]);
  });
  it("releases prepared tensors over 25 consecutive calls", async () => {
    const { model, allocated } = synthetic();
    for (let i = 0; i < 25; ++i) await model.generate({ num_beams: 2, max_new_tokens: 2 });
    expect(allocated).toHaveLength(50);
    expect(allocated.every((t) => t.dispose.mock.calls.length === 1)).toBe(true);
  });
  it.each([1, 2, 4])("encodes a ragged padded batch once, consumes prefill once (%s beams)", async (num_beams) => {
    const { model, forwards, allocated } = synthetic();
    const output = await model.generate({ num_beams, max_new_tokens: 3, eos_token_id: null });
    expect(output.dims).toEqual([2, 6]);
    expect(model.encode_image).toHaveBeenCalledTimes(1);
    expect(model.encode_text).toHaveBeenCalledTimes(1);
    expect(forwards[0].embeds).toEqual([
      ...Array.from({ length: num_beams }, () => [
        [0, 0],
        [2, 3],
        [0, 0],
      ]),
      ...Array.from({ length: num_beams }, () => [
        [4, 5],
        [6, 7],
        [0, 0],
      ]),
    ]);
    expect(forwards[1].embeds).toBeUndefined();
    expect(forwards[2].embeds).toBeUndefined();
    expect(forwards[1].mask[0]).toEqual([0n, 1n, 1n, 1n]);
    expect(allocated.every((t) => t.dispose.mock.calls.length === 1)).toBe(true);
  });
  it("preserves caller embeddings, bypassing both encoders", async () => {
    const inputs_embeds = tensor("float32", new Float32Array(12).fill(7), [2, 3, 2]);
    const dispose = jest.spyOn(inputs_embeds, "dispose");
    const { model, forwards } = synthetic({ inputs_embeds });
    await model.generate({ num_beams: 2, max_new_tokens: 2 });
    expect(model.encode_text).not.toHaveBeenCalled();
    expect(model.encode_image).not.toHaveBeenCalled();
    expect(forwards[1].embeds).toBeUndefined();
    expect(dispose).not.toHaveBeenCalled();
    expect(inputs_embeds.data[0]).toBe(7);
  });
  it.each(["encode", "merge", "forward"])("releases owned allocations after %s failure", async (stage) => {
    const { model, allocated } = synthetic();
    if (stage === "encode")
      model.encode_image = async () => {
        throw new Error("injected");
      };
    if (stage === "merge")
      model._merge_input_ids_with_image_features = () => {
        throw new Error("injected");
      };
    if (stage === "forward")
      model.forward = async () => {
        throw new Error("injected");
      };
    await expect(model.generate({ num_beams: 2, max_new_tokens: 2 })).rejects.toThrow("injected");
    expect(allocated.length).toBeGreaterThan(0);
    expect(allocated.every((t) => t.dispose.mock.calls.length === 1)).toBe(true);
  });
  it("expands Qwen positions on axis 1 and rope deltas on axis 0", async () => {
    const position_ids = tensor("int64", [1n, 2n, 3n, 11n, 12n, 13n, 21n, 22n, 23n, 31n, 32n, 33n, 41n, 42n, 43n, 51n, 52n, 53n], [3, 2, 3]);
    const { model, forwards } = synthetic({ position_ids, rope_deltas: tensor("int64", [7n, 17n], [2, 1]) });
    const config = model._multimodal_generation_config();
    model._multimodal_generation_config = () => ({ ...config, batch_axes: { position_ids: 1, rope_deltas: 0 } });
    await model.generate({ num_beams: 2, max_new_tokens: 2 });
    expect(forwards[0].positions).toEqual(position_ids.tolist().map((rows) => rows.flatMap((row) => [row, row])));
    expect(forwards[0].deltas).toEqual([[7n], [7n], [17n], [17n]]);
    expect(forwards[1].deltas).toEqual(forwards[0].deltas);
  });
  it("does not re-encode modalities for supplied nonempty cache", async () => {
    const cache = new DynamicCache({ "past_key_values.0.key": tensor("float32", new Float32Array(12), [2, 1, 3, 2]) });
    const { model } = synthetic({ past_key_values: cache, input_ids: tensor("int64", [1n, 2n], [2, 1]) });
    await model.generate({ num_beams: 2, max_new_tokens: 1 });
    expect(model.encode_image).not.toHaveBeenCalled();
    expect(cache.get_seq_length()).toBe(3);
  });
  it("merges newly supplied image placeholders after a cached text prefix", async () => {
    const cache = new DynamicCache({ "past_key_values.0.key": tensor("float32", new Float32Array(12), [2, 1, 3, 2]) });
    const { model, forwards } = synthetic({ past_key_values: cache });
    await model.generate({ num_beams: 2, max_new_tokens: 1 });
    expect(model.encode_image).toHaveBeenCalledTimes(1);
    expect(forwards[0].embeds[0][1]).toEqual([2, 3]);
    expect(cache.get_seq_length()).toBe(3);
  });
  it.each(["float32", "float16"])("handles ONNX Uint16-backed embeddings with %s features", (type) => {
    const embeddings = tensor("float16", new Uint16Array(2), [1, 1, 2]);
    // Native ONNX feeds/results may expose Uint16 storage on modern Node too.
    embeddings.ort_tensor.cpuData = new Uint16Array(2);
    const features = type === "float32" ? tensor(type, [1, -2], [1, 2]) : tensor(type, new Uint16Array([0x3c00, 0xc000]), [1, 2]);
    default_merge_input_ids_with_image_features({ image_token_id: 9, input_ids: tensor("int64", [9n], [1, 1]), inputs_embeds: embeddings, image_features: features });
    expect(Array.from(embeddings.data)).toEqual([0x3c00, 0xc000]);
  });
  it("handles Uint16-backed half features with native half embeddings", () => {
    const embeddings = tensor("float16", new Uint16Array(2), [1, 1, 2]);
    const features = tensor("float16", new Uint16Array(2), [1, 2]);
    features.ort_tensor.cpuData = new Uint16Array([0x3c00, 0xc000]);
    default_merge_input_ids_with_image_features({ image_token_id: 9, input_ids: tensor("int64", [9n], [1, 1]), inputs_embeds: embeddings, image_features: features });
    expect(Array.from(embeddings.to("float32").data)).toEqual([1, -2]);
  });
  it("numerically converts float32 features into half embeddings", () => {
    const embeddings = tensor("float16", new Uint16Array(6), [1, 1, 6]);
    default_merge_input_ids_with_image_features({ image_token_id: 9, input_ids: tensor("int64", [9n], [1, 1]), inputs_embeds: embeddings, image_features: tensor("float32", [1, -2, 0.5, 65504, 2 ** -24, -0], [1, 6]) });
    expect(Array.from(new Uint16Array(embeddings.data.buffer))).toEqual([0x3c00, 0xc000, 0x3800, 0x7bff, 1, 0x8000]);
  });
});
