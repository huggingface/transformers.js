import { jest } from "@jest/globals";
import { PreTrainedModel } from "../../src/models/modeling_utils.js";
import { Tensor } from "../../src/utils/tensor.js";
import { getTensorConstructor } from "../../src/backends/onnx.js";

const prepare = (encoded, model_inputs, guidance_scale = null) => PreTrainedModel.prototype._prepare_encoder_decoder_kwargs_for_generation.call({ sessions: { model: { inputNames: ["input_ids"], run: async () => ({ last_hidden_state: encoded }) } } }, { model_inputs, generation_config: { guidance_scale } });

it("releases the encoder GPU allocation if decoder batch validation throws", async () => {
  const dispose = jest.fn();
  const encoded = getTensorConstructor().fromGpuBuffer({ usage: 1 }, { dataType: "float32", dims: [2, 1, 4], dispose });
  const ids = new Tensor("int64", [1n, 2n], [2, 1]);
  await expect(prepare(encoded, { input_ids: ids, decoder_input_ids: [[1], [2], [3]] })).rejects.toThrow("different batch size");
  expect(dispose).toHaveBeenCalledTimes(1);
  expect(ids.tolist()).toEqual([[1n], [2n]]);
});

it.each(["broadcast", "guidance", "unchanged"])("transfers only final encoder output and frees temporary %s tensors", async (mode) => {
  const encoded = new Tensor("float32", [2, 3], [1, 1, 2]);
  const ids = new Tensor("int64", [1n], [1, 1]);
  const mask = new Tensor("int64", [1n], [1, 1]);
  const disposed = [];
  const original = Tensor.prototype.dispose;
  const spy = jest.spyOn(Tensor.prototype, "dispose").mockImplementation(function () {
    disposed.push(this.ort_tensor);
    return original.call(this);
  });
  try {
    const result = await prepare(encoded, { input_ids: ids, attention_mask: mask, decoder_input_ids: mode === "broadcast" ? [[1], [2]] : [[1]] }, mode === "guidance" ? 2 : null);
    expect(result.encoder_outputs.tolist()).toEqual(mode === "guidance" ? [[[2, 3]], [[0, 0]]] : mode === "broadcast" ? [[[2, 3]], [[2, 3]]] : [[[2, 3]]]);
    expect(disposed.filter((x) => x === encoded.ort_tensor)).toHaveLength(mode === "unchanged" ? 0 : 1);
    expect(disposed).not.toContain(result.encoder_outputs.ort_tensor);
    expect(disposed).not.toContain(mask.ort_tensor);
    expect(disposed).not.toContain(ids.ort_tensor);
    expect(disposed).toHaveLength(mode === "guidance" ? 3 : mode === "broadcast" ? 1 : 0);
    expect(mask.tolist()).toEqual([[1n]]);
    if (mode === "guidance") {
      expect(result.attention_mask.tolist()).toEqual([[1n], [0n]]);
      result.attention_mask.dispose();
    }
    result.encoder_outputs.dispose();
  } finally {
    spy.mockRestore();
  }
});

it("releases original, null and expanded encoder states if guidance preparation fails", async () => {
  const encoded = new Tensor("float32", [2, 3], [1, 1, 2]);
  const inputs = {
    input_ids: new Tensor("int64", [1n], [1, 1]),
    get attention_mask() {
      throw Error("mask preparation failed");
    },
  };
  const disposed = [];
  const original = Tensor.prototype.dispose;
  const spy = jest.spyOn(Tensor.prototype, "dispose").mockImplementation(function () {
    disposed.push(this.ort_tensor);
    return original.call(this);
  });
  try {
    await expect(prepare(encoded, inputs, 2)).rejects.toThrow("mask preparation failed");
    expect(disposed).toHaveLength(3);
    expect(new Set(disposed).size).toBe(3);
    expect(disposed).toContain(encoded.ort_tensor);
    expect(inputs.input_ids.tolist()).toEqual([[1n]]);
  } finally {
    spy.mockRestore();
  }
});
