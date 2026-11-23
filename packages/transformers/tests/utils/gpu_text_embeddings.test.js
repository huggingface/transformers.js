import { Tensor as ONNXTensor } from "onnxruntime-node";

import { LlavaForConditionalGeneration, PretrainedConfig, Tensor } from "../../src/transformers.js";

// On WebGPU, the text embeddings of image/audio-text-to-text models are kept on the GPU and passed straight to the
// decoder, unless they are needed on the CPU. These tests use fake sessions that return GPU-resident tensors
// (`fromGpuBuffer` with custom download/dispose callbacks), so no GPU or model weights are needed.

const HIDDEN_SIZE = 4;
const IMAGE_TOKEN_ID = 9;

const config = new PretrainedConfig({
  model_type: "llava",
  image_token_index: IMAGE_TOKEN_ID,
  text_config: { model_type: "llama" },
});

/**
 * Create a model with fake sessions. GPU downloads/disposals and the location of the decoder's `inputs_embeds` are
 * recorded in `events`; the inputs the decoder received are stored in `decoder_inputs`. `liveGpuEmbeds()` returns the
 * number of GPU-resident embeddings that were created (with `gpuEmbeds`) but not released yet.
 * @param {Object} options
 * @param {string} [options.device="webgpu"] The device of the decoder session.
 * @param {boolean} [options.failDownload=false] Whether downloading the embeddings fails.
 * @param {boolean} [options.failDecoder=false] Whether running the decoder fails.
 * @param {boolean} [options.cpuEmbeds=false] Whether the `embed_tokens` session returns its output on the CPU.
 */
function createModel({ device = "webgpu", failDownload = false, failDecoder = false, cpuEmbeds = false } = {}) {
  const events = [];
  const decoder_inputs = {};
  let live = 0;
  const gpuEmbeds = (dims) => {
    live += 1;
    return new Tensor(
      ONNXTensor.fromGpuBuffer(/** @type {any} */ ({}), {
        dataType: "float32",
        dims,
        download: async () => {
          if (failDownload) throw new Error("download failed");
          events.push("download");
          return new Float32Array(dims.reduce((a, b) => a * b, 1)).fill(1);
        },
        dispose: () => {
          live -= 1;
          events.push("dispose");
        },
      }),
    );
  };

  const sessions = {
    embed_tokens: {
      inputNames: ["input_ids"],
      run: async ({ input_ids }) => {
        const dims = [...input_ids.dims, HIDDEN_SIZE];
        if (cpuEmbeds) return { inputs_embeds: new ONNXTensor("float32", new Float32Array(dims.reduce((a, b) => a * b, 1)).fill(1), dims) };
        return { inputs_embeds: gpuEmbeds(dims).ort_tensor };
      },
    },
    vision_encoder: {
      inputNames: ["pixel_values"],
      run: async () => ({ image_features: new ONNXTensor("float32", new Float32Array(HIDDEN_SIZE), [1, 1, HIDDEN_SIZE]) }),
    },
    decoder_model_merged: {
      inputNames: ["inputs_embeds", "attention_mask"],
      inputMetadata: [],
      config: { device },
      run: async ({ inputs_embeds, attention_mask }) => {
        events.push(`decoder:${inputs_embeds.location}`);
        if (failDecoder) throw new Error("decoder failed");
        Object.assign(decoder_inputs, { inputs_embeds, attention_mask });
        return { logits: new ONNXTensor("float32", new Float32Array(HIDDEN_SIZE), [1, 1, HIDDEN_SIZE]) };
      },
    },
  };
  const model = new LlavaForConditionalGeneration(config, sessions, {});
  return { model, events, decoder_inputs, gpuEmbeds, liveGpuEmbeds: () => live };
}

const input_ids = new Tensor("int64", [1n, 2n, 3n], [1, 3]);
const attention_mask = new Tensor("int64", [1n, 1n, 1n], [1, 3]);
const pixel_values = new Tensor("float32", new Float32Array(3), [1, 3]);

// Decode step: a single new token with a cache of length 2
const decode_input_ids = new Tensor("int64", [5n], [1, 1]);
const decode_attention_mask = new Tensor("int64", [1n, 1n], [1, 2]);
const past_key_values = { get_seq_length: () => 2 };

describe("GPU-resident text embeddings", () => {
  describe("encode_text", () => {
    it("downloads GPU-resident embeddings and releases the GPU buffer by default", async () => {
      const { model, events } = createModel();
      const inputs_embeds = await model.encode_text({ input_ids });
      expect(inputs_embeds.location).toBe("cpu");
      expect(inputs_embeds.data).toEqual(new Float32Array(3 * HIDDEN_SIZE).fill(1));
      // ORT's `getData(true)` releases the GPU buffer after a successful download
      expect(events).toEqual(["download", "dispose"]);
    });

    it("releases the GPU buffer when the download fails", async () => {
      const { model, events } = createModel({ failDownload: true });
      await expect(model.encode_text({ input_ids })).rejects.toThrow("download failed");
      expect(events).toEqual(["dispose"]);
    });
  });

  describe("_encode_input", () => {
    it("keeps the output on the GPU when requested", async () => {
      const { model, events } = createModel();
      const inputs_embeds = await model._encode_input("embed_tokens", { input_ids }, "inputs_embeds", { keepOnGpu: true });
      expect(inputs_embeds.location).toBe("gpu-buffer");
      expect(events).toEqual([]);
      inputs_embeds.dispose();
      expect(events).toEqual(["dispose"]);
    });
  });

  describe("forward", () => {
    it("passes the prefill embeddings to a WebGPU decoder on the GPU, and releases them afterwards", async () => {
      const { model, events, liveGpuEmbeds } = createModel();
      await model.forward({ input_ids, attention_mask });
      expect(events).toEqual(["decoder:gpu-buffer", "dispose"]);
      expect(liveGpuEmbeds()).toBe(0);
    });

    it("passes the decode-step embeddings to a WebGPU decoder on the GPU", async () => {
      const { model, events, liveGpuEmbeds } = createModel();
      await model.forward({ input_ids: decode_input_ids, attention_mask: decode_attention_mask, past_key_values });
      expect(events).toEqual(["decoder:gpu-buffer", "dispose"]);
      expect(liveGpuEmbeds()).toBe(0);
    });

    it("keeps the decode-step embeddings on the GPU when modality inputs are still present", async () => {
      const { model, events, decoder_inputs, liveGpuEmbeds } = createModel();
      await model.forward({
        input_ids: decode_input_ids,
        attention_mask: decode_attention_mask,
        pixel_values,
        past_key_values,
      });
      expect(events).toEqual(["decoder:gpu-buffer", "dispose"]);
      expect(liveGpuEmbeds()).toBe(0);
      // The attention mask is extended to the cache length on the CPU, independently of the embeddings
      expect(decoder_inputs.attention_mask.dims).toEqual([1, 3]);
    });

    it("does not release user-provided embeddings", async () => {
      const { model, events, gpuEmbeds, liveGpuEmbeds } = createModel();
      await model.forward({ inputs_embeds: gpuEmbeds([1, 3, HIDDEN_SIZE]), attention_mask });
      expect(events).toEqual(["decoder:gpu-buffer"]);
      // The caller still owns its embeddings
      expect(liveGpuEmbeds()).toBe(1);
    });

    it("does not dispose embeddings that the session returned on the CPU", async () => {
      // Requesting GPU-resident embeddings does not guarantee them, e.g., ORT returns empty tensors on the CPU
      const { model, events, decoder_inputs, liveGpuEmbeds } = createModel({ cpuEmbeds: true });
      await model.forward({ input_ids, attention_mask });
      expect(events).toEqual(["decoder:cpu"]);
      expect(liveGpuEmbeds()).toBe(0);
      // A disposed tensor would have the location "none"
      expect(decoder_inputs.inputs_embeds.location).toBe("cpu");
    });

    it("leaves no GPU embeddings behind when a step before the decoder throws", async () => {
      const { model, events, liveGpuEmbeds } = createModel();
      const promise = model.forward({
        input_ids: decode_input_ids,
        attention_mask: decode_attention_mask,
        pixel_values,
        past_key_values: {
          get_seq_length() {
            throw new Error("cache error");
          },
        },
      });
      await expect(promise).rejects.toThrow("cache error");
      // The GPU-resident embeddings are only computed right before the decoder call
      expect(events).toEqual([]);
      expect(liveGpuEmbeds()).toBe(0);
    });

    it("releases the embeddings when the decoder throws", async () => {
      const { model, events, liveGpuEmbeds } = createModel({ failDecoder: true });
      await expect(model.forward({ input_ids, attention_mask })).rejects.toThrow("decoder failed");
      expect(events).toEqual(["decoder:gpu-buffer", "dispose"]);
      expect(liveGpuEmbeds()).toBe(0);
    });

    it("downloads the embeddings when the decoder does not run on WebGPU", async () => {
      const { model, events, liveGpuEmbeds } = createModel({ device: "wasm" });
      await model.forward({ input_ids, attention_mask });
      expect(events).toEqual(["download", "dispose", "decoder:cpu"]);
      expect(liveGpuEmbeds()).toBe(0);
    });

    it("downloads the embeddings before merging them with image features on the CPU", async () => {
      const { model, events, decoder_inputs, liveGpuEmbeds } = createModel();
      await model.forward({
        input_ids: new Tensor("int64", [1n, BigInt(IMAGE_TOKEN_ID), 3n], [1, 3]),
        attention_mask,
        pixel_values,
      });
      expect(events).toEqual(["download", "dispose", "decoder:cpu"]);
      expect(liveGpuEmbeds()).toBe(0);
      // Text embeddings are all ones; the (single) image feature is all zeros
      expect(decoder_inputs.inputs_embeds.dims).toEqual([1, 3, HIDDEN_SIZE]);
      expect(Array.from(decoder_inputs.inputs_embeds.data)).toEqual([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]);
    });
  });
});
