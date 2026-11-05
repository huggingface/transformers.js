import { MODEL_TYPES, MODEL_SESSION_CONFIG, getSessionsConfig } from "../../src/models/session_config.js";

describe("MODEL_SESSION_CONFIG", () => {
  it("should pin the KV cache of the ImageAudioTextToText decoder (regression: missing cache_sessions)", () => {
    const { cache_sessions } = getSessionsConfig(MODEL_TYPES.ImageAudioTextToText, {});
    expect(cache_sessions?.decoder_model_merged).toBe(true);
  });

  it("should declare cache_sessions for every generating model type", () => {
    // Every model type that loads a generation config runs an autoregressive decode loop,
    // so at least one of its sessions must have its KV-cache outputs marked as cacheable
    // (used on WebGPU to keep `present.*` outputs on-GPU via preferredOutputLocation).
    const missing = [];
    for (const [name, type] of Object.entries(MODEL_TYPES)) {
      const typeConfig = MODEL_SESSION_CONFIG[type];
      if (!typeConfig?.optional_configs?.generation_config) continue;
      const hasCacheSession = Object.values(typeConfig.cache_sessions ?? {}).some(Boolean);
      if (!hasCacheSession) missing.push(name);
    }
    expect(missing).toEqual([]);
  });

  it("should only keep session outputs on the GPU for model types that pass them straight to another session", () => {
    // `gpu_output_sessions` outputs are returned as GPU tensors on WebGPU, so they are only safe for model types whose
    // forward (`generic_text_to_text_forward`) downloads them when needed. Models that read or modify `inputs_embeds`
    // on the CPU directly (e.g., Gemma3n, Voxtral Realtime, Chatterbox) must not use it.
    const enabled = Object.entries(MODEL_TYPES)
      .filter(([, type]) => MODEL_SESSION_CONFIG[type]?.gpu_output_sessions)
      .map(([name]) => name);
    expect(enabled).toEqual(["ImageTextToText", "AudioTextToText"]);
    for (const name of enabled) {
      expect(MODEL_SESSION_CONFIG[MODEL_TYPES[name]].gpu_output_sessions).toEqual({ embed_tokens: ["inputs_embeds"] });
    }
  });
});
