import { getSessionsConfig, MODEL_SESSION_CONFIG, MODEL_TYPES } from "../../src/models/session_config.js";

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
});

describe("model file name overrides", () => {
  it("applies file-name mappings to multi-session models", () => {
    const { sessions } = getSessionsConfig(
      MODEL_TYPES.Seq2Seq,
      {},
      {
        model_file_name: {
          encoder_model: "custom_encoder",
          decoder_model_merged: "custom_decoder",
        },
      },
    );

    expect(sessions).toEqual({
      model: "custom_encoder",
      decoder_model_merged: "custom_decoder",
    });
  });

  it("accepts runtime session names and preserves unspecified defaults", () => {
    const { sessions } = getSessionsConfig(
      MODEL_TYPES.Seq2Seq,
      {},
      {
        model_file_name: { model: "custom_encoder" },
      },
    );

    expect(sessions).toEqual({
      model: "custom_encoder",
      decoder_model_merged: "decoder_model_merged",
    });
  });

  it("keeps string overrides working for single-session models", () => {
    const { sessions } = getSessionsConfig(
      MODEL_TYPES.DecoderOnly,
      {},
      {
        model_file_name: "custom_model",
      },
    );

    expect(sessions).toEqual({ model: "custom_model" });
  });
});
