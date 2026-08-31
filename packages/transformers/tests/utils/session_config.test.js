import { getSessionsConfig, MODEL_TYPES } from "../../src/models/session_config.js";

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
