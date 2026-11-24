// Import registry to populate MODEL_TYPE_MAPPING (side-effect import)
await import("../../src/models/registry.js");

const { get_model_files } = await import("../../src/utils/model_registry/get_model_files.js");

const SEQ2SEQ_CONFIG = {
  architectures: ["T5ForConditionalGeneration"],
  model_type: "t5",
};

describe("get_model_files", () => {
  it("should support custom file names for seq2seq models", async () => {
    const files = await get_model_files("test/model", {
      config: SEQ2SEQ_CONFIG,
      dtype: "fp32",
      model_file_name: {
        encoder_model: "custom_encoder",
        decoder_model_merged: "custom_decoder",
      },
    });

    expect(files).toEqual(["config.json", "onnx/custom_encoder.onnx", "onnx/custom_decoder.onnx", "generation_config.json"]);
  });
});
