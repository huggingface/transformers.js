import { jest } from "@jest/globals";

// Mock get_file_metadata before importing the module under test
const mockGetFileMetadata = jest.fn();
jest.unstable_mockModule("../../src/utils/model_registry/get_file_metadata.js", () => ({
  get_file_metadata: mockGetFileMetadata,
}));

// Import registry to populate MODEL_TYPE_MAPPING (side-effect import)
await import("../../src/models/registry.js");

// Dynamic import after mock setup (required for ESM)
const { get_available_dtypes } = await import("../../src/utils/model_registry/get_available_dtypes.js");

// A minimal config that mimics a BERT-like encoder-only model
const ENCODER_ONLY_CONFIG = {
  architectures: ["BertModel"],
  model_type: "bert",
};

// A minimal config for a decoder-only (causal LM) model
const DECODER_ONLY_CONFIG = {
  architectures: ["LlamaForCausalLM"],
  model_type: "llama",
};

// A minimal config for a Seq2Seq model (encoder + decoder)
const SEQ2SEQ_CONFIG = {
  architectures: ["T5ForConditionalGeneration"],
  model_type: "t5",
};

// A config with an unknown architecture (falls back to EncoderOnly)
const UNKNOWN_ARCH_CONFIG = {
  architectures: ["SomeUnknownArchitecture"],
  model_type: "unknown_type",
};

/**
 * Helper: given a set of files that "exist", returns a mock implementation
 * for get_file_metadata that resolves { exists: true } for those files.
 * @param {string[]} existingFiles
 */
function setupExistingFiles(...existingFiles) {
  mockGetFileMetadata.mockImplementation((_modelId, filename, _options) => {
    return Promise.resolve({
      exists: existingFiles.includes(filename),
      fromCache: false,
    });
  });
}

describe("get_available_dtypes", () => {
  beforeEach(() => {
    mockGetFileMetadata.mockReset();
  });

  it("should detect fp32 and q4 for an encoder-only model", async () => {
    setupExistingFiles(
      "onnx/model.onnx", // fp32
      "onnx/model_q4.onnx", // q4
    );

    const dtypes = await get_available_dtypes("test/model", { config: ENCODER_ONLY_CONFIG });

    expect(dtypes).toContain("fp32");
    expect(dtypes).toContain("q4");
    expect(dtypes).not.toContain("fp16");
    expect(dtypes).not.toContain("q8");
    expect(dtypes).not.toContain("int8");
  });

  it("should detect all dtypes when all files exist", async () => {
    setupExistingFiles(
      "onnx/model.onnx", // fp32
      "onnx/model_fp16.onnx", // fp16
      "onnx/model_int8.onnx", // int8
      "onnx/model_uint8.onnx", // uint8
      "onnx/model_quantized.onnx", // q8
      "onnx/model_q4.onnx", // q4
      "onnx/model_q4f16.onnx", // q4f16
      "onnx/model_bnb4.onnx", // bnb4
    );

    const dtypes = await get_available_dtypes("test/model", { config: ENCODER_ONLY_CONFIG });

    expect(dtypes).toEqual(["fp32", "fp16", "int8", "uint8", "q8", "q4", "q4f16", "bnb4"]);
  });

  it("should return empty array when no ONNX files exist", async () => {
    setupExistingFiles();
    const dtypes = await get_available_dtypes("test/model", { config: ENCODER_ONLY_CONFIG });

    expect(dtypes).toEqual([]);
  });

  it("should require all session files for seq2seq models", async () => {
    // Only encoder has q4, decoder does not — q4 should NOT be available
    setupExistingFiles(
      "onnx/encoder_model.onnx", // fp32 encoder
      "onnx/decoder_model_merged.onnx", // fp32 decoder
      "onnx/encoder_model_q4.onnx", // q4 encoder (but no q4 decoder)
    );

    const dtypes = await get_available_dtypes("test/model", { config: SEQ2SEQ_CONFIG });

    expect(dtypes).toContain("fp32");
    expect(dtypes).not.toContain("q4");
  });

  it("should list dtype only when all session files exist for seq2seq", async () => {
    // Both encoder and decoder have fp32 and q8
    setupExistingFiles("onnx/encoder_model.onnx", "onnx/decoder_model_merged.onnx", "onnx/encoder_model_quantized.onnx", "onnx/decoder_model_merged_quantized.onnx");

    const dtypes = await get_available_dtypes("test/model", { config: SEQ2SEQ_CONFIG });

    expect(dtypes).toContain("fp32");
    expect(dtypes).toContain("q8");
    expect(dtypes).not.toContain("fp16");
    expect(dtypes).not.toContain("q4");
  });

  it("should handle decoder-only models", async () => {
    setupExistingFiles("onnx/model.onnx", "onnx/model_q4.onnx", "onnx/model_q4f16.onnx");

    const dtypes = await get_available_dtypes("test/model", { config: DECODER_ONLY_CONFIG });

    expect(dtypes).toContain("fp32");
    expect(dtypes).toContain("q4");
    expect(dtypes).toContain("q4f16");
    expect(dtypes).toHaveLength(3);
  });

  it("should fall back to EncoderOnly for unknown architectures", async () => {
    setupExistingFiles("onnx/model.onnx", "onnx/model_fp16.onnx");

    const dtypes = await get_available_dtypes("test/model", { config: UNKNOWN_ARCH_CONFIG });

    expect(dtypes).toContain("fp32");
    expect(dtypes).toContain("fp16");
    expect(dtypes).toHaveLength(2);
  });

  it("should support custom model_file_name", async () => {
    setupExistingFiles("onnx/custom_model.onnx", "onnx/custom_model_q4.onnx");

    const dtypes = await get_available_dtypes("test/model", {
      config: ENCODER_ONLY_CONFIG,
      model_file_name: "custom_model",
    });

    expect(dtypes).toContain("fp32");
    expect(dtypes).toContain("q4");
    expect(dtypes).not.toContain("fp16");
  });

  it("should pass revision and cache_dir to get_file_metadata", async () => {
    setupExistingFiles("onnx/model.onnx");

    await get_available_dtypes("test/model", {
      config: ENCODER_ONLY_CONFIG,
      revision: "v2",
      cache_dir: "/tmp/cache",
    });

    // Verify that metadata calls received the correct options
    for (const call of mockGetFileMetadata.mock.calls) {
      expect(call[0]).toBe("test/model");
      expect(call[2]).toMatchObject({ revision: "v2", cache_dir: "/tmp/cache" });
    }
  });

  it("should only return valid dtype strings", async () => {
    setupExistingFiles("onnx/model.onnx", "onnx/model_fp16.onnx");

    const dtypes = await get_available_dtypes("test/model", { config: ENCODER_ONLY_CONFIG });

    const validDtypes = ["fp32", "fp16", "int8", "uint8", "q8", "q4", "q4f16", "bnb4"];
    for (const dtype of dtypes) {
      expect(validDtypes).toContain(dtype);
    }
  });
});
