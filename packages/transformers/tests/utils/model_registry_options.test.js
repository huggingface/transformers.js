import { jest } from "@jest/globals";

// Mock get_file_metadata before importing the modules under test. Only this one: the listing path itself —
// get_files, get_model_files and the config fetch under it — is what these tests exercise, so (unlike
// model_registry.test.js) get_model_files.js is left real.
const mockGetFileMetadata = jest.fn();
jest.unstable_mockModule("../../src/utils/model_registry/get_file_metadata.js", () => ({
  get_file_metadata: mockGetFileMetadata,
}));

// Import registry to populate MODEL_TYPE_MAPPING (side-effect import)
await import("../../src/models/registry.js");

// Dynamic import after mock setup (required for ESM)
const { get_files } = await import("../../src/utils/model_registry/get_files.js");
const { get_model_files } = await import("../../src/utils/model_registry/get_model_files.js");
const { ModelRegistry } = await import("../../src/utils/model_registry/ModelRegistry.js");
const { AutoConfig } = await import("../../src/configs.js");

// A minimal config that mimics a BERT-like encoder-only model
const ENCODER_ONLY_CONFIG = {
  architectures: ["BertModel"],
  model_type: "bert",
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

describe("get_files", () => {
  beforeEach(() => {
    mockGetFileMetadata.mockReset();
  });

  it("should pass revision and cache_dir to every file lookup", async () => {
    // Including the tokenizer and processor lookups: listing those from `main` while the weights come from a
    // pinned revision leaves stray files in the cache, and can list files the pinned revision does not have.
    setupExistingFiles("onnx/model.onnx", "tokenizer_config.json", "tokenizer.json", "preprocessor_config.json");

    const files = await get_files("test/model", {
      config: ENCODER_ONLY_CONFIG,
      revision: "v2",
      cache_dir: "/tmp/cache",
    });

    expect(files).toContain("tokenizer_config.json");
    expect(mockGetFileMetadata.mock.calls.length).toBeGreaterThan(0);
    for (const call of mockGetFileMetadata.mock.calls) {
      expect(call[0]).toBe("test/model");
      expect(call[2]).toMatchObject({ revision: "v2", cache_dir: "/tmp/cache" });
    }
  });

  it("should still work when no options are passed", async () => {
    setupExistingFiles("onnx/model.onnx", "tokenizer_config.json");

    const files = await get_files("test/model", { config: ENCODER_ONLY_CONFIG });

    expect(files).toContain("onnx/model.onnx");
    for (const call of mockGetFileMetadata.mock.calls) {
      expect(call[2]).toBeDefined();
    }
  });
});

describe("ModelRegistry facades", () => {
  // The static ModelRegistry methods are the exported API — the bare helpers are not — so the options
  // have to survive this extra hop too, or public callers stay pinned to `main`.
  beforeEach(() => {
    mockGetFileMetadata.mockReset();
  });

  it("should pass revision and cache_dir through get_tokenizer_files", async () => {
    setupExistingFiles("tokenizer_config.json");

    const files = await ModelRegistry.get_tokenizer_files("test/model", { revision: "v2", cache_dir: "/tmp/cache" });

    expect(files).toEqual(["tokenizer.json", "tokenizer_config.json"]);
    expect(mockGetFileMetadata).toHaveBeenCalledWith("test/model", "tokenizer_config.json", expect.objectContaining({ revision: "v2", cache_dir: "/tmp/cache" }));
  });

  it("should pass revision and cache_dir through get_processor_files", async () => {
    setupExistingFiles("preprocessor_config.json");

    const files = await ModelRegistry.get_processor_files("test/model", { revision: "v2", cache_dir: "/tmp/cache" });

    expect(files).toEqual(["preprocessor_config.json"]);
    expect(mockGetFileMetadata).toHaveBeenCalledWith("test/model", "preprocessor_config.json", expect.objectContaining({ revision: "v2", cache_dir: "/tmp/cache" }));
  });

  it("should still work when no options are passed", async () => {
    setupExistingFiles("tokenizer_config.json");

    const files = await ModelRegistry.get_tokenizer_files("test/model");

    expect(files).toEqual(["tokenizer.json", "tokenizer_config.json"]);
    expect(mockGetFileMetadata).toHaveBeenCalledWith("test/model", "tokenizer_config.json", expect.anything());
  });
});

describe("get_model_files", () => {
  /** @type {import("@jest/globals").jest.SpiedFunction<typeof AutoConfig.from_pretrained>} */
  let configSpy;

  beforeEach(() => {
    mockGetFileMetadata.mockReset();
    // Spied rather than fetched: the assertion is about the options it receives, not its result.
    configSpy = jest.spyOn(AutoConfig, "from_pretrained").mockResolvedValue(ENCODER_ONLY_CONFIG);
  });

  afterEach(() => {
    configSpy.mockRestore();
  });

  it("should read config.json from the requested revision", async () => {
    // The config is fetched separately from the file listing; leaving it on `main` both reads the wrong
    // config for a pinned load and leaves a stray `main` entry in the browser cache.
    setupExistingFiles("onnx/model.onnx");

    await get_model_files("test/model", { revision: "v2", cache_dir: "/tmp/cache" });

    expect(configSpy).toHaveBeenCalledWith("test/model", expect.objectContaining({ revision: "v2", cache_dir: "/tmp/cache" }));
  });
});
