import { jest } from "@jest/globals";

// Mock getModelFile before importing the module under test, so we can
// simulate a failing external-data download without network access.
const mockGetModelFile = jest.fn();
jest.unstable_mockModule("../../src/utils/hub.js", () => ({
  getModelFile: mockGetModelFile,
  MAX_EXTERNAL_DATA_CHUNKS: 10,
}));

// Dynamic import after mock setup (required for ESM)
const { getModelDataFiles } = await import("../../src/utils/model-loader.js");

describe("getModelDataFiles", () => {
  beforeEach(() => {
    mockGetModelFile.mockReset();
  });

  it("rejects when an external data chunk download fails", async () => {
    // Regression for https://github.com/huggingface/transformers.js/issues/1801:
    // the removed `new Promise(async ...)` wrapper swallowed the rejection,
    // leaving Promise.all pending forever (from_pretrained never settles).
    mockGetModelFile.mockRejectedValue(new Error("simulated network error"));

    await expect(getModelDataFiles("onnx-community/test-model", "model", "_q4", { subfolder: "onnx" }, 1)).rejects.toThrow("simulated network error");
  }, 2000);

  it("resolves chunk data with path info on success", async () => {
    const data = new Uint8Array([1, 2, 3]);
    mockGetModelFile.mockResolvedValue(data);

    const result = await getModelDataFiles("onnx-community/test-model", "model", "_q4", { subfolder: "onnx" }, 2);
    expect(result).toEqual([
      { path: "model_q4.onnx_data", data },
      { path: "model_q4.onnx_data_1", data },
    ]);
    // fatal=true so a missing chunk throws instead of resolving silently; the last two are
    // return_path and its complement as_blob, which follow the environment
    expect(mockGetModelFile).toHaveBeenCalledWith("onnx-community/test-model", "onnx/model_q4.onnx_data", true, { subfolder: "onnx" }, expect.any(Boolean), expect.any(Boolean));
  });

  it("resolves to the file path when getModelFile returns a string", async () => {
    mockGetModelFile.mockResolvedValue("/cache/model_q4.onnx_data");

    const result = await getModelDataFiles("onnx-community/test-model", "model", "_q4", {}, 1);
    expect(result).toEqual(["model_q4.onnx_data"]);
  });
});
