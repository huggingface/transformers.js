import { jest } from "@jest/globals";

import { get_file_metadata } from "../../src/utils/model_registry/get_file_metadata.js";

function createBackend(getModelArtifactMetadata) {
  return {
    modelId: "test/metadata-backend",
    load() {},
    getModelArtifactMetadata,
  };
}

describe("get_file_metadata", () => {
  it("deduplicates only in-flight backend metadata requests", async () => {
    let resolveMetadata;
    const pendingMetadata = new Promise((resolve) => {
      resolveMetadata = resolve;
    });
    const getModelArtifactMetadata = jest.fn(() => pendingMetadata);
    const backend = createBackend(getModelArtifactMetadata);

    const first = get_file_metadata(backend, "model.bin");
    const second = get_file_metadata(backend, "model.bin");
    expect(getModelArtifactMetadata).toHaveBeenCalledTimes(1);
    resolveMetadata({ size: 12, fromCache: true });

    await expect(Promise.all([first, second])).resolves.toEqual([
      { exists: true, size: 12, fromCache: true },
      { exists: true, size: 12, fromCache: true },
    ]);
    await get_file_metadata(backend, "model.bin");
    expect(getModelArtifactMetadata).toHaveBeenCalledTimes(2);
  });

  it("returns an asynchronously rejected promise for an already-aborted signal", async () => {
    const reason = new Error("cancelled");
    const controller = new AbortController();
    controller.abort(reason);
    let result;

    expect(() => {
      result = get_file_metadata(createBackend(jest.fn()), "model.bin", { signal: controller.signal });
    }).not.toThrow();
    await expect(result).rejects.toBe(reason);
  });

  it("does not share signaled requests", async () => {
    const getModelArtifactMetadata = jest.fn(async () => ({ size: 1 }));
    const backend = createBackend(getModelArtifactMetadata);

    await Promise.all([get_file_metadata(backend, "model.bin", { signal: new AbortController().signal }), get_file_metadata(backend, "model.bin", { signal: new AbortController().signal })]);
    expect(getModelArtifactMetadata).toHaveBeenCalledTimes(2);
  });
});
