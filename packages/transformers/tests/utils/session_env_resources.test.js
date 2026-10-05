import { jest } from "@jest/globals";

import { CrossOriginStorage } from "../../src/utils/cache/CrossOriginStorageCache.js";
import { loadWasmBinary } from "../../src/backends/utils/cacheWasm.js";
import { env } from "../../src/env.js";
import { getCache } from "../../src/utils/cache.js";

describe("Session-scoped resource loading", () => {
  describe("Cross-origin storage", () => {
    const originalCaches = globalThis.caches;

    beforeEach(() => {
      globalThis.caches = {
        open: async () => ({
          match: async () => undefined,
          put: async () => {},
        }),
      };
    });

    afterAll(() => {
      globalThis.caches = originalCaches;
    });

    it("should not fetch an LFS pointer when remote access is disabled", async () => {
      const fetch = jest.fn();
      const navigatorDescriptor = Object.getOwnPropertyDescriptor(globalThis, "navigator");
      const originalExperimental = env.experimental_useCrossOriginStorage;
      const originalUseBrowserCache = env.useBrowserCache;
      const originalUseFSCache = env.useFSCache;
      const originalUseCustomCache = env.useCustomCache;

      Object.defineProperty(globalThis, "navigator", {
        configurable: true,
        value: { crossOriginStorage: {} },
      });
      env.experimental_useCrossOriginStorage = true;
      env.useBrowserCache = false;
      env.useFSCache = false;
      env.useCustomCache = false;

      try {
        const cache = await getCache(null, {
          env: { fetch, allowRemoteModels: false },
          allowRemote: true,
        });

        await expect(cache.match("https://huggingface.co/org/model/resolve/main/model.onnx")).resolves.toBeUndefined();
        expect(fetch).not.toHaveBeenCalled();
      } finally {
        env.experimental_useCrossOriginStorage = originalExperimental;
        env.useBrowserCache = originalUseBrowserCache;
        env.useFSCache = originalUseFSCache;
        env.useCustomCache = originalUseCustomCache;
        if (navigatorDescriptor) {
          Object.defineProperty(globalThis, "navigator", navigatorDescriptor);
        } else {
          delete globalThis.navigator;
        }
      }
    });

    it("should use the scoped fetch for LFS pointer lookups", async () => {
      const fetch = jest.fn(async () => new Response("version https://git-lfs.github.com/spec/v1\noid sha256:abc123\n"));
      const cache = new CrossOriginStorage({ fetch });

      await cache.match("https://huggingface.co/org/model/resolve/main/model.onnx");

      expect(fetch).toHaveBeenCalledWith("https://huggingface.co/org/model/raw/main/model.onnx", expect.any(Object));
    });
  });

  it("should use the scoped fetch when loading the ONNX WASM binary", async () => {
    const bytes = new Uint8Array([1, 2, 3]);
    const fetch = jest.fn(async () => new Response(bytes));
    const originalUseFSCache = env.useFSCache;
    const originalUseBrowserCache = env.useBrowserCache;
    const originalUseCustomCache = env.useCustomCache;
    env.useFSCache = false;
    env.useBrowserCache = false;
    env.useCustomCache = false;

    try {
      const result = await loadWasmBinary("https://cdn.example/ort.wasm", fetch);

      expect(fetch).toHaveBeenCalledWith("https://cdn.example/ort.wasm");
      expect(new Uint8Array(result)).toEqual(bytes);
    } finally {
      env.useFSCache = originalUseFSCache;
      env.useBrowserCache = originalUseBrowserCache;
      env.useCustomCache = originalUseCustomCache;
    }
  });
});
