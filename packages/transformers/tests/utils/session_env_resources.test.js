import { jest } from "@jest/globals";

import { CrossOriginStorage } from "../../src/utils/cache/CrossOriginStorageCache.js";
import { env } from "../../src/env.js";
import { getCache } from "../../src/utils/cache.js";
import { FileCache } from "../../src/utils/cache/FileCache.js";
import { getFetchHeaders, getFetchOptions } from "../../src/utils/hub.js";
import { Pipeline, getSessionEnv } from "../../src/pipelines/_base.js";

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

  describe("Cache directory", () => {
    const originalUseBrowserCache = env.useBrowserCache;
    const originalUseCustomCache = env.useCustomCache;
    const originalUseFSCache = env.useFSCache;
    const originalExperimental = env.experimental_useCrossOriginStorage;

    beforeEach(() => {
      env.useBrowserCache = false;
      env.useCustomCache = false;
      env.experimental_useCrossOriginStorage = false;
      env.useFSCache = true;
    });

    afterAll(() => {
      env.useBrowserCache = originalUseBrowserCache;
      env.useCustomCache = originalUseCustomCache;
      env.useFSCache = originalUseFSCache;
      env.experimental_useCrossOriginStorage = originalExperimental;
    });

    it("should use the session-scoped cacheDir for the file system cache", async () => {
      const cache = await getCache(null, { env: { cacheDir: "/session-cache/" } });

      expect(cache).toBeInstanceOf(FileCache);
      expect(/** @type {FileCache} */ (cache).path).toBe("/session-cache/");
    });

    it("should let cache_dir override the session cacheDir", async () => {
      const cache = await getCache("/legacy-cache/", { env: { cacheDir: "/session-cache/" } });

      expect(/** @type {FileCache} */ (cache).path).toBe("/legacy-cache/");
    });
  });

  describe("Environment variables", () => {
    const names = ["HF_TOKEN", "HF_ACCESS_TOKEN", "HF_ENDPOINT"];
    let saved;

    beforeEach(() => {
      saved = Object.fromEntries(names.map((name) => [name, process.env[name]]));
      for (const name of names) delete process.env[name];
    });

    afterEach(() => {
      for (const name of names) {
        if (saved[name] === undefined) delete process.env[name];
        else process.env[name] = saved[name];
      }
    });

    it("should read HF_TOKEN and HF_ENDPOINT when accessed, not at import", () => {
      // e.g., `dotenv.config()` called after importing the library
      process.env.HF_TOKEN = "late-token";
      process.env.HF_ENDPOINT = "https://late-hub.example/";

      expect(env.hfToken).toBe("late-token");
      expect(env.remoteHost).toBe("https://late-hub.example/");
      expect(getFetchHeaders("https://late-hub.example/org/model", getFetchOptions(env)).get("Authorization")).toBe("Bearer late-token");
    });

    it("should fall back to HF_ACCESS_TOKEN and the official Hub", () => {
      process.env.HF_ACCESS_TOKEN = "legacy-token";

      expect(env.hfToken).toBe("legacy-token");
      expect(env.remoteHost).toBe("https://huggingface.co/");
    });
  });

  describe("Pipeline env", () => {
    it("should keep session overrides off the public pipeline instance", () => {
      const sessionEnv = { hfToken: "scoped-token" };
      const pipe = new Pipeline({ task: "test", model: null, sessionEnv });

      expect(getSessionEnv(pipe)).toBe(sessionEnv);
      expect(Object.values(pipe)).not.toContain(sessionEnv);
      expect(JSON.stringify(Object.keys(pipe))).not.toMatch(/env/i);
    });
  });
});
