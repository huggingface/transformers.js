import { jest } from "@jest/globals";

import { CrossOriginStorage } from "../../src/utils/cache/CrossOriginStorageCache.js";
import { env } from "../../src/env.js";
import { FileCache } from "../../src/utils/cache/FileCache.js";
import { AssetLoadingContext } from "../../src/utils/hub/AssetLoadingContext.js";
import { Pipeline, getSessionEnv, prepareImages, prepareAudios } from "../../src/pipelines/_base.js";

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

    it("should pass the session fetch to cross-origin storage", async () => {
      const fetch = jest.fn(async () => new Response("version https://git-lfs.github.com/spec/v1\noid sha256:abc123\n"));
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
        const cache = await new AssetLoadingContext({ env: { fetch } }).getCache();

        await expect(cache.match("https://huggingface.co/org/model/resolve/main/model.onnx")).resolves.toBeUndefined();
        expect(fetch).toHaveBeenCalledWith("https://huggingface.co/org/model/raw/main/model.onnx");
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

      expect(fetch).toHaveBeenCalledWith("https://huggingface.co/org/model/raw/main/model.onnx");
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
      const cache = await new AssetLoadingContext({ env: { cacheDir: "/session-cache/" } }).getCache();

      expect(cache).toBeInstanceOf(FileCache);
      expect(/** @type {FileCache} */ (cache).path).toBe("/session-cache/");
    });

    it("should let cache_dir override the session cacheDir", async () => {
      const cache = await new AssetLoadingContext({ cache_dir: "/legacy-cache/", env: { cacheDir: "/session-cache/" } }).getCache();

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
      expect(new AssetLoadingContext().getHeaders("https://late-hub.example/org/model").get("Authorization")).toBe("Bearer late-token");
    });

    it("should restore the environment-variable fallback when assigned undefined", () => {
      env.hfToken = "assigned-token";
      env.remoteHost = "https://assigned-hub.example/";
      env.hfToken = undefined;
      env.remoteHost = undefined;
      process.env.HF_TOKEN = "late-token";
      process.env.HF_ENDPOINT = "https://late-hub.example/";

      expect(env.hfToken).toBe("late-token");
      expect(env.remoteHost).toBe("https://late-hub.example/");
    });

    it("should keep reading HF_TOKEN after an env snapshot without a token is restored", () => {
      const snapshot = { ...env };
      env.hfToken = "assigned-token";
      Object.assign(env, snapshot);
      process.env.HF_TOKEN = "late-token";

      expect(env.hfToken).toBe("late-token");
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

  describe("Pipeline inputs", () => {
    const sessionEnv = {
      remoteHost: "https://private-hub.example/",
      hfToken: "scoped-token",
    };

    it("should download image URL inputs with the session fetch and token", async () => {
      const fetch = jest.fn(async () => ({ status: 404, statusText: "Not Found" }));

      await expect(prepareImages(new Pipeline({ task: "test", model: null, sessionEnv: { ...sessionEnv, fetch } }), "https://private-hub.example/image.png")).rejects.toThrow('Unable to read image from "https://private-hub.example/image.png"');

      expect(fetch).toHaveBeenCalledWith("https://private-hub.example/image.png", expect.any(Object));
      expect(fetch.mock.calls[0][1].headers.get("Authorization")).toBe("Bearer scoped-token");
    });

    it("should download audio URL inputs with the session fetch and token", async () => {
      const originalAudioContext = globalThis.AudioContext;
      globalThis.AudioContext = class {
        constructor({ sampleRate }) {
          this.sampleRate = sampleRate;
        }
        async decodeAudioData() {
          return { numberOfChannels: 1, length: 2, getChannelData: () => new Float32Array([0.25, 0.5]) };
        }
      };
      const fetch = jest.fn(async () => new Response(new ArrayBuffer(8)));

      try {
        const [audio] = await prepareAudios(new Pipeline({ task: "test", model: null, sessionEnv: { ...sessionEnv, fetch } }), "https://private-hub.example/audio.wav", 16000);

        expect(audio).toEqual(new Float32Array([0.25, 0.5]));
        expect(fetch).toHaveBeenCalledWith("https://private-hub.example/audio.wav", expect.any(Object));
        expect(fetch.mock.calls[0][1].headers.get("Authorization")).toBe("Bearer scoped-token");
      } finally {
        globalThis.AudioContext = originalAudioContext;
      }
    });
  });
});
