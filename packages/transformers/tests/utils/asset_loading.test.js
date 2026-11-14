import { jest } from "@jest/globals";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";

const files = ["tokenizer_config.json", "onnx/model.onnx"];
jest.unstable_mockModule("../../src/utils/model_registry/get_files.js", () => ({
  get_files: jest.fn(async () => files),
}));
jest.unstable_mockModule("../../src/utils/model_registry/get_pipeline_files.js", () => ({
  get_pipeline_files: jest.fn(async () => files),
}));

const { env } = await import("../../src/env.js");
const { getModelJSON, getModelFile } = await import("../../src/utils/hub.js");
const { get_file_metadata, get_file_metadata_with_context } = await import("../../src/utils/model_registry/get_file_metadata.js");
const { get_tokenizer_files } = await import("../../src/utils/model_registry/get_tokenizer_files.js");
const { is_cached_files, is_pipeline_cached_files } = await import("../../src/utils/model_registry/is_cached.js");
const { clear_cache, clear_pipeline_cache } = await import("../../src/utils/model_registry/clear_cache.js");
const { FileCache } = await import("../../src/utils/cache/FileCache.js");
const { AssetLoadingContext } = await import("../../src/utils/hub/AssetLoadingContext.js");

const MODEL = "test/asset-loading";
const REMOTE = `https://huggingface.co/${MODEL}/resolve/main/`;
const VERSIONED = `${REMOTE}tokenizer_config.json?__transformersjs_cache_version=1`;
const ORIGINAL_ENV = { ...env };

function jsonResponse(value) {
  const text = JSON.stringify(value);
  return new Response(text, { headers: { "content-type": "application/json", "content-length": String(text.length) } });
}

function memoryCache() {
  const entries = new Map();
  return {
    entries,
    match: jest.fn(async (key) => entries.get(key)?.clone()),
    put: jest.fn(async (key, response) => entries.set(key, response.clone())),
    delete: jest.fn(async (key) => entries.delete(key)),
  };
}

describe("Internal asset loading", () => {
  let cache;
  beforeEach(() => {
    cache = memoryCache();
    Object.assign(env, {
      allowLocalModels: false,
      allowRemoteModels: true,
      remoteHost: "https://huggingface.co/",
      remotePathTemplate: "{model}/resolve/{revision}/",
      localModelPath: "/models/",
      useCustomCache: true,
      customCache: cache,
      useFSCache: false,
      useBrowserCache: false,
      experimental_useCrossOriginStorage: false,
      fetch: jest.fn(async () => jsonResponse({ updated: true })),
    });
  });
  afterEach(() => Object.assign(env, ORIGINAL_ENV));

  it("shallow-copies the environment while preserving transport and cache identities", () => {
    const context = new AssetLoadingContext();
    expect(context.settings).not.toBe(env);
    expect(context.settings).toEqual(env);
    expect(context.settings.fetch).toBe(env.fetch);
    expect(context.settings.customCache).toBe(cache);
    expect(context.settings.backends).toBe(env.backends);

    env.remoteHost = "https://other.example/";
    env.fetch = jest.fn();
    env.customCache = memoryCache();
    expect(context.settings.remoteHost).toBe("https://huggingface.co/");
    expect(context.settings.fetch).not.toBe(env.fetch);
    expect(context.settings.customCache).toBe(cache);
  });

  it("ignores backend, logging, and WASM settings in loading keys", () => {
    const key = new AssetLoadingContext().key(MODEL, "config.json");
    const onnx = {};
    onnx.self = onnx;
    env.backends = { onnx };
    env.logLevel = 50;
    env.useWasmCache = !env.useWasmCache;
    expect(new AssetLoadingContext().key(MODEL, "config.json")).toBe(key);
  });

  it.each([
    ["allowLocalModels", true],
    ["allowRemoteModels", false],
    ["localModelPath", "/other-models/"],
    ["remoteHost", "https://other.example/"],
    ["remotePathTemplate", "{model}/{revision}/"],
    ["useFS", !ORIGINAL_ENV.useFS],
    ["useCustomCache", false],
    ["experimental_useCrossOriginStorage", true],
    ["useBrowserCache", true],
    ["cacheKey", "other-cache"],
    ["useFSCache", true],
    ["cacheDir", "/other-cache/"],
    ["fetch", async () => jsonResponse({ other: true })],
    ["customCache", memoryCache()],
  ])("isolates loading keys when %s changes", (setting, value) => {
    const key = new AssetLoadingContext().key(MODEL, "config.json");
    env[setting] = value;
    expect(new AssetLoadingContext().key(MODEL, "config.json")).not.toBe(key);
  });

  it("isolates loading keys for settings it does not know about", () => {
    const key = new AssetLoadingContext().key(MODEL, "config.json");
    env.someFutureLoadingSetting = true;
    try {
      expect(new AssetLoadingContext().key(MODEL, "config.json")).not.toBe(key);
    } finally {
      delete env.someFutureLoadingSetting;
    }
  });

  it("refreshes only versioned metadata, rejects old local aliases, and remains cache-first afterwards", async () => {
    const oldRemote = `${REMOTE}tokenizer_config.json`;
    const oldLocal = `/models/${MODEL}/tokenizer_config.json`;
    const weights = `${REMOTE}onnx/model.onnx`;
    cache.entries.set(oldRemote, jsonResponse({ stale: true }));
    cache.entries.set(oldLocal, jsonResponse({ stale: true }));
    cache.entries.set(weights, new Response(new Uint8Array([1, 2, 3])));

    expect(await getModelJSON(MODEL, "tokenizer_config.json")).toEqual({ updated: true });
    expect(env.fetch).toHaveBeenCalledWith(oldRemote, expect.objectContaining({ cache: "reload" }));
    expect(cache.entries.has(VERSIONED)).toBe(true);
    expect(cache.entries.has(oldRemote)).toBe(false);
    expect(cache.entries.has(oldLocal)).toBe(false);
    expect(cache.entries.has(weights)).toBe(true);

    expect(await getModelJSON(MODEL, "tokenizer_config.json")).toEqual({ updated: true });
    expect(await getModelFile(MODEL, "onnx/model.onnx")).toEqual(new Uint8Array([1, 2, 3]));
    expect(env.fetch).toHaveBeenCalledTimes(1);
  });

  it("uses normal HTTP caching for a versioned asset on a fresh cache", async () => {
    expect(await getModelJSON(MODEL, "tokenizer_config.json")).toEqual({ updated: true });
    expect(env.fetch).toHaveBeenCalledTimes(1);
    const [url, init] = env.fetch.mock.calls[0];
    expect(url).toBe(`${REMOTE}tokenizer_config.json`);
    expect(init.cache).toBeUndefined();
  });

  it("uses normal HTTP caching when no cache backend is enabled", async () => {
    env.useCustomCache = false;
    expect(await getModelJSON(MODEL, "tokenizer_config.json")).toEqual({ updated: true });
    expect(env.fetch.mock.calls[0][1].cache).toBeUndefined();
  });

  it("uses the same versioned identity for metadata, registry checks, and removal", async () => {
    cache.entries.set(`${REMOTE}tokenizer_config.json`, jsonResponse({ stale: true }));
    cache.entries.set(`${REMOTE}onnx/model.onnx`, new Response("weights"));
    expect((await is_cached_files(MODEL)).allCached).toBe(false);
    expect((await is_pipeline_cached_files("text-generation", MODEL)).allCached).toBe(false);

    await getModelJSON(MODEL, "tokenizer_config.json");
    expect(await get_file_metadata(MODEL, "tokenizer_config.json")).toEqual({
      exists: true,
      size: JSON.stringify({ updated: true }).length,
      contentType: "application/json",
      fromCache: true,
    });
    expect((await is_cached_files(MODEL)).allCached).toBe(true);
    expect((await is_pipeline_cached_files("text-generation", MODEL)).allCached).toBe(true);
    expect((await clear_cache(MODEL)).filesDeleted).toBe(2);
    expect(cache.entries.size).toBe(0);
  });

  describe.each([
    ["clear_cache", (options) => clear_cache(MODEL, options)],
    ["clear_pipeline_cache", (options) => clear_pipeline_cache("text-generation", MODEL, options)],
  ])("%s cache aliases", (_name, clear) => {
    it.each([`${REMOTE}tokenizer_config.json`, `/models/${MODEL}/tokenizer_config.json`])("clears a stale-only entry at %s before any replacement is loaded", async (key) => {
      cache.entries.set(key, jsonResponse({ stale: true }));
      expect((await is_cached_files(MODEL)).files[0].cached).toBe(false);

      expect(await clear()).toEqual({
        filesDeleted: 1,
        filesCached: 1,
        files: [
          { file: "tokenizer_config.json", deleted: true, wasCached: true },
          { file: "onnx/model.onnx", deleted: false, wasCached: false },
        ],
      });
      expect(cache.entries.size).toBe(0);
      expect(cache.put).not.toHaveBeenCalled();
      expect(env.fetch).not.toHaveBeenCalled();
    });

    it("removes every current and obsolete alias but counts each logical file once", async () => {
      const resource = new AssetLoadingContext().resolve(MODEL, "tokenizer_config.json", cache);
      const aliases = [...resource.cacheKeys, ...resource.obsoleteCacheKeys];
      for (const key of aliases) cache.entries.set(key, jsonResponse({ cached: true }));
      const unrelated = `${REMOTE}tokenizer.json`;
      cache.entries.set(unrelated, jsonResponse({ unrelated: true }));

      const result = await clear();
      expect(result.filesCached).toBe(1);
      expect(result.filesDeleted).toBe(1);
      expect(result.files[0]).toEqual({ file: "tokenizer_config.json", deleted: true, wasCached: true });
      expect([...cache.entries.keys()]).toEqual([unrelated]);
      expect(cache.delete).toHaveBeenCalledTimes(new Set(aliases).size);
      expect(env.fetch).not.toHaveBeenCalled();
    });

    it("does not report a file as deleted if one alias cannot be removed", async () => {
      const oldKey = `${REMOTE}tokenizer_config.json`;
      cache.entries.set(oldKey, jsonResponse({ stale: true }));
      cache.entries.set(VERSIONED, jsonResponse({ updated: true }));
      cache.delete.mockImplementation(async (key) => key !== oldKey && cache.entries.delete(key));

      const result = await clear();
      expect(result.filesCached).toBe(1);
      expect(result.filesDeleted).toBe(0);
      expect(result.files[0]).toEqual({ file: "tokenizer_config.json", deleted: false, wasCached: true });
      expect([...cache.entries.keys()]).toEqual([oldKey]);
    });

    it("removes stale-only filesystem entries for the requested revision", async () => {
      const directory = await fs.mkdtemp(path.join(os.tmpdir(), "transformers-clear-assets-"));
      try {
        const fileCache = new FileCache(directory);
        env.customCache = fileCache;
        const oldKey = `${MODEL}/v1/tokenizer_config.json`;
        const otherRevision = `${MODEL}/tokenizer_config.json`;
        await fileCache.put(oldKey, jsonResponse({ stale: true }));
        await fileCache.put(otherRevision, jsonResponse({ unrelated: true }));

        const result = await clear({ revision: "v1" });
        expect(result.filesCached).toBe(1);
        expect(result.filesDeleted).toBe(1);
        expect(await fileCache.match(oldKey)).toBeUndefined();
        expect(await fileCache.match(otherRevision)).toBeDefined();
        expect(env.fetch).not.toHaveBeenCalled();
      } finally {
        await fs.rm(directory, { recursive: true, force: true });
      }
    });
  });

  it("matches resolved cache keys without accepting stale aliases", async () => {
    const context = new AssetLoadingContext();
    const resource = context.resolve(MODEL, "tokenizer_config.json", cache);
    cache.entries.set(resource.localPath, jsonResponse({ stale: true }));
    expect(await context.match(cache, resource)).toBeUndefined();
    cache.entries.set(VERSIONED, jsonResponse({ updated: true }));
    expect(await (await context.match(cache, resource)).response.json()).toEqual({ updated: true });
  });

  it("versions local HTTP cache entries without changing the local request path", async () => {
    env.allowLocalModels = true;
    env.useFS = false;
    const localPath = `/models/${MODEL}/tokenizer_config.json`;
    cache.entries.set(localPath, jsonResponse({ stale: true }));
    expect(await getModelJSON(MODEL, "tokenizer_config.json")).toEqual({ updated: true });
    expect(env.fetch).toHaveBeenCalledWith(localPath, expect.objectContaining({ cache: "reload" }));
    expect(cache.entries.has(`${localPath}?__transformersjs_cache_version=1`)).toBe(true);
    expect(cache.entries.has(localPath)).toBe(false);
    expect(await getModelJSON(MODEL, "tokenizer_config.json")).toEqual({ updated: true });
    expect(env.fetch).toHaveBeenCalledTimes(1);
  });

  it("works with a custom cache that treats keys only as request URLs", async () => {
    const entries = new Map();
    const requestURL = (key) => new Request(new URL(key, "https://app.example/")).url;
    env.customCache = {
      match: async (key) => entries.get(requestURL(key))?.clone(),
      put: async (key, response) => entries.set(requestURL(key), response.clone()),
    };

    expect(await getModelJSON(MODEL, "tokenizer_config.json")).toEqual({ updated: true });
    expect(await getModelJSON(MODEL, "tokenizer_config.json")).toEqual({ updated: true });
    expect([...entries.keys()]).toEqual([VERSIONED]);
    expect(env.fetch).toHaveBeenCalledTimes(1);
    expect(env.fetch.mock.calls[0][0]).toBe(`${REMOTE}tokenizer_config.json`);
  });

  it("does not require custom caches to support deletion", async () => {
    delete cache.delete;
    cache.entries.set(`${REMOTE}tokenizer_config.json`, jsonResponse({ stale: true }));
    expect(await getModelJSON(MODEL, "tokenizer_config.json")).toEqual({ updated: true });
    expect(cache.entries.has(VERSIONED)).toBe(true);
    expect(cache.entries.has(`${REMOTE}tokenizer_config.json`)).toBe(true);
  });

  it("keeps old entries when storing a replacement fails", async () => {
    cache.entries.set(`${REMOTE}tokenizer_config.json`, jsonResponse({ stale: true }));
    cache.put.mockRejectedValue(new Error("cache full"));
    env.logLevel = 50;
    expect(await getModelJSON(MODEL, "tokenizer_config.json")).toEqual({ updated: true });
    expect(cache.delete).not.toHaveBeenCalled();
    expect(cache.entries.has(`${REMOTE}tokenizer_config.json`)).toBe(true);
  });

  it("ignores cleanup failures after successfully storing the replacement", async () => {
    cache.delete.mockRejectedValue(new Error("read-only cache"));
    expect(await getModelJSON(MODEL, "tokenizer_config.json")).toEqual({ updated: true });
    expect(cache.entries.has(VERSIONED)).toBe(true);
  });

  it.each([
    ["env.allowRemoteModels=false", () => (env.allowRemoteModels = false), {}],
    ["local_files_only", () => {}, { local_files_only: true }],
  ])("falls back to the outdated copy when %s", async (_name, configure, options) => {
    env.allowLocalModels = true;
    configure();
    cache.entries.set(`${REMOTE}tokenizer_config.json`, jsonResponse({ stale: true }));
    expect(await getModelJSON(MODEL, "tokenizer_config.json", true, options)).toEqual({ stale: true });
    expect(env.fetch).not.toHaveBeenCalled();
    expect(cache.put).not.toHaveBeenCalled();
    expect(cache.delete).not.toHaveBeenCalled();
  });

  it.each([
    ["remote disabled", { allowRemoteModels: false }, {}],
    ["local files only", {}, { local_files_only: true }],
  ])("discovers the old tokenizer cache with %s", async (_name, settings, options) => {
    env.allowLocalModels = true;
    Object.assign(env, settings);
    cache.entries.set(`${REMOTE}tokenizer_config.json`, jsonResponse({ stale: true }));
    expect(await get_tokenizer_files(MODEL, options)).toEqual(["tokenizer.json", "tokenizer_config.json"]);
    expect(await getModelJSON(MODEL, "tokenizer_config.json", true, options)).toEqual({ stale: true });
    expect(env.fetch).not.toHaveBeenCalled();
  });

  it("discovers an old tokenizer cache when the metadata request fails offline", async () => {
    env.logLevel = 50;
    cache.entries.set(`${REMOTE}tokenizer_config.json`, jsonResponse({ stale: true }));
    env.fetch = jest.fn(async () => {
      throw new TypeError("Failed to fetch");
    });
    expect(await get_tokenizer_files(MODEL)).toEqual(["tokenizer.json", "tokenizer_config.json"]);
    expect(await getModelJSON(MODEL, "tokenizer_config.json")).toEqual({ stale: true });
  });

  it.each([401, 404, 429, 503])("uses an old tokenizer cache after HTTP %i and refreshes it once available", async (status) => {
    env.logLevel = 50;
    cache.entries.set(`${REMOTE}tokenizer_config.json`, jsonResponse({ stale: true }));
    const online = env.fetch;
    env.fetch = jest.fn(async () => new Response(null, { status }));
    expect(await getModelJSON(MODEL, "tokenizer_config.json")).toEqual({ stale: true });
    expect(await get_tokenizer_files(MODEL)).toEqual(["tokenizer.json", "tokenizer_config.json"]);
    expect(cache.put).not.toHaveBeenCalled();
    expect(cache.delete).not.toHaveBeenCalled();
    env.fetch = online;
    expect(await get_tokenizer_files(MODEL)).toEqual(["tokenizer.json", "tokenizer_config.json"]);
    expect(await getModelJSON(MODEL, "tokenizer_config.json")).toEqual({ updated: true });
    expect(cache.entries.has(VERSIONED)).toBe(true);
    expect(cache.entries.has(`${REMOTE}tokenizer_config.json`)).toBe(false);
  });

  it.each([401, 404, 429, 503])("preserves HTTP %i behavior without an old cache", async (status) => {
    env.fetch.mockImplementation(async () => new Response(null, { status }));
    await expect(getModelJSON(MODEL, "tokenizer_config.json")).rejects.toThrow();
    if (status === 404) expect(await get_tokenizer_files(MODEL)).toEqual([]);
    else await expect(get_tokenizer_files(MODEL)).rejects.toThrow();
  });

  it("propagates tokenizer metadata network errors without an old cache", async () => {
    env.fetch = jest.fn(async () => {
      throw new TypeError("Failed to fetch");
    });
    await expect(get_tokenizer_files(MODEL)).rejects.toThrow("Failed to fetch");
  });

  describe.each([
    ["path offline", "/models/foo", null],
    ["path HTTP 404", "/models/foo", 404],
    ["Hub id local offline", MODEL, null],
    ["Hub id local HTTP 404", MODEL, 404],
    ["Hub id local HTTP 403", MODEL, 403],
    ["Hub id local HTTP 503", MODEL, 503],
  ])("locally served tokenizer: %s", (_name, modelId, status) => {
    const configure = () => {
      env.useFS = false;
      env.allowLocalModels = true;
      env.logLevel = 50;
      env.fetch = jest.fn(async () => {
        if (status === null) throw new TypeError("Failed to fetch");
        return new Response(JSON.stringify({ error: status }), { status });
      });
      return new AssetLoadingContext().resolve(modelId, "tokenizer_config.json", cache);
    };

    it("uses the old local cache and replaces it after a successful local read", async () => {
      const resource = configure();
      cache.entries.set(resource.localPath, jsonResponse({ stale: true }));
      expect(await getModelJSON(modelId, "tokenizer_config.json")).toEqual({ stale: true });
      expect(await get_tokenizer_files(modelId)).toEqual(["tokenizer.json", "tokenizer_config.json"]);
      expect(await get_file_metadata(modelId, "tokenizer_config.json")).toMatchObject({ exists: true, fromCache: true });
      expect(cache.put).not.toHaveBeenCalled();
      expect(cache.delete).not.toHaveBeenCalled();
      const canTryHub = modelId === MODEL && (status === null || status === 404);
      expect(env.fetch.mock.calls.some(([url]) => url === resource.remoteURL)).toBe(canTryHub);
      env.fetch = jest.fn(async () => jsonResponse({ updated: true }));
      expect(await getModelJSON(modelId, "tokenizer_config.json")).toEqual({ updated: true });
      expect(cache.entries.has(resource.localCacheKey)).toBe(true);
      expect(cache.entries.has(resource.localPath)).toBe(false);
    });

    it("preserves the existing behavior without an old cache", async () => {
      configure();
      if (modelId.startsWith("/") || status === 404) {
        await expect(getModelJSON(modelId, "tokenizer_config.json")).rejects.toThrow(modelId.startsWith("/") ? "invalid model ID" : "Could not locate file");
        expect(await get_tokenizer_files(modelId)).toEqual([]);
      } else if (status === null) {
        await expect(getModelJSON(modelId, "tokenizer_config.json")).rejects.toThrow("Failed to fetch");
        await expect(get_tokenizer_files(modelId)).rejects.toThrow("Failed to fetch");
      } else {
        // Local error responses without a stale copy are intentionally unchanged.
        expect(await getModelJSON(modelId, "tokenizer_config.json")).toEqual({ error: status });
        expect(await get_tokenizer_files(modelId)).toEqual(["tokenizer.json", "tokenizer_config.json"]);
      }
    });
  });

  it("still refreshes an old Hub cache after an ordinary filesystem miss", async () => {
    env.allowLocalModels = true;
    env.useFS = true;
    cache.entries.set(`${REMOTE}tokenizer_config.json`, jsonResponse({ stale: true }));
    expect(await getModelJSON(MODEL, "tokenizer_config.json")).toEqual({ updated: true });
    expect(env.fetch).toHaveBeenCalledWith(`${REMOTE}tokenizer_config.json`, expect.objectContaining({ cache: "reload" }));
    expect(cache.entries.has(VERSIONED)).toBe(true);
  });

  it("keeps filesystem content namespaces separate by directory, not host or token", async () => {
    const directory = await fs.mkdtemp(path.join(os.tmpdir(), "transformers-namespaces-"));
    try {
      env.useCustomCache = false;
      env.useFSCache = true;
      env.allowLocalModels = true;
      const cacheA = path.join(directory, "host-a");
      const cacheB = path.join(directory, "host-b");
      const fetch = jest.fn(async (url) => jsonResponse({ host: new URL(url).hostname }));
      const options = (host, cacheDir, hfToken = "synthetic-original-token") => ({
        revision: "v1",
        env: { remoteHost: `https://${host}/`, cacheDir, hfToken, fetch },
      });
      const a = options("host-a.example", cacheA);
      const b = options("host-b.example", cacheB);
      expect(await getModelJSON(MODEL, "config.json", true, a)).toEqual({ host: "host-a.example" });
      expect(await getModelJSON(MODEL, "config.json", true, b)).toEqual({ host: "host-b.example" });
      expect(fetch).toHaveBeenCalledTimes(2);
      fetch.mockClear();
      fetch.mockImplementation(async () => {
        throw new Error("offline: network must not be called");
      });
      expect(await getModelJSON(MODEL, "config.json", true, { ...a, local_files_only: true })).toEqual({ host: "host-a.example" });
      expect(await getModelJSON(MODEL, "config.json", true, { ...b, local_files_only: true })).toEqual({ host: "host-b.example" });
      const rotated = options("host-a.example", cacheA, "synthetic-rotated-token");
      expect(await getModelJSON(MODEL, "config.json", true, { ...rotated, local_files_only: true })).toEqual({ host: "host-a.example" });
      // Credentials are not an authorization boundary for persisted files in the same content namespace.
      const sameNamespace = options("another-mirror.example", cacheA, "synthetic-other-token");
      expect(await getModelJSON(MODEL, "config.json", true, { ...sameNamespace, local_files_only: true })).toEqual({ host: "host-a.example" });
      const explicit = { ...b, cache_dir: cacheA, local_files_only: true };
      expect(await getModelJSON(MODEL, "config.json", true, explicit)).toEqual({ host: "host-a.example" });
      expect(fetch).not.toHaveBeenCalled();
      expect(await fs.readFile(path.join(cacheA, MODEL, "v1/config.json"), "utf8")).toBe('{"host":"host-a.example"}');
      expect(await fs.readFile(path.join(cacheB, MODEL, "v1/config.json"), "utf8")).toBe('{"host":"host-b.example"}');
    } finally {
      await fs.rm(directory, { recursive: true, force: true });
    }
  });

  it.each([null, 404])("refreshes an old Hub cache after local %s, including tokenizer discovery", async (localStatus) => {
    env.useFS = false;
    env.allowLocalModels = true;
    env.logLevel = 50;
    cache.entries.set(`${REMOTE}tokenizer_config.json`, jsonResponse({ stale: true }));
    env.fetch = jest.fn(async (url) => {
      if (url === `${REMOTE}tokenizer_config.json`) return jsonResponse({ updated: true });
      if (localStatus === null) throw new TypeError("Failed to fetch locally");
      return new Response(null, { status: localStatus });
    });
    expect(await get_tokenizer_files(MODEL)).toEqual(["tokenizer.json", "tokenizer_config.json"]);
    expect(env.fetch.mock.calls.map(([url]) => url)).toEqual([`/models/${MODEL}/tokenizer_config.json`, `${REMOTE}tokenizer_config.json`]);
    env.fetch.mockClear();
    expect(await getModelJSON(MODEL, "tokenizer_config.json")).toEqual({ updated: true });
    expect(env.fetch.mock.calls.map(([url]) => url)).toEqual([`/models/${MODEL}/tokenizer_config.json`, `${REMOTE}tokenizer_config.json`]);
    expect(cache.entries.has(VERSIONED)).toBe(true);
    expect(cache.entries.has(`${REMOTE}tokenizer_config.json`)).toBe(false);
    env.fetch.mockClear();
    expect(await getModelJSON(MODEL, "tokenizer_config.json")).toEqual({ updated: true });
    expect(await get_tokenizer_files(MODEL)).toEqual(["tokenizer.json", "tokenizer_config.json"]);
    expect(env.fetch).not.toHaveBeenCalled();
  });

  it.each([null, 404])("uses the old Hub cache if local %s is followed by Hub HTTP 503", async (localStatus) => {
    env.useFS = false;
    env.allowLocalModels = true;
    env.logLevel = 50;
    cache.entries.set(`${REMOTE}tokenizer_config.json`, jsonResponse({ stale: true }));
    env.fetch = jest.fn(async (url) => {
      if (url === `${REMOTE}tokenizer_config.json`) return new Response(null, { status: 503 });
      if (localStatus === null) throw new TypeError("Failed to fetch locally");
      return new Response(null, { status: localStatus });
    });
    expect(await getModelJSON(MODEL, "tokenizer_config.json")).toEqual({ stale: true });
    expect(await get_tokenizer_files(MODEL)).toEqual(["tokenizer.json", "tokenizer_config.json"]);
    expect(env.fetch.mock.calls.map(([url]) => url)).toEqual([`/models/${MODEL}/tokenizer_config.json`, `${REMOTE}tokenizer_config.json`, `/models/${MODEL}/tokenizer_config.json`, `${REMOTE}tokenizer_config.json`]);
    expect(cache.put).not.toHaveBeenCalled();
    expect(cache.delete).not.toHaveBeenCalled();
  });

  const permissionTest = process.platform === "win32" || process.getuid?.() === 0 ? it.skip : it;
  permissionTest("returns a refreshed tokenizer config when the old filesystem cache is read-only", async () => {
    const directory = await fs.mkdtemp(path.join(os.tmpdir(), "transformers-readonly-"));
    const modelDirectory = path.join(directory, MODEL);
    try {
      await fs.mkdir(modelDirectory, { recursive: true });
      await fs.writeFile(path.join(modelDirectory, "tokenizer_config.json"), '{"stale":true}');
      await fs.chmod(modelDirectory, 0o555);
      // A child process makes an unhandled stream error an observable failure rather than crashing Jest.
      const script = `
        import { env } from ${JSON.stringify(new URL("../../src/env.js", import.meta.url).href)};
        import { getModelJSON } from ${JSON.stringify(new URL("../../src/utils/hub.js", import.meta.url).href)};
        Object.assign(env, {
          allowLocalModels: false, allowRemoteModels: true, useFSCache: true,
          useCustomCache: false, useBrowserCache: false, experimental_useCrossOriginStorage: false,
          remoteHost: "https://huggingface.co/", cacheDir: ${JSON.stringify(directory)}, logLevel: 50,
          fetch: async () => new Response('{"updated":true}')
        });
        console.log(JSON.stringify(await getModelJSON(${JSON.stringify(MODEL)}, "tokenizer_config.json")));
      `;
      const result = spawnSync(process.execPath, ["--input-type=module", "-e", script], { encoding: "utf8", timeout: 10000 });
      expect({ status: result.status, stderr: result.stderr, error: result.error }).toEqual({ status: 0, stderr: "", error: undefined });
      expect(JSON.parse(result.stdout.trim())).toEqual({ updated: true });
      expect(await fs.readFile(path.join(modelDirectory, "tokenizer_config.json"), "utf8")).toBe('{"stale":true}');
      expect(await fs.readdir(modelDirectory)).toEqual(["tokenizer_config.json"]);
    } finally {
      await fs.chmod(modelDirectory, 0o755);
      await fs.rm(directory, { recursive: true, force: true });
    }
  });

  it("falls back to the outdated copy when the refresh fails, and replaces it once online", async () => {
    env.logLevel = 50;
    cache.entries.set(`${REMOTE}tokenizer_config.json`, jsonResponse({ stale: true }));
    const online = env.fetch;
    env.fetch = jest.fn(async () => {
      throw new TypeError("Failed to fetch");
    });
    expect(await getModelJSON(MODEL, "tokenizer_config.json")).toEqual({ stale: true });
    expect(cache.put).not.toHaveBeenCalled();
    expect(cache.entries.has(`${REMOTE}tokenizer_config.json`)).toBe(true);

    env.fetch = online;
    expect(await getModelJSON(MODEL, "tokenizer_config.json")).toEqual({ updated: true });
    expect(cache.entries.has(VERSIONED)).toBe(true);
    expect(cache.entries.has(`${REMOTE}tokenizer_config.json`)).toBe(false);
  });

  it("still fails offline when there is no cached copy at all", async () => {
    env.fetch = jest.fn(async () => {
      throw new TypeError("Failed to fetch");
    });
    await expect(getModelJSON(MODEL, "tokenizer_config.json")).rejects.toThrow("Failed to fetch");
  });

  it("serves the outdated filesystem copy offline from a cache folder filled by an older version", async () => {
    const directory = await fs.mkdtemp(path.join(os.tmpdir(), "transformers-offline-"));
    try {
      await fs.mkdir(path.join(directory, MODEL), { recursive: true });
      await fs.writeFile(path.join(directory, MODEL, "tokenizer_config.json"), '{"stale":true}');
      env.customCache = new FileCache(directory);
      env.allowRemoteModels = false;
      env.allowLocalModels = true;
      expect(await get_tokenizer_files(MODEL)).toEqual(["tokenizer.json", "tokenizer_config.json"]);
      expect(await getModelJSON(MODEL, "tokenizer_config.json")).toEqual({ stale: true });
      expect(await getModelFile(MODEL, "tokenizer_config.json", true, {}, true)).toBe(path.join(directory, MODEL, "tokenizer_config.json"));
    } finally {
      await fs.rm(directory, { recursive: true, force: true });
    }
  });

  it("captures the source, transport, and cache before asynchronous cache lookup", async () => {
    const fetch = env.fetch;
    const otherCache = memoryCache();
    const otherFetch = jest.fn(async () => jsonResponse({ wrong: true }));
    cache.match.mockImplementation(async () => {
      env.remoteHost = "https://other.example/";
      env.fetch = otherFetch;
      env.customCache = otherCache;
      return undefined;
    });
    expect(await getModelJSON(MODEL, "tokenizer_config.json")).toEqual({ updated: true });
    expect(fetch).toHaveBeenCalledWith(`${REMOTE}tokenizer_config.json`, expect.anything());
    expect(otherFetch).not.toHaveBeenCalled();
    expect(cache.entries.has(VERSIONED)).toBe(true);
    expect(otherCache.put).not.toHaveBeenCalled();
  });

  it("deduplicates equivalent loads but isolates different transports", async () => {
    env.useCustomCache = false;
    let resolveFirst;
    const firstFetch = jest.fn(
      () =>
        new Promise((resolve) => {
          resolveFirst = resolve;
        }),
    );
    env.fetch = firstFetch;
    const first = getModelJSON(MODEL, "config.json");
    const sibling = getModelJSON(MODEL, "config.json");
    env.fetch = jest.fn(async () => jsonResponse({ second: true }));
    const second = getModelJSON(MODEL, "config.json");
    expect(await second).toEqual({ second: true });
    resolveFirst(jsonResponse({ first: true }));
    expect(await first).toEqual({ first: true });
    expect(await sibling).toEqual({ first: true });
    expect(firstFetch).toHaveBeenCalledTimes(1);
    expect(env.fetch).toHaveBeenCalledTimes(1);
  });

  it("reuses an existing loading context for metadata requests", async () => {
    const fetch = env.fetch;
    fetch.mockResolvedValue(new Response("x", { status: 206, headers: { "content-range": "bytes 0-0/42" } }));
    const context = new AssetLoadingContext();
    env.fetch = jest.fn();
    env.customCache = memoryCache();

    expect(await get_file_metadata_with_context(MODEL, "config.json", context)).toEqual({ exists: true, size: 42, contentType: "text/plain;charset=UTF-8", fromCache: false });
    expect(fetch).toHaveBeenCalledWith(`${REMOTE}config.json`, expect.objectContaining({ method: "GET" }));
    expect(env.fetch).not.toHaveBeenCalled();
    expect(cache.match).toHaveBeenCalled();
    expect(env.customCache.match).not.toHaveBeenCalled();
  });

  it("preserves authorization while adding metadata Range headers, without versioning the request URL", async () => {
    const originalToken = process.env.HF_TOKEN;
    process.env.HF_TOKEN = "test-token";
    env.fetch.mockResolvedValue(new Response("x", { status: 206, headers: { "content-range": "bytes 0-0/42" } }));
    try {
      expect(await get_file_metadata(MODEL, "tokenizer_config.json")).toEqual({ exists: true, size: 42, contentType: "text/plain;charset=UTF-8", fromCache: false });
      const [url, init] = env.fetch.mock.calls[0];
      expect(url).toBe(`${REMOTE}tokenizer_config.json`);
      expect(init.headers.get("Range")).toBe("bytes=0-0");
      expect(init.headers.get("Authorization")).toBe("Bearer test-token");
    } finally {
      if (originalToken === undefined) delete process.env.HF_TOKEN;
      else process.env.HF_TOKEN = originalToken;
    }
  });

  it("keeps the repo layout for versioned filesystem entries while preserving revisions and unrelated entries", async () => {
    const directory = await fs.mkdtemp(path.join(os.tmpdir(), "transformers-assets-"));
    try {
      const fileCache = new FileCache(directory);
      env.customCache = fileCache;
      await fileCache.put(`${MODEL}/tokenizer_config.json`, jsonResponse({ stale: true }));
      await fileCache.put(`${MODEL}/onnx/model.onnx`, new Response("weights"));
      expect(await getModelJSON(MODEL, "tokenizer_config.json")).toEqual({ updated: true });
      expect(await fs.readFile(path.join(directory, MODEL, "tokenizer_config.json"), "utf8")).toBe('{"updated":true}');
      expect(await fs.readdir(directory)).toEqual(["test"]);
      expect(await fileCache.match(`${MODEL}/tokenizer_config.json`)).toBeUndefined();
      expect(await fileCache.match(`${MODEL}/onnx/model.onnx`)).toBeDefined();
      const revision = new AssetLoadingContext({ revision: "v1" }).resolve(MODEL, "tokenizer_config.json", fileCache);
      expect(revision.proposedCacheKey).toBe(`${MODEL}/v1/tokenizer_config.json?__transformersjs_cache_version=1`);
      expect(revision.remoteURL).toBe(`https://huggingface.co/${MODEL}/resolve/v1/tokenizer_config.json`);
    } finally {
      await fs.rm(directory, { recursive: true, force: true });
    }
  });

  it.each([
    ["model/config.json?__transformersjs_cache_version=1", "model/config.json"],
    ["model/config.json?download=true&__transformersjs_cache_version=1", "model/config.json?download=true"],
    ["model/config.json?__transformersjs_cache_version=1&download=true#fragment", "model/config.json?download=true#fragment"],
  ])("stores %s at the unversioned path and tracks its version beside it", async (key, originalKey) => {
    const directory = await fs.mkdtemp(path.join(os.tmpdir(), "transformers-file-identity-"));
    try {
      const fileCache = new FileCache(directory);
      await fileCache.put(originalKey, jsonResponse({ old: true }));
      expect(await fileCache.match(key)).toBeUndefined();
      await fileCache.put(key, jsonResponse({ updated: true }));
      const response = await fileCache.match(key);
      expect(response.filePath).toBe(path.join(directory, originalKey));
      expect(await response.json()).toEqual({ updated: true });
      expect(await fileCache.match(originalKey)).toBeUndefined();
      expect(await fileCache.delete(originalKey)).toBe(false);
      expect(await fileCache.match(key)).toBeDefined();
      expect(await fileCache.delete(key)).toBe(true);
      expect(await fileCache.match(key)).toBeUndefined();
      expect(await fs.readdir(path.join(directory, "model"))).toEqual([]);
    } finally {
      await fs.rm(directory, { recursive: true, force: true });
    }
  });

  it("returns the repo-layout filesystem path for a versioned asset on download and cache hit", async () => {
    const directory = await fs.mkdtemp(path.join(os.tmpdir(), "transformers-return-path-"));
    try {
      env.customCache = new FileCache(directory);
      const expectedPath = path.join(directory, MODEL, "tokenizer_config.json");
      expect(await getModelFile(MODEL, "tokenizer_config.json", true, {}, true)).toBe(expectedPath);
      expect(await getModelFile(MODEL, "tokenizer_config.json", true, {}, true)).toBe(expectedPath);
      expect(await fs.readFile(expectedPath, "utf8")).toBe('{"updated":true}');
      expect(env.fetch).toHaveBeenCalledTimes(1);
    } finally {
      await fs.rm(directory, { recursive: true, force: true });
    }
  });

  it("reads actual local source files without renaming, versioning, or deleting them", async () => {
    const directory = await fs.mkdtemp(path.join(os.tmpdir(), "transformers-local-assets-"));
    try {
      const source = path.join(directory, "model");
      await fs.mkdir(source);
      await fs.writeFile(path.join(source, "tokenizer_config.json"), '{"local":true}');
      env.allowLocalModels = true;
      env.allowRemoteModels = false;
      env.customCache = new FileCache(path.join(directory, "cache"));
      expect(await getModelJSON(source, "tokenizer_config.json", true, { local_files_only: true })).toEqual({ local: true });
      expect(await fs.readFile(path.join(source, "tokenizer_config.json"), "utf8")).toBe('{"local":true}');
      expect(env.fetch).not.toHaveBeenCalled();
    } finally {
      await fs.rm(directory, { recursive: true, force: true });
    }
  });
});
