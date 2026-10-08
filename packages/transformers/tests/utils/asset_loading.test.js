import { jest } from "@jest/globals";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";

const files = ["tokenizer_config.json", "onnx/model.onnx"];
jest.unstable_mockModule("../../src/utils/model_registry/get_files.js", () => ({
  get_files: jest.fn(async () => files),
}));
jest.unstable_mockModule("../../src/utils/model_registry/get_pipeline_files.js", () => ({
  get_pipeline_files: jest.fn(async () => files),
}));

const { env } = await import("../../src/env.js");
const { getModelJSON, getModelFile, buildResourcePaths, checkCachedResource } = await import("../../src/utils/hub.js");
const { get_file_metadata } = await import("../../src/utils/model_registry/get_file_metadata.js");
const { is_cached_files, is_pipeline_cached_files } = await import("../../src/utils/model_registry/is_cached.js");
const { clear_cache, clear_pipeline_cache } = await import("../../src/utils/model_registry/clear_cache.js");
const { FileCache } = await import("../../src/utils/cache/FileCache.js");

const MODEL = "test/asset-loading";
const REMOTE = `https://huggingface.co/${MODEL}/resolve/main/`;
const VERSIONED = `${REMOTE}tokenizer_config.json?__transformersjs_cache_version=2`;
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
    it.each([`${REMOTE}tokenizer_config.json`, `/models/${MODEL}/tokenizer_config.json`, `${REMOTE}tokenizer_config.json?__transformersjs_cache_version=1`, `/models/${MODEL}/tokenizer_config.json?__transformersjs_cache_version=1`])("clears a stale-only entry at %s before any replacement is loaded", async (key) => {
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
      const resource = buildResourcePaths(MODEL, "tokenizer_config.json", {}, cache);
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
        const earlierKey = `transformersjs_assets_v1/${oldKey}`;
        const otherRevision = `${MODEL}/tokenizer_config.json`;
        await fileCache.put(oldKey, jsonResponse({ stale: true }));
        await fileCache.put(earlierKey, jsonResponse({ stale: true }));
        await fileCache.put(otherRevision, jsonResponse({ unrelated: true }));

        const result = await clear({ revision: "v1" });
        expect(result.filesCached).toBe(1);
        expect(result.filesDeleted).toBe(1);
        expect(await fileCache.match(oldKey)).toBeUndefined();
        expect(await fileCache.match(earlierKey)).toBeUndefined();
        expect(await fileCache.match(otherRevision)).toBeDefined();
        expect(env.fetch).not.toHaveBeenCalled();
      } finally {
        await fs.rm(directory, { recursive: true, force: true });
      }
    });
  });

  it("keeps the existing path and cache helpers consistent without accepting stale aliases", async () => {
    const resource = buildResourcePaths(MODEL, "tokenizer_config.json", {}, cache);
    cache.entries.set(resource.localPath, jsonResponse({ stale: true }));
    expect(await checkCachedResource(cache, resource.localPath, resource.proposedCacheKey)).toBeUndefined();
    cache.entries.set(VERSIONED, jsonResponse({ updated: true }));
    expect(await (await checkCachedResource(cache, resource.localPath, resource.proposedCacheKey)).json()).toEqual({ updated: true });
  });

  it("versions local HTTP cache entries without changing the local request path", async () => {
    env.allowLocalModels = true;
    env.useFS = false;
    const localPath = `/models/${MODEL}/tokenizer_config.json`;
    cache.entries.set(localPath, jsonResponse({ stale: true }));
    expect(await getModelJSON(MODEL, "tokenizer_config.json")).toEqual({ updated: true });
    expect(env.fetch).toHaveBeenCalledWith(localPath, expect.anything());
    expect(cache.entries.has(`${localPath}?__transformersjs_cache_version=2`)).toBe(true);
    expect(cache.entries.has(localPath)).toBe(false);
    expect(await getModelJSON(MODEL, "tokenizer_config.json")).toEqual({ updated: true });
    expect(env.fetch).toHaveBeenCalledTimes(1);
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

  it("does not accept an invalidated entry when remote access is disabled", async () => {
    env.allowLocalModels = true;
    env.allowRemoteModels = false;
    cache.entries.set(`${REMOTE}tokenizer_config.json`, jsonResponse({ stale: true }));
    await expect(getModelJSON(MODEL, "tokenizer_config.json")).rejects.toThrow("file was not found locally");
    expect(env.fetch).not.toHaveBeenCalled();
    expect(cache.delete).not.toHaveBeenCalled();
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

  it("versions filesystem cache paths while preserving filenames, revisions, and unrelated entries", async () => {
    const directory = await fs.mkdtemp(path.join(os.tmpdir(), "transformers-assets-"));
    try {
      const fileCache = new FileCache(directory);
      env.customCache = fileCache;
      await fileCache.put(`${MODEL}/tokenizer_config.json`, jsonResponse({ stale: true }));
      await fileCache.put(`${MODEL}/onnx/model.onnx`, new Response("weights"));
      expect(await getModelJSON(MODEL, "tokenizer_config.json")).toEqual({ updated: true });
      expect(await fs.readFile(path.join(directory, "transformersjs_assets_v2", MODEL, "tokenizer_config.json"), "utf8")).toBe('{"updated":true}');
      expect(await fileCache.match(`${MODEL}/tokenizer_config.json`)).toBeUndefined();
      expect(await fileCache.match(`${MODEL}/onnx/model.onnx`)).toBeDefined();
      const revision = buildResourcePaths(MODEL, "tokenizer_config.json", { revision: "v1" }, fileCache);
      expect(revision.proposedCacheKey).toBe(`transformersjs_assets_v2/${MODEL}/v1/tokenizer_config.json`);
      expect(revision.remoteURL).toBe(`https://huggingface.co/${MODEL}/resolve/v1/tokenizer_config.json`);
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
