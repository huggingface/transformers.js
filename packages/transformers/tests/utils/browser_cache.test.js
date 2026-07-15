import { env } from "../../src/transformers.js";
import { getModelFile } from "../../src/utils/hub.js";
import { init } from "../init.js";

// Initialise the testing environment
init();

// The browser Cache API is not available in Node.js, so emulate it
const originalCache = globalThis.Cache;
if (typeof globalThis.Cache === "undefined") {
  globalThis.Cache = class Cache {};
}

/**
 * A minimal stand-in for the browser Cache API, which only supports
 * http(s) URLs as keys and throws a TypeError for any other scheme.
 */
class FakeBrowserCache extends Cache {
  constructor() {
    super();
    /** @type {string[]} */
    this.puts = [];
  }

  async match(_request) {
    return undefined;
  }

  async put(request, _response) {
    const key = String(request);
    this.puts.push(key);
    let scheme = null;
    try {
      scheme = new URL(key).protocol;
    } catch {
      // Relative keys are resolved against the page URL by the real Cache API
    }
    if (scheme !== null && scheme !== "http:" && scheme !== "https:") {
      throw new TypeError(`Request scheme '${scheme.slice(0, -1)}' is unsupported`);
    }
  }
}

describe("Browser cache", () => {
  // Store original values so we can restore them after tests
  const originalCaches = globalThis.caches;
  const originalFetch = env.fetch;
  const originalUseFS = env.useFS;
  const originalUseBrowserCache = env.useBrowserCache;
  const originalUseFSCache = env.useFSCache;
  const originalAllowLocalModels = env.allowLocalModels;
  const originalAllowRemoteModels = env.allowRemoteModels;
  const originalLocalModelPath = env.localModelPath;

  /** @type {FakeBrowserCache} */
  let cache;

  beforeEach(() => {
    cache = new FakeBrowserCache();
    globalThis.caches = { open: async () => cache };

    env.useFS = false;
    env.useBrowserCache = true;
    env.useFSCache = false;
    env.allowLocalModels = true;
    env.allowRemoteModels = true;
    env.localModelPath = "chrome-extension://abcdefghijklmnop/models/";
    env.fetch = async (url) => {
      const key = String(url);
      if (key.startsWith("chrome-extension://") || key.startsWith("https://") || key.startsWith("/models/")) {
        return new Response(JSON.stringify({ hello: "world" }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      return new Response(null, { status: 404 });
    };
  });

  afterAll(() => {
    if (originalCache === undefined) {
      delete globalThis.Cache;
    } else {
      globalThis.Cache = originalCache;
    }
    globalThis.caches = originalCaches;
    env.fetch = originalFetch;
    env.useFS = originalUseFS;
    env.useBrowserCache = originalUseBrowserCache;
    env.useFSCache = originalUseFSCache;
    env.allowLocalModels = originalAllowLocalModels;
    env.allowRemoteModels = originalAllowRemoteModels;
    env.localModelPath = originalLocalModelPath;
  });

  it("should not attempt to cache files with non-http(s) schemes (e.g., bundled in a browser extension)", async () => {
    const file = await getModelFile("my-model", "config.json", true, {});

    expect(file).toBeInstanceOf(Uint8Array);
    expect(cache.puts).toEqual([]);
  });

  it("should still cache local files served from a relative path", async () => {
    env.localModelPath = "/models/";

    const file = await getModelFile("my-model", "config.json", true, {});

    expect(file).toBeInstanceOf(Uint8Array);
    expect(cache.puts).toEqual(["/models/my-model/config.json"]);
  });

  it("should still cache remote files", async () => {
    env.allowLocalModels = false;

    const file = await getModelFile("my-org/my-model", "config.json", true, {});

    expect(file).toBeInstanceOf(Uint8Array);
    expect(cache.puts).toHaveLength(1);
    expect(cache.puts[0]).toMatch(/^https:/);
  });
});
