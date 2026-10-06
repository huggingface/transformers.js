import { env } from "../../src/env.js";
import { getModelFile } from "../../src/utils/hub.js";

/**
 * The two things `as_blob` has to get right are a COLD file and a WARM one, and they take different branches:
 * cold streams the body into the cache and reads it back, warm reads the cached `Response` directly. Both are
 * exercised here against an in-memory cache and a stubbed `env.fetch`, so nothing touches the network and the
 * assertions can be about which branch ran rather than about wall-clock behaviour.
 */
class MemoryCache {
  constructor() {
    /** @type {Map<string, Response>} */
    this.entries = new Map();
    this.puts = 0;
    /** @type {Error|null} Set to make `put` reject, standing in for QuotaExceededError. */
    this.putError = null;
  }

  async match(key) {
    const hit = this.entries.get(key);
    return hit ? hit.clone() : undefined;
  }

  async put(key, response) {
    this.puts++;
    if (this.putError) throw this.putError;
    // Read the body here rather than storing the streaming `Response`: a real Cache Storage entry is settled
    // by the time `match` can see it, and holding the live stream would let a test pass on a tee that a real
    // cache would never hand back.
    this.entries.set(key, new Response(await response.arrayBuffer(), { headers: response.headers }));
  }
}

const BYTES = new Uint8Array(4096).fill(7);

describe("External data as a Blob", () => {
  const saved = {};
  let cache;
  let fetches;

  beforeEach(() => {
    for (const k of ["useCustomCache", "customCache", "useBrowserCache", "useFSCache", "allowLocalModels", "fetch"]) {
      saved[k] = env[k];
    }
    cache = new MemoryCache();
    fetches = 0;
    env.useCustomCache = true;
    env.customCache = cache;
    env.useBrowserCache = false;
    env.useFSCache = false;
    env.allowLocalModels = false;
    env.fetch = async () => {
      ++fetches;
      return new Response(BYTES, {
        status: 200,
        headers: { "content-length": String(BYTES.length), "content-type": "application/octet-stream" },
      });
    };
  });

  afterEach(() => {
    for (const [k, v] of Object.entries(saved)) env[k] = v;
  });

  const load = (as_blob, options = {}, filename = "onnx/model.onnx_data") => getModelFile("hf-internal-testing/blob-external-data", filename, true, options, false, as_blob);

  it("streams a cold file into the cache and resolves a Blob, storing it once", async () => {
    const progress = [];
    const out = await load(true, { progress_callback: (e) => e.status === "progress" && progress.push(e) });

    expect(out).toBeInstanceOf(Blob);
    expect(out.size).toBe(BYTES.length);
    expect(cache.puts).toBe(1); // and NOT stored a second time by the block at the end of loadResourceFile
    expect(fetches).toBe(1);

    // Progress survives the streaming path — it is the reason the buffer existed, so losing it silently would
    // be the regression that matters most to a user watching gigabytes arrive.
    expect(progress.length).toBeGreaterThan(0);
    expect(progress.at(-1).loaded).toBe(BYTES.length);
    expect(progress.at(-1).total).toBe(BYTES.length);
  });

  it("reads a warm file straight out of the cache without fetching again", async () => {
    await load(true);
    expect(fetches).toBe(1);

    const progress = [];
    const out = await load(true, { progress_callback: (e) => e.status === "progress" && progress.push(e) });

    expect(out).toBeInstanceOf(Blob);
    expect(out.size).toBe(BYTES.length);
    expect(fetches).toBe(1); // served from the cache
    expect(cache.puts).toBe(1); // and not rewritten
    // The warm branch reports one completed event rather than a stream of them.
    expect(progress.at(-1)).toMatchObject({ progress: 100, loaded: BYTES.length, total: BYTES.length });
  });

  it("falls back to a buffer when the cache refuses the write", async () => {
    cache.putError = Object.assign(new Error("quota"), { name: "QuotaExceededError" });

    const out = await load(true);

    // A cache that cannot accept the file must not fail the load — it costs the memory saving, nothing else.
    expect(out).toBeInstanceOf(Uint8Array);
    expect(out.length).toBe(BYTES.length);
    // Re-fetched, because the streaming attempt consumed the first body.
    expect(fetches).toBe(2);
  });

  it("still resolves a Uint8Array when as_blob is not asked for", async () => {
    const out = await load(false);
    expect(out).toBeInstanceOf(Uint8Array);
    expect(out.length).toBe(BYTES.length);
  });

  // ⛔ THE HALF OF THE MEMORY CLAIM THIS LAYER CAN ACTUALLY TEST. Whether onnxruntime-web copies the Blob at
  // session creation depends on its build — JSPI does not, the default asyncify one does — and that is not
  // observable from here. What IS observable, and is the saving this file is responsible for, is that the
  // DOWNLOAD never materialises the body: no `arrayBuffer()` on the fetched response, on any build.
  //
  // Paired with the negative case on purpose. Asserting "never called" alone would pass just as happily if
  // the spy were wired to the wrong object, so the same spy has to SEE the call on the buffered path.
  it("streams a cold file without materialising the response, and buffers it when not asked for a Blob", async () => {
    let materialised = 0;
    const inner = env.fetch;
    env.fetch = async (...args) => {
      const response = await inner(...args);
      const original = response.arrayBuffer.bind(response);
      response.arrayBuffer = async () => {
        ++materialised;
        return original();
      };
      return response;
    };

    expect(await load(true)).toBeInstanceOf(Blob);
    expect(materialised).toBe(0);

    // Same spy, same fetch, `as_blob` off: the buffered path reads the body onto the heap, which is what the
    // streaming branch is avoiding — and proves the assertion above is watching the right object.
    expect(await load(false, {}, "onnx/other.onnx_data")).toBeInstanceOf(Uint8Array);
    expect(materialised).toBe(1);
  });

  // ⛔ THE REGRESSION THIS EXISTS FOR. `as_blob` changes the RESOLVED TYPE, so it has to be part of the
  // in-flight dedup key. Without it whichever caller arrives first decides for both, and the other gets a
  // `Blob` where it expects a `Uint8Array` or the reverse — silently, and only under concurrency.
  it("does not hand a Blob to a concurrent caller that asked for bytes", async () => {
    await load(true); // warm, so both calls below take the same branch and race deterministically

    const [asBlob, asBytes] = await Promise.all([load(true), load(false)]);

    expect(asBlob).toBeInstanceOf(Blob);
    expect(asBytes).toBeInstanceOf(Uint8Array);
  });

  it("does not hand bytes to a concurrent caller that asked for a Blob", async () => {
    await load(true);

    // The reverse order, because a key bug is only visible in whichever direction loses the race.
    const [asBytes, asBlob] = await Promise.all([load(false), load(true)]);

    expect(asBytes).toBeInstanceOf(Uint8Array);
    expect(asBlob).toBeInstanceOf(Blob);
  });
});
