import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";

import { __calls as nativeFsCalls } from "./native-universal-fs.mock.js";

// These must be applied before any src/ module is imported: env.js derives
// `apis.IS_REACT_NATIVE_ENV` and `apis.IS_NODE_ENV` at module-evaluation time.
// React Native's Metro polyfills `process` but not `process.release`, so a faithful
// simulation has to clear it -- otherwise IS_NODE_ENV stays true and the RN-only
// branches are never reached.
Object.defineProperty(globalThis, "navigator", { value: { product: "ReactNative" }, configurable: true, writable: true });
Object.defineProperty(process, "release", { value: undefined, configurable: true });

const { apis, env } = await import("../src/env.js");
const { FileResponse } = await import("../src/utils/hub/FileResponse.js");
const { readResponse } = await import("../src/utils/hub/utils.js");
const { getCoreModelFile } = await import("../src/utils/model-loader.js");
const { FileCache } = await import("../src/utils/cache/FileCache.js");

const MODEL_ID = "testorg/testmodel";
const CONFIG_BODY = JSON.stringify({ model_type: "bert" });
// Large enough that a chunked read would produce more than one progress event.
const ONNX_BODY = Buffer.alloc(64 * 1024, 7);

let tmpDir;
let modelDir;

beforeAll(() => {
  tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "transformers-rn-test-"));
  modelDir = path.join(tmpDir, "models", MODEL_ID);
  fs.mkdirSync(path.join(modelDir, "onnx"), { recursive: true });
  fs.writeFileSync(path.join(modelDir, "config.json"), CONFIG_BODY);
  fs.writeFileSync(path.join(modelDir, "onnx", "encoder_model.onnx"), ONNX_BODY);

  env.localModelPath = path.join(tmpDir, "models") + path.sep;
  env.allowRemoteModels = false;
  env.useFSCache = false;
  env.useBrowserCache = false;
});

afterAll(() => {
  fs.rmSync(tmpDir, { recursive: true, force: true });
});

describe("React Native environment", () => {
  // Everything below is meaningless if the simulated environment isn't actually RN.
  it("is detected as React Native and not as Node", () => {
    expect(apis.IS_REACT_NATIVE_ENV).toBe(true);
    expect(apis.IS_NODE_ENV).toBe(false);
  });

  describe("FileResponse", () => {
    it("resolves metadata for an existing file via create()", async () => {
      const response = await FileResponse.create(path.join(modelDir, "config.json"));
      expect(response.exists).toBe(true);
      expect(response.status).toBe(200);
      expect(response.headers.get("content-length")).toBe(String(CONFIG_BODY.length));
      expect(response.headers.get("content-type")).toBe("application/json");
    });

    it("reports a missing file as 404", async () => {
      const response = await FileResponse.create(path.join(modelDir, "does-not-exist.json"));
      expect(response.exists).toBe(false);
      expect(response.status).toBe(404);
    });

    it("reads contents through the native filesystem", async () => {
      const response = await FileResponse.create(path.join(modelDir, "config.json"));
      expect(await response.text()).toBe(CONFIG_BODY);
      expect(await response.json()).toEqual({ model_type: "bert" });
      expect(new Uint8Array(await response.arrayBuffer())).toEqual(new Uint8Array(Buffer.from(CONFIG_BODY)));
    });
  });

  describe("readResponse", () => {
    // RN FileResponses carry no `body` stream -- reading is deferred until something
    // actually asks for the bytes -- so readResponse has to cope with a null body.
    it("reads a FileResponse that has no body stream", async () => {
      const response = await FileResponse.create(path.join(modelDir, "onnx", "encoder_model.onnx"));
      expect(response.body).toBeNull();

      const progress = [];
      const buffer = await readResponse(response, (data) => progress.push(data));

      expect(buffer).toEqual(new Uint8Array(ONNX_BODY));
      expect(progress.length).toBeGreaterThan(0);
      expect(progress.at(-1)).toMatchObject({ progress: 100, loaded: ONNX_BODY.length, total: ONNX_BODY.length });
    });
  });

  describe("getCoreModelFile", () => {
    // This is the path PreTrainedModel.from_pretrained() takes to load the .onnx graph.
    // In RN it asks for a path (return_path=true) so onnxruntime-react-native can load
    // straight from disk instead of through a JS buffer.
    it("returns a path rather than a buffer", async () => {
      const result = await getCoreModelFile(MODEL_ID, "encoder_model", { subfolder: "onnx" }, "");
      expect(typeof result).toBe("string");
      expect(result).toContain("encoder_model.onnx");
    });

    it("returns a path when a progress_callback is supplied", async () => {
      const events = [];
      const result = await getCoreModelFile(MODEL_ID, "encoder_model", { subfolder: "onnx", progress_callback: (e) => events.push(e) }, "");
      expect(typeof result).toBe("string");
      expect(result).toContain("encoder_model.onnx");
      expect(events.some((e) => e.status === "done")).toBe(true);
    });

    it("returns a file:// URL, which is what the native filesystem layer uses", async () => {
      const result = await getCoreModelFile(MODEL_ID, "encoder_model", { subfolder: "onnx" }, "");
      expect(result.startsWith("file://")).toBe(true);
    });
  });

  describe("FileCache", () => {
    // On RN the cache writes through native-universal-fs (base64) rather than a Node
    // write stream, so the round trip is worth pinning down.
    it("stores and retrieves a file, reporting download progress", async () => {
      const cache = new FileCache(path.join(tmpDir, "cache"));
      const key = "testorg/testmodel/onnx/encoder_model.onnx";

      expect(await cache.match(key)).toBeUndefined();

      const progress = [];
      await cache.put(key, new Response(ONNX_BODY, { headers: { "Content-Length": String(ONNX_BODY.length) } }), (data) => progress.push(data));

      const cached = await cache.match(key);
      expect(cached).toBeDefined();
      expect(cached.exists).toBe(true);
      expect(new Uint8Array(await cached.arrayBuffer())).toEqual(new Uint8Array(ONNX_BODY));
      expect(progress.at(-1)).toMatchObject({ progress: 100, loaded: ONNX_BODY.length, total: ONNX_BODY.length });

      // No temp files should survive a successful write.
      expect(fs.readdirSync(path.dirname(path.join(tmpDir, "cache", key))).filter((f) => f.includes(".tmp."))).toEqual([]);
    });

    // React Native's `fetch` is the whatwg-fetch polyfill, whose `Response` has no `body`
    // stream at all. The cache write therefore cannot go through `body.getReader()`.
    it("stores a Response that exposes no body stream", async () => {
      const cache = new FileCache(path.join(tmpDir, "cache-bodyless"));
      const key = "testorg/testmodel/config.json";
      const bodyless = {
        headers: new Headers({ "Content-Length": String(CONFIG_BODY.length) }),
        body: undefined,
        arrayBuffer: async () => new TextEncoder().encode(CONFIG_BODY).buffer,
      };

      await cache.put(key, /** @type {any} */ (bodyless));

      const cached = await cache.match(key);
      expect(cached).toBeDefined();
      expect(await cached.text()).toBe(CONFIG_BODY);
    });

    it("leaves no temp file behind when the write fails", async () => {
      const cache = new FileCache(path.join(tmpDir, "cache-fail"));
      const key = "testorg/testmodel/broken.onnx";
      const body = new ReadableStream({
        start(controller) {
          controller.error(new Error("connection reset"));
        },
      });

      await expect(cache.put(key, new Response(body))).rejects.toThrow("connection reset");
      expect(await cache.match(key)).toBeUndefined();
    });
  });

  describe("remote model download", () => {
    const REMOTE_MODEL_ID = "testorg/remotemodel";
    let server;
    let baseURL;
    let previousEnv;

    beforeAll(async () => {
      server = http.createServer((req, res) => {
        if (req.url.endsWith("/onnx/encoder_model.onnx")) {
          res.writeHead(200, { "Content-Type": "application/octet-stream", "Content-Length": String(ONNX_BODY.length) });
          res.end(ONNX_BODY);
        } else {
          res.writeHead(404);
          res.end();
        }
      });
      await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
      baseURL = `http://127.0.0.1:${server.address().port}/`;

      previousEnv = { remoteHost: env.remoteHost, allowRemoteModels: env.allowRemoteModels, useFSCache: env.useFSCache, cacheDir: env.cacheDir };
      env.remoteHost = baseURL;
      env.allowRemoteModels = true;
      env.useFSCache = true;
      env.cacheDir = path.join(tmpDir, "model-cache");
    });

    afterAll(async () => {
      Object.assign(env, previousEnv);
      await new Promise((resolve) => server.close(resolve));
    });

    // A .onnx graph can be gigabytes. Pulling it through `fetch` would materialise the
    // whole thing in the JS heap (React Native's fetch buffers into an ArrayBuffer, and
    // its Response has no stream to read incrementally), which is exactly what exhausts
    // the heap on device. The native downloader writes straight to disk instead.
    it("streams the model to disk without routing it through fetch", async () => {
      nativeFsCalls.downloadFile.length = 0;
      // In this environment `env.fetch` is the XHR-backed RN implementation, which has no
      // XMLHttpRequest to call -- so if the model is fetched rather than streamed, this
      // throws rather than quietly buffering.
      const result = await getCoreModelFile(REMOTE_MODEL_ID, "encoder_model", { subfolder: "onnx" }, "");

      expect(typeof result).toBe("string");
      expect(result.startsWith("file://")).toBe(true);
      expect(nativeFsCalls.downloadFile).toHaveLength(1);
      expect(nativeFsCalls.downloadFile[0].fromUrl).toContain("encoder_model.onnx");

      const onDisk = fs.readFileSync(String(result).replace(/^file:\/\//, ""));
      expect(new Uint8Array(onDisk)).toEqual(new Uint8Array(ONNX_BODY));
    });

    it("reuses the downloaded file on a second load", async () => {
      nativeFsCalls.downloadFile.length = 0;
      const result = await getCoreModelFile(REMOTE_MODEL_ID, "encoder_model", { subfolder: "onnx" }, "");

      expect(result.startsWith("file://")).toBe(true);
      expect(nativeFsCalls.downloadFile).toHaveLength(0);
    });
  });
});
