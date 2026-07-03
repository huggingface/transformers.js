import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { FileCache } from "../../src/utils/cache/FileCache.js";

/**
 * Build a streaming `Response` from an array of `Uint8Array` chunks.
 * @param {Uint8Array[]} chunks
 * @param {{status?: number, headers?: Record<string,string>}} [init]
 */
function streamingResponse(chunks, { status = 200, headers = {} } = {}) {
  let i = 0;
  const body = new ReadableStream({
    pull(controller) {
      if (i >= chunks.length) {
        controller.close();
        return;
      }
      controller.enqueue(chunks[i++]);
    },
  });
  return new Response(body, { status, headers });
}

const bytes = (str) => new TextEncoder().encode(str);

describe("FileCache resumable downloads", () => {
  let dir;
  let cache;

  beforeEach(() => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), "tjs-filecache-"));
    cache = new FileCache(dir);
  });

  afterEach(() => {
    fs.rmSync(dir, { recursive: true, force: true });
  });

  it("getResumeInfo returns undefined when there is no partial", async () => {
    expect(await cache.getResumeInfo("model.onnx")).toBeUndefined();
  });

  it("writes a full 200 response and leaves no partial artifacts", async () => {
    const content = "hello world";
    await cache.put(
      "model.onnx",
      streamingResponse([bytes(content)], {
        status: 200,
        headers: { "Content-Length": String(content.length), etag: '"abc"' },
      }),
    );

    expect(fs.readFileSync(path.join(dir, "model.onnx"), "utf-8")).toBe(content);
    expect(fs.existsSync(path.join(dir, "model.onnx.incomplete"))).toBe(false);
    expect(fs.existsSync(path.join(dir, "model.onnx.incomplete.json"))).toBe(false);
    expect(fs.existsSync(path.join(dir, "model.onnx.incomplete.lock"))).toBe(false);
  });

  it("keeps the partial and sidecar when the body is truncated", async () => {
    const key = "big.onnx";
    // Server promises 10 bytes but only delivers 4.
    await expect(
      cache.put(
        key,
        streamingResponse([bytes("0123")], {
          status: 200,
          headers: { "Content-Length": "10", etag: '"v1"' },
        }),
      ),
    ).rejects.toThrow(/Incomplete download/);

    // Final file must NOT be published; partial + sidecar remain for resume.
    expect(fs.existsSync(path.join(dir, key))).toBe(false);
    expect(fs.readFileSync(path.join(dir, key + ".incomplete"), "utf-8")).toBe("0123");
    expect(fs.existsSync(path.join(dir, key + ".incomplete.lock"))).toBe(false);

    const resume = await cache.getResumeInfo(key);
    expect(resume).toEqual({ size: 4, etag: '"v1"', total: 10 });
  });

  it("appends a 206 partial-content response onto the existing partial", async () => {
    const key = "weights.onnx";
    const full = "0123456789";

    // Seed a prior interrupted download: first 4 bytes + sidecar.
    fs.writeFileSync(path.join(dir, key + ".incomplete"), full.slice(0, 4));
    fs.writeFileSync(path.join(dir, key + ".incomplete.json"), JSON.stringify({ etag: '"v1"', total: 10 }));

    // Server continues from byte 4 with a 206.
    await cache.put(
      key,
      streamingResponse([bytes(full.slice(4))], {
        status: 206,
        headers: { "Content-Range": "bytes 4-9/10", "Content-Length": "6", etag: '"v1"' },
      }),
    );

    expect(fs.readFileSync(path.join(dir, key), "utf-8")).toBe(full);
    expect(fs.existsSync(path.join(dir, key + ".incomplete"))).toBe(false);
    expect(fs.existsSync(path.join(dir, key + ".incomplete.json"))).toBe(false);
    expect(await cache.getResumeInfo(key)).toBeUndefined();
  });

  it("restarts from scratch on a 200 even if a stale partial exists", async () => {
    const key = "restart.onnx";
    fs.writeFileSync(path.join(dir, key + ".incomplete"), "STALE-JUNK");
    fs.writeFileSync(path.join(dir, key + ".incomplete.json"), JSON.stringify({ etag: '"old"', total: 10 }));

    const content = "freshdata!";
    await cache.put(
      key,
      streamingResponse([bytes(content)], {
        status: 200,
        headers: { "Content-Length": String(content.length), etag: '"new"' },
      }),
    );

    expect(fs.readFileSync(path.join(dir, key), "utf-8")).toBe(content);
  });
});
