import { spawn } from "node:child_process";
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

/**
 * A `Response` whose body stalls until `release()` is called, so two writers can
 * be held in flight at the same time.
 * @param {string} content
 */
function stallingResponse(content) {
  let release;
  const gate = new Promise((resolve) => (release = resolve));
  let sent = false;
  const body = new ReadableStream({
    async pull(controller) {
      if (sent) {
        controller.close();
        return;
      }
      await gate;
      sent = true;
      controller.enqueue(bytes(content));
    },
  });
  return {
    response: new Response(body, { status: 200, headers: { "Content-Length": String(content.length), etag: '"v1"' } }),
    release: () => release(),
  };
}

const bytes = (str) => new TextEncoder().encode(str);

/** A pid that is guaranteed to be valid but no longer running. */
async function deadPid() {
  const child = spawn(process.execPath, ["-e", ""], { stdio: "ignore" });
  const pid = child.pid;
  await new Promise((resolve) => child.on("exit", resolve));
  return pid;
}

describe("FileCache resumable downloads", () => {
  let dir;
  let cache;

  const p = (...parts) => path.join(dir, ...parts);

  /** Seed an interrupted download: a partial plus its sidecar. */
  const seedPartial = (key, content, sidecar) => {
    fs.writeFileSync(p(key + ".incomplete"), content);
    fs.writeFileSync(p(key + ".incomplete.json"), JSON.stringify(sidecar));
  };

  /** Write a foreign lock file with an explicit owner and age. */
  const seedLock = (key, owner, ageMs = 0) => {
    const lockPath = p(key + ".incomplete.lock");
    fs.writeFileSync(lockPath, JSON.stringify(owner));
    const when = new Date(Date.now() - ageMs);
    fs.utimesSync(lockPath, when, when);
    return lockPath;
  };

  beforeEach(() => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), "tjs-filecache-"));
    cache = new FileCache(dir);
  });

  afterEach(() => {
    fs.rmSync(dir, { recursive: true, force: true });
  });

  describe("basic write paths", () => {
    it("reserveResume returns undefined when there is no partial", async () => {
      expect(await cache.reserveResume("model.onnx")).toBeUndefined();
      // A failed reservation must not leave a lock behind.
      expect(fs.existsSync(p("model.onnx.incomplete.lock"))).toBe(false);
    });

    it("writes a full 200 response and leaves no partial artifacts", async () => {
      const content = "hello world";
      await cache.put("model.onnx", streamingResponse([bytes(content)], { status: 200, headers: { "Content-Length": String(content.length), etag: '"abc"' } }));

      expect(fs.readFileSync(p("model.onnx"), "utf-8")).toBe(content);
      expect(fs.existsSync(p("model.onnx.incomplete"))).toBe(false);
      expect(fs.existsSync(p("model.onnx.incomplete.json"))).toBe(false);
      expect(fs.existsSync(p("model.onnx.incomplete.lock"))).toBe(false);
    });

    it("keeps the partial and sidecar when the body is truncated", async () => {
      const key = "big.onnx";
      // Server promises 10 bytes but only delivers 4.
      await expect(cache.put(key, streamingResponse([bytes("0123")], { status: 200, headers: { "Content-Length": "10", etag: '"v1"' } }))).rejects.toThrow(/Incomplete download/);

      // Final file must NOT be published; partial + sidecar remain for resume.
      expect(fs.existsSync(p(key))).toBe(false);
      expect(fs.readFileSync(p(key + ".incomplete"), "utf-8")).toBe("0123");
      expect(fs.existsSync(p(key + ".incomplete.lock"))).toBe(false);

      expect(await cache.reserveResume(key)).toEqual({ size: 4, etag: '"v1"', total: 10 });
      await cache.releaseResume(key);
    });

    it("appends a 206 partial-content response onto the existing partial", async () => {
      const key = "weights.onnx";
      const full = "0123456789";
      seedPartial(key, full.slice(0, 4), { etag: '"v1"', total: 10 });

      expect(await cache.reserveResume(key)).toEqual({ size: 4, etag: '"v1"', total: 10 });
      await cache.put(key, streamingResponse([bytes(full.slice(4))], { status: 206, headers: { "Content-Range": "bytes 4-9/10", "Content-Length": "6", etag: '"v1"' } }));

      expect(fs.readFileSync(p(key), "utf-8")).toBe(full);
      expect(fs.existsSync(p(key + ".incomplete"))).toBe(false);
      expect(fs.existsSync(p(key + ".incomplete.json"))).toBe(false);
      expect(fs.existsSync(p(key + ".incomplete.lock"))).toBe(false);
      expect(await cache.reserveResume(key)).toBeUndefined();
    });

    it("restarts from scratch on a 200 even if a stale partial exists", async () => {
      const key = "restart.onnx";
      seedPartial(key, "STALE-JUNK", { etag: '"old"', total: 10 });

      const content = "freshdata!";
      await cache.put(key, streamingResponse([bytes(content)], { status: 200, headers: { "Content-Length": String(content.length), etag: '"new"' } }));

      expect(fs.readFileSync(p(key), "utf-8")).toBe(content);
      expect(fs.existsSync(p(key + ".incomplete"))).toBe(false);
    });

    it("reports a null etag when the sidecar has no validator", async () => {
      const key = "novalidator.onnx";
      seedPartial(key, "0123", { etag: null, total: 10 });

      // The caller uses this to decide *not* to send `Range` at all.
      expect(await cache.reserveResume(key)).toEqual({ size: 4, etag: null, total: 10 });
      await cache.releaseResume(key);
    });
  });

  describe("partial responses are confined to the writer that owns the partial", () => {
    it("refuses a 206 when no resume was reserved", async () => {
      const key = "unreserved.onnx";
      seedPartial(key, "0123", { etag: '"v1"', total: 10 });

      // No `reserveResume` call: this writer does not own the partial.
      await expect(cache.put(key, streamingResponse([bytes("456789")], { status: 206, headers: { "Content-Range": "bytes 4-9/10", "Content-Length": "6" } }))).rejects.toThrow(/without a resume reservation/);

      // Crucially, the trailing bytes were not published as the whole file.
      expect(fs.existsSync(p(key))).toBe(false);
      // The other writer's partial is untouched.
      expect(fs.readFileSync(p(key + ".incomplete"), "utf-8")).toBe("0123");
    });

    it("refuses a 206 that arrives while another writer holds the key", async () => {
      const key = "contended.onnx";
      seedPartial(key, "0123", { etag: '"v1"', total: 10 });

      // A second FileCache instance stands in for another process holding the key.
      const other = new FileCache(dir);
      expect(await other.reserveResume(key)).toEqual({ size: 4, etag: '"v1"', total: 10 });

      // This writer could not reserve, so it must never produce a 206...
      expect(await cache.reserveResume(key)).toBeUndefined();
      // ...and if one arrives anyway it is refused rather than written to a
      // unique temp file and renamed over the final path.
      await expect(cache.put(key, streamingResponse([bytes("456789")], { status: 206, headers: { "Content-Range": "bytes 4-9/10", "Content-Length": "6" } }))).rejects.toThrow(/without a resume reservation/);

      expect(fs.existsSync(p(key))).toBe(false);
      expect(fs.readFileSync(p(key + ".incomplete"), "utf-8")).toBe("0123");
      await other.releaseResume(key);
    });
  });

  describe("Content-Range validation", () => {
    const key = "range.onnx";
    // Partial holds bytes 0-3 of a 10-byte file.
    const reserve = async () => {
      seedPartial(key, "0123", { etag: '"v1"', total: 10 });
      expect(await cache.reserveResume(key)).toBeDefined();
    };

    it.each([
      ["a gap (server resumed past the partial)", { "Content-Range": "bytes 6-9/10", "Content-Length": "4" }, "456789", /gap/],
      ["an overlap (server resumed before the partial)", { "Content-Range": "bytes 2-9/10", "Content-Length": "8" }, "23456789", /overlap/],
      ["a changed total size", { "Content-Range": "bytes 4-9/20", "Content-Length": "6" }, "456789", /total size changed/],
      ["an unsatisfied range", { "Content-Range": "bytes */10" }, "", /unusable Content-Range/],
      ["a suffix range", { "Content-Range": "bytes -6/10", "Content-Length": "6" }, "456789", /unusable Content-Range/],
      ["an unknown total", { "Content-Range": "bytes 4-9/*", "Content-Length": "6" }, "456789", /unusable Content-Range/],
      ["a missing Content-Range", {}, "456789", /unusable Content-Range/],
      ["an end before the start", { "Content-Range": "bytes 4-2/10" }, "", /inconsistent Content-Range/],
      ["an end past the total", { "Content-Range": "bytes 4-99/10" }, "", /inconsistent Content-Range/],
      ["a Content-Length that disagrees with the span", { "Content-Range": "bytes 4-9/10", "Content-Length": "99" }, "456789", /does not match Content-Range span/],
    ])("rejects %s", async (_label, headers, body, expected) => {
      await reserve();

      await expect(cache.put(key, streamingResponse([bytes(body)], { status: 206, headers }))).rejects.toThrow(expected);

      // Nothing is published, and the suspect partial is dropped so the next
      // attempt starts clean instead of retrying onto bad data forever.
      expect(fs.existsSync(p(key))).toBe(false);
      expect(fs.existsSync(p(key + ".incomplete"))).toBe(false);
      expect(fs.existsSync(p(key + ".incomplete.lock"))).toBe(false);
      expect(await cache.reserveResume(key)).toBeUndefined();
    });

    it("accepts a valid range whose Content-Length is absent", async () => {
      await reserve();
      await cache.put(key, streamingResponse([bytes("456789")], { status: 206, headers: { "Content-Range": "bytes 4-9/10" } }));
      expect(fs.readFileSync(p(key), "utf-8")).toBe("0123456789");
    });
  });

  describe("concurrent writers", () => {
    it("both writers finish, and the file is complete exactly once", async () => {
      const key = "parallel.onnx";
      const content = "0123456789";
      const a = stallingResponse(content);
      const b = stallingResponse(content);

      const pending = [cache.put(key, a.response), cache.put(key, b.response)];
      // Both are now past lock acquisition: one holds the deterministic partial,
      // the other fell back to a unique temp file.
      a.release();
      b.release();
      await Promise.all(pending);

      expect(fs.readFileSync(p(key), "utf-8")).toBe(content);
      // No partial, sidecar, lock, or temp file survives either writer.
      expect(fs.readdirSync(dir)).toEqual([key]);
    });

    it("a failing writer does not strand the key or the other writer", async () => {
      const key = "parallel-fail.onnx";
      const content = "0123456789";
      const good = stallingResponse(content);

      const failing = cache.put(key, streamingResponse([bytes("012")], { status: 200, headers: { "Content-Length": "10", etag: '"v1"' } }));
      await expect(failing).rejects.toThrow(/Incomplete download/);

      good.release();
      await cache.put(key, good.response);

      expect(fs.readFileSync(p(key), "utf-8")).toBe(content);
      expect(fs.existsSync(p(key + ".incomplete.lock"))).toBe(false);
    });

    it("releaseResume hands the key back without writing", async () => {
      const key = "released.onnx";
      seedPartial(key, "0123", { etag: '"v1"', total: 10 });

      expect(await cache.reserveResume(key)).toBeDefined();
      const other = new FileCache(dir);
      expect(await other.reserveResume(key)).toBeUndefined();

      await cache.releaseResume(key);
      expect(fs.existsSync(p(key + ".incomplete.lock"))).toBe(false);
      expect(await other.reserveResume(key)).toEqual({ size: 4, etag: '"v1"', total: 10 });
      await other.releaseResume(key);
    });

    it("releaseResume is a no-op when nothing is held", async () => {
      await expect(cache.releaseResume("never-reserved.onnx")).resolves.toBeUndefined();
    });
  });

  describe("completion requires the exact declared size", () => {
    it("refuses to publish an oversized body and discards the partial", async () => {
      const key = "oversized.onnx";
      // Server declares 4 bytes and sends 10. Appending the surplus and
      // renaming would publish a file that is not the one we asked for.
      await expect(cache.put(key, streamingResponse([bytes("0123456789")], { status: 200, headers: { "Content-Length": "4", etag: '"v1"' } }))).rejects.toThrow(/Oversized download/);

      expect(fs.existsSync(p(key))).toBe(false);
      // An oversized partial can never be resumed onto (it already runs past
      // `total`), so it is dropped rather than kept like a truncated one.
      expect(fs.existsSync(p(key + ".incomplete"))).toBe(false);
      expect(fs.existsSync(p(key + ".incomplete.json"))).toBe(false);
      expect(fs.existsSync(p(key + ".incomplete.lock"))).toBe(false);
      expect(await cache.reserveResume(key)).toBeUndefined();
    });

    it("refuses to publish a 206 that overshoots the total", async () => {
      const key = "oversized-range.onnx";
      seedPartial(key, "0123", { etag: '"v1"', total: 10 });
      expect(await cache.reserveResume(key)).toBeDefined();

      // Content-Range is internally consistent and lines up with the partial,
      // but the body carries more bytes than it promised.
      await expect(cache.put(key, streamingResponse([bytes("456789EXTRA")], { status: 206, headers: { "Content-Range": "bytes 4-9/10", "Content-Length": "6" } }))).rejects.toThrow(/Oversized download/);

      expect(fs.existsSync(p(key))).toBe(false);
      expect(fs.existsSync(p(key + ".incomplete"))).toBe(false);
      expect(fs.existsSync(p(key + ".incomplete.lock"))).toBe(false);
    });

    it("still keeps the partial when the body falls short", async () => {
      const key = "short.onnx";
      await expect(cache.put(key, streamingResponse([bytes("012")], { status: 200, headers: { "Content-Length": "10", etag: '"v1"' } }))).rejects.toThrow(/Incomplete download/);

      expect(fs.readFileSync(p(key + ".incomplete"), "utf-8")).toBe("012");
      expect(await cache.reserveResume(key)).toEqual({ size: 3, etag: '"v1"', total: 10 });
      await cache.releaseResume(key);
    });
  });

  describe("the output stream is closed before the key is handed on", () => {
    /**
     * Assert nothing is still holding `target` open.
     *
     * On Linux the leaked descriptor is directly observable via `/proc/self/fd`,
     * which is the assertion that actually bites — POSIX happily unlinks a file
     * that is still open, so the removable check alone would pass either way.
     * Elsewhere (notably Windows, where an open handle does block the unlink)
     * the removable check is what is left.
     */
    const expectNotHeldOpen = (target) => {
      const fdDir = "/proc/self/fd";
      if (fs.existsSync(fdDir)) {
        const holding = fs.readdirSync(fdDir).filter((fd) => {
          try {
            return fs.readlinkSync(path.join(fdDir, fd)) === target;
          } catch {
            // The descriptor used to walk the directory may already be gone.
            return false;
          }
        });
        expect(holding).toEqual([]);
      }
      expect(() => fs.unlinkSync(target)).not.toThrow();
    };

    it("closes the stream when the reader errors mid-body", async () => {
      const key = "reader-error.onnx";
      const body = new ReadableStream({
        pull(controller) {
          controller.enqueue(bytes("0123"));
          controller.error(new Error("connection reset"));
        },
      });
      const response = new Response(body, { status: 200, headers: { "Content-Length": "10", etag: '"v1"' } });

      await expect(cache.put(key, response)).rejects.toThrow(/connection reset/);

      // The lock is gone, so another writer may take this key immediately —
      // it must not find a partial we are still writing into.
      expect(fs.existsSync(p(key + ".incomplete.lock"))).toBe(false);
      expect(fs.existsSync(p(key))).toBe(false);
      expectNotHeldOpen(p(key + ".incomplete"));
    });

    it("closes the stream when the progress callback throws", async () => {
      const key = "callback-error.onnx";
      const response = streamingResponse([bytes("0123"), bytes("456789")], { status: 200, headers: { "Content-Length": "10", etag: '"v1"' } });

      await expect(
        cache.put(key, response, () => {
          throw new Error("callback blew up");
        }),
      ).rejects.toThrow(/callback blew up/);

      expect(fs.existsSync(p(key + ".incomplete.lock"))).toBe(false);
      expect(fs.existsSync(p(key))).toBe(false);
      expectNotHeldOpen(p(key + ".incomplete"));
    });

    it("closes the stream when the writer errors", async () => {
      const key = "writer-error.onnx";
      // Make the partial path a directory: opening it for writing fails, and
      // the failure surfaces asynchronously through the stream.
      fs.mkdirSync(p(key + ".incomplete"));

      await expect(cache.put(key, streamingResponse([bytes("0123")], { status: 200, headers: { "Content-Length": "10" } }))).rejects.toThrow();

      // The error must not escape as an unhandled 'error' event, and the key
      // must be released for the next writer.
      expect(fs.existsSync(p(key + ".incomplete.lock"))).toBe(false);
      expect(fs.existsSync(p(key))).toBe(false);
    });

    it("cleans up the temp file when the fallback path fails", async () => {
      const key = "fallback-error.onnx";
      seedPartial(key, "0123", { etag: '"v1"', total: 10 });
      // Hold the key so `put` takes the unique-temp fallback.
      seedLock(key, { instance: "other", pid: process.pid, startedAt: Date.now() }, 0);

      await expect(cache.put(key, streamingResponse([bytes("012")], { status: 200, headers: { "Content-Length": "10" } }))).rejects.toThrow(/Incomplete download/);

      // No `.tmp.*` left behind, and the other writer's artifacts are untouched.
      expect(fs.readdirSync(dir).filter((f) => f.includes(".tmp."))).toEqual([]);
      expect(fs.existsSync(p(key))).toBe(false);
      expect(fs.readFileSync(p(key + ".incomplete"), "utf-8")).toBe("0123");
    });
  });

  describe("delete removes every artifact of an entry", () => {
    it("removes the completed file", async () => {
      const key = "done.onnx";
      fs.writeFileSync(p(key), "0123456789");

      expect(await cache.delete(key)).toBe(true);
      expect(fs.readdirSync(dir)).toEqual([]);
    });

    it("removes a partial, its sidecar, and a stale lock", async () => {
      const key = "half.onnx";
      seedPartial(key, "0123", { etag: '"v1"', total: 10 });
      seedLock(key, { instance: "other", pid: await deadPid(), startedAt: Date.now() }, 10 * 60 * 1000);

      expect(await cache.delete(key)).toBe(true);
      expect(fs.readdirSync(dir)).toEqual([]);
      // Nothing is left for a later request to resume onto.
      expect(await cache.reserveResume(key)).toBeUndefined();
    });

    it("removes the partial alongside the completed file", async () => {
      const key = "both.onnx";
      fs.writeFileSync(p(key), "0123456789");
      seedPartial(key, "0123", { etag: '"v2"', total: 10 });

      expect(await cache.delete(key)).toBe(true);
      expect(fs.readdirSync(dir)).toEqual([]);
    });

    it("releases a reservation this instance holds", async () => {
      const key = "reserved.onnx";
      seedPartial(key, "0123", { etag: '"v1"', total: 10 });
      expect(await cache.reserveResume(key)).toBeDefined();

      expect(await cache.delete(key)).toBe(true);
      // The lock we owned is gone, not merely unlinked out from under a live
      // heartbeat — a second delete finds nothing at all.
      expect(fs.readdirSync(dir)).toEqual([]);
      expect(await cache.delete(key)).toBe(false);
    });

    it("returns false when there is nothing cached", async () => {
      expect(await cache.delete("absent.onnx")).toBe(false);
    });

    it("does not report a bare leftover lock as a deleted entry", async () => {
      // `clear_cache` falls back to a second key when `delete` returns false,
      // so a stray lock must not masquerade as a real entry...
      const key = "lock-only.onnx";
      seedLock(key, { instance: "other", pid: process.pid, startedAt: Date.now() }, 0);

      expect(await cache.delete(key)).toBe(false);
      // ...but it is still swept up.
      expect(fs.readdirSync(dir)).toEqual([]);
    });
  });

  describe("lock ownership", () => {
    const key = "locked.onnx";

    beforeEach(() => seedPartial(key, "0123", { etag: '"v1"', total: 10 }));

    it("does not steal a lock that is being refreshed", async () => {
      seedLock(key, { instance: "other", pid: await deadPid(), startedAt: Date.now() }, 0);
      // Recently touched: a writer is actively downloading, dead-looking pid or
      // not. We must wait rather than write concurrently.
      expect(await cache.reserveResume(key)).toBeUndefined();
      expect(fs.existsSync(p(key + ".incomplete.lock"))).toBe(true);
    });

    it("does not steal a stale-looking lock whose owner is still alive", async () => {
      // This is the slow-download case: the lock has not been refreshed for
      // longer than the stale window, but its owner process still exists.
      seedLock(key, { instance: "other", pid: process.pid, startedAt: Date.now() }, 10 * 60 * 1000);
      expect(await cache.reserveResume(key)).toBeUndefined();
      expect(fs.existsSync(p(key + ".incomplete.lock"))).toBe(true);
    });

    it("steals a stale lock whose owner is gone", async () => {
      seedLock(key, { instance: "other", pid: await deadPid(), startedAt: Date.now() }, 10 * 60 * 1000);
      expect(await cache.reserveResume(key)).toEqual({ size: 4, etag: '"v1"', total: 10 });
      await cache.releaseResume(key);
    });

    it("steals a lock left untouched past the abandoned ceiling", async () => {
      // Owner pid resolves (it is us), but nothing has refreshed the lock for
      // an hour, so it cannot be an in-progress download.
      seedLock(key, { instance: "other", pid: process.pid, startedAt: Date.now() }, 2 * 60 * 60 * 1000);
      expect(await cache.reserveResume(key)).toEqual({ size: 4, etag: '"v1"', total: 10 });
      await cache.releaseResume(key);
    });

    it("steals an unreadable lock", async () => {
      // A writer that died midway through creating its lock file.
      seedLock(key, "{not-json", 10 * 60 * 1000);
      expect(await cache.reserveResume(key)).toEqual({ size: 4, etag: '"v1"', total: 10 });
      await cache.releaseResume(key);
    });

    it("falls back to the unique-temp path for a 200 while the key is locked", async () => {
      seedLock(key, { instance: "other", pid: process.pid, startedAt: Date.now() }, 0);

      const content = "0123456789";
      await cache.put(key, streamingResponse([bytes(content)], { status: 200, headers: { "Content-Length": "10", etag: '"v1"' } }));

      // The full download still lands correctly...
      expect(fs.readFileSync(p(key), "utf-8")).toBe(content);
      // ...without disturbing the other writer's partial or lock.
      expect(fs.readFileSync(p(key + ".incomplete"), "utf-8")).toBe("0123");
      expect(fs.existsSync(p(key + ".incomplete.lock"))).toBe(true);
    });
  });
});
