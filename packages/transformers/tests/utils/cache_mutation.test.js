import { jest } from "@jest/globals";
import fs from "node:fs";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { FileCache } from "../../src/utils/cache/FileCache.js";

describe("filesystem cache mutations", () => {
  it("preserves a separate writer's replacement during loader cleanup", () => {
    const result = spawnSync(process.execPath, [fileURLToPath(new URL("./fixtures/cache_mutation_process.mjs", import.meta.url)), "loader"], { encoding: "utf8", timeout: 15000 });
    expect({ status: result.status, error: result.error?.message, stdout: result.stdout, stderr: result.stderr }).toEqual({ status: 0, error: undefined, stdout: "", stderr: "" });
  }, 20000);

  let directory;
  let cache;
  beforeEach(async () => {
    directory = await fs.promises.mkdtemp(join(tmpdir(), "cache-controls-"));
    cache = new FileCache(directory);
  });
  afterEach(async () => {
    jest.restoreAllMocks();
    await fs.promises.rm(directory, { recursive: true, force: true });
  });
  const key = "org/model/file?__transformersjs_cache_version=1";
  const debris = async () => (await fs.promises.readdir(directory, { recursive: true })).filter((name) => name.includes(".tmp."));

  it("keeps the live marker intact until an atomic replacement", async () => {
    await cache.put(key, new Response("old"));
    const { markerPath } = cache.resolve(key);
    const write = fs.promises.writeFile.bind(fs.promises);
    const rename = fs.promises.rename.bind(fs.promises);
    let observed = false;
    jest.spyOn(fs.promises, "writeFile").mockImplementation(async (path, ...args) => {
      expect(path).not.toBe(markerPath);
      return write(path, ...args);
    });
    jest.spyOn(fs.promises, "rename").mockImplementation(async (source, target) => {
      if (target === markerPath) {
        observed = true;
        expect(await fs.promises.readFile(markerPath, "utf8")).toBe("1");
      }
      return rename(source, target);
    });
    await cache.put(key, new Response("replacement"));
    expect(observed).toBe(true);
    expect(await debris()).toEqual([]);
  });

  it("cleans both staged files after publication failure", async () => {
    const rename = fs.promises.rename.bind(fs.promises);
    jest.spyOn(fs.promises, "rename").mockImplementation(async (source, target) => {
      if (target === cache.resolve(key).markerPath) throw new Error("marker rename failed");
      return rename(source, target);
    });
    await expect(cache.put(key, new Response("failed"))).rejects.toThrow("marker rename failed");
    expect(await debris()).toEqual([]);
    jest.restoreAllMocks();
    await cache.put(key, new Response("retry"));
    expect(await (await cache.match(key)).text()).toBe("retry");
  });

  it("preserves legacy reads, harmless misses and ordinary conditional deletion", async () => {
    await cache.put("legacy/file", new Response("legacy"));
    expect(await (await cache.match("legacy/file")).text()).toBe("legacy");
    expect(await cache.delete("absent/file")).toBe(false);
    await cache.put(key, new Response("current"));
    expect(await cache.delete("org/model/file")).toBe(false);
    expect(await (await cache.match(key)).text()).toBe("current");
    expect(await cache.delete(key)).toBe(true);
    expect(await cache.match(key)).toBeUndefined();
    expect(await debris()).toEqual([]);
  });

  it("reports a failed deletion when unlink fails", async () => {
    await cache.put(key, new Response("current"));
    const unlink = fs.promises.unlink.bind(fs.promises);
    jest.spyOn(fs.promises, "unlink").mockImplementation(async (path) => {
      if (path === cache.resolve(key).filePath) throw new Error("unlink denied");
      return unlink(path);
    });
    expect(await cache.delete(key)).toBe(false);
    expect(await debris()).toEqual([]);
    jest.restoreAllMocks();
    expect(await cache.delete(key)).toBe(true);
  });

  it("cleans a failed staged stream without leaving temporary files", async () => {
    const body = new ReadableStream({
      start(controller) {
        controller.enqueue(new Uint8Array([1]));
        controller.error(new Error("stream failed"));
      },
    });
    await expect(cache.put(key, new Response(body))).rejects.toThrow("stream failed");
    expect(await debris()).toEqual([]);
    await cache.put(key, new Response("retry"));
    expect(await (await cache.match(key)).text()).toBe("retry");
  });

  it("keeps cache reads usable when the directory is read-only", async () => {
    await cache.put(key, new Response("read-only"));
    const parent = dirname(cache.resolve(key).filePath);
    await fs.promises.chmod(parent, 0o555);
    try {
      expect(await (await cache.match(key)).text()).toBe("read-only");
      expect(await cache.delete(key)).toBe(false);
      await expect(cache.put(key, new Response("unwritable"))).rejects.toThrow();
      expect(await (await cache.match(key)).text()).toBe("read-only");
    } finally {
      await fs.promises.chmod(parent, 0o755);
    }
    expect(await debris()).toEqual([]);
  });
});
