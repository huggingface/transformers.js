import assert from "node:assert/strict";
import fs from "node:fs";
import { fork } from "node:child_process";
import { once } from "node:events";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { FileCache } from "../../../src/utils/cache/FileCache.js";
import { AssetLoadingContext } from "../../../src/utils/hub/AssetLoadingContext.js";
import { getModelFile } from "../../../src/utils/hub.js";
import { env } from "../../../src/env.js";

const [, , mode, directory, key, schedule] = process.argv;
const pause = async (event) => {
  const resume = once(process, "message");
  process.send(event);
  await resume;
};

if (mode === "write") {
  const cache = new FileCache(directory);
  const { markerPath, filePath } = cache.resolve(key);
  const rename = fs.promises.rename.bind(fs.promises);
  const writeFile = fs.promises.writeFile.bind(fs.promises);
  const open = fs.promises.open.bind(fs.promises);
  let attempted = false;
  fs.promises.open = async (path, ...args) => {
    if (schedule === "direct" && String(path) === `${markerPath}.lock` && !attempted) {
      attempted = true;
      process.send("lock-attempt");
    }
    return open(path, ...args);
  };
  fs.promises.rename = async (source, target) => {
    if (schedule === "marker" && target === markerPath) await pause("marker-ready");
    if (schedule === "direct" && target === filePath) process.send("publishing");
    return rename(source, target);
  };
  // Baseline adapter only: expose the live write's actual truncate/write boundary. Corrected
  // writers never take this path; their barrier is before the staged marker's atomic rename.
  fs.promises.writeFile = async (path, data, ...args) => {
    if (schedule === "marker" && path === markerPath) {
      const handle = await open(path, "w");
      try {
        await pause("marker-ready");
        await handle.writeFile(data);
      } finally {
        await handle.close();
      }
    } else return writeFile(path, data, ...args);
  };
  await cache.put(key, new Response("replacement"));
  process.disconnect();
} else if (mode === "delete") {
  const cache = new FileCache(directory);
  const readVersion = cache.readVersion.bind(cache);
  cache.readVersion = async (...args) => {
    const version = await readVersion(...args);
    await pause("checked");
    return version;
  };
  assert.equal(await cache.delete(key), true);
  process.disconnect();
} else {
  const root = await fs.promises.mkdtemp(join(tmpdir(), "cache-mutation-"));
  const children = [];
  function job(args) {
    const child = fork(new URL(import.meta.url), args, { stdio: ["ignore", "inherit", "inherit", "ipc"] });
    children.push(child);
    const messages = [];
    const waiters = [];
    child.on("message", (message) => (waiters.length ? waiters.shift()(message) : messages.push(message)));
    const exited = once(child, "exit").then(([code, signal]) => assert.equal(code, 0, `child failed: ${signal}`));
    return { child, exited, next: () => (messages.length ? Promise.resolve(messages.shift()) : new Promise((resolve) => waiters.push(resolve))) };
  }
  try {
    const cache = new FileCache(root);
    const oldKey = "org/model/file";
    const current = oldKey + "?__transformersjs_cache_version=1";
    if (mode === "direct") {
      await cache.put(oldKey, new Response("old"));
      const deletion = job(["delete", root, oldKey]);
      assert.equal(await deletion.next(), "checked");
      const writer = job(["write", root, current, "direct"]);
      const event = await writer.next();
      if (event === "lock-attempt") {
        // Do not wait for a writer blocked behind the paused deleter. Its payload is still staged.
        assert.equal(await fs.promises.readFile(cache.resolve(current).filePath, "utf8"), "old");
        deletion.child.send("continue");
        await Promise.all([deletion.exited, writer.exited]);
      } else {
        assert.equal(event, "publishing");
        await writer.exited;
        deletion.child.send("continue");
        await deletion.exited;
      }
      assert.equal(await (await cache.match(current))?.text(), "replacement", "replacement must survive obsolete deletion");
      assert.equal(await fs.promises.readFile(cache.resolve(current).markerPath, "utf8"), "1");
    } else if (mode === "loader") {
      Object.assign(env, { useCustomCache: true, customCache: cache, useFSCache: false, useBrowserCache: false });
      const options = { env: { allowLocalModels: false, fetch: async () => new Response("loader-result") } };
      const resource = new AssetLoadingContext(options).resolve("org/loader", "tokenizer_config.json", cache);
      const put = cache.put.bind(cache);
      const match = cache.match.bind(cache);
      let stored = false;
      let writer;
      cache.put = async (...args) => {
        await put(...args);
        stored = true;
      };
      cache.match = async (request) => {
        const result = await match(request);
        if (stored && !writer && result && request === resource.proposedCacheKey) {
          writer = job(["write", root, request, "marker"]);
          assert.equal(await writer.next(), "marker-ready");
        }
        return result;
      };
      assert.equal(new TextDecoder().decode(await getModelFile("org/loader", "tokenizer_config.json", true, options)), "loader-result");
      assert.ok(writer, "actual loader cleanup must reach its current-entry guard");
      writer.child.send("continue");
      await writer.exited;
      assert.equal(await (await match(resource.proposedCacheKey))?.text(), "replacement", "replacement must survive loader cleanup");
      assert.equal(await fs.promises.readFile(cache.resolve(resource.proposedCacheKey).markerPath, "utf8"), "1");
      const offline = await getModelFile("org/loader", "tokenizer_config.json", true, { local_files_only: true, env: { allowRemoteModels: false } });
      assert.equal(new TextDecoder().decode(offline), "replacement");
    } else throw new Error(`Unknown mode ${mode}`);
    const files = await fs.promises.readdir(root, { recursive: true });
    assert.equal(
      files.some((name) => name.includes(".tmp.") || name.endsWith(".lock")),
      false,
      "no mutation debris",
    );
  } finally {
    for (const child of children) if (child.exitCode === null) child.kill("SIGKILL");
    await fs.promises.rm(root, { recursive: true, force: true });
  }
}
