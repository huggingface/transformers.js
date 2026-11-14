// Standalone evidence, not a passing regression: demonstrates the unresolved F20 races.
// Run only through the approved no-network sandbox.
import assert from "node:assert/strict";
import fs from "node:fs";
import { fork } from "node:child_process";
import { once } from "node:events";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { FileCache } from "../../src/utils/cache/FileCache.js";
import { AssetLoadingContext } from "../../src/utils/hub/AssetLoadingContext.js";
import { getModelFile } from "../../src/utils/hub.js";
import { env } from "../../src/env.js";

if (process.argv[2] === "writer") {
  const [, , , directory, key] = process.argv;
  const cache = new FileCache(directory);
  const writeFile = fs.promises.writeFile;
  fs.promises.writeFile = async (path, data, ...args) => {
    if (String(path).endsWith(".transformersjs_cache_version")) {
      const handle = await fs.promises.open(path, "w");
      process.send("truncated");
      await once(process, "message");
      await handle.writeFile(data);
      await handle.close();
    } else await writeFile(path, data, ...args);
  };
  await cache.put(key, new Response("replacement"));
  process.send("done");
  process.disconnect();
} else if (process.argv[2] === "replace") {
  await new FileCache(process.argv[3]).put(process.argv[4], new Response("replacement"));
} else {
  const directory = await fs.promises.mkdtemp(join(tmpdir(), "f20-barriers-"));
  try {
    // API-level schedule: pause after delete's version check, then replace in another process.
    const cache = new FileCache(directory);
    await cache.put("control/file", new Response("offline-readable"));
    assert.equal(await (await cache.match("control/file")).text(), "offline-readable");
    assert.equal(await cache.delete("control/file"), true);
    assert.equal(await cache.match("control/file"), undefined);
    console.log("F20 controls: ordinary offline read and explicit deletion work");
    const oldKey = "direct/file";
    const newKey = oldKey + "?__transformersjs_cache_version=1";
    await cache.put(oldKey, new Response("old"));
    const readVersion = cache.readVersion.bind(cache);
    cache.readVersion = async (...args) => {
      const result = await readVersion(...args);
      const child = fork(new URL(import.meta.url), ["replace", directory, newKey], { stdio: ["ignore", "inherit", "inherit", "ipc"] });
      const [code] = await once(child, "exit");
      assert.equal(code, 0);
      const resolved = cache.resolve(newKey);
      assert.equal(await fs.promises.readFile(resolved.filePath, "utf8"), "replacement");
      assert.equal(await fs.promises.readFile(resolved.markerPath, "utf8"), "1");
      return result;
    };
    assert.equal(await cache.delete(oldKey), true);
    assert.equal(await cache.match(newKey), undefined);
    console.log("F20 direct API: cross-process replacement after version check was deleted (loader reachability not claimed)");

    // Loader-level marker schedule: hold cleanup's current-entry guard, then truncate the marker
    // in a separate writer and run obsolete cleanup before permitting marker publication.
    const loaderCache = new FileCache(join(directory, "loader"));
    const settings = { allowLocalModels: false, allowRemoteModels: true, useFSCache: false, useBrowserCache: false, useCustomCache: true, customCache: loaderCache, fetch: async () => new Response("loader-result") };
    Object.assign(env, { useFSCache: false, useBrowserCache: false, useCustomCache: true, customCache: loaderCache });
    const context = new AssetLoadingContext({ env: settings });
    const resource = context.resolve("org/model", "tokenizer_config.json", loaderCache);
    let writer;
    let publications = 0;
    const put = loaderCache.put.bind(loaderCache);
    loaderCache.put = async (...args) => {
      await put(...args);
      ++publications;
    };
    const match = loaderCache.match.bind(loaderCache);
    loaderCache.match = async (key) => {
      const result = await match(key);
      if (publications && writer === undefined && result && key === resource.proposedCacheKey) {
        writer = fork(new URL(import.meta.url), ["writer", loaderCache.path, key], { stdio: ["ignore", "inherit", "inherit", "ipc"] });
        const [message] = await once(writer, "message");
        assert.equal(message, "truncated");
      }
      return result;
    };
    const loaded = await getModelFile("org/model", "tokenizer_config.json", true, { env: settings });
    assert.equal(new TextDecoder().decode(loaded), "loader-result");
    assert.ok(writer, "loader cleanup reached the controlled current-entry guard");
    const done = once(writer, "exit");
    writer.send("continue");
    assert.equal((await done)[0], 0);
    assert.equal(await match(resource.proposedCacheKey), undefined);
    console.log("F20 loader: successful load returned; stale cleanup deleted separate writer's replacement during marker truncation");
    const files = await fs.promises.readdir(directory, { recursive: true });
    assert.equal(
      files.some((name) => name.includes(".tmp.")),
      false,
    );
    console.log("No temporary payload debris; these barriers establish reachability, not real-world race frequency.");
  } finally {
    await fs.promises.rm(directory, { recursive: true, force: true });
  }
}
