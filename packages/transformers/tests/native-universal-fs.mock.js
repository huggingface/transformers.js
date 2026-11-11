// Mock for `native-universal-fs`, backed by the real Node filesystem.
//
// The React Native code paths in src/ call this module with `file://` URIs (see
// FileResponse, FileCache, io.js). Mapping the module to `node:fs/promises` -- as the
// jest config used to -- leaves `exists`/`moveFile`/`DocumentDirectoryPath` undefined, so
// those branches could never be exercised by a test. This mock implements the subset of
// the API that src/ actually uses, which is what lets tests/react_native.test.js run.
import fs from "node:fs";
import fsp from "node:fs/promises";
import os from "node:os";
import path from "node:path";

/** Strip the `file://` prefix that the RN code paths pass in. */
const toPath = (p) => String(p).replace(/^file:\/\//, "");

export const DocumentDirectoryPath = path.join(os.tmpdir(), "transformers-js-rn-documents");
export const CachesDirectoryPath = path.join(os.tmpdir(), "transformers-js-rn-caches");
export const TemporaryDirectoryPath = os.tmpdir();

export async function exists(p) {
  return fs.existsSync(toPath(p));
}

export async function stat(p) {
  const target = toPath(p);
  const s = await fsp.stat(target);
  return {
    mtime: s.mtime,
    ctime: s.ctime,
    name: path.basename(target),
    path: target,
    size: s.size,
    isDirectory: () => s.isDirectory(),
    isFile: () => s.isFile(),
  };
}

export async function readFile(p, encoding = "utf8") {
  return await fsp.readFile(toPath(p), encoding);
}

export async function writeFile(p, data, encoding = "utf8") {
  return await fsp.writeFile(toPath(p), data, encoding);
}

export async function appendFile(p, data, encoding = "utf8") {
  return await fsp.appendFile(toPath(p), data, encoding);
}

export async function mkdir(p) {
  await fsp.mkdir(toPath(p), { recursive: true });
}

export async function unlink(p) {
  await fsp.unlink(toPath(p));
}

export async function rename(from, to) {
  await fsp.rename(toPath(from), toPath(to));
}

export async function moveFile(from, to) {
  await fsp.rename(toPath(from), toPath(to));
}

export async function copyFile(from, to) {
  await fsp.copyFile(toPath(from), toPath(to));
}

export async function readdir(p) {
  return await fsp.readdir(toPath(p));
}

export const readDir = readdir;

/**
 * Calls recorded for assertions. Tests use this to check that model downloads went
 * through the native streaming path instead of being pulled into the JS heap.
 * @type {{ downloadFile: {fromUrl: string, toFile: string}[] }}
 */
export const __calls = { downloadFile: [] };

/**
 * Streaming download straight to disk, mirroring the native implementation's contract:
 * returns `{ jobId, promise }` rather than a plain promise.
 */
export function downloadFile({ fromUrl, toFile, headers, begin, progress }) {
  __calls.downloadFile.push({ fromUrl: String(fromUrl), toFile: String(toFile) });
  const jobId = downloadFile._nextJobId++;
  const run = async () => {
    const response = await fetch(String(fromUrl), { headers });
    const contentLength = parseInt(response.headers.get("content-length") ?? "0", 10);
    begin?.({ jobId, statusCode: response.status, contentLength, headers: response.headers });

    await fsp.mkdir(path.dirname(toPath(toFile)), { recursive: true });
    const handle = await fsp.open(toPath(toFile), "w");
    let bytesWritten = 0;
    try {
      const reader = response.body.getReader();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        await handle.write(value);
        bytesWritten += value.length;
        progress?.({ jobId, contentLength, bytesWritten });
      }
    } finally {
      await handle.close();
    }
    return { jobId, bytesWritten, statusCode: response.status };
  };
  return { jobId, promise: run() };
}
downloadFile._nextJobId = 1;

export async function stopDownload() {}
