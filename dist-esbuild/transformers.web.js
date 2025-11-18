var __defProp = Object.defineProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};

// src/env.js
import fs from "fs";
import path from "path";
import url from "url";
var VERSION = "3.7.6";
var IS_BROWSER_ENV = typeof window !== "undefined" && typeof window.document !== "undefined";
var IS_WEBWORKER_ENV = typeof self !== "undefined" && ["DedicatedWorkerGlobalScope", "ServiceWorkerGlobalScope", "SharedWorkerGlobalScope"].includes(self.constructor?.name);
var IS_WEB_CACHE_AVAILABLE = typeof self !== "undefined" && "caches" in self;
var IS_WEBGPU_AVAILABLE = typeof navigator !== "undefined" && "gpu" in navigator;
var IS_WEBNN_AVAILABLE = typeof navigator !== "undefined" && "ml" in navigator;
var IS_PROCESS_AVAILABLE = typeof process !== "undefined";
var IS_NODE_ENV = IS_PROCESS_AVAILABLE && process?.release?.name === "node";
var IS_FS_AVAILABLE = !isEmpty(fs);
var IS_PATH_AVAILABLE = !isEmpty(path);
var IS_DENO_RUNTIME = typeof globalThis.Deno !== "undefined";
var IS_BUN_RUNTIME = typeof globalThis.Bun !== "undefined";
var apis = Object.freeze({
  /** Whether we are running in a browser environment (and not a web worker) */
  IS_BROWSER_ENV,
  /** Whether we are running in a web worker environment */
  IS_WEBWORKER_ENV,
  /** Whether the Cache API is available */
  IS_WEB_CACHE_AVAILABLE,
  /** Whether the WebGPU API is available */
  IS_WEBGPU_AVAILABLE,
  /** Whether the WebNN API is available */
  IS_WEBNN_AVAILABLE,
  /** Whether the Node.js process API is available */
  IS_PROCESS_AVAILABLE,
  /** Whether we are running in a Node.js-like environment (node, deno, bun) */
  IS_NODE_ENV,
  /** Whether the filesystem API is available */
  IS_FS_AVAILABLE,
  /** Whether the path API is available */
  IS_PATH_AVAILABLE
});
var RUNNING_LOCALLY = IS_FS_AVAILABLE && IS_PATH_AVAILABLE;
var dirname__ = "./";
if (RUNNING_LOCALLY) {
  const _import_meta_url = Object(import.meta).url;
  if (_import_meta_url) {
    dirname__ = path.dirname(path.dirname(url.fileURLToPath(_import_meta_url)));
  } else if (typeof __dirname !== "undefined") {
    dirname__ = path.dirname(__dirname);
  }
}
var DEFAULT_CACHE_DIR = RUNNING_LOCALLY ? path.join(dirname__, "/.cache/") : null;
var DEFAULT_LOCAL_MODEL_PATH = "/models/";
var localModelPath = RUNNING_LOCALLY ? path.join(dirname__, DEFAULT_LOCAL_MODEL_PATH) : DEFAULT_LOCAL_MODEL_PATH;
var env = {
  version: VERSION,
  /////////////////// Backends settings ///////////////////
  // NOTE: These will be populated later by the backends themselves.
  backends: {
    // onnxruntime-web/onnxruntime-node
    onnx: {}
  },
  /////////////////// Model settings ///////////////////
  allowRemoteModels: true,
  remoteHost: "https://huggingface.co/",
  remotePathTemplate: "{model}/resolve/{revision}/",
  allowLocalModels: !(IS_BROWSER_ENV || IS_WEBWORKER_ENV),
  localModelPath,
  useFS: IS_FS_AVAILABLE,
  /////////////////// Cache settings ///////////////////
  useBrowserCache: IS_WEB_CACHE_AVAILABLE && !IS_DENO_RUNTIME,
  useFSCache: IS_FS_AVAILABLE,
  cacheDir: DEFAULT_CACHE_DIR,
  useCustomCache: false,
  customCache: null
  //////////////////////////////////////////////////////
};
function isEmpty(obj) {
  return Object.keys(obj).length === 0;
}

// src/utils/generic.js
var Callable = (
  /** @type {any} */
  class {
    /**
    * Creates a new instance of the Callable class.
    */
    constructor() {
      let closure = function(...args) {
        return closure._call(...args);
      };
      return Object.setPrototypeOf(closure, new.target.prototype);
    }
    /**
     * This method should be implemented in subclasses to provide the
     * functionality of the callable object.
     *
     * @param {any[]} args
     * @throws {Error} If the subclass does not implement the `_call` method.
     */
    _call(...args) {
      throw Error("Must implement _call method in subclass");
    }
  }
);

// src/utils/core.js
function dispatchCallback(progress_callback, data) {
  if (progress_callback) progress_callback(data);
}
function reverseDictionary(data) {
  return Object.fromEntries(Object.entries(data).map(([key, value]) => [value, key]));
}
function escapeRegExp(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
function isIntegralNumber(x) {
  return Number.isInteger(x) || typeof x === "bigint";
}
function isNullishDimension(x) {
  return x === null || x === void 0 || x === -1;
}
function calculateDimensions(arr) {
  const dimensions = [];
  let current = arr;
  while (Array.isArray(current)) {
    dimensions.push(current.length);
    current = current[0];
  }
  return dimensions;
}
function mergeArrays(...arrs) {
  return Array.prototype.concat.apply([], arrs);
}
function product(...a) {
  return a.reduce((a2, b) => a2.flatMap((d) => b.map((e) => [d, e])));
}
function calculateReflectOffset(i, w) {
  return Math.abs((i + w) % (2 * w) - w);
}
function saveBlob(path3, blob) {
  const dataURL = URL.createObjectURL(blob);
  const downloadLink = document.createElement("a");
  downloadLink.href = dataURL;
  downloadLink.download = path3;
  downloadLink.click();
  downloadLink.remove();
  URL.revokeObjectURL(dataURL);
}
function pick(o, props) {
  return Object.assign(
    {},
    ...props.map((prop) => {
      if (o[prop] !== void 0) {
        return { [prop]: o[prop] };
      }
    })
  );
}
function len(s) {
  let length = 0;
  for (const c of s) ++length;
  return length;
}
function count(arr, value) {
  let count2 = 0;
  for (const v of arr) {
    if (v === value) ++count2;
  }
  return count2;
}

// src/utils/hub.js
import fs2 from "fs";
import path2 from "path";
var MAX_EXTERNAL_DATA_CHUNKS = 100;
var CONTENT_TYPE_MAP = {
  "txt": "text/plain",
  "html": "text/html",
  "css": "text/css",
  "js": "text/javascript",
  "json": "application/json",
  "png": "image/png",
  "jpg": "image/jpeg",
  "jpeg": "image/jpeg",
  "gif": "image/gif"
};
var FileResponse = class _FileResponse {
  /**
   * Creates a new `FileResponse` object.
   * @param {string} filePath
   */
  constructor(filePath) {
    this.filePath = filePath;
    this.headers = new Headers();
    this.exists = fs2.existsSync(filePath);
    if (this.exists) {
      this.status = 200;
      this.statusText = "OK";
      let stats = fs2.statSync(filePath);
      this.headers.set("content-length", stats.size.toString());
      this.updateContentType();
      const stream = fs2.createReadStream(filePath);
      this.body = new ReadableStream({
        start(controller) {
          stream.on("data", (chunk2) => controller.enqueue(chunk2));
          stream.on("end", () => controller.close());
          stream.on("error", (err) => controller.error(err));
        },
        cancel() {
          stream.destroy();
        }
      });
    } else {
      this.status = 404;
      this.statusText = "Not Found";
      this.body = null;
    }
  }
  /**
   * Updates the 'content-type' header property of the response based on the extension of
   * the file specified by the filePath property of the current object.
   * @returns {void}
   */
  updateContentType() {
    const extension = this.filePath.toString().split(".").pop().toLowerCase();
    this.headers.set("content-type", CONTENT_TYPE_MAP[extension] ?? "application/octet-stream");
  }
  /**
   * Clone the current FileResponse object.
   * @returns {FileResponse} A new FileResponse object with the same properties as the current object.
   */
  clone() {
    let response = new _FileResponse(this.filePath);
    response.exists = this.exists;
    response.status = this.status;
    response.statusText = this.statusText;
    response.headers = new Headers(this.headers);
    return response;
  }
  /**
   * Reads the contents of the file specified by the filePath property and returns a Promise that
   * resolves with an ArrayBuffer containing the file's contents.
   * @returns {Promise<ArrayBuffer>} A Promise that resolves with an ArrayBuffer containing the file's contents.
   * @throws {Error} If the file cannot be read.
   */
  async arrayBuffer() {
    const data = await fs2.promises.readFile(this.filePath);
    return (
      /** @type {ArrayBuffer} */
      data.buffer
    );
  }
  /**
   * Reads the contents of the file specified by the filePath property and returns a Promise that
   * resolves with a Blob containing the file's contents.
   * @returns {Promise<Blob>} A Promise that resolves with a Blob containing the file's contents.
   * @throws {Error} If the file cannot be read.
   */
  async blob() {
    const data = await fs2.promises.readFile(this.filePath);
    return new Blob([data], { type: this.headers.get("content-type") });
  }
  /**
   * Reads the contents of the file specified by the filePath property and returns a Promise that
   * resolves with a string containing the file's contents.
   * @returns {Promise<string>} A Promise that resolves with a string containing the file's contents.
   * @throws {Error} If the file cannot be read.
   */
  async text() {
    const data = await fs2.promises.readFile(this.filePath, "utf8");
    return data;
  }
  /**
   * Reads the contents of the file specified by the filePath property and returns a Promise that
   * resolves with a parsed JavaScript object containing the file's contents.
   *
   * @returns {Promise<Object>} A Promise that resolves with a parsed JavaScript object containing the file's contents.
   * @throws {Error} If the file cannot be read.
   */
  async json() {
    return JSON.parse(await this.text());
  }
};
function isValidUrl(string, protocols = null, validHosts = null) {
  let url2;
  try {
    url2 = new URL(string);
  } catch (_) {
    return false;
  }
  if (protocols && !protocols.includes(url2.protocol)) {
    return false;
  }
  if (validHosts && !validHosts.includes(url2.hostname)) {
    return false;
  }
  return true;
}
var REPO_ID_REGEX = /^(\b[\w\-.]+\b\/)?\b[\w\-.]{1,96}\b$/;
function isValidHfModelId(string) {
  if (!REPO_ID_REGEX.test(string)) return false;
  if (string.includes("..") || string.includes("--")) return false;
  if (string.endsWith(".git") || string.endsWith(".ipynb")) return false;
  return true;
}
async function getFile(urlOrPath) {
  if (env.useFS && !isValidUrl(urlOrPath, ["http:", "https:", "blob:"])) {
    return new FileResponse(
      urlOrPath instanceof URL ? urlOrPath.protocol === "file:" ? urlOrPath.pathname : urlOrPath.toString() : urlOrPath
    );
  } else if (typeof process !== "undefined" && process?.release?.name === "node") {
    const IS_CI = !!process.env?.TESTING_REMOTELY;
    const version = env.version;
    const headers = new Headers();
    headers.set("User-Agent", `transformers.js/${version}; is_ci/${IS_CI};`);
    const isHFURL = isValidUrl(urlOrPath, ["http:", "https:"], ["huggingface.co", "hf.co"]);
    if (isHFURL) {
      const token = process.env?.HF_TOKEN ?? process.env?.HF_ACCESS_TOKEN;
      if (token) {
        headers.set("Authorization", `Bearer ${token}`);
      }
    }
    return fetch(urlOrPath, { headers });
  } else {
    return fetch(urlOrPath);
  }
}
var ERROR_MAPPING = {
  // 4xx errors (https://developer.mozilla.org/en-US/docs/Web/HTTP/Status#client_error_responses)
  400: "Bad request error occurred while trying to load file",
  401: "Unauthorized access to file",
  403: "Forbidden access to file",
  404: "Could not locate file",
  408: "Request timeout error occurred while trying to load file",
  // 5xx errors (https://developer.mozilla.org/en-US/docs/Web/HTTP/Status#server_error_responses)
  500: "Internal server error error occurred while trying to load file",
  502: "Bad gateway error occurred while trying to load file",
  503: "Service unavailable error occurred while trying to load file",
  504: "Gateway timeout error occurred while trying to load file"
};
function handleError(status, remoteURL, fatal) {
  if (!fatal) {
    return null;
  }
  const message = ERROR_MAPPING[status] ?? `Error (${status}) occurred while trying to load file`;
  throw Error(`${message}: "${remoteURL}".`);
}
var FileCache = class {
  /**
   * Instantiate a `FileCache` object.
   * @param {string} path
   */
  constructor(path3) {
    this.path = path3;
  }
  /**
   * Checks whether the given request is in the cache.
   * @param {string} request
   * @returns {Promise<FileResponse | undefined>}
   */
  async match(request) {
    let filePath = path2.join(this.path, request);
    let file = new FileResponse(filePath);
    if (file.exists) {
      return file;
    } else {
      return void 0;
    }
  }
  /**
   * Adds the given response to the cache.
   * @param {string} request
   * @param {Response} response
   * @param {(data: {progress: number, loaded: number, total: number}) => void} [progress_callback] Optional.
   * The function to call with progress updates
   * @returns {Promise<void>}
   */
  async put(request, response, progress_callback = void 0) {
    let filePath = path2.join(this.path, request);
    try {
      const contentLength = response.headers.get("Content-Length");
      const total = parseInt(contentLength ?? "0");
      let loaded = 0;
      await fs2.promises.mkdir(path2.dirname(filePath), { recursive: true });
      const fileStream = fs2.createWriteStream(filePath);
      const reader = response.body.getReader();
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        await new Promise((resolve, reject) => {
          fileStream.write(value, (err) => {
            if (err) {
              reject(err);
              return;
            }
            resolve();
          });
        });
        loaded += value.length;
        const progress = total ? loaded / total * 100 : 0;
        progress_callback?.({ progress, loaded, total });
      }
      fileStream.close();
    } catch (error) {
      try {
        await fs2.promises.unlink(filePath);
      } catch {
      }
      throw error;
    }
  }
  // TODO add the rest?
  // addAll(requests: RequestInfo[]): Promise<void>;
  // delete(request: RequestInfo | URL, options?: CacheQueryOptions): Promise<boolean>;
  // keys(request?: RequestInfo | URL, options?: CacheQueryOptions): Promise<ReadonlyArray<Request>>;
  // match(request: RequestInfo | URL, options?: CacheQueryOptions): Promise<Response | undefined>;
  // matchAll(request?: RequestInfo | URL, options?: CacheQueryOptions): Promise<ReadonlyArray<Response>>;
};
async function tryCache(cache, ...names) {
  for (let name of names) {
    try {
      let result = await cache.match(name);
      if (result) return result;
    } catch (e) {
      continue;
    }
  }
  return void 0;
}
async function getModelFile(path_or_repo_id, filename, fatal = true, options = {}, return_path = false) {
  if (!env.allowLocalModels) {
    if (options.local_files_only) {
      throw Error("Invalid configuration detected: local models are disabled (`env.allowLocalModels=false`) but you have requested to only use local models (`local_files_only=true`).");
    } else if (!env.allowRemoteModels) {
      throw Error("Invalid configuration detected: both local and remote models are disabled. Fix by setting `env.allowLocalModels` or `env.allowRemoteModels` to `true`.");
    }
  }
  dispatchCallback(options.progress_callback, {
    status: "initiate",
    name: path_or_repo_id,
    file: filename
  });
  let cache;
  if (!cache && env.useCustomCache) {
    if (!env.customCache) {
      throw Error("`env.useCustomCache=true`, but `env.customCache` is not defined.");
    }
    if (!env.customCache.match || !env.customCache.put) {
      throw new Error(
        "`env.customCache` must be an object which implements the `match` and `put` functions of the Web Cache API. For more information, see https://developer.mozilla.org/en-US/docs/Web/API/Cache"
      );
    }
    cache = env.customCache;
  }
  if (!cache && env.useBrowserCache) {
    if (typeof caches === "undefined") {
      throw Error("Browser cache is not available in this environment.");
    }
    try {
      cache = await caches.open("transformers-cache");
    } catch (e) {
      console.warn("An error occurred while opening the browser cache:", e);
    }
  }
  if (!cache && env.useFSCache) {
    if (!apis.IS_FS_AVAILABLE) {
      throw Error("File System Cache is not available in this environment.");
    }
    cache = new FileCache(options.cache_dir ?? env.cacheDir);
  }
  const revision = options.revision ?? "main";
  const requestURL = pathJoin(path_or_repo_id, filename);
  const validModelId = isValidHfModelId(path_or_repo_id);
  const localPath = validModelId ? pathJoin(env.localModelPath, requestURL) : requestURL;
  const remoteURL = pathJoin(
    env.remoteHost,
    env.remotePathTemplate.replaceAll("{model}", path_or_repo_id).replaceAll("{revision}", encodeURIComponent(revision)),
    filename
  );
  let cacheKey;
  const proposedCacheKey = cache instanceof FileCache ? revision === "main" ? requestURL : pathJoin(path_or_repo_id, revision, filename) : remoteURL;
  let toCacheResponse = false;
  let response;
  if (cache) {
    response = await tryCache(cache, localPath, proposedCacheKey);
  }
  const cacheHit = response !== void 0;
  if (response === void 0) {
    if (env.allowLocalModels) {
      const isURL = isValidUrl(requestURL, ["http:", "https:"]);
      if (!isURL) {
        try {
          response = await getFile(localPath);
          cacheKey = localPath;
        } catch (e) {
          console.warn(`Unable to load from local path "${localPath}": "${e}"`);
        }
      } else if (options.local_files_only) {
        throw new Error(`\`local_files_only=true\`, but attempted to load a remote file from: ${requestURL}.`);
      } else if (!env.allowRemoteModels) {
        throw new Error(`\`env.allowRemoteModels=false\`, but attempted to load a remote file from: ${requestURL}.`);
      }
    }
    if (response === void 0 || response.status === 404) {
      if (options.local_files_only || !env.allowRemoteModels) {
        if (fatal) {
          throw Error(`\`local_files_only=true\` or \`env.allowRemoteModels=false\` and file was not found locally at "${localPath}".`);
        } else {
          return null;
        }
      }
      if (!validModelId) {
        throw Error(`Local file missing at "${localPath}" and download aborted due to invalid model ID "${path_or_repo_id}".`);
      }
      response = await getFile(remoteURL);
      if (response.status !== 200) {
        return handleError(response.status, remoteURL, fatal);
      }
      cacheKey = proposedCacheKey;
    }
    toCacheResponse = cache && typeof Response !== "undefined" && response instanceof Response && response.status === 200;
  }
  dispatchCallback(options.progress_callback, {
    status: "download",
    name: path_or_repo_id,
    file: filename
  });
  let result;
  if (!(apis.IS_NODE_ENV && return_path)) {
    let buffer;
    if (!options.progress_callback) {
      buffer = new Uint8Array(await response.arrayBuffer());
    } else if (cacheHit && typeof navigator !== "undefined" && /firefox/i.test(navigator.userAgent)) {
      buffer = new Uint8Array(await response.arrayBuffer());
      dispatchCallback(options.progress_callback, {
        status: "progress",
        name: path_or_repo_id,
        file: filename,
        progress: 100,
        loaded: buffer.length,
        total: buffer.length
      });
    } else {
      buffer = await readResponse(response, (data) => {
        dispatchCallback(options.progress_callback, {
          status: "progress",
          name: path_or_repo_id,
          file: filename,
          ...data
        });
      });
    }
    result = buffer;
  }
  if (
    // Only cache web responses
    // i.e., do not cache FileResponses (prevents duplication)
    toCacheResponse && cacheKey && // Check again whether request is in cache. If not, we add the response to the cache
    await cache.match(cacheKey) === void 0
  ) {
    if (!result) {
      const wrapped_progress = options.progress_callback ? (data) => dispatchCallback(options.progress_callback, {
        status: "progress",
        name: path_or_repo_id,
        file: filename,
        ...data
      }) : void 0;
      await cache.put(
        cacheKey,
        /** @type {Response} */
        response,
        wrapped_progress
      );
    } else {
      await cache.put(cacheKey, new Response(result, {
        headers: response.headers
      })).catch((err) => {
        console.warn(`Unable to add response to browser cache: ${err}.`);
      });
    }
  }
  dispatchCallback(options.progress_callback, {
    status: "done",
    name: path_or_repo_id,
    file: filename
  });
  if (result) {
    if (!apis.IS_NODE_ENV && return_path) {
      throw new Error("Cannot return path in a browser environment.");
    }
    return result;
  }
  if (response instanceof FileResponse) {
    return response.filePath;
  }
  const cachedResponse = await cache?.match(cacheKey);
  if (cachedResponse instanceof FileResponse) {
    return cachedResponse.filePath;
  } else if (cachedResponse instanceof Response) {
    return new Uint8Array(await cachedResponse.arrayBuffer());
  } else if (typeof cachedResponse === "string") {
    return cachedResponse;
  }
  throw new Error("Unable to get model file path or buffer.");
}
async function getModelText(modelPath, fileName, fatal = true, options = {}) {
  const buffer = await getModelFile(modelPath, fileName, fatal, options, false);
  if (buffer === null) {
    return null;
  }
  const decoder = new TextDecoder("utf-8");
  return decoder.decode(
    /** @type {Uint8Array} */
    buffer
  );
}
async function getModelJSON(modelPath, fileName, fatal = true, options = {}) {
  const text = await getModelText(modelPath, fileName, fatal, options);
  if (text === null) {
    return {};
  }
  return JSON.parse(text);
}
async function readResponse(response, progress_callback) {
  const contentLength = response.headers.get("Content-Length");
  if (contentLength === null) {
    console.warn("Unable to determine content-length from response headers. Will expand buffer when needed.");
  }
  let total = parseInt(contentLength ?? "0");
  let buffer = new Uint8Array(total);
  let loaded = 0;
  const reader = response.body.getReader();
  async function read() {
    const { done, value } = await reader.read();
    if (done) return;
    const newLoaded = loaded + value.length;
    if (newLoaded > total) {
      total = newLoaded;
      const newBuffer = new Uint8Array(total);
      newBuffer.set(buffer);
      buffer = newBuffer;
    }
    buffer.set(value, loaded);
    loaded = newLoaded;
    const progress = loaded / total * 100;
    progress_callback({ progress, loaded, total });
    return read();
  }
  await read();
  return buffer;
}
function pathJoin(...parts) {
  parts = parts.map((part, index) => {
    if (index) {
      part = part.replace(new RegExp("^/"), "");
    }
    if (index !== parts.length - 1) {
      part = part.replace(new RegExp("/$"), "");
    }
    return part;
  });
  return parts.join("/");
}

// src/utils/maths.js
function interpolate_data(input, [in_channels, in_height, in_width], [out_height, out_width], mode = "bilinear", align_corners = false) {
  const x_scale = out_width / in_width;
  const y_scale = out_height / in_height;
  const out_img = new input.constructor(out_height * out_width * in_channels);
  const inStride = in_height * in_width;
  const outStride = out_height * out_width;
  for (let i = 0; i < out_height; ++i) {
    for (let j = 0; j < out_width; ++j) {
      const outOffset = i * out_width + j;
      const x = (j + 0.5) / x_scale - 0.5;
      const y = (i + 0.5) / y_scale - 0.5;
      let x1 = Math.floor(x);
      let y1 = Math.floor(y);
      const x2 = Math.min(x1 + 1, in_width - 1);
      const y2 = Math.min(y1 + 1, in_height - 1);
      x1 = Math.max(x1, 0);
      y1 = Math.max(y1, 0);
      const s = x - x1;
      const t = y - y1;
      const w1 = (1 - s) * (1 - t);
      const w2 = s * (1 - t);
      const w3 = (1 - s) * t;
      const w4 = s * t;
      const yStride = y1 * in_width;
      const xStride = y2 * in_width;
      const idx1 = yStride + x1;
      const idx2 = yStride + x2;
      const idx3 = xStride + x1;
      const idx4 = xStride + x2;
      for (let k = 0; k < in_channels; ++k) {
        const cOffset = k * inStride;
        out_img[k * outStride + outOffset] = w1 * input[cOffset + idx1] + w2 * input[cOffset + idx2] + w3 * input[cOffset + idx3] + w4 * input[cOffset + idx4];
      }
    }
  }
  return out_img;
}
function permute_data(array, dims, axes) {
  const shape = new Array(axes.length);
  const stride = new Array(axes.length);
  for (let i = axes.length - 1, s = 1; i >= 0; --i) {
    stride[i] = s;
    shape[i] = dims[axes[i]];
    s *= shape[i];
  }
  const invStride = axes.map((_, i) => stride[axes.indexOf(i)]);
  const permutedData = new array.constructor(array.length);
  for (let i = 0; i < array.length; ++i) {
    let newIndex = 0;
    for (let j = dims.length - 1, k = i; j >= 0; --j) {
      newIndex += k % dims[j] * invStride[j];
      k = Math.floor(k / dims[j]);
    }
    permutedData[newIndex] = array[i];
  }
  return [permutedData, shape];
}
function softmax(arr) {
  const maxVal = max(arr)[0];
  const exps = arr.map((x) => Math.exp(x - maxVal));
  const sumExps = exps.reduce((acc, val) => acc + val, 0);
  const softmaxArr = exps.map((x) => x / sumExps);
  return (
    /** @type {T} */
    softmaxArr
  );
}
function log_softmax(arr) {
  const maxVal = max(arr)[0];
  let sumExps = 0;
  for (let i = 0; i < arr.length; ++i) {
    sumExps += Math.exp(arr[i] - maxVal);
  }
  const logSum = Math.log(sumExps);
  const logSoftmaxArr = arr.map((x) => x - maxVal - logSum);
  return (
    /** @type {T} */
    logSoftmaxArr
  );
}
function dot(arr1, arr2) {
  let result = 0;
  for (let i = 0; i < arr1.length; ++i) {
    result += arr1[i] * arr2[i];
  }
  return result;
}
function cos_sim(arr1, arr2) {
  const dotProduct = dot(arr1, arr2);
  const magnitudeA = magnitude(arr1);
  const magnitudeB = magnitude(arr2);
  const cosineSimilarity = dotProduct / (magnitudeA * magnitudeB);
  return cosineSimilarity;
}
function magnitude(arr) {
  return Math.sqrt(arr.reduce((acc, val) => acc + val * val, 0));
}
function min(arr) {
  if (arr.length === 0) throw Error("Array must not be empty");
  let min2 = arr[0];
  let indexOfMin = 0;
  for (let i = 1; i < arr.length; ++i) {
    if (arr[i] < min2) {
      min2 = arr[i];
      indexOfMin = i;
    }
  }
  return (
    /** @type {T extends bigint[]|BigTypedArray ? [bigint, number] : [number, number]} */
    [min2, indexOfMin]
  );
}
function max(arr) {
  if (arr.length === 0) throw Error("Array must not be empty");
  let max2 = arr[0];
  let indexOfMax = 0;
  for (let i = 1; i < arr.length; ++i) {
    if (arr[i] > max2) {
      max2 = arr[i];
      indexOfMax = i;
    }
  }
  return (
    /** @type {T extends bigint[]|BigTypedArray ? [bigint, number] : [number, number]} */
    [max2, indexOfMax]
  );
}
function isPowerOfTwo(number) {
  return number > 0 && (number & number - 1) === 0;
}
var P2FFT = class {
  /**
   * @param {number} size The size of the input array. Must be a power of two larger than 1.
   * @throws {Error} FFT size must be a power of two larger than 1.
   */
  constructor(size) {
    this.size = size | 0;
    if (this.size <= 1 || !isPowerOfTwo(this.size))
      throw new Error("FFT size must be a power of two larger than 1");
    this._csize = size << 1;
    this.table = new Float64Array(this.size * 2);
    for (let i = 0; i < this.table.length; i += 2) {
      const angle = Math.PI * i / this.size;
      this.table[i] = Math.cos(angle);
      this.table[i + 1] = -Math.sin(angle);
    }
    let power = 0;
    for (let t = 1; this.size > t; t <<= 1)
      ++power;
    this._width = power % 2 === 0 ? power - 1 : power;
    this._bitrev = new Int32Array(1 << this._width);
    for (let j = 0; j < this._bitrev.length; ++j) {
      this._bitrev[j] = 0;
      for (let shift = 0; shift < this._width; shift += 2) {
        const revShift = this._width - shift - 2;
        this._bitrev[j] |= (j >>> shift & 3) << revShift;
      }
    }
  }
  /**
   * Create a complex number array with size `2 * size`
   *
   * @returns {Float64Array} A complex number array with size `2 * size`
   */
  createComplexArray() {
    return new Float64Array(this._csize);
  }
  /**
   * Converts a complex number representation stored in a Float64Array to an array of real numbers.
   * 
   * @param {Float64Array} complex The complex number representation to be converted.
   * @param {number[]} [storage] An optional array to store the result in.
   * @returns {number[]} An array of real numbers representing the input complex number representation.
   */
  fromComplexArray(complex, storage) {
    const res = storage || new Array(complex.length >>> 1);
    for (let i = 0; i < complex.length; i += 2)
      res[i >>> 1] = complex[i];
    return res;
  }
  /**
   * Convert a real-valued input array to a complex-valued output array.
   * @param {Float64Array} input The real-valued input array.
   * @param {Float64Array} [storage] Optional buffer to store the output array.
   * @returns {Float64Array} The complex-valued output array.
   */
  toComplexArray(input, storage) {
    const res = storage || this.createComplexArray();
    for (let i = 0; i < res.length; i += 2) {
      res[i] = input[i >>> 1];
      res[i + 1] = 0;
    }
    return res;
  }
  /**
   * Performs a Fast Fourier Transform (FFT) on the given input data and stores the result in the output buffer.
   * 
   * @param {Float64Array} out The output buffer to store the result.
   * @param {Float64Array} data The input data to transform.
   * 
   * @throws {Error} Input and output buffers must be different.
   * 
   * @returns {void}
   */
  transform(out, data) {
    if (out === data)
      throw new Error("Input and output buffers must be different");
    this._transform4(
      out,
      data,
      1
      /* DONE */
    );
  }
  /**
   * Performs a real-valued forward FFT on the given input buffer and stores the result in the given output buffer.
   * The input buffer must contain real values only, while the output buffer will contain complex values. The input and
   * output buffers must be different.
   *
   * @param {Float64Array} out The output buffer.
   * @param {Float64Array} data The input buffer containing real values.
   *
   * @throws {Error} If the input and output buffers are the same.
   */
  realTransform(out, data) {
    if (out === data)
      throw new Error("Input and output buffers must be different");
    this._realTransform4(
      out,
      data,
      1
      /* DONE */
    );
  }
  /**
   * Performs an inverse FFT transformation on the given `data` array, and stores the result in `out`.
   * The `out` array must be a different buffer than the `data` array. The `out` array will contain the
   * result of the transformation. The `data` array will not be modified.
   * 
   * @param {Float64Array} out The output buffer for the transformed data.
   * @param {Float64Array} data The input data to transform.
   * @throws {Error} If `out` and `data` refer to the same buffer.
   * @returns {void}
   */
  inverseTransform(out, data) {
    if (out === data)
      throw new Error("Input and output buffers must be different");
    this._transform4(
      out,
      data,
      -1
      /* DONE */
    );
    for (let i = 0; i < out.length; ++i)
      out[i] /= this.size;
  }
  /**
   * Performs a radix-4 implementation of a discrete Fourier transform on a given set of data.
   *
   * @param {Float64Array} out The output buffer for the transformed data.
   * @param {Float64Array} data The input buffer of data to be transformed.
   * @param {number} inv A scaling factor to apply to the transform.
   * @returns {void}
   */
  _transform4(out, data, inv) {
    const size = this._csize;
    const width = this._width;
    let step = 1 << width;
    let len2 = size / step << 1;
    let outOff;
    let t;
    const bitrev = this._bitrev;
    if (len2 === 4) {
      for (outOff = 0, t = 0; outOff < size; outOff += len2, ++t) {
        const off = bitrev[t];
        this._singleTransform2(data, out, outOff, off, step);
      }
    } else {
      for (outOff = 0, t = 0; outOff < size; outOff += len2, ++t) {
        const off = bitrev[t];
        this._singleTransform4(data, out, outOff, off, step, inv);
      }
    }
    const table = this.table;
    for (step >>= 2; step >= 2; step >>= 2) {
      len2 = size / step << 1;
      const quarterLen = len2 >>> 2;
      for (outOff = 0; outOff < size; outOff += len2) {
        const limit = outOff + quarterLen - 1;
        for (let i = outOff, k = 0; i < limit; i += 2, k += step) {
          const A = i;
          const B = A + quarterLen;
          const C = B + quarterLen;
          const D = C + quarterLen;
          const Ar = out[A];
          const Ai = out[A + 1];
          const Br = out[B];
          const Bi = out[B + 1];
          const Cr = out[C];
          const Ci = out[C + 1];
          const Dr = out[D];
          const Di = out[D + 1];
          const tableBr = table[k];
          const tableBi = inv * table[k + 1];
          const MBr = Br * tableBr - Bi * tableBi;
          const MBi = Br * tableBi + Bi * tableBr;
          const tableCr = table[2 * k];
          const tableCi = inv * table[2 * k + 1];
          const MCr = Cr * tableCr - Ci * tableCi;
          const MCi = Cr * tableCi + Ci * tableCr;
          const tableDr = table[3 * k];
          const tableDi = inv * table[3 * k + 1];
          const MDr = Dr * tableDr - Di * tableDi;
          const MDi = Dr * tableDi + Di * tableDr;
          const T0r = Ar + MCr;
          const T0i = Ai + MCi;
          const T1r = Ar - MCr;
          const T1i = Ai - MCi;
          const T2r = MBr + MDr;
          const T2i = MBi + MDi;
          const T3r = inv * (MBr - MDr);
          const T3i = inv * (MBi - MDi);
          out[A] = T0r + T2r;
          out[A + 1] = T0i + T2i;
          out[B] = T1r + T3i;
          out[B + 1] = T1i - T3r;
          out[C] = T0r - T2r;
          out[C + 1] = T0i - T2i;
          out[D] = T1r - T3i;
          out[D + 1] = T1i + T3r;
        }
      }
    }
  }
  /**
   * Performs a radix-2 implementation of a discrete Fourier transform on a given set of data.
   *
   * @param {Float64Array} data The input buffer of data to be transformed.
   * @param {Float64Array} out The output buffer for the transformed data.
   * @param {number} outOff The offset at which to write the output data.
   * @param {number} off The offset at which to begin reading the input data.
   * @param {number} step The step size for indexing the input data.
   * @returns {void}
   */
  _singleTransform2(data, out, outOff, off, step) {
    const evenR = data[off];
    const evenI = data[off + 1];
    const oddR = data[off + step];
    const oddI = data[off + step + 1];
    out[outOff] = evenR + oddR;
    out[outOff + 1] = evenI + oddI;
    out[outOff + 2] = evenR - oddR;
    out[outOff + 3] = evenI - oddI;
  }
  /**
   * Performs radix-4 transformation on input data of length 8
   *
   * @param {Float64Array} data Input data array of length 8
   * @param {Float64Array} out Output data array of length 8
   * @param {number} outOff Index of output array to start writing from
   * @param {number} off Index of input array to start reading from
   * @param {number} step Step size between elements in input array
   * @param {number} inv Scaling factor for inverse transform
   * 
   * @returns {void}
   */
  _singleTransform4(data, out, outOff, off, step, inv) {
    const step2 = step * 2;
    const step3 = step * 3;
    const Ar = data[off];
    const Ai = data[off + 1];
    const Br = data[off + step];
    const Bi = data[off + step + 1];
    const Cr = data[off + step2];
    const Ci = data[off + step2 + 1];
    const Dr = data[off + step3];
    const Di = data[off + step3 + 1];
    const T0r = Ar + Cr;
    const T0i = Ai + Ci;
    const T1r = Ar - Cr;
    const T1i = Ai - Ci;
    const T2r = Br + Dr;
    const T2i = Bi + Di;
    const T3r = inv * (Br - Dr);
    const T3i = inv * (Bi - Di);
    out[outOff] = T0r + T2r;
    out[outOff + 1] = T0i + T2i;
    out[outOff + 2] = T1r + T3i;
    out[outOff + 3] = T1i - T3r;
    out[outOff + 4] = T0r - T2r;
    out[outOff + 5] = T0i - T2i;
    out[outOff + 6] = T1r - T3i;
    out[outOff + 7] = T1i + T3r;
  }
  /**
   * Real input radix-4 implementation
   * @param {Float64Array} out Output array for the transformed data
   * @param {Float64Array} data Input array of real data to be transformed
   * @param {number} inv The scale factor used to normalize the inverse transform
   */
  _realTransform4(out, data, inv) {
    const size = this._csize;
    const width = this._width;
    let step = 1 << width;
    let len2 = size / step << 1;
    let outOff;
    let t;
    const bitrev = this._bitrev;
    if (len2 === 4) {
      for (outOff = 0, t = 0; outOff < size; outOff += len2, ++t) {
        const off = bitrev[t];
        this._singleRealTransform2(data, out, outOff, off >>> 1, step >>> 1);
      }
    } else {
      for (outOff = 0, t = 0; outOff < size; outOff += len2, ++t) {
        const off = bitrev[t];
        this._singleRealTransform4(data, out, outOff, off >>> 1, step >>> 1, inv);
      }
    }
    const table = this.table;
    for (step >>= 2; step >= 2; step >>= 2) {
      len2 = size / step << 1;
      const halfLen = len2 >>> 1;
      const quarterLen = halfLen >>> 1;
      const hquarterLen = quarterLen >>> 1;
      for (outOff = 0; outOff < size; outOff += len2) {
        for (let i = 0, k = 0; i <= hquarterLen; i += 2, k += step) {
          const A = outOff + i;
          const B = A + quarterLen;
          const C = B + quarterLen;
          const D = C + quarterLen;
          const Ar = out[A];
          const Ai = out[A + 1];
          const Br = out[B];
          const Bi = out[B + 1];
          const Cr = out[C];
          const Ci = out[C + 1];
          const Dr = out[D];
          const Di = out[D + 1];
          const MAr = Ar;
          const MAi = Ai;
          const tableBr = table[k];
          const tableBi = inv * table[k + 1];
          const MBr = Br * tableBr - Bi * tableBi;
          const MBi = Br * tableBi + Bi * tableBr;
          const tableCr = table[2 * k];
          const tableCi = inv * table[2 * k + 1];
          const MCr = Cr * tableCr - Ci * tableCi;
          const MCi = Cr * tableCi + Ci * tableCr;
          const tableDr = table[3 * k];
          const tableDi = inv * table[3 * k + 1];
          const MDr = Dr * tableDr - Di * tableDi;
          const MDi = Dr * tableDi + Di * tableDr;
          const T0r = MAr + MCr;
          const T0i = MAi + MCi;
          const T1r = MAr - MCr;
          const T1i = MAi - MCi;
          const T2r = MBr + MDr;
          const T2i = MBi + MDi;
          const T3r = inv * (MBr - MDr);
          const T3i = inv * (MBi - MDi);
          out[A] = T0r + T2r;
          out[A + 1] = T0i + T2i;
          out[B] = T1r + T3i;
          out[B + 1] = T1i - T3r;
          if (i === 0) {
            out[C] = T0r - T2r;
            out[C + 1] = T0i - T2i;
            continue;
          }
          if (i === hquarterLen)
            continue;
          const SA = outOff + quarterLen - i;
          const SB = outOff + halfLen - i;
          out[SA] = T1r - inv * T3i;
          out[SA + 1] = -T1i - inv * T3r;
          out[SB] = T0r - inv * T2r;
          out[SB + 1] = -T0i + inv * T2i;
        }
      }
    }
    const half = size >>> 1;
    for (let i = 2; i < half; i += 2) {
      out[size - i] = out[i];
      out[size - i + 1] = -out[i + 1];
    }
  }
  /**
   * Performs a single real input radix-2 transformation on the provided data
   * 
   * @param {Float64Array} data The input data array
   * @param {Float64Array} out The output data array
   * @param {number} outOff The output offset
   * @param {number} off The input offset
   * @param {number} step The step
   * 
   * @returns {void}
   */
  _singleRealTransform2(data, out, outOff, off, step) {
    const evenR = data[off];
    const oddR = data[off + step];
    out[outOff] = evenR + oddR;
    out[outOff + 1] = 0;
    out[outOff + 2] = evenR - oddR;
    out[outOff + 3] = 0;
  }
  /**
   * Computes a single real-valued transform using radix-4 algorithm.
   * This method is only called for len=8.
   *
   * @param {Float64Array} data The input data array.
   * @param {Float64Array} out The output data array.
   * @param {number} outOff The offset into the output array.
   * @param {number} off The offset into the input array.
   * @param {number} step The step size for the input array.
   * @param {number} inv The value of inverse.
   */
  _singleRealTransform4(data, out, outOff, off, step, inv) {
    const step2 = step * 2;
    const step3 = step * 3;
    const Ar = data[off];
    const Br = data[off + step];
    const Cr = data[off + step2];
    const Dr = data[off + step3];
    const T0r = Ar + Cr;
    const T1r = Ar - Cr;
    const T2r = Br + Dr;
    const T3r = inv * (Br - Dr);
    out[outOff] = T0r + T2r;
    out[outOff + 1] = 0;
    out[outOff + 2] = T1r;
    out[outOff + 3] = -T3r;
    out[outOff + 4] = T0r - T2r;
    out[outOff + 5] = 0;
    out[outOff + 6] = T1r;
    out[outOff + 7] = T3r;
  }
};
var NP2FFT = class {
  /**
   * Constructs a new NP2FFT object.
   * @param {number} fft_length The length of the FFT
   */
  constructor(fft_length) {
    const a = 2 * (fft_length - 1);
    const b = 2 * (2 * fft_length - 1);
    const nextP2 = 2 ** Math.ceil(Math.log2(b));
    this.bufferSize = nextP2;
    this._a = a;
    const chirp = new Float64Array(b);
    const ichirp = new Float64Array(nextP2);
    this._chirpBuffer = new Float64Array(nextP2);
    this._buffer1 = new Float64Array(nextP2);
    this._buffer2 = new Float64Array(nextP2);
    this._outBuffer1 = new Float64Array(nextP2);
    this._outBuffer2 = new Float64Array(nextP2);
    const theta = -2 * Math.PI / fft_length;
    const baseR = Math.cos(theta);
    const baseI = Math.sin(theta);
    for (let i = 0; i < b >> 1; ++i) {
      const e = (i + 1 - fft_length) ** 2 / 2;
      const result_mod = Math.sqrt(baseR ** 2 + baseI ** 2) ** e;
      const result_arg = e * Math.atan2(baseI, baseR);
      const i2 = 2 * i;
      chirp[i2] = result_mod * Math.cos(result_arg);
      chirp[i2 + 1] = result_mod * Math.sin(result_arg);
      ichirp[i2] = chirp[i2];
      ichirp[i2 + 1] = -chirp[i2 + 1];
    }
    this._slicedChirpBuffer = chirp.subarray(a, b);
    this._f = new P2FFT(nextP2 >> 1);
    this._f.transform(this._chirpBuffer, ichirp);
  }
  _transform(output, input, real) {
    const ib1 = this._buffer1;
    const ib2 = this._buffer2;
    const ob2 = this._outBuffer1;
    const ob3 = this._outBuffer2;
    const cb = this._chirpBuffer;
    const sb = this._slicedChirpBuffer;
    const a = this._a;
    if (real) {
      for (let j = 0; j < sb.length; j += 2) {
        const j2 = j + 1;
        const j3 = j >> 1;
        const a_real = input[j3];
        ib1[j] = a_real * sb[j];
        ib1[j2] = a_real * sb[j2];
      }
    } else {
      for (let j = 0; j < sb.length; j += 2) {
        const j2 = j + 1;
        ib1[j] = input[j] * sb[j] - input[j2] * sb[j2];
        ib1[j2] = input[j] * sb[j2] + input[j2] * sb[j];
      }
    }
    this._f.transform(ob2, ib1);
    for (let j = 0; j < cb.length; j += 2) {
      const j2 = j + 1;
      ib2[j] = ob2[j] * cb[j] - ob2[j2] * cb[j2];
      ib2[j2] = ob2[j] * cb[j2] + ob2[j2] * cb[j];
    }
    this._f.inverseTransform(ob3, ib2);
    for (let j = 0; j < ob3.length; j += 2) {
      const a_real = ob3[j + a];
      const a_imag = ob3[j + a + 1];
      const b_real = sb[j];
      const b_imag = sb[j + 1];
      output[j] = a_real * b_real - a_imag * b_imag;
      output[j + 1] = a_real * b_imag + a_imag * b_real;
    }
  }
  transform(output, input) {
    this._transform(output, input, false);
  }
  realTransform(output, input) {
    this._transform(output, input, true);
  }
};
var FFT = class {
  constructor(fft_length) {
    this.fft_length = fft_length;
    this.isPowerOfTwo = isPowerOfTwo(fft_length);
    if (this.isPowerOfTwo) {
      this.fft = new P2FFT(fft_length);
      this.outputBufferSize = 2 * fft_length;
    } else {
      this.fft = new NP2FFT(fft_length);
      this.outputBufferSize = this.fft.bufferSize;
    }
  }
  realTransform(out, input) {
    this.fft.realTransform(out, input);
  }
  transform(out, input) {
    this.fft.transform(out, input);
  }
};
function medianFilter(data, windowSize) {
  if (windowSize % 2 === 0 || windowSize <= 0) {
    throw new Error("Window size must be a positive odd number");
  }
  const outputArray = new data.constructor(data.length);
  const buffer = new data.constructor(windowSize);
  const halfWindowSize = Math.floor(windowSize / 2);
  for (let i = 0; i < data.length; ++i) {
    let valuesIndex = 0;
    for (let j = -halfWindowSize; j <= halfWindowSize; ++j) {
      let index = i + j;
      if (index < 0) {
        index = Math.abs(index);
      } else if (index >= data.length) {
        index = 2 * (data.length - 1) - index;
      }
      buffer[valuesIndex++] = data[index];
    }
    buffer.sort();
    outputArray[i] = buffer[halfWindowSize];
  }
  return outputArray;
}
function round(num, decimals) {
  const pow = Math.pow(10, decimals);
  return Math.round(num * pow) / pow;
}
function bankers_round(x) {
  const r = Math.round(x);
  const br = Math.abs(x) % 1 === 0.5 ? r % 2 === 0 ? r : r - 1 : r;
  return br;
}
function dynamic_time_warping(matrix) {
  const output_length = matrix.length;
  const input_length = matrix[0].length;
  const outputShape = [output_length + 1, input_length + 1];
  const cost = Array.from(
    { length: outputShape[0] },
    () => Array(outputShape[1]).fill(Infinity)
  );
  cost[0][0] = 0;
  const trace = Array.from(
    { length: outputShape[0] },
    () => Array(outputShape[1]).fill(-1)
  );
  for (let j2 = 1; j2 < outputShape[1]; ++j2) {
    for (let i2 = 1; i2 < outputShape[0]; ++i2) {
      const c0 = cost[i2 - 1][j2 - 1];
      const c1 = cost[i2 - 1][j2];
      const c2 = cost[i2][j2 - 1];
      let c, t;
      if (c0 < c1 && c0 < c2) {
        c = c0;
        t = 0;
      } else if (c1 < c0 && c1 < c2) {
        c = c1;
        t = 1;
      } else {
        c = c2;
        t = 2;
      }
      cost[i2][j2] = matrix[i2 - 1][j2 - 1] + c;
      trace[i2][j2] = t;
    }
  }
  for (let i2 = 0; i2 < outputShape[1]; ++i2) {
    trace[0][i2] = 2;
  }
  for (let i2 = 0; i2 < outputShape[0]; ++i2) {
    trace[i2][0] = 1;
  }
  let i = output_length;
  let j = input_length;
  let text_indices = [];
  let time_indices = [];
  while (i > 0 || j > 0) {
    text_indices.push(i - 1);
    time_indices.push(j - 1);
    switch (trace[i][j]) {
      case 0:
        --i;
        --j;
        break;
      case 1:
        --i;
        break;
      case 2:
        --j;
        break;
      default:
        throw new Error(
          `Internal error in dynamic time warping. Unexpected trace[${i}, ${j}]. Please file a bug report.`
        );
    }
  }
  text_indices.reverse();
  time_indices.reverse();
  return [text_indices, time_indices];
}

// ignore-modules:onnxruntime-node
var onnxruntime_node_exports = {};
__export(onnxruntime_node_exports, {
  default: () => onnxruntime_node_default
});
var onnxruntime_node_default = {};

// src/backends/onnx.js
import * as ONNX_WEB from "onnxruntime-web";
import { Tensor } from "onnxruntime-common";
var DEVICE_TO_EXECUTION_PROVIDER_MAPPING = Object.freeze({
  auto: null,
  // Auto-detect based on device and environment
  gpu: null,
  // Auto-detect GPU
  cpu: "cpu",
  // CPU
  wasm: "wasm",
  // WebAssembly
  webgpu: "webgpu",
  // WebGPU
  cuda: "cuda",
  // CUDA
  dml: "dml",
  // DirectML
  webnn: { name: "webnn", deviceType: "cpu" },
  // WebNN (default)
  "webnn-npu": { name: "webnn", deviceType: "npu" },
  // WebNN NPU
  "webnn-gpu": { name: "webnn", deviceType: "gpu" },
  // WebNN GPU
  "webnn-cpu": { name: "webnn", deviceType: "cpu" }
  // WebNN CPU
});
var supportedDevices = [];
var defaultDevices;
var ONNX;
var ORT_SYMBOL = Symbol.for("onnxruntime");
if (ORT_SYMBOL in globalThis) {
  ONNX = globalThis[ORT_SYMBOL];
} else if (apis.IS_NODE_ENV) {
  ONNX = onnxruntime_node_default ?? onnxruntime_node_exports;
  switch (process.platform) {
    case "win32":
      supportedDevices.push("dml");
      break;
    case "linux":
      if (process.arch === "x64") {
        supportedDevices.push("cuda");
      }
      break;
    case "darwin":
      break;
  }
  supportedDevices.push("cpu");
  defaultDevices = ["cpu"];
} else {
  ONNX = ONNX_WEB;
  if (apis.IS_WEBNN_AVAILABLE) {
    supportedDevices.push("webnn-npu", "webnn-gpu", "webnn-cpu", "webnn");
  }
  if (apis.IS_WEBGPU_AVAILABLE) {
    supportedDevices.push("webgpu");
  }
  supportedDevices.push("wasm");
  defaultDevices = ["wasm"];
}
var InferenceSession = ONNX.InferenceSession;
function deviceToExecutionProviders(device = null) {
  if (!device) return defaultDevices;
  switch (device) {
    case "auto":
      return supportedDevices;
    case "gpu":
      return supportedDevices.filter(
        (x) => ["webgpu", "cuda", "dml", "webnn-gpu"].includes(x)
      );
  }
  if (supportedDevices.includes(device)) {
    return [DEVICE_TO_EXECUTION_PROVIDER_MAPPING[device] ?? device];
  }
  throw new Error(`Unsupported device: "${device}". Should be one of: ${supportedDevices.join(", ")}.`);
}
var wasmInitPromise = null;
async function createInferenceSession(buffer_or_path, session_options, session_config) {
  if (wasmInitPromise) {
    await wasmInitPromise;
  }
  const sessionPromise = InferenceSession.create(buffer_or_path, session_options);
  wasmInitPromise ??= sessionPromise;
  const session = await sessionPromise;
  session.config = session_config;
  return session;
}
var webInferenceChain = Promise.resolve();
var IS_WEB_ENV = apis.IS_BROWSER_ENV || apis.IS_WEBWORKER_ENV;
async function runInferenceSession(session, ortFeed) {
  const run = () => session.run(ortFeed);
  const output = await (IS_WEB_ENV ? webInferenceChain = webInferenceChain.then(run) : run());
  return output;
}
function isONNXTensor(x) {
  return x instanceof ONNX.Tensor;
}
var ONNX_ENV = ONNX?.env;
if (ONNX_ENV?.wasm) {
  if (
    // @ts-ignore Cannot find name 'ServiceWorkerGlobalScope'.ts(2304)
    !(typeof ServiceWorkerGlobalScope !== "undefined" && self instanceof ServiceWorkerGlobalScope) && !ONNX_ENV.wasm.wasmPaths
  ) {
    ONNX_ENV.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/@huggingface/transformers@${env.version}/dist/`;
  }
  ONNX_ENV.wasm.proxy = false;
}
if (ONNX_ENV?.webgpu) {
  ONNX_ENV.webgpu.powerPreference = "high-performance";
}
function isONNXProxy() {
  return ONNX_ENV?.wasm?.proxy;
}
env.backends.onnx = ONNX_ENV;

// src/ops/registry.js
var wrap = async (session_bytes, session_options, names) => {
  const session = await createInferenceSession(
    new Uint8Array(session_bytes),
    session_options
  );
  return (
    /** @type {any} */
    (async (inputs) => {
      const proxied = isONNXProxy();
      const ortFeed = Object.fromEntries(Object.entries(inputs).map(([k, v]) => [k, (proxied ? v.clone() : v).ort_tensor]));
      const outputs = await runInferenceSession(session, ortFeed);
      if (Array.isArray(names)) {
        return names.map((n) => new Tensor2(outputs[n]));
      } else {
        return new Tensor2(outputs[
          /** @type {string} */
          names
        ]);
      }
    })
  );
};
var TensorOpRegistry = class {
  static session_options = {
    // TODO: Allow for multiple execution providers
    // executionProviders: ['webgpu'],
  };
  static get nearest_interpolate_4d() {
    if (!this._nearest_interpolate_4d) {
      this._nearest_interpolate_4d = wrap(
        [8, 10, 18, 0, 58, 129, 1, 10, 41, 10, 1, 120, 10, 0, 10, 0, 10, 1, 115, 18, 1, 121, 34, 6, 82, 101, 115, 105, 122, 101, 42, 18, 10, 4, 109, 111, 100, 101, 34, 7, 110, 101, 97, 114, 101, 115, 116, 160, 1, 3, 18, 1, 114, 90, 31, 10, 1, 120, 18, 26, 10, 24, 8, 1, 18, 20, 10, 3, 18, 1, 98, 10, 3, 18, 1, 99, 10, 3, 18, 1, 104, 10, 3, 18, 1, 119, 90, 15, 10, 1, 115, 18, 10, 10, 8, 8, 7, 18, 4, 10, 2, 8, 4, 98, 31, 10, 1, 121, 18, 26, 10, 24, 8, 1, 18, 20, 10, 3, 18, 1, 98, 10, 3, 18, 1, 99, 10, 3, 18, 1, 104, 10, 3, 18, 1, 119, 66, 2, 16, 21],
        this.session_options,
        "y"
      );
    }
    return this._nearest_interpolate_4d;
  }
  static get bilinear_interpolate_4d() {
    if (!this._bilinear_interpolate_4d) {
      this._bilinear_interpolate_4d = wrap(
        [8, 9, 18, 0, 58, 128, 1, 10, 40, 10, 1, 120, 10, 0, 10, 0, 10, 1, 115, 18, 1, 121, 34, 6, 82, 101, 115, 105, 122, 101, 42, 17, 10, 4, 109, 111, 100, 101, 34, 6, 108, 105, 110, 101, 97, 114, 160, 1, 3, 18, 1, 114, 90, 31, 10, 1, 120, 18, 26, 10, 24, 8, 1, 18, 20, 10, 3, 18, 1, 98, 10, 3, 18, 1, 99, 10, 3, 18, 1, 104, 10, 3, 18, 1, 119, 90, 15, 10, 1, 115, 18, 10, 10, 8, 8, 7, 18, 4, 10, 2, 8, 4, 98, 31, 10, 1, 121, 18, 26, 10, 24, 8, 1, 18, 20, 10, 3, 18, 1, 98, 10, 3, 18, 1, 99, 10, 3, 18, 1, 104, 10, 3, 18, 1, 119, 66, 2, 16, 20],
        this.session_options,
        "y"
      );
    }
    return this._bilinear_interpolate_4d;
  }
  static get bicubic_interpolate_4d() {
    if (!this._bicubic_interpolate_4d) {
      this._bicubic_interpolate_4d = wrap(
        [8, 9, 18, 0, 58, 127, 10, 39, 10, 1, 120, 10, 0, 10, 0, 10, 1, 115, 18, 1, 121, 34, 6, 82, 101, 115, 105, 122, 101, 42, 16, 10, 4, 109, 111, 100, 101, 34, 5, 99, 117, 98, 105, 99, 160, 1, 3, 18, 1, 114, 90, 31, 10, 1, 120, 18, 26, 10, 24, 8, 1, 18, 20, 10, 3, 18, 1, 98, 10, 3, 18, 1, 99, 10, 3, 18, 1, 104, 10, 3, 18, 1, 119, 90, 15, 10, 1, 115, 18, 10, 10, 8, 8, 7, 18, 4, 10, 2, 8, 4, 98, 31, 10, 1, 121, 18, 26, 10, 24, 8, 1, 18, 20, 10, 3, 18, 1, 98, 10, 3, 18, 1, 99, 10, 3, 18, 1, 104, 10, 3, 18, 1, 119, 66, 2, 16, 20],
        this.session_options,
        "y"
      );
    }
    return this._bicubic_interpolate_4d;
  }
  static get matmul() {
    if (!this._matmul) {
      this._matmul = wrap(
        [8, 9, 18, 0, 58, 55, 10, 17, 10, 1, 97, 10, 1, 98, 18, 1, 99, 34, 6, 77, 97, 116, 77, 117, 108, 18, 1, 114, 90, 9, 10, 1, 97, 18, 4, 10, 2, 8, 1, 90, 9, 10, 1, 98, 18, 4, 10, 2, 8, 1, 98, 9, 10, 1, 99, 18, 4, 10, 2, 8, 1, 66, 2, 16, 20],
        this.session_options,
        "c"
      );
    }
    return this._matmul;
  }
  static get stft() {
    if (!this._stft) {
      this._stft = wrap(
        [8, 7, 18, 0, 58, 148, 1, 10, 38, 10, 1, 115, 10, 1, 106, 10, 1, 119, 10, 1, 108, 18, 1, 111, 34, 4, 83, 84, 70, 84, 42, 15, 10, 8, 111, 110, 101, 115, 105, 100, 101, 100, 24, 1, 160, 1, 2, 18, 1, 115, 90, 26, 10, 1, 115, 18, 21, 10, 19, 8, 1, 18, 15, 10, 3, 18, 1, 98, 10, 3, 18, 1, 115, 10, 3, 18, 1, 99, 90, 11, 10, 1, 106, 18, 6, 10, 4, 8, 7, 18, 0, 90, 16, 10, 1, 119, 18, 11, 10, 9, 8, 1, 18, 5, 10, 3, 18, 1, 119, 90, 11, 10, 1, 108, 18, 6, 10, 4, 8, 7, 18, 0, 98, 31, 10, 1, 111, 18, 26, 10, 24, 8, 1, 18, 20, 10, 3, 18, 1, 98, 10, 3, 18, 1, 102, 10, 3, 18, 1, 100, 10, 3, 18, 1, 99, 66, 2, 16, 17],
        this.session_options,
        "o"
      );
    }
    return this._stft;
  }
  static get rfft() {
    if (!this._rfft) {
      this._rfft = wrap(
        [8, 9, 18, 0, 58, 97, 10, 33, 10, 1, 120, 10, 0, 10, 1, 97, 18, 1, 121, 34, 3, 68, 70, 84, 42, 15, 10, 8, 111, 110, 101, 115, 105, 100, 101, 100, 24, 1, 160, 1, 2, 18, 1, 100, 90, 21, 10, 1, 120, 18, 16, 10, 14, 8, 1, 18, 10, 10, 3, 18, 1, 115, 10, 3, 18, 1, 99, 90, 11, 10, 1, 97, 18, 6, 10, 4, 8, 7, 18, 0, 98, 21, 10, 1, 121, 18, 16, 10, 14, 8, 1, 18, 10, 10, 3, 18, 1, 115, 10, 3, 18, 1, 99, 66, 2, 16, 20],
        this.session_options,
        "y"
      );
    }
    return this._rfft;
  }
  static get top_k() {
    if (!this._top_k) {
      this._top_k = wrap(
        [8, 10, 18, 0, 58, 73, 10, 18, 10, 1, 120, 10, 1, 107, 18, 1, 118, 18, 1, 105, 34, 4, 84, 111, 112, 75, 18, 1, 116, 90, 9, 10, 1, 120, 18, 4, 10, 2, 8, 1, 90, 15, 10, 1, 107, 18, 10, 10, 8, 8, 7, 18, 4, 10, 2, 8, 1, 98, 9, 10, 1, 118, 18, 4, 10, 2, 8, 1, 98, 9, 10, 1, 105, 18, 4, 10, 2, 8, 7, 66, 2, 16, 21],
        this.session_options,
        [
          /* Values */
          "v",
          /* Indices */
          "i"
        ]
      );
    }
    return this._top_k;
  }
  static get slice() {
    if (!this._slice) {
      this._slice = wrap(
        [8, 7, 18, 0, 58, 96, 10, 25, 10, 1, 120, 10, 1, 115, 10, 1, 101, 10, 1, 97, 10, 1, 116, 18, 1, 121, 34, 5, 83, 108, 105, 99, 101, 18, 1, 114, 90, 9, 10, 1, 120, 18, 4, 10, 2, 8, 1, 90, 9, 10, 1, 115, 18, 4, 10, 2, 8, 7, 90, 9, 10, 1, 101, 18, 4, 10, 2, 8, 7, 90, 9, 10, 1, 97, 18, 4, 10, 2, 8, 7, 90, 9, 10, 1, 116, 18, 4, 10, 2, 8, 7, 98, 9, 10, 1, 121, 18, 4, 10, 2, 8, 1, 66, 2, 16, 13],
        this.session_options,
        "y"
      );
    }
    return this._slice;
  }
};

// src/utils/tensor.js
var DataTypeMap = Object.freeze({
  float32: Float32Array,
  // @ts-ignore ts(2552) Limited availability of Float16Array across browsers:
  // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Float16Array
  float16: typeof Float16Array !== "undefined" ? Float16Array : Uint16Array,
  float64: Float64Array,
  string: Array,
  // string[]
  int8: Int8Array,
  uint8: Uint8Array,
  int16: Int16Array,
  uint16: Uint16Array,
  int32: Int32Array,
  uint32: Uint32Array,
  int64: BigInt64Array,
  uint64: BigUint64Array,
  bool: Uint8Array,
  uint4: Uint8Array,
  int4: Int8Array
});
var Tensor2 = class _Tensor {
  /** @type {number[]} Dimensions of the tensor. */
  get dims() {
    return this.ort_tensor.dims;
  }
  set dims(value) {
    this.ort_tensor.dims = value;
  }
  /** @type {DataType} Type of the tensor. */
  get type() {
    return this.ort_tensor.type;
  }
  /** @type {DataArray} The data stored in the tensor. */
  get data() {
    return this.ort_tensor.data;
  }
  /** @type {number} The number of elements in the tensor. */
  get size() {
    return this.ort_tensor.size;
  }
  /** @type {string} The location of the tensor data. */
  get location() {
    return this.ort_tensor.location;
  }
  ort_tensor;
  /**
   * Create a new Tensor or copy an existing Tensor.
   * @param {[DataType, DataArray, number[]]|[ONNXTensor]} args
   */
  constructor(...args) {
    if (isONNXTensor(args[0])) {
      this.ort_tensor = /** @type {ONNXTensor} */
      args[0];
    } else {
      this.ort_tensor = new Tensor(
        /** @type {DataType} */
        args[0],
        // @ts-expect-error ts(2769) Type 'number' is not assignable to type 'bigint'.
        /** @type {Exclude<import('./maths.js').AnyTypedArray, Uint8ClampedArray>} */
        args[1],
        args[2]
      );
    }
    return new Proxy(this, {
      get: (obj, key) => {
        if (typeof key === "string") {
          let index = Number(key);
          if (Number.isInteger(index)) {
            return obj._getitem(index);
          }
        }
        return obj[key];
      },
      set: (obj, key, value) => {
        return obj[key] = value;
      }
    });
  }
  dispose() {
    this.ort_tensor.dispose();
  }
  /**
   * Returns an iterator object for iterating over the tensor data in row-major order.
   * If the tensor has more than one dimension, the iterator will yield subarrays.
   * @returns {Iterator} An iterator object for iterating over the tensor data in row-major order.
   */
  *[Symbol.iterator]() {
    const [iterLength, ...iterDims] = this.dims;
    if (iterDims.length > 0) {
      const iterSize = iterDims.reduce((a, b) => a * b);
      for (let i = 0; i < iterLength; ++i) {
        yield this._subarray(i, iterSize, iterDims);
      }
    } else {
      yield* this.data;
    }
  }
  /**
   * Index into a Tensor object.
   * @param {number} index The index to access.
   * @returns {Tensor} The data at the specified index.
   */
  _getitem(index) {
    const [iterLength, ...iterDims] = this.dims;
    index = safeIndex(index, iterLength);
    if (iterDims.length > 0) {
      const iterSize = iterDims.reduce((a, b) => a * b);
      return this._subarray(index, iterSize, iterDims);
    } else {
      return new _Tensor(this.type, [this.data[index]], iterDims);
    }
  }
  /**
   * @param {number|bigint} item The item to search for in the tensor
   * @returns {number} The index of the first occurrence of item in the tensor data.
   */
  indexOf(item) {
    const this_data = this.data;
    for (let index = 0; index < this_data.length; ++index) {
      if (this_data[index] == item) {
        return index;
      }
    }
    return -1;
  }
  /**
   * @param {number} index
   * @param {number} iterSize
   * @param {any} iterDims
   * @returns {Tensor}
   */
  _subarray(index, iterSize, iterDims) {
    const o1 = index * iterSize;
    const o2 = (index + 1) * iterSize;
    const data = "subarray" in this.data ? this.data.subarray(o1, o2) : this.data.slice(o1, o2);
    return new _Tensor(this.type, data, iterDims);
  }
  /**
   * Returns the value of this tensor as a standard JavaScript Number. This only works
   * for tensors with one element. For other cases, see `Tensor.tolist()`.
   * @returns {number|bigint} The value of this tensor as a standard JavaScript Number.
   * @throws {Error} If the tensor has more than one element.
   */
  item() {
    const this_data = this.data;
    if (this_data.length !== 1) {
      throw new Error(`a Tensor with ${this_data.length} elements cannot be converted to Scalar`);
    }
    return this_data[0];
  }
  /**
   * Convert tensor data to a n-dimensional JS list
   * @returns {Array}
   */
  tolist() {
    return reshape(this.data, this.dims);
  }
  /**
   * Return a new Tensor with the sigmoid function applied to each element.
   * @returns {Tensor} The tensor with the sigmoid function applied.
   */
  sigmoid() {
    return this.clone().sigmoid_();
  }
  /**
   * Applies the sigmoid function to the tensor in place.
   * @returns {Tensor} Returns `this`.
   */
  sigmoid_() {
    const this_data = this.data;
    for (let i = 0; i < this_data.length; ++i) {
      this_data[i] = 1 / (1 + Math.exp(-this_data[i]));
    }
    return this;
  }
  /**
   * Return a new Tensor with a callback function applied to each element.
   * @param {Function} callback - The function to apply to each element. It should take three arguments:
   *                              the current element, its index, and the tensor's data array.
   * @returns {Tensor} A new Tensor with the callback function applied to each element.
   */
  map(callback) {
    return this.clone().map_(callback);
  }
  /**
   * Apply a callback function to each element of the tensor in place.
   * @param {Function} callback - The function to apply to each element. It should take three arguments:
   *                              the current element, its index, and the tensor's data array.
   * @returns {Tensor} Returns `this`.
   */
  map_(callback) {
    const this_data = this.data;
    for (let i = 0; i < this_data.length; ++i) {
      this_data[i] = callback(this_data[i], i, this_data);
    }
    return this;
  }
  /**
   * Return a new Tensor with every element multiplied by a constant.
   * @param {number} val The value to multiply by.
   * @returns {Tensor} The new tensor.
   */
  mul(val) {
    return this.clone().mul_(val);
  }
  /**
   * Multiply the tensor by a constant in place.
   * @param {number} val The value to multiply by.
   * @returns {Tensor} Returns `this`.
   */
  mul_(val) {
    const this_data = this.data;
    for (let i = 0; i < this_data.length; ++i) {
      this_data[i] *= val;
    }
    return this;
  }
  /**
   * Return a new Tensor with every element divided by a constant.
   * @param {number} val The value to divide by.
   * @returns {Tensor} The new tensor.
   */
  div(val) {
    return this.clone().div_(val);
  }
  /**
   * Divide the tensor by a constant in place.
   * @param {number} val The value to divide by.
   * @returns {Tensor} Returns `this`.
   */
  div_(val) {
    const this_data = this.data;
    for (let i = 0; i < this_data.length; ++i) {
      this_data[i] /= val;
    }
    return this;
  }
  /**
   * Return a new Tensor with every element added by a constant.
   * @param {number} val The value to add by.
   * @returns {Tensor} The new tensor.
   */
  add(val) {
    return this.clone().add_(val);
  }
  /**
   * Add the tensor by a constant in place.
   * @param {number} val The value to add by.
   * @returns {Tensor} Returns `this`.
   */
  add_(val) {
    const this_data = this.data;
    for (let i = 0; i < this_data.length; ++i) {
      this_data[i] += val;
    }
    return this;
  }
  /**
   * Return a new Tensor with every element subtracted by a constant.
   * @param {number} val The value to subtract by.
   * @returns {Tensor} The new tensor.
   */
  sub(val) {
    return this.clone().sub_(val);
  }
  /**
   * Subtract the tensor by a constant in place.
   * @param {number} val The value to subtract by.
   * @returns {Tensor} Returns `this`.
   */
  sub_(val) {
    const this_data = this.data;
    for (let i = 0; i < this_data.length; ++i) {
      this_data[i] -= val;
    }
    return this;
  }
  /**
   * Creates a deep copy of the current Tensor.
   * @returns {Tensor} A new Tensor with the same type, data, and dimensions as the original.
   */
  clone() {
    return new _Tensor(this.type, this.data.slice(), this.dims.slice());
  }
  /**
   * Performs a slice operation on the Tensor along specified dimensions.
   *
   * Consider a Tensor that has a dimension of [4, 7]:
   * ```
   * [ 1,  2,  3,  4,  5,  6,  7]
   * [ 8,  9, 10, 11, 12, 13, 14]
   * [15, 16, 17, 18, 19, 20, 21]
   * [22, 23, 24, 25, 26, 27, 28]
   * ```
   * We can slice against the two dims of row and column, for instance in this
   * case we can start at the second element, and return to the second last,
   * like this:
   * ```
   * tensor.slice([1, -1], [1, -1]);
   * ```
   * which would return:
   * ```
   * [  9, 10, 11, 12, 13 ]
   * [ 16, 17, 18, 19, 20 ]
   * ```
   *
   * @param {...(number|number[]|null)} slices The slice specifications for each dimension.
   * - If a number is given, then a single element is selected.
   * - If an array of two numbers is given, then a range of elements [start, end (exclusive)] is selected.
   * - If null is given, then the entire dimension is selected.
   * @returns {Tensor} A new Tensor containing the selected elements.
   * @throws {Error} If the slice input is invalid.
   */
  slice(...slices) {
    const newTensorDims = [];
    const newOffsets = [];
    for (let sliceIndex = 0; sliceIndex < this.dims.length; ++sliceIndex) {
      let slice3 = slices[sliceIndex];
      if (slice3 === null || slice3 === void 0) {
        newOffsets.push([0, this.dims[sliceIndex]]);
        newTensorDims.push(this.dims[sliceIndex]);
      } else if (typeof slice3 === "number") {
        slice3 = safeIndex(slice3, this.dims[sliceIndex], sliceIndex);
        newOffsets.push([slice3, slice3 + 1]);
      } else if (Array.isArray(slice3) && slice3.length === 2) {
        let [start, end] = slice3;
        start = start === null ? 0 : safeIndex(start, this.dims[sliceIndex], sliceIndex, false);
        end = end === null ? this.dims[sliceIndex] : safeIndex(end, this.dims[sliceIndex], sliceIndex, false);
        if (start > end) {
          throw new Error(`Invalid slice: ${slice3}`);
        }
        const offsets = [
          Math.max(start, 0),
          Math.min(end, this.dims[sliceIndex])
        ];
        newOffsets.push(offsets);
        newTensorDims.push(offsets[1] - offsets[0]);
      } else {
        throw new Error(`Invalid slice: ${slice3}`);
      }
    }
    const newDims = newOffsets.map(([start, end]) => end - start);
    const newBufferSize = newDims.reduce((a, b) => a * b);
    const this_data = this.data;
    const data = new this_data.constructor(newBufferSize);
    const stride = this.stride();
    let isContiguous = true;
    for (let i = 1; i < newDims.length; ++i) {
      if (newOffsets[i][0] !== 0 || newOffsets[i][1] !== this.dims[i]) {
        isContiguous = false;
        break;
      }
    }
    if (isContiguous) {
      const start = newOffsets[0][0] * stride[0];
      const end = newOffsets[0][1] * stride[0];
      if (ArrayBuffer.isView(this_data)) {
        data.set(this_data.subarray(start, end));
      } else if (Array.isArray(this_data)) {
        const slicedData = this_data.slice(start, end);
        for (let i = 0; i < slicedData.length; ++i) {
          data[i] = slicedData[i];
        }
      } else {
        throw new Error("Unsupported data type for slicing");
      }
    } else {
      for (let i = 0; i < newBufferSize; ++i) {
        let originalIndex = 0;
        for (let j = newDims.length - 1, num = i; j >= 0; --j) {
          const size = newDims[j];
          originalIndex += (num % size + newOffsets[j][0]) * stride[j];
          num = Math.floor(num / size);
        }
        data[i] = this_data[originalIndex];
      }
    }
    return new _Tensor(this.type, data, newTensorDims);
  }
  /**
   * Return a permuted version of this Tensor, according to the provided dimensions.
   * @param  {...number} dims Dimensions to permute.
   * @returns {Tensor} The permuted tensor.
   */
  permute(...dims) {
    return permute(this, dims);
  }
  // TODO: implement transpose. For now (backwards compatibility), it's just an alias for permute()
  transpose(...dims) {
    return this.permute(...dims);
  }
  /**
   * Returns the sum of each row of the input tensor in the given dimension dim.
   *
   * @param {number} [dim=null] The dimension or dimensions to reduce. If `null`, all dimensions are reduced.
   * @param {boolean} keepdim Whether the output tensor has `dim` retained or not.
   * @returns The summed tensor
   */
  sum(dim = null, keepdim = false) {
    return this.norm(1, dim, keepdim);
  }
  /**
   * Returns the matrix norm or vector norm of a given tensor.
   * @param {number|string} [p='fro'] The order of norm
   * @param {number} [dim=null] Specifies which dimension of the tensor to calculate the norm across.
   * If dim is None, the norm will be calculated across all dimensions of input.
   * @param {boolean} [keepdim=false] Whether the output tensors have dim retained or not.
   * @returns {Tensor} The norm of the tensor.
   */
  norm(p = "fro", dim = null, keepdim = false) {
    if (p === "fro") {
      p = 2;
    } else if (typeof p === "string") {
      throw Error(`Unsupported norm: ${p}`);
    }
    const this_data = this.data;
    const fn = (a, b) => a + b ** p;
    if (dim === null) {
      const val = this_data.reduce(fn, 0) ** (1 / p);
      return new _Tensor(this.type, [val], []);
    }
    const [type, result, resultDims] = reduce_helper(fn, this, dim, keepdim);
    if (p !== 1) {
      for (let i = 0; i < result.length; ++i) {
        result[i] = result[i] ** (1 / p);
      }
    }
    return new _Tensor(type, result, resultDims);
  }
  /**
   * Performs `L_p` normalization of inputs over specified dimension. Operates in place.
   * @param {number} [p=2] The exponent value in the norm formulation
   * @param {number} [dim=1] The dimension to reduce
   * @returns {Tensor} `this` for operation chaining.
   */
  normalize_(p = 2, dim = 1) {
    dim = safeIndex(dim, this.dims.length);
    const norm = this.norm(p, dim, true);
    const this_data = this.data;
    const norm_data = norm.data;
    for (let i = 0; i < this_data.length; ++i) {
      let resultIndex = 0;
      for (let j = this.dims.length - 1, num = i, resultMultiplier = 1; j >= 0; --j) {
        const size = this.dims[j];
        if (j !== dim) {
          const index = num % size;
          resultIndex += index * resultMultiplier;
          resultMultiplier *= this.dims[j];
        }
        num = Math.floor(num / size);
      }
      this_data[i] /= norm_data[resultIndex];
    }
    return this;
  }
  /**
   * Performs `L_p` normalization of inputs over specified dimension.
   * @param {number} [p=2] The exponent value in the norm formulation
   * @param {number} [dim=1] The dimension to reduce
   * @returns {Tensor} The normalized tensor.
   */
  normalize(p = 2, dim = 1) {
    return this.clone().normalize_(p, dim);
  }
  /**
   * Compute and return the stride of this tensor.
   * Stride is the jump necessary to go from one element to the next one in the specified dimension dim.
   * @returns {number[]} The stride of this tensor.
   */
  stride() {
    return dimsToStride(this.dims);
  }
  /**
   * Returns a tensor with all specified dimensions of input of size 1 removed.
   *
   * NOTE: The returned tensor shares the storage with the input tensor, so changing the contents of one will change the contents of the other.
   * If you would like a copy, use `tensor.clone()` before squeezing.
   *
   * @param {number|number[]} [dim=null] If given, the input will be squeezed only in the specified dimensions.
   * @returns {Tensor} The squeezed tensor
   */
  squeeze(dim = null) {
    return new _Tensor(
      this.type,
      this.data,
      calc_squeeze_dims(this.dims, dim)
    );
  }
  /**
   * In-place version of @see {@link Tensor.squeeze}
   */
  squeeze_(dim = null) {
    this.dims = calc_squeeze_dims(this.dims, dim);
    return this;
  }
  /**
   * Returns a new tensor with a dimension of size one inserted at the specified position.
   *
   * NOTE: The returned tensor shares the same underlying data with this tensor.
   *
   * @param {number} dim The index at which to insert the singleton dimension
   * @returns {Tensor} The unsqueezed tensor
   */
  unsqueeze(dim = null) {
    return new _Tensor(
      this.type,
      this.data,
      calc_unsqueeze_dims(this.dims, dim)
    );
  }
  /**
   * In-place version of @see {@link Tensor.unsqueeze}
   */
  unsqueeze_(dim = null) {
    this.dims = calc_unsqueeze_dims(this.dims, dim);
    return this;
  }
  /**
   * In-place version of @see {@link Tensor.flatten}
   */
  flatten_(start_dim = 0, end_dim = -1) {
    end_dim = (end_dim + this.dims.length) % this.dims.length;
    let dimsToKeepBefore = this.dims.slice(0, start_dim);
    let dimsToFlatten = this.dims.slice(start_dim, end_dim + 1);
    let dimsToKeepAfter = this.dims.slice(end_dim + 1);
    this.dims = [...dimsToKeepBefore, dimsToFlatten.reduce((a, b) => a * b, 1), ...dimsToKeepAfter];
    return this;
  }
  /**
   * Flattens input by reshaping it into a one-dimensional tensor.
   * If `start_dim` or `end_dim` are passed, only dimensions starting with `start_dim`
   * and ending with `end_dim` are flattened. The order of elements in input is unchanged.
   * @param {number} start_dim the first dim to flatten
   * @param {number} end_dim the last dim to flatten
   * @returns {Tensor} The flattened tensor.
   */
  flatten(start_dim = 0, end_dim = -1) {
    return this.clone().flatten_(start_dim, end_dim);
  }
  /**
   * Returns a new tensor with the same data as the `self` tensor but of a different `shape`.
   * @param  {...number} dims the desired size
   * @returns {Tensor} The tensor with the same data but different shape
   */
  view(...dims) {
    let inferredIndex = -1;
    for (let i = 0; i < dims.length; ++i) {
      if (dims[i] === -1) {
        if (inferredIndex !== -1) {
          throw new Error("Only one dimension can be inferred");
        }
        inferredIndex = i;
      }
    }
    const this_data = this.data;
    if (inferredIndex !== -1) {
      const productOther = dims.reduce((product2, curr, index) => {
        return index !== inferredIndex ? product2 * curr : product2;
      }, 1);
      dims[inferredIndex] = this_data.length / productOther;
    }
    return new _Tensor(this.type, this_data, dims);
  }
  neg_() {
    const this_data = this.data;
    for (let i = 0; i < this_data.length; ++i) {
      this_data[i] = -this_data[i];
    }
    return this;
  }
  neg() {
    return this.clone().neg_();
  }
  /**
   * Computes input > val element-wise.
   * @param {number} val The value to compare with.
   * @returns {Tensor} A boolean tensor that is `true` where input is greater than other and `false` elsewhere.
   */
  gt(val) {
    const mask = new Uint8Array(this.data.length);
    const this_data = this.data;
    for (let i = 0; i < this_data.length; ++i) {
      mask[i] = this_data[i] > val ? 1 : 0;
    }
    return new _Tensor("bool", mask, this.dims);
  }
  /**
   * Computes input < val element-wise.
   * @param {number} val The value to compare with.
   * @returns {Tensor} A boolean tensor that is `true` where input is less than other and `false` elsewhere.
   */
  lt(val) {
    const mask = new Uint8Array(this.data.length);
    const this_data = this.data;
    for (let i = 0; i < this_data.length; ++i) {
      mask[i] = this_data[i] < val ? 1 : 0;
    }
    return new _Tensor("bool", mask, this.dims);
  }
  /**
   * In-place version of @see {@link Tensor.clamp}
   */
  clamp_(min2, max2) {
    const this_data = this.data;
    for (let i = 0; i < this_data.length; ++i) {
      this_data[i] = Math.min(Math.max(this_data[i], min2), max2);
    }
    return this;
  }
  /**
   * Clamps all elements in input into the range [ min, max ]
   * @param {number} min lower-bound of the range to be clamped to
   * @param {number} max upper-bound of the range to be clamped to
   * @returns {Tensor} the output tensor.
   */
  clamp(min2, max2) {
    return this.clone().clamp_(min2, max2);
  }
  /**
   * In-place version of @see {@link Tensor.round}
   */
  round_() {
    const this_data = this.data;
    for (let i = 0; i < this_data.length; ++i) {
      this_data[i] = Math.round(this_data[i]);
    }
    return this;
  }
  /**
   * Rounds elements of input to the nearest integer.
   * @returns {Tensor} the output tensor.
   */
  round() {
    return this.clone().round_();
  }
  mean(dim = null, keepdim = false) {
    return mean(this, dim, keepdim);
  }
  min(dim = null, keepdim = false) {
    if (dim === null) {
      const val = min(this.data)[0];
      return new _Tensor(this.type, [val], [
        /* scalar */
      ]);
    }
    const [type, result, resultDims] = reduce_helper((a, b) => Math.min(a, b), this, dim, keepdim, Infinity);
    return new _Tensor(type, result, resultDims);
  }
  max(dim = null, keepdim = false) {
    if (dim === null) {
      const val = max(this.data)[0];
      return new _Tensor(this.type, [val], [
        /* scalar */
      ]);
    }
    const [type, result, resultDims] = reduce_helper((a, b) => Math.max(a, b), this, dim, keepdim, -Infinity);
    return new _Tensor(type, result, resultDims);
  }
  argmin(dim = null, keepdim = false) {
    if (dim !== null) {
      throw new Error("`dim !== null` not yet implemented.");
    }
    const index = min(this.data)[1];
    return new _Tensor("int64", [BigInt(index)], []);
  }
  argmax(dim = null, keepdim = false) {
    if (dim !== null) {
      throw new Error("`dim !== null` not yet implemented.");
    }
    const index = max(this.data)[1];
    return new _Tensor("int64", [BigInt(index)], []);
  }
  /**
   * Performs Tensor dtype conversion.
   * @param {DataType} type The desired data type.
   * @returns {Tensor} The converted tensor.
   */
  to(type) {
    if (this.type === type) return this;
    if (!DataTypeMap.hasOwnProperty(type)) {
      throw new Error(`Unsupported type: ${type}`);
    }
    let map_fn;
    const is_source_bigint = ["int64", "uint64"].includes(this.type);
    const is_dest_bigint = ["int64", "uint64"].includes(type);
    if (is_source_bigint && !is_dest_bigint) {
      map_fn = Number;
    } else if (!is_source_bigint && is_dest_bigint) {
      map_fn = BigInt;
    }
    return new _Tensor(type, DataTypeMap[type].from(this.data, map_fn), this.dims);
  }
};
function reshape(data, dimensions) {
  const totalElements = data.length;
  const dimensionSize = dimensions.reduce((a, b) => a * b);
  if (totalElements !== dimensionSize) {
    throw Error(`cannot reshape array of size ${totalElements} into shape (${dimensions})`);
  }
  let reshapedArray = data;
  for (let i = dimensions.length - 1; i >= 0; i--) {
    reshapedArray = reshapedArray.reduce((acc, val) => {
      let lastArray = acc[acc.length - 1];
      if (lastArray.length < dimensions[i]) {
        lastArray.push(val);
      } else {
        acc.push([val]);
      }
      return acc;
    }, [[]]);
  }
  return reshapedArray[0];
}
function permute(tensor, axes) {
  const [permutedData, shape] = permute_data(tensor.data, tensor.dims, axes);
  return new Tensor2(tensor.type, permutedData, shape);
}
function interpolate(input, [out_height, out_width], mode = "bilinear", align_corners = false) {
  const in_channels = input.dims.at(-3) ?? 1;
  const in_height = input.dims.at(-2);
  const in_width = input.dims.at(-1);
  let output = interpolate_data(
    /** @type {import('./maths.js').TypedArray}*/
    input.data,
    [in_channels, in_height, in_width],
    [out_height, out_width],
    mode,
    align_corners
  );
  return new Tensor2(input.type, output, [in_channels, out_height, out_width]);
}
async function interpolate_4d(input, {
  size = null,
  mode = "bilinear"
} = {}) {
  if (input.dims.length !== 4) {
    throw new Error("`interpolate_4d` currently only supports 4D input.");
  }
  if (!size) {
    throw new Error("`interpolate_4d` requires a `size` argument.");
  }
  let targetDims;
  if (size.length === 2) {
    targetDims = [...input.dims.slice(0, 2), ...size];
  } else if (size.length === 3) {
    targetDims = [input.dims[0], ...size];
  } else if (size.length === 4) {
    targetDims = size;
  } else {
    throw new Error("`size` must be of length 2, 3, or 4.");
  }
  let op;
  if (mode === "nearest") {
    op = await TensorOpRegistry.nearest_interpolate_4d;
  } else if (mode === "bilinear") {
    op = await TensorOpRegistry.bilinear_interpolate_4d;
  } else if (mode === "bicubic") {
    op = await TensorOpRegistry.bicubic_interpolate_4d;
  } else {
    throw new Error(`Unsupported mode: ${mode}`);
  }
  const sizeTensor = new Tensor2("int64", new BigInt64Array(targetDims.map(BigInt)), [targetDims.length]);
  return await op({ x: input, s: sizeTensor });
}
async function matmul(a, b) {
  const op = await TensorOpRegistry.matmul;
  return await op({ a, b });
}
async function rfft(x, a) {
  const op = await TensorOpRegistry.rfft;
  return await op({ x, a });
}
async function topk(x, k) {
  const op = await TensorOpRegistry.top_k;
  if (k == null) {
    k = x.dims.at(-1);
  } else {
    k = Math.min(k, x.dims.at(-1));
  }
  return await op({
    x,
    k: new Tensor2(
      "int64",
      [BigInt(k)],
      [1]
    )
  });
}
var arrayToIndexTensor = (array) => new Tensor2("int64", array, [array.length]);
async function slice(data, starts, ends, axes, steps) {
  const op = await TensorOpRegistry.slice;
  return await op({
    x: data,
    s: arrayToIndexTensor(starts),
    e: arrayToIndexTensor(ends),
    a: arrayToIndexTensor(axes),
    t: arrayToIndexTensor(steps ?? new Array(axes.length).fill(1))
  });
}
function mean_pooling(last_hidden_state, attention_mask) {
  const lastHiddenStateData = last_hidden_state.data;
  const attentionMaskData = attention_mask.data;
  const shape = [last_hidden_state.dims[0], last_hidden_state.dims[2]];
  const returnedData = new lastHiddenStateData.constructor(shape[0] * shape[1]);
  const [batchSize, seqLength, embedDim] = last_hidden_state.dims;
  let outIndex = 0;
  for (let i = 0; i < batchSize; ++i) {
    const offset = i * embedDim * seqLength;
    for (let k = 0; k < embedDim; ++k) {
      let sum = 0;
      let count2 = 0;
      const attnMaskOffset = i * seqLength;
      const offset2 = offset + k;
      for (let j = 0; j < seqLength; ++j) {
        const attn = Number(attentionMaskData[attnMaskOffset + j]);
        count2 += attn;
        sum += lastHiddenStateData[offset2 + j * embedDim] * attn;
      }
      const avg = sum / count2;
      returnedData[outIndex++] = avg;
    }
  }
  return new Tensor2(
    last_hidden_state.type,
    returnedData,
    shape
  );
}
function layer_norm(input, normalized_shape, {
  eps = 1e-5
} = {}) {
  if (input.dims.length !== 2) {
    throw new Error("`layer_norm` currently only supports 2D input.");
  }
  const [batchSize, featureDim] = input.dims;
  if (normalized_shape.length !== 1 && normalized_shape[0] !== featureDim) {
    throw new Error("`normalized_shape` must be a 1D array with shape `[input.dims[1]]`.");
  }
  const [std, mean2] = std_mean(input, 1, 0, true);
  const stdData = (
    /** @type {Float32Array} */
    std.data
  );
  const meanData = (
    /** @type {Float32Array} */
    mean2.data
  );
  const inputData = (
    /** @type {Float32Array} */
    input.data
  );
  const returnedData = new inputData.constructor(inputData.length);
  for (let i = 0; i < batchSize; ++i) {
    const offset = i * featureDim;
    for (let j = 0; j < featureDim; ++j) {
      const offset2 = offset + j;
      returnedData[offset2] = (inputData[offset2] - meanData[i]) / (stdData[i] + eps);
    }
  }
  return new Tensor2(input.type, returnedData, input.dims);
}
function calc_squeeze_dims(dims, dim) {
  dims = dims.slice();
  if (dim === null) {
    dims = dims.filter((d) => d !== 1);
  } else if (typeof dim === "number") {
    if (dims[dim] === 1) {
      dims.splice(dim, 1);
    }
  } else if (Array.isArray(dim)) {
    dims = dims.filter((x, i) => {
      return x !== 1 || !dim.includes(i);
    });
  }
  return dims;
}
function calc_unsqueeze_dims(dims, dim) {
  dim = safeIndex(dim, dims.length + 1);
  dims = dims.slice();
  dims.splice(dim, 0, 1);
  return dims;
}
function safeIndex(index, size, dimension = null, boundsCheck = true) {
  if (index < -size || index >= size) {
    if (boundsCheck) {
      throw new Error(`IndexError: index ${index} is out of bounds for dimension${dimension === null ? "" : " " + dimension} with size ${size}`);
    } else {
      return index < -size ? 0 : size;
    }
  }
  if (index < 0) {
    index = (index % size + size) % size;
  }
  return index;
}
function cat(tensors, dim = 0) {
  dim = safeIndex(dim, tensors[0].dims.length);
  const resultDims = tensors[0].dims.slice();
  resultDims[dim] = tensors.reduce((a, b) => a + b.dims[dim], 0);
  const resultSize = resultDims.reduce((a, b) => a * b, 1);
  const result = new tensors[0].data.constructor(resultSize);
  const resultType = tensors[0].type;
  if (dim === 0) {
    let offset = 0;
    for (const tensor of tensors) {
      const tensorData = tensor.data;
      result.set(tensorData, offset);
      offset += tensorData.length;
    }
  } else {
    let currentDim = 0;
    for (let t = 0; t < tensors.length; ++t) {
      const { data, dims } = tensors[t];
      for (let i = 0; i < data.length; ++i) {
        let resultIndex = 0;
        for (let j = dims.length - 1, num = i, resultMultiplier = 1; j >= 0; --j) {
          const size = dims[j];
          let index = num % size;
          if (j === dim) {
            index += currentDim;
          }
          resultIndex += index * resultMultiplier;
          resultMultiplier *= resultDims[j];
          num = Math.floor(num / size);
        }
        result[resultIndex] = data[i];
      }
      currentDim += dims[dim];
    }
  }
  return new Tensor2(resultType, result, resultDims);
}
function stack(tensors, dim = 0) {
  return cat(tensors.map((t) => t.unsqueeze(dim)), dim);
}
function reduce_helper(callbackfn, input, dim = null, keepdim = false, initialValue = null) {
  const inputData = input.data;
  const inputDims = input.dims;
  dim = safeIndex(dim, inputDims.length);
  const resultDims = inputDims.slice();
  resultDims[dim] = 1;
  const result = new inputData.constructor(inputData.length / inputDims[dim]);
  if (initialValue !== null) {
    result.fill(initialValue);
  }
  for (let i = 0; i < inputData.length; ++i) {
    let resultIndex = 0;
    for (let j = inputDims.length - 1, num = i, resultMultiplier = 1; j >= 0; --j) {
      const size = inputDims[j];
      if (j !== dim) {
        const index = num % size;
        resultIndex += index * resultMultiplier;
        resultMultiplier *= resultDims[j];
      }
      num = Math.floor(num / size);
    }
    result[resultIndex] = callbackfn(result[resultIndex], inputData[i], i, resultIndex);
  }
  if (!keepdim) resultDims.splice(dim, 1);
  return [input.type, result, resultDims];
}
function std_mean(input, dim = null, correction = 1, keepdim = false) {
  const inputData = (
    /** @type {Float32Array} */
    input.data
  );
  const inputDims = input.dims;
  if (dim === null) {
    const sum = inputData.reduce((a, b) => a + b, 0);
    const mean2 = sum / inputData.length;
    const std = Math.sqrt(inputData.reduce((a, b) => a + (b - mean2) ** 2, 0) / (inputData.length - correction));
    const meanTensor2 = new Tensor2(input.type, [mean2], [
      /* scalar */
    ]);
    const stdTensor2 = new Tensor2(input.type, [std], [
      /* scalar */
    ]);
    return [stdTensor2, meanTensor2];
  }
  dim = safeIndex(dim, inputDims.length);
  const meanTensor = mean(input, dim, keepdim);
  const meanTensorData = meanTensor.data;
  const [type, result, resultDims] = reduce_helper((a, b, i, j) => a + (b - meanTensorData[j]) ** 2, input, dim, keepdim);
  for (let i = 0; i < result.length; ++i) {
    result[i] = Math.sqrt(result[i] / (inputDims[dim] - correction));
  }
  const stdTensor = new Tensor2(type, result, resultDims);
  return [stdTensor, meanTensor];
}
function mean(input, dim = null, keepdim = false) {
  const inputDims = input.dims;
  const inputData = (
    /** @type {Float32Array} */
    input.data
  );
  if (dim === null) {
    const val = inputData.reduce((a, b) => a + b, 0);
    return new Tensor2(input.type, [val / inputData.length], [
      /* scalar */
    ]);
  }
  dim = safeIndex(dim, inputDims.length);
  const [type, result, resultDims] = reduce_helper((a, b) => a + b, input, dim, keepdim);
  if (inputDims[dim] !== 1) {
    for (let i = 0; i < result.length; ++i) {
      result[i] /= inputDims[dim];
    }
  }
  return new Tensor2(type, result, resultDims);
}
function dimsToStride(dims) {
  const stride = new Array(dims.length);
  for (let i = dims.length - 1, s2 = 1; i >= 0; --i) {
    stride[i] = s2;
    s2 *= dims[i];
  }
  return stride;
}
function fullHelper(size, fill_value, dtype, cls) {
  const numElements = size.reduce((a, b) => a * b, 1);
  return new Tensor2(
    dtype,
    new cls(numElements).fill(fill_value),
    size
  );
}
function full(size, fill_value) {
  let dtype;
  let typedArrayCls;
  if (typeof fill_value === "number") {
    dtype = "float32";
    typedArrayCls = Float32Array;
  } else if (typeof fill_value === "bigint") {
    dtype = "int64";
    typedArrayCls = BigInt64Array;
  } else if (typeof fill_value === "boolean") {
    dtype = "bool";
    typedArrayCls = Uint8Array;
  } else {
    throw new Error(`Unsupported data type: ${typeof fill_value}`);
  }
  return fullHelper(size, fill_value, dtype, typedArrayCls);
}
function full_like(tensor, fill_value) {
  return full(tensor.dims, fill_value);
}
function ones(size) {
  return fullHelper(size, 1n, "int64", BigInt64Array);
}
function ones_like(tensor) {
  return ones(tensor.dims);
}
function zeros(size) {
  return fullHelper(size, 0n, "int64", BigInt64Array);
}
function zeros_like(tensor) {
  return zeros(tensor.dims);
}
function rand(size) {
  const length = size.reduce((a, b) => a * b, 1);
  return new Tensor2(
    "float32",
    Float32Array.from({ length }, () => Math.random()),
    size
  );
}
function quantize_embeddings(tensor, precision) {
  if (tensor.dims.length !== 2) {
    throw new Error("The tensor must have 2 dimensions");
  }
  if (tensor.dims.at(-1) % 8 !== 0) {
    throw new Error("The last dimension of the tensor must be a multiple of 8");
  }
  if (!["binary", "ubinary"].includes(precision)) {
    throw new Error("The precision must be either 'binary' or 'ubinary'");
  }
  const signed = precision === "binary";
  const dtype = signed ? "int8" : "uint8";
  const cls = signed ? Int8Array : Uint8Array;
  const inputData = tensor.data;
  const outputData = new cls(inputData.length / 8);
  for (let i = 0; i < inputData.length; ++i) {
    const bit = inputData[i] > 0 ? 1 : 0;
    const arrayIndex = Math.floor(i / 8);
    const bitPosition = i % 8;
    outputData[arrayIndex] |= bit << 7 - bitPosition;
    if (signed && bitPosition === 0) {
      outputData[arrayIndex] -= 128;
    }
  }
  ;
  return new Tensor2(dtype, outputData, [tensor.dims[0], tensor.dims[1] / 8]);
}

// src/utils/data-structures.js
var PriorityQueue = class {
  /**
   * Create a new PriorityQueue.
   * @param {function(any, any): boolean} comparator Comparator function to determine priority. Defaults to a MaxHeap.
   */
  constructor(comparator = (a, b) => a > b, maxSize = Infinity) {
    this._heap = [];
    this._comparator = comparator;
    this._maxSize = maxSize;
  }
  /**
   * The size of the queue
   */
  get size() {
    return this._heap.length;
  }
  /**
   * Check if the queue is empty.
   * @returns {boolean} `true` if the queue is empty, `false` otherwise.
   */
  isEmpty() {
    return this.size === 0;
  }
  /**
   * Return the element with the highest priority in the queue.
   * @returns {any} The highest priority element in the queue.
   */
  peek() {
    return this._heap[0];
  }
  /**
   * Add one or more elements to the queue.
   * @param  {...any} values The values to push into the queue.
   * @returns {number} The new size of the queue.
   */
  push(...values) {
    return this.extend(values);
  }
  /**
   * Add multiple elements to the queue.
   * @param {any[]} values The values to push into the queue.
   * @returns {number} The new size of the queue.
   */
  extend(values) {
    for (const value of values) {
      if (this.size < this._maxSize) {
        this._heap.push(value);
        this._siftUp();
      } else {
        const smallest = this._smallest();
        if (this._comparator(value, this._heap[smallest])) {
          this._heap[smallest] = value;
          this._siftUpFrom(smallest);
        }
      }
    }
    return this.size;
  }
  /**
   * Remove and return the element with the highest priority in the queue.
   * @returns {any} The element with the highest priority in the queue.
   */
  pop() {
    const poppedValue = this.peek();
    const bottom = this.size - 1;
    if (bottom > 0) {
      this._swap(0, bottom);
    }
    this._heap.pop();
    this._siftDown();
    return poppedValue;
  }
  /**
   * Replace the element with the highest priority in the queue with a new value.
   * @param {*} value The new value.
   * @returns {*} The replaced value.
   */
  replace(value) {
    const replacedValue = this.peek();
    this._heap[0] = value;
    this._siftDown();
    return replacedValue;
  }
  /**
   * Compute the index for the parent of the node at index `i`.
   * @param {number} i The index of the node to get the parent of.
   * @returns {number} The index of the parent node.
   * @private
   */
  _parent(i) {
    return (i + 1 >>> 1) - 1;
  }
  /**
   * Compute the index for the left child of the node at index `i`.
   * @param {number} i The index of the node to get the left child of.
   * @returns {number} The index of the left child.
   * @private
   */
  _left(i) {
    return (i << 1) + 1;
  }
  /**
   * Compute the index for the right child of the node at index `i`.
   * @param {number} i The index of the node to get the right child of.
   * @returns {number} The index of the right child.
   * @private
   */
  _right(i) {
    return i + 1 << 1;
  }
  /**
   * Check if the element at index `i` is greater than the element at index `j`.
   * @param {number} i The index of the first element to compare.
   * @param {number} j The index of the second element to compare.
   * @returns {boolean} `true` if the element at index `i` is greater than the element at index `j`, `false` otherwise.
   * @private
   */
  _greater(i, j) {
    return this._comparator(this._heap[i], this._heap[j]);
  }
  /**
   * Swap the elements at indices `i` and `j`.
   * @param {number} i The index of the first element to swap.
   * @param {number} j The index of the second element to swap.
   * @private
   */
  _swap(i, j) {
    const temp = this._heap[i];
    this._heap[i] = this._heap[j];
    this._heap[j] = temp;
  }
  /**
   * Maintain the heap property by updating positions in the heap,
   * starting at the last element and moving up the heap.
   * @private
   */
  _siftUp() {
    this._siftUpFrom(this.size - 1);
  }
  /**
   * Helper function to sift up from a given node.
   * @param {number} node The index of the node to start sifting up from.
   */
  _siftUpFrom(node) {
    while (node > 0 && this._greater(node, this._parent(node))) {
      this._swap(node, this._parent(node));
      node = this._parent(node);
    }
  }
  /**
   * Maintain the heap property by updating positions in the heap,
   * starting at the first element and moving down the heap.
   * @private
   */
  _siftDown() {
    let node = 0;
    while (this._left(node) < this.size && this._greater(this._left(node), node) || this._right(node) < this.size && this._greater(this._right(node), node)) {
      const maxChild = this._right(node) < this.size && this._greater(this._right(node), this._left(node)) ? this._right(node) : this._left(node);
      this._swap(node, maxChild);
      node = maxChild;
    }
  }
  /**
   * Get the index of the smallest element in the heap. Since we use an array-based heap,
   * the index can be computed without needing to traverse the heap.
   * @private
   */
  _smallest() {
    return 2 ** Math.floor(Math.log2(this.size)) - 1;
  }
};
var CharTrie = class {
  constructor() {
    this.root = CharTrieNode.default();
  }
  /**
   * Adds one or more `texts` to the trie.
   * @param {string[]} texts The strings to add to the trie.
   */
  extend(texts) {
    for (const text of texts) {
      this.push(text);
    }
  }
  /**
   * Adds text to the trie.
   * @param {string} text The string to add to the trie.
   */
  push(text) {
    let node = this.root;
    for (const ch of text) {
      let child = node.children.get(ch);
      if (child === void 0) {
        child = CharTrieNode.default();
        node.children.set(ch, child);
      }
      node = child;
    }
    node.isLeaf = true;
  }
  /**
   * Searches the trie for all strings with a common prefix of `text`.
   * @param {string} text The common prefix to search for.
   * @yields {string} Each string in the trie that has `text` as a prefix.
   */
  *commonPrefixSearch(text) {
    let node = this.root;
    if (node === void 0) return;
    let prefix = "";
    for (const ch of text) {
      prefix += ch;
      node = node.children.get(ch);
      if (node === void 0) return;
      if (node.isLeaf) {
        yield prefix;
      }
    }
  }
};
var CharTrieNode = class _CharTrieNode {
  /**
   * Create a new CharTrieNode.
   * @param {boolean} isLeaf Whether the node is a leaf node or not.
   * @param {Map<string, CharTrieNode>} children A map containing the node's children, where the key is a character and the value is a `CharTrieNode`.
   */
  constructor(isLeaf, children) {
    this.isLeaf = isLeaf;
    this.children = children;
  }
  /**
   * Returns a new `CharTrieNode` instance with default values.
   * @returns {CharTrieNode} A new `CharTrieNode` instance with `isLeaf` set to `false` and an empty `children` map.
   */
  static default() {
    return new _CharTrieNode(false, /* @__PURE__ */ new Map());
  }
};
var TokenLattice = class {
  /**
   * Creates a new TokenLattice instance.
   *
   * @param {string} sentence The input sentence to be tokenized.
   * @param {number} bosTokenId The beginning-of-sequence token ID.
   * @param {number} eosTokenId The end-of-sequence token ID.
   */
  constructor(sentence, bosTokenId, eosTokenId) {
    this.chars = Array.from(sentence);
    this.len = this.chars.length;
    this.bosTokenId = bosTokenId;
    this.eosTokenId = eosTokenId;
    this.nodes = [];
    this.beginNodes = Array.from({ length: this.len + 1 }, () => []);
    this.endNodes = Array.from({ length: this.len + 1 }, () => []);
    const bos = new TokenLatticeNode(this.bosTokenId, 0, 0, 0, 0);
    const eos = new TokenLatticeNode(this.eosTokenId, 1, this.len, 0, 0);
    this.nodes.push(bos.clone());
    this.nodes.push(eos.clone());
    this.beginNodes[this.len].push(eos);
    this.endNodes[0].push(bos);
  }
  /**
   * Inserts a new token node into the token lattice.
   *
   * @param {number} pos The starting position of the token.
   * @param {number} length The length of the token.
   * @param {number} score The score of the token.
   * @param {number} tokenId The token ID of the token.
   */
  insert(pos, length, score, tokenId) {
    const nodeId = this.nodes.length;
    const node = new TokenLatticeNode(tokenId, nodeId, pos, length, score);
    this.beginNodes[pos].push(node);
    this.endNodes[pos + length].push(node);
    this.nodes.push(node);
  }
  /**
   * Implements the Viterbi algorithm to compute the most likely sequence of tokens.
   *
   * @returns {TokenLatticeNode[]} The most likely sequence of tokens.
   */
  viterbi() {
    const len2 = this.len;
    let pos = 0;
    while (pos <= len2) {
      if (this.beginNodes[pos].length == 0) {
        return [];
      }
      for (let rnode of this.beginNodes[pos]) {
        rnode.prev = null;
        let bestScore = 0;
        let bestNode = null;
        for (let lnode of this.endNodes[pos]) {
          const score = lnode.backtraceScore + rnode.score;
          if (bestNode === null || score > bestScore) {
            bestNode = lnode.clone();
            bestScore = score;
          }
        }
        if (bestNode !== null) {
          rnode.prev = bestNode;
          rnode.backtraceScore = bestScore;
        } else {
          return [];
        }
      }
      ++pos;
    }
    const results = [];
    const root = this.beginNodes[len2][0];
    const prev = root.prev;
    if (prev === null) {
      return [];
    }
    let node = prev.clone();
    while (node.prev !== null) {
      results.push(node.clone());
      const n = node.clone();
      node = n.prev.clone();
    }
    results.reverse();
    return results;
  }
  /**
   * @param {TokenLatticeNode} node
   * @returns {string} The array of nodes representing the most likely sequence of tokens.
   */
  piece(node) {
    return this.chars.slice(node.pos, node.pos + node.length).join("");
  }
  /**
   * @returns {string[]} The most likely sequence of tokens.
   */
  tokens() {
    const nodes = this.viterbi();
    return nodes.map((x) => this.piece(x));
  }
  /**
   * @returns {number[]} The most likely sequence of token ids.
   */
  tokenIds() {
    const nodes = this.viterbi();
    return nodes.map((x) => x.tokenId);
  }
};
var TokenLatticeNode = class _TokenLatticeNode {
  /**
   * Represents a node in a token lattice for a given sentence.
   * @param {number} tokenId The ID of the token associated with this node.
   * @param {number} nodeId The ID of this node.
   * @param {number} pos The starting position of the token in the sentence.
   * @param {number} length The length of the token.
   * @param {number} score The score associated with the token.
   */
  constructor(tokenId, nodeId, pos, length, score) {
    this.tokenId = tokenId;
    this.nodeId = nodeId;
    this.pos = pos;
    this.length = length;
    this.score = score;
    this.prev = null;
    this.backtraceScore = 0;
  }
  /**
   * Returns a clone of this node.
   * @returns {TokenLatticeNode} A clone of this node.
   */
  clone() {
    const n = new _TokenLatticeNode(this.tokenId, this.nodeId, this.pos, this.length, this.score);
    n.prev = this.prev;
    n.backtraceScore = this.backtraceScore;
    return n;
  }
};
var DictionarySplitter = class {
  /**
   * @param {string[]} dictionary The dictionary of words to use for splitting.
   */
  constructor(dictionary) {
    this.trie = this._buildTrie(dictionary);
  }
  /**
   * Builds a trie from the given dictionary.
   * @param {string[]} dictionary The dictionary of words to build the trie from.
   * @returns {Object} The root node of the trie.
   * @private
   */
  _buildTrie(dictionary) {
    const trie = /* @__PURE__ */ Object.create(null);
    for (const word of dictionary) {
      let node = trie;
      for (let i = 0; i < word.length; ++i) {
        node = node[word[i]] ??= /* @__PURE__ */ Object.create(null);
      }
      node.end = word;
    }
    return trie;
  }
  /**
   * Splits the input text into tokens based on the dictionary.
   * @param {string} text The input text to split.
   * @returns {string[]} An array of tokens.
   */
  split(text) {
    const result = [];
    const n = text.length;
    let start = 0;
    let i = 0;
    while (i < n) {
      let node = this.trie;
      let match = null;
      let j = i;
      while (j < n && (node = node[text[j]])) {
        if (node.end) {
          match = node.end;
        }
        ++j;
      }
      if (match) {
        if (i > start) {
          result.push(text.slice(start, i));
        }
        result.push(match);
        i += match.length;
        start = i;
      } else {
        ++i;
      }
    }
    if (start < n) {
      result.push(text.slice(start));
    }
    return result;
  }
};
var LRUCache = class {
  /**
   * Creates an LRUCache instance.
   * @param {number} capacity The maximum number of items the cache can hold.
   */
  constructor(capacity) {
    this.capacity = capacity;
    this.cache = /* @__PURE__ */ new Map();
  }
  /**
   * Retrieves the value associated with the given key and marks the key as recently used.
   * @param {any} key The key to retrieve.
   * @returns {any} The value associated with the key, or undefined if the key does not exist.
   */
  get(key) {
    if (!this.cache.has(key)) return void 0;
    const value = this.cache.get(key);
    this.cache.delete(key);
    this.cache.set(key, value);
    return value;
  }
  /**
   * Inserts or updates the key-value pair in the cache.
   * If the key already exists, it is updated and marked as recently used.
   * If the cache exceeds its capacity, the least recently used item is evicted.
   * @param {any} key The key to add or update.
   * @param {any} value The value to associate with the key.
   */
  put(key, value) {
    if (this.cache.has(key)) {
      this.cache.delete(key);
    }
    this.cache.set(key, value);
    if (this.cache.size > this.capacity) {
      this.cache.delete(this.cache.keys().next().value);
    }
  }
  /**
   * Clears the cache.
   */
  clear() {
    this.cache.clear();
  }
};

// node_modules/@huggingface/jinja/dist/index.js
var TOKEN_TYPES = Object.freeze({
  Text: "Text",
  // The text between Jinja statements or expressions
  NumericLiteral: "NumericLiteral",
  // e.g., 123, 1.0
  StringLiteral: "StringLiteral",
  // 'string'
  Identifier: "Identifier",
  // Variables, functions, statements, booleans, etc.
  Equals: "Equals",
  // =
  OpenParen: "OpenParen",
  // (
  CloseParen: "CloseParen",
  // )
  OpenStatement: "OpenStatement",
  // {%
  CloseStatement: "CloseStatement",
  // %}
  OpenExpression: "OpenExpression",
  // {{
  CloseExpression: "CloseExpression",
  // }}
  OpenSquareBracket: "OpenSquareBracket",
  // [
  CloseSquareBracket: "CloseSquareBracket",
  // ]
  OpenCurlyBracket: "OpenCurlyBracket",
  // {
  CloseCurlyBracket: "CloseCurlyBracket",
  // }
  Comma: "Comma",
  // ,
  Dot: "Dot",
  // .
  Colon: "Colon",
  // :
  Pipe: "Pipe",
  // |
  CallOperator: "CallOperator",
  // ()
  AdditiveBinaryOperator: "AdditiveBinaryOperator",
  // + - ~
  MultiplicativeBinaryOperator: "MultiplicativeBinaryOperator",
  // * / %
  ComparisonBinaryOperator: "ComparisonBinaryOperator",
  // < > <= >= == !=
  UnaryOperator: "UnaryOperator",
  // ! - +
  Comment: "Comment"
  // {# ... #}
});
var Token = class {
  /**
   * Constructs a new Token.
   * @param {string} value The raw value as seen inside the source code.
   * @param {TokenType} type The type of token.
   */
  constructor(value, type) {
    this.value = value;
    this.type = type;
  }
};
function isWord(char) {
  return /\w/.test(char);
}
function isInteger(char) {
  return /[0-9]/.test(char);
}
var ORDERED_MAPPING_TABLE = [
  // Control sequences
  ["{%", TOKEN_TYPES.OpenStatement],
  ["%}", TOKEN_TYPES.CloseStatement],
  ["{{", TOKEN_TYPES.OpenExpression],
  ["}}", TOKEN_TYPES.CloseExpression],
  // Single character tokens
  ["(", TOKEN_TYPES.OpenParen],
  [")", TOKEN_TYPES.CloseParen],
  ["{", TOKEN_TYPES.OpenCurlyBracket],
  ["}", TOKEN_TYPES.CloseCurlyBracket],
  ["[", TOKEN_TYPES.OpenSquareBracket],
  ["]", TOKEN_TYPES.CloseSquareBracket],
  [",", TOKEN_TYPES.Comma],
  [".", TOKEN_TYPES.Dot],
  [":", TOKEN_TYPES.Colon],
  ["|", TOKEN_TYPES.Pipe],
  // Comparison operators
  ["<=", TOKEN_TYPES.ComparisonBinaryOperator],
  [">=", TOKEN_TYPES.ComparisonBinaryOperator],
  ["==", TOKEN_TYPES.ComparisonBinaryOperator],
  ["!=", TOKEN_TYPES.ComparisonBinaryOperator],
  ["<", TOKEN_TYPES.ComparisonBinaryOperator],
  [">", TOKEN_TYPES.ComparisonBinaryOperator],
  // Arithmetic operators
  ["+", TOKEN_TYPES.AdditiveBinaryOperator],
  ["-", TOKEN_TYPES.AdditiveBinaryOperator],
  ["~", TOKEN_TYPES.AdditiveBinaryOperator],
  ["*", TOKEN_TYPES.MultiplicativeBinaryOperator],
  ["/", TOKEN_TYPES.MultiplicativeBinaryOperator],
  ["%", TOKEN_TYPES.MultiplicativeBinaryOperator],
  // Assignment operator
  ["=", TOKEN_TYPES.Equals]
];
var ESCAPE_CHARACTERS = /* @__PURE__ */ new Map([
  ["n", "\n"],
  // New line
  ["t", "	"],
  // Horizontal tab
  ["r", "\r"],
  // Carriage return
  ["b", "\b"],
  // Backspace
  ["f", "\f"],
  // Form feed
  ["v", "\v"],
  // Vertical tab
  ["'", "'"],
  // Single quote
  ['"', '"'],
  // Double quote
  ["\\", "\\"]
  // Backslash
]);
function preprocess(template, options = {}) {
  if (template.endsWith("\n")) {
    template = template.slice(0, -1);
  }
  if (options.lstrip_blocks) {
    template = template.replace(/^[ \t]*({[#%-])/gm, "$1");
  }
  if (options.trim_blocks) {
    template = template.replace(/([#%-]})\n/g, "$1");
  }
  return template.replace(/-%}\s*/g, "%}").replace(/\s*{%-/g, "{%").replace(/-}}\s*/g, "}}").replace(/\s*{{-/g, "{{").replace(/-#}\s*/g, "#}").replace(/\s*{#-/g, "{#").replace(/{%\s*(end)?generation\s*%}/gs, "");
}
function tokenize(source, options = {}) {
  const tokens = [];
  const src = preprocess(source, options);
  let cursorPosition = 0;
  let curlyBracketDepth = 0;
  const consumeWhile = (predicate) => {
    let str = "";
    while (predicate(src[cursorPosition])) {
      if (src[cursorPosition] === "\\") {
        ++cursorPosition;
        if (cursorPosition >= src.length)
          throw new SyntaxError("Unexpected end of input");
        const escaped = src[cursorPosition++];
        const unescaped = ESCAPE_CHARACTERS.get(escaped);
        if (unescaped === void 0) {
          throw new SyntaxError(`Unexpected escaped character: ${escaped}`);
        }
        str += unescaped;
        continue;
      }
      str += src[cursorPosition++];
      if (cursorPosition >= src.length)
        throw new SyntaxError("Unexpected end of input");
    }
    return str;
  };
  main:
    while (cursorPosition < src.length) {
      const lastTokenType = tokens.at(-1)?.type;
      if (lastTokenType === void 0 || lastTokenType === TOKEN_TYPES.CloseStatement || lastTokenType === TOKEN_TYPES.CloseExpression || lastTokenType === TOKEN_TYPES.Comment) {
        let text = "";
        while (cursorPosition < src.length && // Keep going until we hit the next Jinja statement or expression
        !(src[cursorPosition] === "{" && (src[cursorPosition + 1] === "%" || src[cursorPosition + 1] === "{" || src[cursorPosition + 1] === "#"))) {
          text += src[cursorPosition++];
        }
        if (text.length > 0) {
          tokens.push(new Token(text, TOKEN_TYPES.Text));
          continue;
        }
      }
      if (src[cursorPosition] === "{" && src[cursorPosition + 1] === "#") {
        cursorPosition += 2;
        let comment = "";
        while (src[cursorPosition] !== "#" || src[cursorPosition + 1] !== "}") {
          if (cursorPosition + 2 >= src.length) {
            throw new SyntaxError("Missing end of comment tag");
          }
          comment += src[cursorPosition++];
        }
        tokens.push(new Token(comment, TOKEN_TYPES.Comment));
        cursorPosition += 2;
        continue;
      }
      consumeWhile((char2) => /\s/.test(char2));
      const char = src[cursorPosition];
      if (char === "-" || char === "+") {
        const lastTokenType2 = tokens.at(-1)?.type;
        if (lastTokenType2 === TOKEN_TYPES.Text || lastTokenType2 === void 0) {
          throw new SyntaxError(`Unexpected character: ${char}`);
        }
        switch (lastTokenType2) {
          case TOKEN_TYPES.Identifier:
          case TOKEN_TYPES.NumericLiteral:
          case TOKEN_TYPES.StringLiteral:
          case TOKEN_TYPES.CloseParen:
          case TOKEN_TYPES.CloseSquareBracket:
            break;
          default: {
            ++cursorPosition;
            const num = consumeWhile(isInteger);
            tokens.push(
              new Token(`${char}${num}`, num.length > 0 ? TOKEN_TYPES.NumericLiteral : TOKEN_TYPES.UnaryOperator)
            );
            continue;
          }
        }
      }
      for (const [seq, type] of ORDERED_MAPPING_TABLE) {
        if (seq === "}}" && curlyBracketDepth > 0) {
          continue;
        }
        const slice22 = src.slice(cursorPosition, cursorPosition + seq.length);
        if (slice22 === seq) {
          tokens.push(new Token(seq, type));
          if (type === TOKEN_TYPES.OpenExpression) {
            curlyBracketDepth = 0;
          } else if (type === TOKEN_TYPES.OpenCurlyBracket) {
            ++curlyBracketDepth;
          } else if (type === TOKEN_TYPES.CloseCurlyBracket) {
            --curlyBracketDepth;
          }
          cursorPosition += seq.length;
          continue main;
        }
      }
      if (char === "'" || char === '"') {
        ++cursorPosition;
        const str = consumeWhile((c) => c !== char);
        tokens.push(new Token(str, TOKEN_TYPES.StringLiteral));
        ++cursorPosition;
        continue;
      }
      if (isInteger(char)) {
        let num = consumeWhile(isInteger);
        if (src[cursorPosition] === "." && isInteger(src[cursorPosition + 1])) {
          ++cursorPosition;
          const frac = consumeWhile(isInteger);
          num = `${num}.${frac}`;
        }
        tokens.push(new Token(num, TOKEN_TYPES.NumericLiteral));
        continue;
      }
      if (isWord(char)) {
        const word = consumeWhile(isWord);
        tokens.push(new Token(word, TOKEN_TYPES.Identifier));
        continue;
      }
      throw new SyntaxError(`Unexpected character: ${char}`);
    }
  return tokens;
}
var Statement = class {
  type = "Statement";
};
var Program = class extends Statement {
  constructor(body) {
    super();
    this.body = body;
  }
  type = "Program";
};
var If = class extends Statement {
  constructor(test, body, alternate) {
    super();
    this.test = test;
    this.body = body;
    this.alternate = alternate;
  }
  type = "If";
};
var For = class extends Statement {
  constructor(loopvar, iterable, body, defaultBlock) {
    super();
    this.loopvar = loopvar;
    this.iterable = iterable;
    this.body = body;
    this.defaultBlock = defaultBlock;
  }
  type = "For";
};
var Break = class extends Statement {
  type = "Break";
};
var Continue = class extends Statement {
  type = "Continue";
};
var SetStatement = class extends Statement {
  constructor(assignee, value, body) {
    super();
    this.assignee = assignee;
    this.value = value;
    this.body = body;
  }
  type = "Set";
};
var Macro = class extends Statement {
  constructor(name, args, body) {
    super();
    this.name = name;
    this.args = args;
    this.body = body;
  }
  type = "Macro";
};
var Comment = class extends Statement {
  constructor(value) {
    super();
    this.value = value;
  }
  type = "Comment";
};
var Expression = class extends Statement {
  type = "Expression";
};
var MemberExpression = class extends Expression {
  constructor(object, property, computed) {
    super();
    this.object = object;
    this.property = property;
    this.computed = computed;
  }
  type = "MemberExpression";
};
var CallExpression = class extends Expression {
  constructor(callee, args) {
    super();
    this.callee = callee;
    this.args = args;
  }
  type = "CallExpression";
};
var Identifier = class extends Expression {
  /**
   * @param {string} value The name of the identifier
   */
  constructor(value) {
    super();
    this.value = value;
  }
  type = "Identifier";
};
var Literal = class extends Expression {
  constructor(value) {
    super();
    this.value = value;
  }
  type = "Literal";
};
var IntegerLiteral = class extends Literal {
  type = "IntegerLiteral";
};
var FloatLiteral = class extends Literal {
  type = "FloatLiteral";
};
var StringLiteral = class extends Literal {
  type = "StringLiteral";
};
var ArrayLiteral = class extends Literal {
  type = "ArrayLiteral";
};
var TupleLiteral = class extends Literal {
  type = "TupleLiteral";
};
var ObjectLiteral = class extends Literal {
  type = "ObjectLiteral";
};
var BinaryExpression = class extends Expression {
  constructor(operator, left, right) {
    super();
    this.operator = operator;
    this.left = left;
    this.right = right;
  }
  type = "BinaryExpression";
};
var FilterExpression = class extends Expression {
  constructor(operand, filter) {
    super();
    this.operand = operand;
    this.filter = filter;
  }
  type = "FilterExpression";
};
var FilterStatement = class extends Statement {
  constructor(filter, body) {
    super();
    this.filter = filter;
    this.body = body;
  }
  type = "FilterStatement";
};
var SelectExpression = class extends Expression {
  constructor(lhs, test) {
    super();
    this.lhs = lhs;
    this.test = test;
  }
  type = "SelectExpression";
};
var TestExpression = class extends Expression {
  constructor(operand, negate, test) {
    super();
    this.operand = operand;
    this.negate = negate;
    this.test = test;
  }
  type = "TestExpression";
};
var UnaryExpression = class extends Expression {
  constructor(operator, argument) {
    super();
    this.operator = operator;
    this.argument = argument;
  }
  type = "UnaryExpression";
};
var SliceExpression = class extends Expression {
  constructor(start = void 0, stop = void 0, step = void 0) {
    super();
    this.start = start;
    this.stop = stop;
    this.step = step;
  }
  type = "SliceExpression";
};
var KeywordArgumentExpression = class extends Expression {
  constructor(key, value) {
    super();
    this.key = key;
    this.value = value;
  }
  type = "KeywordArgumentExpression";
};
var SpreadExpression = class extends Expression {
  constructor(argument) {
    super();
    this.argument = argument;
  }
  type = "SpreadExpression";
};
var CallStatement = class extends Statement {
  constructor(call, callerArgs, body) {
    super();
    this.call = call;
    this.callerArgs = callerArgs;
    this.body = body;
  }
  type = "CallStatement";
};
var Ternary = class extends Expression {
  constructor(condition, trueExpr, falseExpr) {
    super();
    this.condition = condition;
    this.trueExpr = trueExpr;
    this.falseExpr = falseExpr;
  }
  type = "Ternary";
};
function parse(tokens) {
  const program = new Program([]);
  let current = 0;
  function expect(type, error) {
    const prev = tokens[current++];
    if (!prev || prev.type !== type) {
      throw new Error(`Parser Error: ${error}. ${prev.type} !== ${type}.`);
    }
    return prev;
  }
  function expectIdentifier(name) {
    if (!isIdentifier(name)) {
      throw new SyntaxError(`Expected ${name}`);
    }
    ++current;
  }
  function parseAny() {
    switch (tokens[current].type) {
      case TOKEN_TYPES.Comment:
        return new Comment(tokens[current++].value);
      case TOKEN_TYPES.Text:
        return parseText();
      case TOKEN_TYPES.OpenStatement:
        return parseJinjaStatement();
      case TOKEN_TYPES.OpenExpression:
        return parseJinjaExpression();
      default:
        throw new SyntaxError(`Unexpected token type: ${tokens[current].type}`);
    }
  }
  function is(...types) {
    return current + types.length <= tokens.length && types.every((type, i) => type === tokens[current + i].type);
  }
  function isStatement(...names) {
    return tokens[current]?.type === TOKEN_TYPES.OpenStatement && tokens[current + 1]?.type === TOKEN_TYPES.Identifier && names.includes(tokens[current + 1]?.value);
  }
  function isIdentifier(...names) {
    return current + names.length <= tokens.length && names.every((name, i) => tokens[current + i].type === "Identifier" && name === tokens[current + i].value);
  }
  function parseText() {
    return new StringLiteral(expect(TOKEN_TYPES.Text, "Expected text token").value);
  }
  function parseJinjaStatement() {
    expect(TOKEN_TYPES.OpenStatement, "Expected opening statement token");
    if (tokens[current].type !== TOKEN_TYPES.Identifier) {
      throw new SyntaxError(`Unknown statement, got ${tokens[current].type}`);
    }
    const name = tokens[current].value;
    let result;
    switch (name) {
      case "set":
        ++current;
        result = parseSetStatement();
        break;
      case "if":
        ++current;
        result = parseIfStatement();
        expect(TOKEN_TYPES.OpenStatement, "Expected {% token");
        expectIdentifier("endif");
        expect(TOKEN_TYPES.CloseStatement, "Expected %} token");
        break;
      case "macro":
        ++current;
        result = parseMacroStatement();
        expect(TOKEN_TYPES.OpenStatement, "Expected {% token");
        expectIdentifier("endmacro");
        expect(TOKEN_TYPES.CloseStatement, "Expected %} token");
        break;
      case "for":
        ++current;
        result = parseForStatement();
        expect(TOKEN_TYPES.OpenStatement, "Expected {% token");
        expectIdentifier("endfor");
        expect(TOKEN_TYPES.CloseStatement, "Expected %} token");
        break;
      case "call": {
        ++current;
        let callerArgs = null;
        if (is(TOKEN_TYPES.OpenParen)) {
          callerArgs = parseArgs();
        }
        const callee = parsePrimaryExpression();
        if (callee.type !== "Identifier") {
          throw new SyntaxError(`Expected identifier following call statement`);
        }
        const callArgs = parseArgs();
        expect(TOKEN_TYPES.CloseStatement, "Expected closing statement token");
        const body = [];
        while (!isStatement("endcall")) {
          body.push(parseAny());
        }
        expect(TOKEN_TYPES.OpenStatement, "Expected '{%'");
        expectIdentifier("endcall");
        expect(TOKEN_TYPES.CloseStatement, "Expected closing statement token");
        const callExpr = new CallExpression(callee, callArgs);
        result = new CallStatement(callExpr, callerArgs, body);
        break;
      }
      case "break":
        ++current;
        expect(TOKEN_TYPES.CloseStatement, "Expected closing statement token");
        result = new Break();
        break;
      case "continue":
        ++current;
        expect(TOKEN_TYPES.CloseStatement, "Expected closing statement token");
        result = new Continue();
        break;
      case "filter": {
        ++current;
        let filterNode = parsePrimaryExpression();
        if (filterNode instanceof Identifier && is(TOKEN_TYPES.OpenParen)) {
          filterNode = parseCallExpression(filterNode);
        }
        expect(TOKEN_TYPES.CloseStatement, "Expected closing statement token");
        const filterBody = [];
        while (!isStatement("endfilter")) {
          filterBody.push(parseAny());
        }
        expect(TOKEN_TYPES.OpenStatement, "Expected '{%'");
        expectIdentifier("endfilter");
        expect(TOKEN_TYPES.CloseStatement, "Expected '%}'");
        result = new FilterStatement(filterNode, filterBody);
        break;
      }
      default:
        throw new SyntaxError(`Unknown statement type: ${name}`);
    }
    return result;
  }
  function parseJinjaExpression() {
    expect(TOKEN_TYPES.OpenExpression, "Expected opening expression token");
    const result = parseExpression();
    expect(TOKEN_TYPES.CloseExpression, "Expected closing expression token");
    return result;
  }
  function parseSetStatement() {
    const left = parseExpressionSequence();
    let value = null;
    const body = [];
    if (is(TOKEN_TYPES.Equals)) {
      ++current;
      value = parseExpressionSequence();
    } else {
      expect(TOKEN_TYPES.CloseStatement, "Expected %} token");
      while (!isStatement("endset")) {
        body.push(parseAny());
      }
      expect(TOKEN_TYPES.OpenStatement, "Expected {% token");
      expectIdentifier("endset");
    }
    expect(TOKEN_TYPES.CloseStatement, "Expected closing statement token");
    return new SetStatement(left, value, body);
  }
  function parseIfStatement() {
    const test = parseExpression();
    expect(TOKEN_TYPES.CloseStatement, "Expected closing statement token");
    const body = [];
    const alternate = [];
    while (!isStatement("elif", "else", "endif")) {
      body.push(parseAny());
    }
    if (isStatement("elif")) {
      ++current;
      ++current;
      const result = parseIfStatement();
      alternate.push(result);
    } else if (isStatement("else")) {
      ++current;
      ++current;
      expect(TOKEN_TYPES.CloseStatement, "Expected closing statement token");
      while (!isStatement("endif")) {
        alternate.push(parseAny());
      }
    }
    return new If(test, body, alternate);
  }
  function parseMacroStatement() {
    const name = parsePrimaryExpression();
    if (name.type !== "Identifier") {
      throw new SyntaxError(`Expected identifier following macro statement`);
    }
    const args = parseArgs();
    expect(TOKEN_TYPES.CloseStatement, "Expected closing statement token");
    const body = [];
    while (!isStatement("endmacro")) {
      body.push(parseAny());
    }
    return new Macro(name, args, body);
  }
  function parseExpressionSequence(primary = false) {
    const fn = primary ? parsePrimaryExpression : parseExpression;
    const expressions = [fn()];
    const isTuple = is(TOKEN_TYPES.Comma);
    while (isTuple) {
      ++current;
      expressions.push(fn());
      if (!is(TOKEN_TYPES.Comma)) {
        break;
      }
    }
    return isTuple ? new TupleLiteral(expressions) : expressions[0];
  }
  function parseForStatement() {
    const loopVariable = parseExpressionSequence(true);
    if (!(loopVariable instanceof Identifier || loopVariable instanceof TupleLiteral)) {
      throw new SyntaxError(`Expected identifier/tuple for the loop variable, got ${loopVariable.type} instead`);
    }
    if (!isIdentifier("in")) {
      throw new SyntaxError("Expected `in` keyword following loop variable");
    }
    ++current;
    const iterable = parseExpression();
    expect(TOKEN_TYPES.CloseStatement, "Expected closing statement token");
    const body = [];
    while (!isStatement("endfor", "else")) {
      body.push(parseAny());
    }
    const alternative = [];
    if (isStatement("else")) {
      ++current;
      ++current;
      expect(TOKEN_TYPES.CloseStatement, "Expected closing statement token");
      while (!isStatement("endfor")) {
        alternative.push(parseAny());
      }
    }
    return new For(loopVariable, iterable, body, alternative);
  }
  function parseExpression() {
    return parseIfExpression();
  }
  function parseIfExpression() {
    const a = parseLogicalOrExpression();
    if (isIdentifier("if")) {
      ++current;
      const test = parseLogicalOrExpression();
      if (isIdentifier("else")) {
        ++current;
        const falseExpr = parseIfExpression();
        return new Ternary(test, a, falseExpr);
      } else {
        return new SelectExpression(a, test);
      }
    }
    return a;
  }
  function parseLogicalOrExpression() {
    let left = parseLogicalAndExpression();
    while (isIdentifier("or")) {
      const operator = tokens[current];
      ++current;
      const right = parseLogicalAndExpression();
      left = new BinaryExpression(operator, left, right);
    }
    return left;
  }
  function parseLogicalAndExpression() {
    let left = parseLogicalNegationExpression();
    while (isIdentifier("and")) {
      const operator = tokens[current];
      ++current;
      const right = parseLogicalNegationExpression();
      left = new BinaryExpression(operator, left, right);
    }
    return left;
  }
  function parseLogicalNegationExpression() {
    let right;
    while (isIdentifier("not")) {
      const operator = tokens[current];
      ++current;
      const arg = parseLogicalNegationExpression();
      right = new UnaryExpression(operator, arg);
    }
    return right ?? parseComparisonExpression();
  }
  function parseComparisonExpression() {
    let left = parseAdditiveExpression();
    while (true) {
      let operator;
      if (isIdentifier("not", "in")) {
        operator = new Token("not in", TOKEN_TYPES.Identifier);
        current += 2;
      } else if (isIdentifier("in")) {
        operator = tokens[current++];
      } else if (is(TOKEN_TYPES.ComparisonBinaryOperator)) {
        operator = tokens[current++];
      } else {
        break;
      }
      const right = parseAdditiveExpression();
      left = new BinaryExpression(operator, left, right);
    }
    return left;
  }
  function parseAdditiveExpression() {
    let left = parseMultiplicativeExpression();
    while (is(TOKEN_TYPES.AdditiveBinaryOperator)) {
      const operator = tokens[current];
      ++current;
      const right = parseMultiplicativeExpression();
      left = new BinaryExpression(operator, left, right);
    }
    return left;
  }
  function parseCallMemberExpression() {
    const member = parseMemberExpression(parsePrimaryExpression());
    if (is(TOKEN_TYPES.OpenParen)) {
      return parseCallExpression(member);
    }
    return member;
  }
  function parseCallExpression(callee) {
    let expression = new CallExpression(callee, parseArgs());
    expression = parseMemberExpression(expression);
    if (is(TOKEN_TYPES.OpenParen)) {
      expression = parseCallExpression(expression);
    }
    return expression;
  }
  function parseArgs() {
    expect(TOKEN_TYPES.OpenParen, "Expected opening parenthesis for arguments list");
    const args = parseArgumentsList();
    expect(TOKEN_TYPES.CloseParen, "Expected closing parenthesis for arguments list");
    return args;
  }
  function parseArgumentsList() {
    const args = [];
    while (!is(TOKEN_TYPES.CloseParen)) {
      let argument;
      if (tokens[current].type === TOKEN_TYPES.MultiplicativeBinaryOperator && tokens[current].value === "*") {
        ++current;
        const expr = parseExpression();
        argument = new SpreadExpression(expr);
      } else {
        argument = parseExpression();
        if (is(TOKEN_TYPES.Equals)) {
          ++current;
          if (!(argument instanceof Identifier)) {
            throw new SyntaxError(`Expected identifier for keyword argument`);
          }
          const value = parseExpression();
          argument = new KeywordArgumentExpression(argument, value);
        }
      }
      args.push(argument);
      if (is(TOKEN_TYPES.Comma)) {
        ++current;
      }
    }
    return args;
  }
  function parseMemberExpressionArgumentsList() {
    const slices = [];
    let isSlice = false;
    while (!is(TOKEN_TYPES.CloseSquareBracket)) {
      if (is(TOKEN_TYPES.Colon)) {
        slices.push(void 0);
        ++current;
        isSlice = true;
      } else {
        slices.push(parseExpression());
        if (is(TOKEN_TYPES.Colon)) {
          ++current;
          isSlice = true;
        }
      }
    }
    if (slices.length === 0) {
      throw new SyntaxError(`Expected at least one argument for member/slice expression`);
    }
    if (isSlice) {
      if (slices.length > 3) {
        throw new SyntaxError(`Expected 0-3 arguments for slice expression`);
      }
      return new SliceExpression(...slices);
    }
    return slices[0];
  }
  function parseMemberExpression(object) {
    while (is(TOKEN_TYPES.Dot) || is(TOKEN_TYPES.OpenSquareBracket)) {
      const operator = tokens[current];
      ++current;
      let property;
      const computed = operator.type === TOKEN_TYPES.OpenSquareBracket;
      if (computed) {
        property = parseMemberExpressionArgumentsList();
        expect(TOKEN_TYPES.CloseSquareBracket, "Expected closing square bracket");
      } else {
        property = parsePrimaryExpression();
        if (property.type !== "Identifier") {
          throw new SyntaxError(`Expected identifier following dot operator`);
        }
      }
      object = new MemberExpression(object, property, computed);
    }
    return object;
  }
  function parseMultiplicativeExpression() {
    let left = parseTestExpression();
    while (is(TOKEN_TYPES.MultiplicativeBinaryOperator)) {
      const operator = tokens[current++];
      const right = parseTestExpression();
      left = new BinaryExpression(operator, left, right);
    }
    return left;
  }
  function parseTestExpression() {
    let operand = parseFilterExpression();
    while (isIdentifier("is")) {
      ++current;
      const negate = isIdentifier("not");
      if (negate) {
        ++current;
      }
      const filter = parsePrimaryExpression();
      if (!(filter instanceof Identifier)) {
        throw new SyntaxError(`Expected identifier for the test`);
      }
      operand = new TestExpression(operand, negate, filter);
    }
    return operand;
  }
  function parseFilterExpression() {
    let operand = parseCallMemberExpression();
    while (is(TOKEN_TYPES.Pipe)) {
      ++current;
      let filter = parsePrimaryExpression();
      if (!(filter instanceof Identifier)) {
        throw new SyntaxError(`Expected identifier for the filter`);
      }
      if (is(TOKEN_TYPES.OpenParen)) {
        filter = parseCallExpression(filter);
      }
      operand = new FilterExpression(operand, filter);
    }
    return operand;
  }
  function parsePrimaryExpression() {
    const token = tokens[current++];
    switch (token.type) {
      case TOKEN_TYPES.NumericLiteral: {
        const num = token.value;
        return num.includes(".") ? new FloatLiteral(Number(num)) : new IntegerLiteral(Number(num));
      }
      case TOKEN_TYPES.StringLiteral: {
        let value = token.value;
        while (is(TOKEN_TYPES.StringLiteral)) {
          value += tokens[current++].value;
        }
        return new StringLiteral(value);
      }
      case TOKEN_TYPES.Identifier:
        return new Identifier(token.value);
      case TOKEN_TYPES.OpenParen: {
        const expression = parseExpressionSequence();
        expect(TOKEN_TYPES.CloseParen, "Expected closing parenthesis, got ${tokens[current].type} instead.");
        return expression;
      }
      case TOKEN_TYPES.OpenSquareBracket: {
        const values = [];
        while (!is(TOKEN_TYPES.CloseSquareBracket)) {
          values.push(parseExpression());
          if (is(TOKEN_TYPES.Comma)) {
            ++current;
          }
        }
        ++current;
        return new ArrayLiteral(values);
      }
      case TOKEN_TYPES.OpenCurlyBracket: {
        const values = /* @__PURE__ */ new Map();
        while (!is(TOKEN_TYPES.CloseCurlyBracket)) {
          const key = parseExpression();
          expect(TOKEN_TYPES.Colon, "Expected colon between key and value in object literal");
          const value = parseExpression();
          values.set(key, value);
          if (is(TOKEN_TYPES.Comma)) {
            ++current;
          }
        }
        ++current;
        return new ObjectLiteral(values);
      }
      default:
        throw new SyntaxError(`Unexpected token: ${token.type}`);
    }
  }
  while (current < tokens.length) {
    program.body.push(parseAny());
  }
  return program;
}
function range(start, stop, step = 1) {
  if (stop === void 0) {
    stop = start;
    start = 0;
  }
  const result = [];
  for (let i = start; i < stop; i += step) {
    result.push(i);
  }
  return result;
}
function slice2(array, start, stop, step = 1) {
  const direction = Math.sign(step);
  if (direction >= 0) {
    start = (start ??= 0) < 0 ? Math.max(array.length + start, 0) : Math.min(start, array.length);
    stop = (stop ??= array.length) < 0 ? Math.max(array.length + stop, 0) : Math.min(stop, array.length);
  } else {
    start = (start ??= array.length - 1) < 0 ? Math.max(array.length + start, -1) : Math.min(start, array.length - 1);
    stop = (stop ??= -1) < -1 ? Math.max(array.length + stop, -1) : Math.min(stop, array.length - 1);
  }
  const result = [];
  for (let i = start; direction * i < direction * stop; i += step) {
    result.push(array[i]);
  }
  return result;
}
function titleCase(value) {
  return value.replace(/\b\w/g, (c) => c.toUpperCase());
}
function strftime_now(format2) {
  return strftime(/* @__PURE__ */ new Date(), format2);
}
function strftime(date, format2) {
  const monthFormatterLong = new Intl.DateTimeFormat(void 0, { month: "long" });
  const monthFormatterShort = new Intl.DateTimeFormat(void 0, { month: "short" });
  const pad2 = (n) => n < 10 ? "0" + n : n.toString();
  return format2.replace(/%[YmdbBHM%]/g, (token) => {
    switch (token) {
      case "%Y":
        return date.getFullYear().toString();
      case "%m":
        return pad2(date.getMonth() + 1);
      case "%d":
        return pad2(date.getDate());
      case "%b":
        return monthFormatterShort.format(date);
      case "%B":
        return monthFormatterLong.format(date);
      case "%H":
        return pad2(date.getHours());
      case "%M":
        return pad2(date.getMinutes());
      case "%%":
        return "%";
      default:
        return token;
    }
  });
}
function escapeRegExp2(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
function replace(str, oldvalue, newvalue, count2) {
  if (count2 === 0)
    return str;
  let remaining = count2 == null || count2 < 0 ? Infinity : count2;
  const pattern = oldvalue.length === 0 ? new RegExp("(?=)", "gu") : new RegExp(escapeRegExp2(oldvalue), "gu");
  return str.replaceAll(pattern, (match) => {
    if (remaining > 0) {
      --remaining;
      return newvalue;
    }
    return match;
  });
}
var BreakControl = class extends Error {
};
var ContinueControl = class extends Error {
};
var RuntimeValue = class {
  type = "RuntimeValue";
  value;
  /**
   * A collection of built-in functions for this type.
   */
  builtins = /* @__PURE__ */ new Map();
  /**
   * Creates a new RuntimeValue.
   */
  constructor(value = void 0) {
    this.value = value;
  }
  /**
   * Determines truthiness or falsiness of the runtime value.
   * This function should be overridden by subclasses if it has custom truthiness criteria.
   * @returns {BooleanValue} BooleanValue(true) if the value is truthy, BooleanValue(false) otherwise.
   */
  __bool__() {
    return new BooleanValue(!!this.value);
  }
  toString() {
    return String(this.value);
  }
};
var IntegerValue = class extends RuntimeValue {
  type = "IntegerValue";
};
var FloatValue = class extends RuntimeValue {
  type = "FloatValue";
  toString() {
    return this.value % 1 === 0 ? this.value.toFixed(1) : this.value.toString();
  }
};
var StringValue = class extends RuntimeValue {
  type = "StringValue";
  builtins = /* @__PURE__ */ new Map([
    [
      "upper",
      new FunctionValue(() => {
        return new StringValue(this.value.toUpperCase());
      })
    ],
    [
      "lower",
      new FunctionValue(() => {
        return new StringValue(this.value.toLowerCase());
      })
    ],
    [
      "strip",
      new FunctionValue(() => {
        return new StringValue(this.value.trim());
      })
    ],
    [
      "title",
      new FunctionValue(() => {
        return new StringValue(titleCase(this.value));
      })
    ],
    [
      "capitalize",
      new FunctionValue(() => {
        return new StringValue(this.value.charAt(0).toUpperCase() + this.value.slice(1));
      })
    ],
    ["length", new IntegerValue(this.value.length)],
    [
      "rstrip",
      new FunctionValue(() => {
        return new StringValue(this.value.trimEnd());
      })
    ],
    [
      "lstrip",
      new FunctionValue(() => {
        return new StringValue(this.value.trimStart());
      })
    ],
    [
      "startswith",
      new FunctionValue((args) => {
        if (args.length === 0) {
          throw new Error("startswith() requires at least one argument");
        }
        const pattern = args[0];
        if (pattern instanceof StringValue) {
          return new BooleanValue(this.value.startsWith(pattern.value));
        } else if (pattern instanceof ArrayValue) {
          for (const item of pattern.value) {
            if (!(item instanceof StringValue)) {
              throw new Error("startswith() tuple elements must be strings");
            }
            if (this.value.startsWith(item.value)) {
              return new BooleanValue(true);
            }
          }
          return new BooleanValue(false);
        }
        throw new Error("startswith() argument must be a string or tuple of strings");
      })
    ],
    [
      "endswith",
      new FunctionValue((args) => {
        if (args.length === 0) {
          throw new Error("endswith() requires at least one argument");
        }
        const pattern = args[0];
        if (pattern instanceof StringValue) {
          return new BooleanValue(this.value.endsWith(pattern.value));
        } else if (pattern instanceof ArrayValue) {
          for (const item of pattern.value) {
            if (!(item instanceof StringValue)) {
              throw new Error("endswith() tuple elements must be strings");
            }
            if (this.value.endsWith(item.value)) {
              return new BooleanValue(true);
            }
          }
          return new BooleanValue(false);
        }
        throw new Error("endswith() argument must be a string or tuple of strings");
      })
    ],
    [
      "split",
      // follows Python's `str.split(sep=None, maxsplit=-1)` function behavior
      // https://docs.python.org/3.13/library/stdtypes.html#str.split
      new FunctionValue((args) => {
        const sep = args[0] ?? new NullValue();
        if (!(sep instanceof StringValue || sep instanceof NullValue)) {
          throw new Error("sep argument must be a string or null");
        }
        const maxsplit = args[1] ?? new IntegerValue(-1);
        if (!(maxsplit instanceof IntegerValue)) {
          throw new Error("maxsplit argument must be a number");
        }
        let result = [];
        if (sep instanceof NullValue) {
          const text = this.value.trimStart();
          for (const { 0: match, index } of text.matchAll(/\S+/g)) {
            if (maxsplit.value !== -1 && result.length >= maxsplit.value && index !== void 0) {
              result.push(match + text.slice(index + match.length));
              break;
            }
            result.push(match);
          }
        } else {
          if (sep.value === "") {
            throw new Error("empty separator");
          }
          result = this.value.split(sep.value);
          if (maxsplit.value !== -1 && result.length > maxsplit.value) {
            result.push(result.splice(maxsplit.value).join(sep.value));
          }
        }
        return new ArrayValue(result.map((part) => new StringValue(part)));
      })
    ],
    [
      "replace",
      new FunctionValue((args) => {
        if (args.length < 2) {
          throw new Error("replace() requires at least two arguments");
        }
        const oldValue = args[0];
        const newValue = args[1];
        if (!(oldValue instanceof StringValue && newValue instanceof StringValue)) {
          throw new Error("replace() arguments must be strings");
        }
        let count2;
        if (args.length > 2) {
          if (args[2].type === "KeywordArgumentsValue") {
            count2 = args[2].value.get("count") ?? new NullValue();
          } else {
            count2 = args[2];
          }
        } else {
          count2 = new NullValue();
        }
        if (!(count2 instanceof IntegerValue || count2 instanceof NullValue)) {
          throw new Error("replace() count argument must be a number or null");
        }
        return new StringValue(replace(this.value, oldValue.value, newValue.value, count2.value));
      })
    ]
  ]);
};
var BooleanValue = class extends RuntimeValue {
  type = "BooleanValue";
};
var ObjectValue = class extends RuntimeValue {
  type = "ObjectValue";
  /**
   * NOTE: necessary to override since all JavaScript arrays are considered truthy,
   * while only non-empty Python arrays are consider truthy.
   *
   * e.g.,
   *  - JavaScript:  {} && 5 -> 5
   *  - Python:      {} and 5 -> {}
   */
  __bool__() {
    return new BooleanValue(this.value.size > 0);
  }
  builtins = /* @__PURE__ */ new Map([
    [
      "get",
      new FunctionValue(([key, defaultValue]) => {
        if (!(key instanceof StringValue)) {
          throw new Error(`Object key must be a string: got ${key.type}`);
        }
        return this.value.get(key.value) ?? defaultValue ?? new NullValue();
      })
    ],
    ["items", new FunctionValue(() => this.items())],
    ["keys", new FunctionValue(() => this.keys())],
    ["values", new FunctionValue(() => this.values())]
  ]);
  items() {
    return new ArrayValue(
      Array.from(this.value.entries()).map(([key, value]) => new ArrayValue([new StringValue(key), value]))
    );
  }
  keys() {
    return new ArrayValue(Array.from(this.value.keys()).map((key) => new StringValue(key)));
  }
  values() {
    return new ArrayValue(Array.from(this.value.values()));
  }
};
var KeywordArgumentsValue = class extends ObjectValue {
  type = "KeywordArgumentsValue";
};
var ArrayValue = class extends RuntimeValue {
  type = "ArrayValue";
  builtins = /* @__PURE__ */ new Map([["length", new IntegerValue(this.value.length)]]);
  /**
   * NOTE: necessary to override since all JavaScript arrays are considered truthy,
   * while only non-empty Python arrays are consider truthy.
   *
   * e.g.,
   *  - JavaScript:  [] && 5 -> 5
   *  - Python:      [] and 5 -> []
   */
  __bool__() {
    return new BooleanValue(this.value.length > 0);
  }
};
var TupleValue = class extends ArrayValue {
  type = "TupleValue";
};
var FunctionValue = class extends RuntimeValue {
  type = "FunctionValue";
};
var NullValue = class extends RuntimeValue {
  type = "NullValue";
};
var UndefinedValue = class extends RuntimeValue {
  type = "UndefinedValue";
};
var Environment = class {
  constructor(parent) {
    this.parent = parent;
  }
  /**
   * The variables declared in this environment.
   */
  variables = /* @__PURE__ */ new Map([
    [
      "namespace",
      new FunctionValue((args) => {
        if (args.length === 0) {
          return new ObjectValue(/* @__PURE__ */ new Map());
        }
        if (args.length !== 1 || !(args[0] instanceof ObjectValue)) {
          throw new Error("`namespace` expects either zero arguments or a single object argument");
        }
        return args[0];
      })
    ]
  ]);
  /**
   * The tests available in this environment.
   */
  tests = /* @__PURE__ */ new Map([
    ["boolean", (operand) => operand.type === "BooleanValue"],
    ["callable", (operand) => operand instanceof FunctionValue],
    [
      "odd",
      (operand) => {
        if (!(operand instanceof IntegerValue)) {
          throw new Error(`cannot odd on ${operand.type}`);
        }
        return operand.value % 2 !== 0;
      }
    ],
    [
      "even",
      (operand) => {
        if (!(operand instanceof IntegerValue)) {
          throw new Error(`cannot even on ${operand.type}`);
        }
        return operand.value % 2 === 0;
      }
    ],
    ["false", (operand) => operand.type === "BooleanValue" && !operand.value],
    ["true", (operand) => operand.type === "BooleanValue" && operand.value],
    ["none", (operand) => operand.type === "NullValue"],
    ["string", (operand) => operand.type === "StringValue"],
    ["number", (operand) => operand instanceof IntegerValue || operand instanceof FloatValue],
    ["integer", (operand) => operand instanceof IntegerValue],
    ["iterable", (operand) => operand.type === "ArrayValue" || operand.type === "StringValue"],
    ["mapping", (operand) => operand.type === "ObjectValue"],
    [
      "lower",
      (operand) => {
        const str = operand.value;
        return operand.type === "StringValue" && str === str.toLowerCase();
      }
    ],
    [
      "upper",
      (operand) => {
        const str = operand.value;
        return operand.type === "StringValue" && str === str.toUpperCase();
      }
    ],
    ["none", (operand) => operand.type === "NullValue"],
    ["defined", (operand) => operand.type !== "UndefinedValue"],
    ["undefined", (operand) => operand.type === "UndefinedValue"],
    ["equalto", (a, b) => a.value === b.value],
    ["eq", (a, b) => a.value === b.value]
  ]);
  /**
   * Set the value of a variable in the current environment.
   */
  set(name, value) {
    return this.declareVariable(name, convertToRuntimeValues(value));
  }
  declareVariable(name, value) {
    if (this.variables.has(name)) {
      throw new SyntaxError(`Variable already declared: ${name}`);
    }
    this.variables.set(name, value);
    return value;
  }
  // private assignVariable(name: string, value: AnyRuntimeValue): AnyRuntimeValue {
  // 	const env = this.resolve(name);
  // 	env.variables.set(name, value);
  // 	return value;
  // }
  /**
   * Set variable in the current scope.
   * See https://jinja.palletsprojects.com/en/3.0.x/templates/#assignments for more information.
   */
  setVariable(name, value) {
    this.variables.set(name, value);
    return value;
  }
  /**
   * Resolve the environment in which the variable is declared.
   * @param {string} name The name of the variable.
   * @returns {Environment} The environment in which the variable is declared.
   */
  resolve(name) {
    if (this.variables.has(name)) {
      return this;
    }
    if (this.parent) {
      return this.parent.resolve(name);
    }
    throw new Error(`Unknown variable: ${name}`);
  }
  lookupVariable(name) {
    try {
      return this.resolve(name).variables.get(name) ?? new UndefinedValue();
    } catch {
      return new UndefinedValue();
    }
  }
};
function setupGlobals(env2) {
  env2.set("false", false);
  env2.set("true", true);
  env2.set("none", null);
  env2.set("raise_exception", (args) => {
    throw new Error(args);
  });
  env2.set("range", range);
  env2.set("strftime_now", strftime_now);
  env2.set("True", true);
  env2.set("False", false);
  env2.set("None", null);
}
var Interpreter = class {
  global;
  constructor(env2) {
    this.global = env2 ?? new Environment();
  }
  /**
   * Run the program.
   */
  run(program) {
    return this.evaluate(program, this.global);
  }
  /**
   * Evaluates expressions following the binary operation type.
   */
  evaluateBinaryExpression(node, environment) {
    const left = this.evaluate(node.left, environment);
    switch (node.operator.value) {
      case "and":
        return left.__bool__().value ? this.evaluate(node.right, environment) : left;
      case "or":
        return left.__bool__().value ? left : this.evaluate(node.right, environment);
    }
    const right = this.evaluate(node.right, environment);
    switch (node.operator.value) {
      case "==":
        return new BooleanValue(left.value == right.value);
      case "!=":
        return new BooleanValue(left.value != right.value);
    }
    if (left instanceof UndefinedValue || right instanceof UndefinedValue) {
      if (right instanceof UndefinedValue && ["in", "not in"].includes(node.operator.value)) {
        return new BooleanValue(node.operator.value === "not in");
      }
      throw new Error(`Cannot perform operation ${node.operator.value} on undefined values`);
    } else if (left instanceof NullValue || right instanceof NullValue) {
      throw new Error("Cannot perform operation on null values");
    } else if (node.operator.value === "~") {
      return new StringValue(left.value.toString() + right.value.toString());
    } else if ((left instanceof IntegerValue || left instanceof FloatValue) && (right instanceof IntegerValue || right instanceof FloatValue)) {
      const a = left.value, b = right.value;
      switch (node.operator.value) {
        case "+":
        case "-":
        case "*": {
          const res = node.operator.value === "+" ? a + b : node.operator.value === "-" ? a - b : a * b;
          const isFloat = left instanceof FloatValue || right instanceof FloatValue;
          return isFloat ? new FloatValue(res) : new IntegerValue(res);
        }
        case "/":
          return new FloatValue(a / b);
        case "%": {
          const rem = a % b;
          const isFloat = left instanceof FloatValue || right instanceof FloatValue;
          return isFloat ? new FloatValue(rem) : new IntegerValue(rem);
        }
        case "<":
          return new BooleanValue(a < b);
        case ">":
          return new BooleanValue(a > b);
        case ">=":
          return new BooleanValue(a >= b);
        case "<=":
          return new BooleanValue(a <= b);
      }
    } else if (left instanceof ArrayValue && right instanceof ArrayValue) {
      switch (node.operator.value) {
        case "+":
          return new ArrayValue(left.value.concat(right.value));
      }
    } else if (right instanceof ArrayValue) {
      const member = right.value.find((x) => x.value === left.value) !== void 0;
      switch (node.operator.value) {
        case "in":
          return new BooleanValue(member);
        case "not in":
          return new BooleanValue(!member);
      }
    }
    if (left instanceof StringValue || right instanceof StringValue) {
      switch (node.operator.value) {
        case "+":
          return new StringValue(left.value.toString() + right.value.toString());
      }
    }
    if (left instanceof StringValue && right instanceof StringValue) {
      switch (node.operator.value) {
        case "in":
          return new BooleanValue(right.value.includes(left.value));
        case "not in":
          return new BooleanValue(!right.value.includes(left.value));
      }
    }
    if (left instanceof StringValue && right instanceof ObjectValue) {
      switch (node.operator.value) {
        case "in":
          return new BooleanValue(right.value.has(left.value));
        case "not in":
          return new BooleanValue(!right.value.has(left.value));
      }
    }
    throw new SyntaxError(`Unknown operator "${node.operator.value}" between ${left.type} and ${right.type}`);
  }
  evaluateArguments(args, environment) {
    const positionalArguments = [];
    const keywordArguments = /* @__PURE__ */ new Map();
    for (const argument of args) {
      if (argument.type === "SpreadExpression") {
        const spreadNode = argument;
        const val = this.evaluate(spreadNode.argument, environment);
        if (!(val instanceof ArrayValue)) {
          throw new Error(`Cannot unpack non-iterable type: ${val.type}`);
        }
        for (const item of val.value) {
          positionalArguments.push(item);
        }
      } else if (argument.type === "KeywordArgumentExpression") {
        const kwarg = argument;
        keywordArguments.set(kwarg.key.value, this.evaluate(kwarg.value, environment));
      } else {
        if (keywordArguments.size > 0) {
          throw new Error("Positional arguments must come before keyword arguments");
        }
        positionalArguments.push(this.evaluate(argument, environment));
      }
    }
    return [positionalArguments, keywordArguments];
  }
  applyFilter(operand, filterNode, environment) {
    if (filterNode.type === "Identifier") {
      const filter = filterNode;
      if (filter.value === "tojson") {
        return new StringValue(toJSON(operand));
      }
      if (operand instanceof ArrayValue) {
        switch (filter.value) {
          case "list":
            return operand;
          case "first":
            return operand.value[0];
          case "last":
            return operand.value[operand.value.length - 1];
          case "length":
            return new IntegerValue(operand.value.length);
          case "reverse":
            return new ArrayValue(operand.value.reverse());
          case "sort":
            return new ArrayValue(
              operand.value.sort((a, b) => {
                if (a.type !== b.type) {
                  throw new Error(`Cannot compare different types: ${a.type} and ${b.type}`);
                }
                switch (a.type) {
                  case "IntegerValue":
                  case "FloatValue":
                    return a.value - b.value;
                  case "StringValue":
                    return a.value.localeCompare(b.value);
                  default:
                    throw new Error(`Cannot compare type: ${a.type}`);
                }
              })
            );
          case "join":
            return new StringValue(operand.value.map((x) => x.value).join(""));
          case "string":
            return new StringValue(toJSON(operand));
          case "unique": {
            const seen = /* @__PURE__ */ new Set();
            const output = [];
            for (const item of operand.value) {
              if (!seen.has(item.value)) {
                seen.add(item.value);
                output.push(item);
              }
            }
            return new ArrayValue(output);
          }
          default:
            throw new Error(`Unknown ArrayValue filter: ${filter.value}`);
        }
      } else if (operand instanceof StringValue) {
        switch (filter.value) {
          case "length":
          case "upper":
          case "lower":
          case "title":
          case "capitalize": {
            const builtin = operand.builtins.get(filter.value);
            if (builtin instanceof FunctionValue) {
              return builtin.value(
                /* no arguments */
                [],
                environment
              );
            } else if (builtin instanceof IntegerValue) {
              return builtin;
            } else {
              throw new Error(`Unknown StringValue filter: ${filter.value}`);
            }
          }
          case "trim":
            return new StringValue(operand.value.trim());
          case "indent":
            return new StringValue(
              operand.value.split("\n").map(
                (x, i) => (
                  // By default, don't indent the first line or empty lines
                  i === 0 || x.length === 0 ? x : "    " + x
                )
              ).join("\n")
            );
          case "join":
          case "string":
            return operand;
          case "int": {
            const val = parseInt(operand.value, 10);
            return new IntegerValue(isNaN(val) ? 0 : val);
          }
          case "float": {
            const val = parseFloat(operand.value);
            return new FloatValue(isNaN(val) ? 0 : val);
          }
          default:
            throw new Error(`Unknown StringValue filter: ${filter.value}`);
        }
      } else if (operand instanceof IntegerValue || operand instanceof FloatValue) {
        switch (filter.value) {
          case "abs":
            return operand instanceof IntegerValue ? new IntegerValue(Math.abs(operand.value)) : new FloatValue(Math.abs(operand.value));
          case "int":
            return new IntegerValue(Math.floor(operand.value));
          case "float":
            return new FloatValue(operand.value);
          default:
            throw new Error(`Unknown NumericValue filter: ${filter.value}`);
        }
      } else if (operand instanceof ObjectValue) {
        switch (filter.value) {
          case "items":
            return new ArrayValue(
              Array.from(operand.value.entries()).map(([key, value]) => new ArrayValue([new StringValue(key), value]))
            );
          case "length":
            return new IntegerValue(operand.value.size);
          default:
            throw new Error(`Unknown ObjectValue filter: ${filter.value}`);
        }
      } else if (operand instanceof BooleanValue) {
        switch (filter.value) {
          case "bool":
            return new BooleanValue(operand.value);
          case "int":
            return new IntegerValue(operand.value ? 1 : 0);
          case "float":
            return new FloatValue(operand.value ? 1 : 0);
          case "string":
            return new StringValue(operand.value ? "true" : "false");
          default:
            throw new Error(`Unknown BooleanValue filter: ${filter.value}`);
        }
      }
      throw new Error(`Cannot apply filter "${filter.value}" to type: ${operand.type}`);
    } else if (filterNode.type === "CallExpression") {
      const filter = filterNode;
      if (filter.callee.type !== "Identifier") {
        throw new Error(`Unknown filter: ${filter.callee.type}`);
      }
      const filterName = filter.callee.value;
      if (filterName === "tojson") {
        const [, kwargs] = this.evaluateArguments(filter.args, environment);
        const indent = kwargs.get("indent") ?? new NullValue();
        if (!(indent instanceof IntegerValue || indent instanceof NullValue)) {
          throw new Error("If set, indent must be a number");
        }
        return new StringValue(toJSON(operand, indent.value));
      } else if (filterName === "join") {
        let value;
        if (operand instanceof StringValue) {
          value = Array.from(operand.value);
        } else if (operand instanceof ArrayValue) {
          value = operand.value.map((x) => x.value);
        } else {
          throw new Error(`Cannot apply filter "${filterName}" to type: ${operand.type}`);
        }
        const [args, kwargs] = this.evaluateArguments(filter.args, environment);
        const separator = args.at(0) ?? kwargs.get("separator") ?? new StringValue("");
        if (!(separator instanceof StringValue)) {
          throw new Error("separator must be a string");
        }
        return new StringValue(value.join(separator.value));
      } else if (filterName === "int" || filterName === "float") {
        const [args, kwargs] = this.evaluateArguments(filter.args, environment);
        const defaultValue = args.at(0) ?? kwargs.get("default") ?? (filterName === "int" ? new IntegerValue(0) : new FloatValue(0));
        if (operand instanceof StringValue) {
          const val = filterName === "int" ? parseInt(operand.value, 10) : parseFloat(operand.value);
          return isNaN(val) ? defaultValue : filterName === "int" ? new IntegerValue(val) : new FloatValue(val);
        } else if (operand instanceof IntegerValue || operand instanceof FloatValue) {
          return operand;
        } else if (operand instanceof BooleanValue) {
          return filterName === "int" ? new IntegerValue(operand.value ? 1 : 0) : new FloatValue(operand.value ? 1 : 0);
        } else {
          throw new Error(`Cannot apply filter "${filterName}" to type: ${operand.type}`);
        }
      } else if (filterName === "default") {
        const [args, kwargs] = this.evaluateArguments(filter.args, environment);
        const defaultValue = args[0] ?? new StringValue("");
        const booleanValue = args[1] ?? kwargs.get("boolean") ?? new BooleanValue(false);
        if (!(booleanValue instanceof BooleanValue)) {
          throw new Error("`default` filter flag must be a boolean");
        }
        if (operand instanceof UndefinedValue || booleanValue.value && !operand.__bool__().value) {
          return defaultValue;
        }
        return operand;
      }
      if (operand instanceof ArrayValue) {
        switch (filterName) {
          case "selectattr":
          case "rejectattr": {
            const select = filterName === "selectattr";
            if (operand.value.some((x) => !(x instanceof ObjectValue))) {
              throw new Error(`\`${filterName}\` can only be applied to array of objects`);
            }
            if (filter.args.some((x) => x.type !== "StringLiteral")) {
              throw new Error(`arguments of \`${filterName}\` must be strings`);
            }
            const [attr, testName, value] = filter.args.map((x) => this.evaluate(x, environment));
            let testFunction;
            if (testName) {
              const test = environment.tests.get(testName.value);
              if (!test) {
                throw new Error(`Unknown test: ${testName.value}`);
              }
              testFunction = test;
            } else {
              testFunction = (...x) => x[0].__bool__().value;
            }
            const filtered = operand.value.filter((item) => {
              const a = item.value.get(attr.value);
              const result = a ? testFunction(a, value) : false;
              return select ? result : !result;
            });
            return new ArrayValue(filtered);
          }
          case "map": {
            const [, kwargs] = this.evaluateArguments(filter.args, environment);
            if (kwargs.has("attribute")) {
              const attr = kwargs.get("attribute");
              if (!(attr instanceof StringValue)) {
                throw new Error("attribute must be a string");
              }
              const defaultValue = kwargs.get("default");
              const mapped = operand.value.map((item) => {
                if (!(item instanceof ObjectValue)) {
                  throw new Error("items in map must be an object");
                }
                return item.value.get(attr.value) ?? defaultValue ?? new UndefinedValue();
              });
              return new ArrayValue(mapped);
            } else {
              throw new Error("`map` expressions without `attribute` set are not currently supported.");
            }
          }
        }
        throw new Error(`Unknown ArrayValue filter: ${filterName}`);
      } else if (operand instanceof StringValue) {
        switch (filterName) {
          case "indent": {
            const [args, kwargs] = this.evaluateArguments(filter.args, environment);
            const width = args.at(0) ?? kwargs.get("width") ?? new IntegerValue(4);
            if (!(width instanceof IntegerValue)) {
              throw new Error("width must be a number");
            }
            const first = args.at(1) ?? kwargs.get("first") ?? new BooleanValue(false);
            const blank = args.at(2) ?? kwargs.get("blank") ?? new BooleanValue(false);
            const lines = operand.value.split("\n");
            const indent = " ".repeat(width.value);
            const indented = lines.map(
              (x, i) => !first.value && i === 0 || !blank.value && x.length === 0 ? x : indent + x
            );
            return new StringValue(indented.join("\n"));
          }
          case "replace": {
            const replaceFn = operand.builtins.get("replace");
            if (!(replaceFn instanceof FunctionValue)) {
              throw new Error("replace filter not available");
            }
            const [args, kwargs] = this.evaluateArguments(filter.args, environment);
            return replaceFn.value([...args, new KeywordArgumentsValue(kwargs)], environment);
          }
        }
        throw new Error(`Unknown StringValue filter: ${filterName}`);
      } else {
        throw new Error(`Cannot apply filter "${filterName}" to type: ${operand.type}`);
      }
    }
    throw new Error(`Unknown filter: ${filterNode.type}`);
  }
  /**
   * Evaluates expressions following the filter operation type.
   */
  evaluateFilterExpression(node, environment) {
    const operand = this.evaluate(node.operand, environment);
    return this.applyFilter(operand, node.filter, environment);
  }
  /**
   * Evaluates expressions following the test operation type.
   */
  evaluateTestExpression(node, environment) {
    const operand = this.evaluate(node.operand, environment);
    const test = environment.tests.get(node.test.value);
    if (!test) {
      throw new Error(`Unknown test: ${node.test.value}`);
    }
    const result = test(operand);
    return new BooleanValue(node.negate ? !result : result);
  }
  /**
   * Evaluates expressions following the select operation type.
   */
  evaluateSelectExpression(node, environment) {
    const predicate = this.evaluate(node.test, environment);
    if (!predicate.__bool__().value) {
      return new UndefinedValue();
    }
    return this.evaluate(node.lhs, environment);
  }
  /**
   * Evaluates expressions following the unary operation type.
   */
  evaluateUnaryExpression(node, environment) {
    const argument = this.evaluate(node.argument, environment);
    switch (node.operator.value) {
      case "not":
        return new BooleanValue(!argument.value);
      default:
        throw new SyntaxError(`Unknown operator: ${node.operator.value}`);
    }
  }
  evaluateTernaryExpression(node, environment) {
    const cond = this.evaluate(node.condition, environment);
    return cond.__bool__().value ? this.evaluate(node.trueExpr, environment) : this.evaluate(node.falseExpr, environment);
  }
  evalProgram(program, environment) {
    return this.evaluateBlock(program.body, environment);
  }
  evaluateBlock(statements, environment) {
    let result = "";
    for (const statement of statements) {
      const lastEvaluated = this.evaluate(statement, environment);
      if (lastEvaluated.type !== "NullValue" && lastEvaluated.type !== "UndefinedValue") {
        result += lastEvaluated.toString();
      }
    }
    return new StringValue(result);
  }
  evaluateIdentifier(node, environment) {
    return environment.lookupVariable(node.value);
  }
  evaluateCallExpression(expr, environment) {
    const [args, kwargs] = this.evaluateArguments(expr.args, environment);
    if (kwargs.size > 0) {
      args.push(new KeywordArgumentsValue(kwargs));
    }
    const fn = this.evaluate(expr.callee, environment);
    if (fn.type !== "FunctionValue") {
      throw new Error(`Cannot call something that is not a function: got ${fn.type}`);
    }
    return fn.value(args, environment);
  }
  evaluateSliceExpression(object, expr, environment) {
    if (!(object instanceof ArrayValue || object instanceof StringValue)) {
      throw new Error("Slice object must be an array or string");
    }
    const start = this.evaluate(expr.start, environment);
    const stop = this.evaluate(expr.stop, environment);
    const step = this.evaluate(expr.step, environment);
    if (!(start instanceof IntegerValue || start instanceof UndefinedValue)) {
      throw new Error("Slice start must be numeric or undefined");
    }
    if (!(stop instanceof IntegerValue || stop instanceof UndefinedValue)) {
      throw new Error("Slice stop must be numeric or undefined");
    }
    if (!(step instanceof IntegerValue || step instanceof UndefinedValue)) {
      throw new Error("Slice step must be numeric or undefined");
    }
    if (object instanceof ArrayValue) {
      return new ArrayValue(slice2(object.value, start.value, stop.value, step.value));
    } else {
      return new StringValue(slice2(Array.from(object.value), start.value, stop.value, step.value).join(""));
    }
  }
  evaluateMemberExpression(expr, environment) {
    const object = this.evaluate(expr.object, environment);
    let property;
    if (expr.computed) {
      if (expr.property.type === "SliceExpression") {
        return this.evaluateSliceExpression(object, expr.property, environment);
      } else {
        property = this.evaluate(expr.property, environment);
      }
    } else {
      property = new StringValue(expr.property.value);
    }
    let value;
    if (object instanceof ObjectValue) {
      if (!(property instanceof StringValue)) {
        throw new Error(`Cannot access property with non-string: got ${property.type}`);
      }
      value = object.value.get(property.value) ?? object.builtins.get(property.value);
    } else if (object instanceof ArrayValue || object instanceof StringValue) {
      if (property instanceof IntegerValue) {
        value = object.value.at(property.value);
        if (object instanceof StringValue) {
          value = new StringValue(object.value.at(property.value));
        }
      } else if (property instanceof StringValue) {
        value = object.builtins.get(property.value);
      } else {
        throw new Error(`Cannot access property with non-string/non-number: got ${property.type}`);
      }
    } else {
      if (!(property instanceof StringValue)) {
        throw new Error(`Cannot access property with non-string: got ${property.type}`);
      }
      value = object.builtins.get(property.value);
    }
    return value instanceof RuntimeValue ? value : new UndefinedValue();
  }
  evaluateSet(node, environment) {
    const rhs = node.value ? this.evaluate(node.value, environment) : this.evaluateBlock(node.body, environment);
    if (node.assignee.type === "Identifier") {
      const variableName = node.assignee.value;
      environment.setVariable(variableName, rhs);
    } else if (node.assignee.type === "TupleLiteral") {
      const tuple = node.assignee;
      if (!(rhs instanceof ArrayValue)) {
        throw new Error(`Cannot unpack non-iterable type in set: ${rhs.type}`);
      }
      const arr = rhs.value;
      if (arr.length !== tuple.value.length) {
        throw new Error(`Too ${tuple.value.length > arr.length ? "few" : "many"} items to unpack in set`);
      }
      for (let i = 0; i < tuple.value.length; ++i) {
        const elem = tuple.value[i];
        if (elem.type !== "Identifier") {
          throw new Error(`Cannot unpack to non-identifier in set: ${elem.type}`);
        }
        environment.setVariable(elem.value, arr[i]);
      }
    } else if (node.assignee.type === "MemberExpression") {
      const member = node.assignee;
      const object = this.evaluate(member.object, environment);
      if (!(object instanceof ObjectValue)) {
        throw new Error("Cannot assign to member of non-object");
      }
      if (member.property.type !== "Identifier") {
        throw new Error("Cannot assign to member with non-identifier property");
      }
      object.value.set(member.property.value, rhs);
    } else {
      throw new Error(`Invalid LHS inside assignment expression: ${JSON.stringify(node.assignee)}`);
    }
    return new NullValue();
  }
  evaluateIf(node, environment) {
    const test = this.evaluate(node.test, environment);
    return this.evaluateBlock(test.__bool__().value ? node.body : node.alternate, environment);
  }
  evaluateFor(node, environment) {
    const scope = new Environment(environment);
    let test, iterable;
    if (node.iterable.type === "SelectExpression") {
      const select = node.iterable;
      iterable = this.evaluate(select.lhs, scope);
      test = select.test;
    } else {
      iterable = this.evaluate(node.iterable, scope);
    }
    if (!(iterable instanceof ArrayValue || iterable instanceof ObjectValue)) {
      throw new Error(`Expected iterable or object type in for loop: got ${iterable.type}`);
    }
    if (iterable instanceof ObjectValue) {
      iterable = iterable.keys();
    }
    const items = [];
    const scopeUpdateFunctions = [];
    for (let i = 0; i < iterable.value.length; ++i) {
      const loopScope = new Environment(scope);
      const current = iterable.value[i];
      let scopeUpdateFunction;
      if (node.loopvar.type === "Identifier") {
        scopeUpdateFunction = (scope2) => scope2.setVariable(node.loopvar.value, current);
      } else if (node.loopvar.type === "TupleLiteral") {
        const loopvar = node.loopvar;
        if (current.type !== "ArrayValue") {
          throw new Error(`Cannot unpack non-iterable type: ${current.type}`);
        }
        const c = current;
        if (loopvar.value.length !== c.value.length) {
          throw new Error(`Too ${loopvar.value.length > c.value.length ? "few" : "many"} items to unpack`);
        }
        scopeUpdateFunction = (scope2) => {
          for (let j = 0; j < loopvar.value.length; ++j) {
            if (loopvar.value[j].type !== "Identifier") {
              throw new Error(`Cannot unpack non-identifier type: ${loopvar.value[j].type}`);
            }
            scope2.setVariable(loopvar.value[j].value, c.value[j]);
          }
        };
      } else {
        throw new Error(`Invalid loop variable(s): ${node.loopvar.type}`);
      }
      if (test) {
        scopeUpdateFunction(loopScope);
        const testValue = this.evaluate(test, loopScope);
        if (!testValue.__bool__().value) {
          continue;
        }
      }
      items.push(current);
      scopeUpdateFunctions.push(scopeUpdateFunction);
    }
    let result = "";
    let noIteration = true;
    for (let i = 0; i < items.length; ++i) {
      const loop = /* @__PURE__ */ new Map([
        ["index", new IntegerValue(i + 1)],
        ["index0", new IntegerValue(i)],
        ["revindex", new IntegerValue(items.length - i)],
        ["revindex0", new IntegerValue(items.length - i - 1)],
        ["first", new BooleanValue(i === 0)],
        ["last", new BooleanValue(i === items.length - 1)],
        ["length", new IntegerValue(items.length)],
        ["previtem", i > 0 ? items[i - 1] : new UndefinedValue()],
        ["nextitem", i < items.length - 1 ? items[i + 1] : new UndefinedValue()]
      ]);
      scope.setVariable("loop", new ObjectValue(loop));
      scopeUpdateFunctions[i](scope);
      try {
        const evaluated = this.evaluateBlock(node.body, scope);
        result += evaluated.value;
      } catch (err) {
        if (err instanceof ContinueControl) {
          continue;
        }
        if (err instanceof BreakControl) {
          break;
        }
        throw err;
      }
      noIteration = false;
    }
    if (noIteration) {
      const defaultEvaluated = this.evaluateBlock(node.defaultBlock, scope);
      result += defaultEvaluated.value;
    }
    return new StringValue(result);
  }
  /**
   * See https://jinja.palletsprojects.com/en/3.1.x/templates/#macros for more information.
   */
  evaluateMacro(node, environment) {
    environment.setVariable(
      node.name.value,
      new FunctionValue((args, scope) => {
        const macroScope = new Environment(scope);
        args = args.slice();
        let kwargs;
        if (args.at(-1)?.type === "KeywordArgumentsValue") {
          kwargs = args.pop();
        }
        for (let i = 0; i < node.args.length; ++i) {
          const nodeArg = node.args[i];
          const passedArg = args[i];
          if (nodeArg.type === "Identifier") {
            const identifier = nodeArg;
            if (!passedArg) {
              throw new Error(`Missing positional argument: ${identifier.value}`);
            }
            macroScope.setVariable(identifier.value, passedArg);
          } else if (nodeArg.type === "KeywordArgumentExpression") {
            const kwarg = nodeArg;
            const value = passedArg ?? // Try positional arguments first
            kwargs?.value.get(kwarg.key.value) ?? // Look in user-passed kwargs
            this.evaluate(kwarg.value, macroScope);
            macroScope.setVariable(kwarg.key.value, value);
          } else {
            throw new Error(`Unknown argument type: ${nodeArg.type}`);
          }
        }
        return this.evaluateBlock(node.body, macroScope);
      })
    );
    return new NullValue();
  }
  evaluateCallStatement(node, environment) {
    const callerFn = new FunctionValue((callerArgs, callerEnv) => {
      const callBlockEnv = new Environment(callerEnv);
      if (node.callerArgs) {
        for (let i = 0; i < node.callerArgs.length; ++i) {
          const param = node.callerArgs[i];
          if (param.type !== "Identifier") {
            throw new Error(`Caller parameter must be an identifier, got ${param.type}`);
          }
          callBlockEnv.setVariable(param.value, callerArgs[i] ?? new UndefinedValue());
        }
      }
      return this.evaluateBlock(node.body, callBlockEnv);
    });
    const [macroArgs, macroKwargs] = this.evaluateArguments(node.call.args, environment);
    macroArgs.push(new KeywordArgumentsValue(macroKwargs));
    const fn = this.evaluate(node.call.callee, environment);
    if (fn.type !== "FunctionValue") {
      throw new Error(`Cannot call something that is not a function: got ${fn.type}`);
    }
    const newEnv = new Environment(environment);
    newEnv.setVariable("caller", callerFn);
    return fn.value(macroArgs, newEnv);
  }
  evaluateFilterStatement(node, environment) {
    const rendered = this.evaluateBlock(node.body, environment);
    return this.applyFilter(rendered, node.filter, environment);
  }
  evaluate(statement, environment) {
    if (!statement)
      return new UndefinedValue();
    switch (statement.type) {
      case "Program":
        return this.evalProgram(statement, environment);
      case "Set":
        return this.evaluateSet(statement, environment);
      case "If":
        return this.evaluateIf(statement, environment);
      case "For":
        return this.evaluateFor(statement, environment);
      case "Macro":
        return this.evaluateMacro(statement, environment);
      case "CallStatement":
        return this.evaluateCallStatement(statement, environment);
      case "Break":
        throw new BreakControl();
      case "Continue":
        throw new ContinueControl();
      case "IntegerLiteral":
        return new IntegerValue(statement.value);
      case "FloatLiteral":
        return new FloatValue(statement.value);
      case "StringLiteral":
        return new StringValue(statement.value);
      case "ArrayLiteral":
        return new ArrayValue(statement.value.map((x) => this.evaluate(x, environment)));
      case "TupleLiteral":
        return new TupleValue(statement.value.map((x) => this.evaluate(x, environment)));
      case "ObjectLiteral": {
        const mapping = /* @__PURE__ */ new Map();
        for (const [key, value] of statement.value) {
          const evaluatedKey = this.evaluate(key, environment);
          if (!(evaluatedKey instanceof StringValue)) {
            throw new Error(`Object keys must be strings: got ${evaluatedKey.type}`);
          }
          mapping.set(evaluatedKey.value, this.evaluate(value, environment));
        }
        return new ObjectValue(mapping);
      }
      case "Identifier":
        return this.evaluateIdentifier(statement, environment);
      case "CallExpression":
        return this.evaluateCallExpression(statement, environment);
      case "MemberExpression":
        return this.evaluateMemberExpression(statement, environment);
      case "UnaryExpression":
        return this.evaluateUnaryExpression(statement, environment);
      case "BinaryExpression":
        return this.evaluateBinaryExpression(statement, environment);
      case "FilterExpression":
        return this.evaluateFilterExpression(statement, environment);
      case "FilterStatement":
        return this.evaluateFilterStatement(statement, environment);
      case "TestExpression":
        return this.evaluateTestExpression(statement, environment);
      case "SelectExpression":
        return this.evaluateSelectExpression(statement, environment);
      case "Ternary":
        return this.evaluateTernaryExpression(statement, environment);
      case "Comment":
        return new NullValue();
      default:
        throw new SyntaxError(`Unknown node type: ${statement.type}`);
    }
  }
};
function convertToRuntimeValues(input) {
  switch (typeof input) {
    case "number":
      return Number.isInteger(input) ? new IntegerValue(input) : new FloatValue(input);
    case "string":
      return new StringValue(input);
    case "boolean":
      return new BooleanValue(input);
    case "undefined":
      return new UndefinedValue();
    case "object":
      if (input === null) {
        return new NullValue();
      } else if (Array.isArray(input)) {
        return new ArrayValue(input.map(convertToRuntimeValues));
      } else {
        return new ObjectValue(
          new Map(Object.entries(input).map(([key, value]) => [key, convertToRuntimeValues(value)]))
        );
      }
    case "function":
      return new FunctionValue((args, _scope) => {
        const result = input(...args.map((x) => x.value)) ?? null;
        return convertToRuntimeValues(result);
      });
    default:
      throw new Error(`Cannot convert to runtime value: ${input}`);
  }
}
function toJSON(input, indent, depth) {
  const currentDepth = depth ?? 0;
  switch (input.type) {
    case "NullValue":
    case "UndefinedValue":
      return "null";
    case "IntegerValue":
    case "FloatValue":
    case "StringValue":
    case "BooleanValue":
      return JSON.stringify(input.value);
    case "ArrayValue":
    case "ObjectValue": {
      const indentValue = indent ? " ".repeat(indent) : "";
      const basePadding = "\n" + indentValue.repeat(currentDepth);
      const childrenPadding = basePadding + indentValue;
      if (input.type === "ArrayValue") {
        const core = input.value.map((x) => toJSON(x, indent, currentDepth + 1));
        return indent ? `[${childrenPadding}${core.join(`,${childrenPadding}`)}${basePadding}]` : `[${core.join(", ")}]`;
      } else {
        const core = Array.from(input.value.entries()).map(([key, value]) => {
          const v = `"${key}": ${toJSON(value, indent, currentDepth + 1)}`;
          return indent ? `${childrenPadding}${v}` : v;
        });
        return indent ? `{${core.join(",")}${basePadding}}` : `{${core.join(", ")}}`;
      }
    }
    default:
      throw new Error(`Cannot convert to JSON: ${input.type}`);
  }
}
var NEWLINE = "\n";
var OPEN_STATEMENT = "{%- ";
var CLOSE_STATEMENT = " -%}";
function getBinaryOperatorPrecedence(expr) {
  switch (expr.operator.type) {
    case "MultiplicativeBinaryOperator":
      return 4;
    case "AdditiveBinaryOperator":
      return 3;
    case "ComparisonBinaryOperator":
      return 2;
    case "Identifier":
      if (expr.operator.value === "and")
        return 1;
      if (expr.operator.value === "in" || expr.operator.value === "not in")
        return 2;
      return 0;
  }
  return 0;
}
function format(program, indent = "	") {
  const indentStr = typeof indent === "number" ? " ".repeat(indent) : indent;
  const body = formatStatements(program.body, 0, indentStr);
  return body.replace(/\n$/, "");
}
function createStatement(...text) {
  return OPEN_STATEMENT + text.join(" ") + CLOSE_STATEMENT;
}
function formatStatements(stmts, depth, indentStr) {
  return stmts.map((stmt) => formatStatement(stmt, depth, indentStr)).join(NEWLINE);
}
function formatStatement(node, depth, indentStr) {
  const pad = indentStr.repeat(depth);
  switch (node.type) {
    case "Program":
      return formatStatements(node.body, depth, indentStr);
    case "If":
      return formatIf(node, depth, indentStr);
    case "For":
      return formatFor(node, depth, indentStr);
    case "Set":
      return formatSet(node, depth, indentStr);
    case "Macro":
      return formatMacro(node, depth, indentStr);
    case "Break":
      return pad + createStatement("break");
    case "Continue":
      return pad + createStatement("continue");
    case "CallStatement":
      return formatCallStatement(node, depth, indentStr);
    case "FilterStatement":
      return formatFilterStatement(node, depth, indentStr);
    case "Comment":
      return pad + "{# " + node.value + " #}";
    default:
      return pad + "{{- " + formatExpression(node) + " -}}";
  }
}
function formatIf(node, depth, indentStr) {
  const pad = indentStr.repeat(depth);
  const clauses = [];
  let current = node;
  while (current) {
    clauses.push({ test: current.test, body: current.body });
    if (current.alternate.length === 1 && current.alternate[0].type === "If") {
      current = current.alternate[0];
    } else {
      break;
    }
  }
  let out = pad + createStatement("if", formatExpression(clauses[0].test)) + NEWLINE + formatStatements(clauses[0].body, depth + 1, indentStr);
  for (let i = 1; i < clauses.length; ++i) {
    out += NEWLINE + pad + createStatement("elif", formatExpression(clauses[i].test)) + NEWLINE + formatStatements(clauses[i].body, depth + 1, indentStr);
  }
  if (current && current.alternate.length > 0) {
    out += NEWLINE + pad + createStatement("else") + NEWLINE + formatStatements(current.alternate, depth + 1, indentStr);
  }
  out += NEWLINE + pad + createStatement("endif");
  return out;
}
function formatFor(node, depth, indentStr) {
  const pad = indentStr.repeat(depth);
  let formattedIterable = "";
  if (node.iterable.type === "SelectExpression") {
    const n = node.iterable;
    formattedIterable = `${formatExpression(n.lhs)} if ${formatExpression(n.test)}`;
  } else {
    formattedIterable = formatExpression(node.iterable);
  }
  let out = pad + createStatement("for", formatExpression(node.loopvar), "in", formattedIterable) + NEWLINE + formatStatements(node.body, depth + 1, indentStr);
  if (node.defaultBlock.length > 0) {
    out += NEWLINE + pad + createStatement("else") + NEWLINE + formatStatements(node.defaultBlock, depth + 1, indentStr);
  }
  out += NEWLINE + pad + createStatement("endfor");
  return out;
}
function formatSet(node, depth, indentStr) {
  const pad = indentStr.repeat(depth);
  const left = formatExpression(node.assignee);
  const right = node.value ? formatExpression(node.value) : "";
  const value = pad + createStatement("set", `${left}${node.value ? " = " + right : ""}`);
  if (node.body.length === 0) {
    return value;
  }
  return value + NEWLINE + formatStatements(node.body, depth + 1, indentStr) + NEWLINE + pad + createStatement("endset");
}
function formatMacro(node, depth, indentStr) {
  const pad = indentStr.repeat(depth);
  const args = node.args.map(formatExpression).join(", ");
  return pad + createStatement("macro", `${node.name.value}(${args})`) + NEWLINE + formatStatements(node.body, depth + 1, indentStr) + NEWLINE + pad + createStatement("endmacro");
}
function formatCallStatement(node, depth, indentStr) {
  const pad = indentStr.repeat(depth);
  const params = node.callerArgs && node.callerArgs.length > 0 ? `(${node.callerArgs.map(formatExpression).join(", ")})` : "";
  const callExpr = formatExpression(node.call);
  let out = pad + createStatement(`call${params}`, callExpr) + NEWLINE;
  out += formatStatements(node.body, depth + 1, indentStr) + NEWLINE;
  out += pad + createStatement("endcall");
  return out;
}
function formatFilterStatement(node, depth, indentStr) {
  const pad = indentStr.repeat(depth);
  const spec = node.filter.type === "Identifier" ? node.filter.value : formatExpression(node.filter);
  let out = pad + createStatement("filter", spec) + NEWLINE;
  out += formatStatements(node.body, depth + 1, indentStr) + NEWLINE;
  out += pad + createStatement("endfilter");
  return out;
}
function formatExpression(node, parentPrec = -1) {
  switch (node.type) {
    case "SpreadExpression": {
      const n = node;
      return `*${formatExpression(n.argument)}`;
    }
    case "Identifier":
      return node.value;
    case "IntegerLiteral":
      return `${node.value}`;
    case "FloatLiteral":
      return `${node.value}`;
    case "StringLiteral":
      return JSON.stringify(node.value);
    case "BinaryExpression": {
      const n = node;
      const thisPrecedence = getBinaryOperatorPrecedence(n);
      const left = formatExpression(n.left, thisPrecedence);
      const right = formatExpression(n.right, thisPrecedence + 1);
      const expr = `${left} ${n.operator.value} ${right}`;
      return thisPrecedence < parentPrec ? `(${expr})` : expr;
    }
    case "UnaryExpression": {
      const n = node;
      const val = n.operator.value + (n.operator.value === "not" ? " " : "") + formatExpression(n.argument, Infinity);
      return val;
    }
    case "CallExpression": {
      const n = node;
      const args = n.args.map(formatExpression).join(", ");
      return `${formatExpression(n.callee)}(${args})`;
    }
    case "MemberExpression": {
      const n = node;
      let obj = formatExpression(n.object);
      if (![
        "Identifier",
        "MemberExpression",
        "CallExpression",
        "StringLiteral",
        "IntegerLiteral",
        "FloatLiteral",
        "ArrayLiteral",
        "TupleLiteral",
        "ObjectLiteral"
      ].includes(n.object.type)) {
        obj = `(${obj})`;
      }
      let prop = formatExpression(n.property);
      if (!n.computed && n.property.type !== "Identifier") {
        prop = `(${prop})`;
      }
      return n.computed ? `${obj}[${prop}]` : `${obj}.${prop}`;
    }
    case "FilterExpression": {
      const n = node;
      const operand = formatExpression(n.operand, Infinity);
      if (n.filter.type === "CallExpression") {
        return `${operand} | ${formatExpression(n.filter)}`;
      }
      return `${operand} | ${n.filter.value}`;
    }
    case "SelectExpression": {
      const n = node;
      return `${formatExpression(n.lhs)} if ${formatExpression(n.test)}`;
    }
    case "TestExpression": {
      const n = node;
      return `${formatExpression(n.operand)} is${n.negate ? " not" : ""} ${n.test.value}`;
    }
    case "ArrayLiteral":
    case "TupleLiteral": {
      const elems = node.value.map(formatExpression);
      const brackets = node.type === "ArrayLiteral" ? "[]" : "()";
      return `${brackets[0]}${elems.join(", ")}${brackets[1]}`;
    }
    case "ObjectLiteral": {
      const entries = Array.from(node.value.entries()).map(
        ([k, v]) => `${formatExpression(k)}: ${formatExpression(v)}`
      );
      return `{${entries.join(", ")}}`;
    }
    case "SliceExpression": {
      const n = node;
      const s = n.start ? formatExpression(n.start) : "";
      const t = n.stop ? formatExpression(n.stop) : "";
      const st = n.step ? `:${formatExpression(n.step)}` : "";
      return `${s}:${t}${st}`;
    }
    case "KeywordArgumentExpression": {
      const n = node;
      return `${n.key.value}=${formatExpression(n.value)}`;
    }
    case "Ternary": {
      const n = node;
      const expr = `${formatExpression(n.trueExpr)} if ${formatExpression(n.condition, 0)} else ${formatExpression(
        n.falseExpr
      )}`;
      return parentPrec > -1 ? `(${expr})` : expr;
    }
    default:
      throw new Error(`Unknown expression type: ${node.type}`);
  }
}
var Template = class {
  parsed;
  /**
   * @param {string} template The template string
   */
  constructor(template) {
    const tokens = tokenize(template, {
      lstrip_blocks: true,
      trim_blocks: true
    });
    this.parsed = parse(tokens);
  }
  render(items) {
    const env2 = new Environment();
    setupGlobals(env2);
    if (items) {
      for (const [key, value] of Object.entries(items)) {
        env2.set(key, value);
      }
    }
    const interpreter = new Interpreter(env2);
    const result = interpreter.run(this.parsed);
    return result.value;
  }
  format(options) {
    return format(this.parsed, options?.indent || "	");
  }
};

// src/models/whisper/common_whisper.js
var WHISPER_LANGUAGES = [
  ["en", "english"],
  ["zh", "chinese"],
  ["de", "german"],
  ["es", "spanish"],
  ["ru", "russian"],
  ["ko", "korean"],
  ["fr", "french"],
  ["ja", "japanese"],
  ["pt", "portuguese"],
  ["tr", "turkish"],
  ["pl", "polish"],
  ["ca", "catalan"],
  ["nl", "dutch"],
  ["ar", "arabic"],
  ["sv", "swedish"],
  ["it", "italian"],
  ["id", "indonesian"],
  ["hi", "hindi"],
  ["fi", "finnish"],
  ["vi", "vietnamese"],
  ["he", "hebrew"],
  ["uk", "ukrainian"],
  ["el", "greek"],
  ["ms", "malay"],
  ["cs", "czech"],
  ["ro", "romanian"],
  ["da", "danish"],
  ["hu", "hungarian"],
  ["ta", "tamil"],
  ["no", "norwegian"],
  ["th", "thai"],
  ["ur", "urdu"],
  ["hr", "croatian"],
  ["bg", "bulgarian"],
  ["lt", "lithuanian"],
  ["la", "latin"],
  ["mi", "maori"],
  ["ml", "malayalam"],
  ["cy", "welsh"],
  ["sk", "slovak"],
  ["te", "telugu"],
  ["fa", "persian"],
  ["lv", "latvian"],
  ["bn", "bengali"],
  ["sr", "serbian"],
  ["az", "azerbaijani"],
  ["sl", "slovenian"],
  ["kn", "kannada"],
  ["et", "estonian"],
  ["mk", "macedonian"],
  ["br", "breton"],
  ["eu", "basque"],
  ["is", "icelandic"],
  ["hy", "armenian"],
  ["ne", "nepali"],
  ["mn", "mongolian"],
  ["bs", "bosnian"],
  ["kk", "kazakh"],
  ["sq", "albanian"],
  ["sw", "swahili"],
  ["gl", "galician"],
  ["mr", "marathi"],
  ["pa", "punjabi"],
  ["si", "sinhala"],
  ["km", "khmer"],
  ["sn", "shona"],
  ["yo", "yoruba"],
  ["so", "somali"],
  ["af", "afrikaans"],
  ["oc", "occitan"],
  ["ka", "georgian"],
  ["be", "belarusian"],
  ["tg", "tajik"],
  ["sd", "sindhi"],
  ["gu", "gujarati"],
  ["am", "amharic"],
  ["yi", "yiddish"],
  ["lo", "lao"],
  ["uz", "uzbek"],
  ["fo", "faroese"],
  ["ht", "haitian creole"],
  ["ps", "pashto"],
  ["tk", "turkmen"],
  ["nn", "nynorsk"],
  ["mt", "maltese"],
  ["sa", "sanskrit"],
  ["lb", "luxembourgish"],
  ["my", "myanmar"],
  ["bo", "tibetan"],
  ["tl", "tagalog"],
  ["mg", "malagasy"],
  ["as", "assamese"],
  ["tt", "tatar"],
  ["haw", "hawaiian"],
  ["ln", "lingala"],
  ["ha", "hausa"],
  ["ba", "bashkir"],
  ["jw", "javanese"],
  ["su", "sundanese"]
];
var WHISPER_LANGUAGE_MAPPING = new Map(WHISPER_LANGUAGES);
var WHISPER_TO_LANGUAGE_CODE_MAPPING = new Map([
  ...WHISPER_LANGUAGES.map(([k, v]) => [v, k]),
  ...[
    ["burmese", "my"],
    ["valencian", "ca"],
    ["flemish", "nl"],
    ["haitian", "ht"],
    ["letzeburgesch", "lb"],
    ["pushto", "ps"],
    ["panjabi", "pa"],
    ["moldavian", "ro"],
    ["moldovan", "ro"],
    ["sinhalese", "si"],
    ["castilian", "es"]
  ]
]);
function whisper_language_to_code(language) {
  language = language.toLowerCase();
  let language_code = WHISPER_TO_LANGUAGE_CODE_MAPPING.get(language);
  if (language_code === void 0) {
    const language_special_token = language.match(/^<\|([a-z]{2})\|>$/);
    if (language_special_token) {
      language = language_special_token[1];
    }
    if (WHISPER_LANGUAGE_MAPPING.has(language)) {
      language_code = language;
    } else {
      const is_language_code = language.length === 2;
      const langs = is_language_code ? WHISPER_LANGUAGE_MAPPING.keys() : WHISPER_LANGUAGE_MAPPING.values();
      throw new Error(`Language "${language}" is not supported. Must be one of: ${JSON.stringify(Array.from(langs))}`);
    }
  }
  return language_code;
}

// src/tokenizers.js
async function loadTokenizer(pretrained_model_name_or_path, options) {
  const info = await Promise.all([
    getModelJSON(pretrained_model_name_or_path, "tokenizer.json", true, options),
    getModelJSON(pretrained_model_name_or_path, "tokenizer_config.json", true, options)
  ]);
  if (options.legacy !== null) {
    info[1].legacy = options.legacy;
  }
  return info;
}
function regexSplit(text, regex) {
  const result = [];
  let prev = 0;
  for (const match of text.matchAll(regex)) {
    const fullMatch = match[0];
    if (prev < match.index) {
      result.push(text.slice(prev, match.index));
    }
    if (fullMatch.length > 0) {
      result.push(fullMatch);
    }
    prev = match.index + fullMatch.length;
  }
  if (prev < text.length) {
    result.push(text.slice(prev));
  }
  return result;
}
function createPattern(pattern, invert = true) {
  if (pattern.Regex !== void 0) {
    let regex = pattern.Regex.replace(/\\([#&~])/g, "$1");
    for (const [key, value] of PROBLEMATIC_REGEX_MAP) {
      regex = regex.replaceAll(key, value);
    }
    return new RegExp(regex, "gu");
  } else if (pattern.String !== void 0) {
    const escaped = escapeRegExp(pattern.String);
    return new RegExp(invert ? escaped : `(${escaped})`, "gu");
  } else {
    console.warn("Unknown pattern type:", pattern);
    return null;
  }
}
function objectToMap(obj) {
  return new Map(Object.entries(obj));
}
function prepareTensorForDecode(tensor) {
  const dims = tensor.dims;
  switch (dims.length) {
    case 1:
      return tensor.tolist();
    case 2:
      if (dims[0] !== 1) {
        throw new Error("Unable to decode tensor with `batch size !== 1`. Use `tokenizer.batch_decode(...)` for batched inputs.");
      }
      return tensor.tolist()[0];
    default:
      throw new Error(`Expected tensor to have 1-2 dimensions, got ${dims.length}.`);
  }
}
function clean_up_tokenization(text) {
  return text.replace(/ \./g, ".").replace(/ \?/g, "?").replace(/ \!/g, "!").replace(/ ,/g, ",").replace(/ \' /g, "'").replace(/ n\'t/g, "n't").replace(/ \'m/g, "'m").replace(/ \'s/g, "'s").replace(/ \'ve/g, "'ve").replace(/ \'re/g, "'re");
}
function remove_accents(text) {
  return text.replace(/\p{M}/gu, "");
}
function lowercase_and_remove_accent(text) {
  return remove_accents(text.toLowerCase());
}
function is_chinese_char(cp) {
  return cp >= 19968 && cp <= 40959 || cp >= 13312 && cp <= 19903 || cp >= 131072 && cp <= 173791 || cp >= 173824 && cp <= 177983 || cp >= 177984 && cp <= 178207 || cp >= 178208 && cp <= 183983 || cp >= 63744 && cp <= 64255 || cp >= 194560 && cp <= 195103;
}
function fuse_unk(arr, tokens_to_ids, unk_token_id) {
  const fused = [];
  let i = 0;
  while (i < arr.length) {
    fused.push(arr[i]);
    if ((tokens_to_ids.get(arr[i]) ?? unk_token_id) !== unk_token_id) {
      ++i;
      continue;
    }
    while (++i < arr.length && (tokens_to_ids.get(arr[i]) ?? unk_token_id) === unk_token_id) {
      if (tokens_to_ids.get(fused.at(-1)) !== unk_token_id) {
        fused[fused.length - 1] += arr[i];
      }
    }
  }
  return fused;
}
function whitespace_split(text) {
  return text.match(/\S+/g) || [];
}
var PUNCTUATION_REGEX = "\\p{P}\\u0021-\\u002F\\u003A-\\u0040\\u005B-\\u0060\\u007B-\\u007E";
var PUNCTUATION_ONLY_REGEX = new RegExp(`^[${PUNCTUATION_REGEX}]+$`, "gu");
var BLOOM_SPLIT_CHARS = ".,!?\u2026\u3002\uFF0C\u3001\u0964\u06D4\u060C";
var PROBLEMATIC_REGEX_MAP = /* @__PURE__ */ new Map([
  // These use the case insensitive group modifier, which is not supported in JavaScript.
  // When parsing the regex, an "Invalid group" error is thrown.
  ["(?i:'s|'t|'re|'ve|'m|'ll|'d)", "(?:'([sS]|[tT]|[rR][eE]|[vV][eE]|[mM]|[lL][lL]|[dD]))"],
  ["(?i:[sdmt]|ll|ve|re)", "(?:[sS]|[dD]|[mM]|[tT]|[lL][lL]|[vV][eE]|[rR][eE])"],
  // JS doesn't support possessive quantifiers (these are used in recent OpenAI tokenizers).
  ["[^\\r\\n\\p{L}\\p{N}]?+", "[^\\r\\n\\p{L}\\p{N}]?"],
  ["[^\\s\\p{L}\\p{N}]++", "[^\\s\\p{L}\\p{N}]+"],
  // Used to override the default (invalid) regex of the bloom pretokenizer.
  // For more information, see https://github.com/huggingface/transformers.js/issues/94
  [` ?[^(\\s|[${BLOOM_SPLIT_CHARS}])]+`, ` ?[^\\s${BLOOM_SPLIT_CHARS}]+`]
]);
var AddedToken = class {
  /**
   * Creates a new instance of AddedToken.
   * @param {Object} config Added token configuration object.
   * @param {string} config.content The content of the added token.
   * @param {number} config.id The id of the added token.
   * @param {boolean} [config.single_word=false] Whether this token must be a single word or can break words.
   * @param {boolean} [config.lstrip=false] Whether this token should strip whitespaces on its left.
   * @param {boolean} [config.rstrip=false] Whether this token should strip whitespaces on its right.
   * @param {boolean} [config.normalized=false] Whether this token should be normalized.
   * @param {boolean} [config.special=false] Whether this token is special.
   */
  constructor(config) {
    this.content = config.content;
    this.id = config.id;
    this.single_word = config.single_word ?? false;
    this.lstrip = config.lstrip ?? false;
    this.rstrip = config.rstrip ?? false;
    this.special = config.special ?? false;
    this.normalized = config.normalized ?? null;
  }
};
var TokenizerModel = class extends Callable {
  /**
   * Creates a new instance of TokenizerModel.
   * @param {Object} config The configuration object for the TokenizerModel.
   */
  constructor(config) {
    super();
    this.config = config;
    this.vocab = [];
    this.tokens_to_ids = /* @__PURE__ */ new Map();
    this.unk_token_id = void 0;
    this.unk_token = void 0;
    this.end_of_word_suffix = void 0;
    this.fuse_unk = this.config.fuse_unk ?? false;
  }
  /**
   * Instantiates a new TokenizerModel instance based on the configuration object provided.
   * @param {Object} config The configuration object for the TokenizerModel.
   * @param {...*} args Optional arguments to pass to the specific TokenizerModel constructor.
   * @returns {TokenizerModel} A new instance of a TokenizerModel.
   * @throws Will throw an error if the TokenizerModel type in the config is not recognized.
   */
  static fromConfig(config, ...args) {
    switch (config.type) {
      case "WordPiece":
        return new WordPieceTokenizer(config);
      case "Unigram":
        return new Unigram(config, ...args);
      case "BPE":
        return new BPE(config);
      default:
        if (config.vocab) {
          if (Array.isArray(config.vocab)) {
            return new Unigram(config, ...args);
          } else if (Object.hasOwn(config, "continuing_subword_prefix") && Object.hasOwn(config, "unk_token")) {
            if (Object.hasOwn(config, "merges")) {
              return new BPE(config);
            } else {
              return new WordPieceTokenizer(config);
            }
          } else {
            return new LegacyTokenizerModel(config, ...args);
          }
        }
        throw new Error(`Unknown TokenizerModel type: ${config.type}`);
    }
  }
  /**
   * Internal function to call the TokenizerModel instance.
   * @param {string[]} tokens The tokens to encode.
   * @returns {string[]} The encoded tokens.
   */
  _call(tokens) {
    tokens = this.encode(tokens);
    if (this.fuse_unk) {
      tokens = fuse_unk(tokens, this.tokens_to_ids, this.unk_token_id);
    }
    return tokens;
  }
  /**
   * Encodes a list of tokens into a list of token IDs.
   * @param {string[]} tokens The tokens to encode.
   * @returns {string[]} The encoded tokens.
   * @throws Will throw an error if not implemented in a subclass.
   */
  encode(tokens) {
    throw Error("encode should be implemented in subclass.");
  }
  /**
   * Converts a list of tokens into a list of token IDs.
   * @param {string[]} tokens The tokens to convert.
   * @returns {number[]} The converted token IDs.
   */
  convert_tokens_to_ids(tokens) {
    return tokens.map((t) => this.tokens_to_ids.get(t) ?? this.unk_token_id);
  }
  /**
   * Converts a list of token IDs into a list of tokens.
   * @param {number[]|bigint[]} ids The token IDs to convert.
   * @returns {string[]} The converted tokens.
   */
  convert_ids_to_tokens(ids) {
    return ids.map((i) => this.vocab[i] ?? this.unk_token);
  }
};
var WordPieceTokenizer = class extends TokenizerModel {
  /**
   * @param {Object} config The configuration object.
   * @param {Object} config.vocab A mapping of tokens to ids.
   * @param {string} config.unk_token The unknown token string.
   * @param {string} config.continuing_subword_prefix The prefix to use for continuing subwords.
   * @param {number} [config.max_input_chars_per_word=100] The maximum number of characters per word.
   */
  constructor(config) {
    super(config);
    this.tokens_to_ids = objectToMap(config.vocab);
    this.unk_token_id = this.tokens_to_ids.get(config.unk_token);
    this.unk_token = config.unk_token;
    this.max_input_chars_per_word = config.max_input_chars_per_word ?? 100;
    this.vocab = new Array(this.tokens_to_ids.size);
    for (const [key, value] of this.tokens_to_ids) {
      this.vocab[value] = key;
    }
  }
  /**
   * Encodes an array of tokens using WordPiece encoding.
   * @param {string[]} tokens The tokens to encode.
   * @returns {string[]} An array of encoded tokens.
   */
  encode(tokens) {
    const outputTokens = [];
    for (const token of tokens) {
      const chars = [...token];
      if (chars.length > this.max_input_chars_per_word) {
        outputTokens.push(this.unk_token);
        continue;
      }
      let isUnknown = false;
      let start = 0;
      const subTokens = [];
      while (start < chars.length) {
        let end = chars.length;
        let currentSubstring = null;
        while (start < end) {
          let substr = chars.slice(start, end).join("");
          if (start > 0) {
            substr = this.config.continuing_subword_prefix + substr;
          }
          if (this.tokens_to_ids.has(substr)) {
            currentSubstring = substr;
            break;
          }
          --end;
        }
        if (currentSubstring === null) {
          isUnknown = true;
          break;
        }
        subTokens.push(currentSubstring);
        start = end;
      }
      if (isUnknown) {
        outputTokens.push(this.unk_token);
      } else {
        outputTokens.push(...subTokens);
      }
    }
    return outputTokens;
  }
};
var Unigram = class extends TokenizerModel {
  /**
   * Create a new Unigram tokenizer model.
   * @param {Object} config The configuration object for the Unigram model.
   * @param {number} config.unk_id The ID of the unknown token
   * @param {[string, number][]} config.vocab A 2D array representing a mapping of tokens to scores.
   * @param {Object} moreConfig Additional configuration object for the Unigram model.
   */
  constructor(config, moreConfig) {
    super(config);
    const vocabSize = config.vocab.length;
    this.vocab = new Array(vocabSize);
    this.scores = new Array(vocabSize);
    for (let i = 0; i < vocabSize; ++i) {
      [this.vocab[i], this.scores[i]] = config.vocab[i];
    }
    this.unk_token_id = config.unk_id;
    this.unk_token = this.vocab[config.unk_id];
    this.tokens_to_ids = new Map(this.vocab.map((x, i) => [x, i]));
    this.bos_token = " ";
    this.bos_token_id = this.tokens_to_ids.get(this.bos_token);
    this.eos_token = moreConfig.eos_token;
    this.eos_token_id = this.tokens_to_ids.get(this.eos_token);
    this.unk_token = this.vocab[this.unk_token_id];
    this.minScore = min(this.scores)[0];
    this.unk_score = this.minScore - 10;
    this.scores[this.unk_token_id] = this.unk_score;
    this.trie = new CharTrie();
    this.trie.extend(this.vocab);
    this.fuse_unk = true;
  }
  /**
   * Populates lattice nodes.
   * @param {TokenLattice} lattice The token lattice to populate with nodes.
   */
  populateNodes(lattice) {
    const chars = lattice.chars;
    const mblen = 1;
    let beginPos = 0;
    while (beginPos < chars.length) {
      let hasSingleNode = false;
      const tokens = [];
      const sliced = chars.slice(beginPos).join("");
      const prefixedTokens = this.trie.commonPrefixSearch(sliced);
      for (const token of prefixedTokens) {
        tokens.push(token);
        const tokenId = this.tokens_to_ids.get(token);
        const tokenScore = this.scores[tokenId];
        const n = len(token);
        lattice.insert(beginPos, n, tokenScore, tokenId);
        if (!hasSingleNode && n === mblen) {
          hasSingleNode = true;
        }
      }
      if (!hasSingleNode) {
        lattice.insert(beginPos, mblen, this.unk_score, this.unk_token_id);
      }
      beginPos += mblen;
    }
  }
  /**
   * Encodes an array of tokens into an array of subtokens using the unigram model.
   *
   * @param {string} normalized The normalized string.
   * @returns {string[]} An array of subtokens obtained by encoding the input tokens using the unigram model.
   */
  tokenize(normalized) {
    const lattice = new TokenLattice(normalized, this.bos_token_id, this.eos_token_id);
    this.populateNodes(lattice);
    return lattice.tokens();
  }
  /**
   * Encodes an array of tokens using Unigram encoding.
   * @param {string[]} tokens The tokens to encode.
   * @returns {string[]} An array of encoded tokens.
   */
  encode(tokens) {
    const toReturn = [];
    for (const token of tokens) {
      const tokenized = this.tokenize(token);
      toReturn.push(...tokenized);
    }
    return toReturn;
  }
};
var BYTES_TO_UNICODE = (() => {
  const bs = [
    ...Array.from({ length: "~".charCodeAt(0) - "!".charCodeAt(0) + 1 }, (_, i) => i + "!".charCodeAt(0)),
    ...Array.from({ length: "\xAC".charCodeAt(0) - "\xA1".charCodeAt(0) + 1 }, (_, i) => i + "\xA1".charCodeAt(0)),
    ...Array.from({ length: "\xFF".charCodeAt(0) - "\xAE".charCodeAt(0) + 1 }, (_, i) => i + "\xAE".charCodeAt(0))
  ];
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; ++b) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n += 1;
    }
  }
  const ccs = cs.map((n2) => String.fromCharCode(n2));
  return Object.fromEntries(bs.map((b, i) => [b, ccs[i]]));
})();
var UNICODE_TO_BYTES = reverseDictionary(BYTES_TO_UNICODE);
var BPE = class extends TokenizerModel {
  /**
   * Create a BPE instance.
   * @param {Object} config The configuration object for BPE.
   * @param {Object} config.vocab A mapping of tokens to ids.
   * @param {string[]|[string, string][]} config.merges An array of BPE merges as strings.
   * @param {string} config.unk_token The unknown token used for out of vocabulary words.
   * @param {string} config.end_of_word_suffix The suffix to place at the end of each word.
   * @param {string} [config.continuing_subword_suffix] The suffix to insert between words.
   * @param {boolean} [config.byte_fallback=false] Whether to use spm byte-fallback trick (defaults to False)
   * @param {boolean} [config.ignore_merges=false] Whether or not to match tokens with the vocab before using merges.
   */
  constructor(config) {
    super(config);
    this.tokens_to_ids = objectToMap(config.vocab);
    this.unk_token_id = this.tokens_to_ids.get(config.unk_token);
    this.unk_token = config.unk_token;
    this.vocab = new Array(this.tokens_to_ids.size);
    for (const [key, value] of this.tokens_to_ids) {
      this.vocab[value] = key;
    }
    const use_new_merge_format = Array.isArray(config.merges[0]);
    this.merges = use_new_merge_format ? (
      /** @type {[string, string][]} */
      config.merges
    ) : (
      /** @type {string[]} */
      config.merges.map((x) => (
        /** @type {[string, string]} */
        x.split(" ", 2)
      ))
    );
    this.bpe_ranks = new Map(this.merges.map((x, i) => [JSON.stringify(x), i]));
    this.end_of_word_suffix = config.end_of_word_suffix;
    this.continuing_subword_suffix = config.continuing_subword_suffix ?? null;
    this.byte_fallback = this.config.byte_fallback ?? false;
    if (this.byte_fallback) {
      this.text_encoder = new TextEncoder();
    }
    this.ignore_merges = this.config.ignore_merges ?? false;
    this.max_length_to_cache = 256;
    this.cache_capacity = 1e4;
    this.cache = new LRUCache(this.cache_capacity);
  }
  /**
   * Clears the cache.
   */
  clear_cache() {
    this.cache.clear();
  }
  /**
   * Apply Byte-Pair-Encoding (BPE) to a given token. Efficient heap-based priority
   * queue implementation adapted from https://github.com/belladoreai/llama-tokenizer-js.
   * @param {string} token The token to encode.
   * @returns {string[]} The BPE encoded tokens.
   */
  bpe(token) {
    if (token.length === 0) {
      return [];
    }
    const cached = this.cache.get(token);
    if (cached !== void 0) {
      return cached;
    }
    const word = Array.from(token);
    if (this.end_of_word_suffix) {
      word[word.length - 1] += this.end_of_word_suffix;
    }
    let result = [];
    if (word.length > 1) {
      const queue = new PriorityQueue((a, b) => a.score < b.score);
      let startingNode = {
        token: word[0],
        bias: 0,
        prev: null,
        next: null
      };
      let previousNode = startingNode;
      for (let i = 1; i < word.length; ++i) {
        const currentNode = {
          bias: i / word.length,
          // Add fractional component to break ties
          token: word[i],
          prev: previousNode,
          next: null
        };
        previousNode.next = currentNode;
        this._add_node(queue, previousNode);
        previousNode = currentNode;
      }
      while (!queue.isEmpty()) {
        const node = queue.pop();
        if (node.deleted || !node.next || node.next.deleted) continue;
        node.deleted = true;
        node.next.deleted = true;
        if (node.prev) {
          const newPreviousNode = { ...node.prev };
          node.prev.deleted = true;
          node.prev = newPreviousNode;
          if (newPreviousNode.prev) {
            newPreviousNode.prev.next = newPreviousNode;
          } else {
            startingNode = newPreviousNode;
          }
        }
        const merged = {
          token: node.token + node.next.token,
          bias: node.bias,
          prev: node.prev,
          next: node.next.next
        };
        if (merged.prev) {
          merged.prev.next = merged;
          this._add_node(queue, merged.prev);
        } else {
          startingNode = merged;
        }
        if (merged.next) {
          merged.next.prev = merged;
          this._add_node(queue, merged);
        }
      }
      for (let currentNode = startingNode; currentNode !== null; currentNode = currentNode.next) {
        result.push(currentNode.token);
      }
    } else {
      result = word;
    }
    if (this.continuing_subword_suffix) {
      for (let i = 0; i < result.length - 1; ++i) {
        result[i] += this.continuing_subword_suffix;
      }
    }
    if (token.length < this.max_length_to_cache) {
      this.cache.put(token, result);
    }
    return result;
  }
  /**
   * Helper function to add a node to the priority queue.
   * @param {PriorityQueue} queue 
   * @param {BPENode} node
   * @private
   */
  _add_node(queue, node) {
    const rank = this.bpe_ranks.get(JSON.stringify([node.token, node.next.token]));
    if (rank !== void 0) {
      node.score = rank + node.bias;
      queue.push(node);
    }
  }
  /**
   * Encodes the input sequence of tokens using the BPE algorithm and returns the resulting subword tokens.
   * @param {string[]} tokens The input sequence of tokens to encode.
   * @returns {string[]} The resulting subword tokens after applying the BPE algorithm to the input sequence of tokens.
   */
  encode(tokens) {
    const outputTokens = [];
    for (const token of tokens) {
      if (this.ignore_merges && this.tokens_to_ids.has(token)) {
        outputTokens.push(token);
        continue;
      }
      const bpe_token_list = this.bpe(token);
      for (const t of bpe_token_list) {
        if (this.tokens_to_ids.has(t)) {
          outputTokens.push(t);
        } else if (this.byte_fallback) {
          const byteTokens = Array.from(this.text_encoder.encode(t)).map((x) => `<0x${x.toString(16).toUpperCase().padStart(2, "0")}>`);
          if (byteTokens.every((x) => this.tokens_to_ids.has(x))) {
            outputTokens.push(...byteTokens);
          } else {
            outputTokens.push(this.unk_token);
          }
        } else {
          outputTokens.push(this.unk_token);
        }
      }
    }
    return outputTokens;
  }
};
var LegacyTokenizerModel = class extends TokenizerModel {
  /**
   * Create a LegacyTokenizerModel instance.
   * @param {Object} config The configuration object for LegacyTokenizerModel.
   * @param {Object} config.vocab A (possibly nested) mapping of tokens to ids.
   * @param {Object} moreConfig Additional configuration object for the LegacyTokenizerModel model.
   */
  constructor(config, moreConfig) {
    super(config);
    this.tokens_to_ids = objectToMap(
      moreConfig.target_lang ? config.vocab[moreConfig.target_lang] : config.vocab
    );
    this.bos_token = moreConfig.bos_token;
    this.bos_token_id = this.tokens_to_ids.get(this.bos_token);
    this.eos_token = moreConfig.eos_token;
    this.eos_token_id = this.tokens_to_ids.get(this.eos_token);
    this.pad_token = moreConfig.pad_token;
    this.pad_token_id = this.tokens_to_ids.get(this.pad_token);
    this.unk_token = moreConfig.unk_token;
    this.unk_token_id = this.tokens_to_ids.get(this.unk_token);
    this.vocab = new Array(this.tokens_to_ids.size);
    for (const [key, value] of this.tokens_to_ids) {
      this.vocab[value] = key;
    }
  }
  encode(tokens) {
    return tokens;
  }
};
var Normalizer = class extends Callable {
  /**
   * @param {Object} config The configuration object for the normalizer.
   */
  constructor(config) {
    super();
    this.config = config;
  }
  /**
   * Factory method for creating normalizers from config objects.
   * @static
   * @param {Object} config The configuration object for the normalizer.
   * @returns {Normalizer} A Normalizer object.
   * @throws {Error} If an unknown Normalizer type is specified in the config.
   */
  static fromConfig(config) {
    if (config === null) return null;
    switch (config.type) {
      case "BertNormalizer":
        return new BertNormalizer(config);
      case "Precompiled":
        return new Precompiled(config);
      case "Sequence":
        return new NormalizerSequence(config);
      case "Replace":
        return new Replace(config);
      case "NFC":
        return new NFC(config);
      case "NFD":
        return new NFD(config);
      case "NFKC":
        return new NFKC(config);
      case "NFKD":
        return new NFKD(config);
      case "Strip":
        return new StripNormalizer(config);
      case "StripAccents":
        return new StripAccents(config);
      case "Lowercase":
        return new Lowercase(config);
      case "Prepend":
        return new Prepend(config);
      default:
        throw new Error(`Unknown Normalizer type: ${config.type}`);
    }
  }
  /**
   * Normalize the input text.
   * @abstract
   * @param {string} text The text to normalize.
   * @returns {string} The normalized text.
   * @throws {Error} If this method is not implemented in a subclass.
   */
  normalize(text) {
    throw Error("normalize should be implemented in subclass.");
  }
  /**
   * Alias for {@link Normalizer#normalize}.
   * @param {string} text The text to normalize.
   * @returns {string} The normalized text.
   */
  _call(text) {
    return this.normalize(text);
  }
};
var Replace = class extends Normalizer {
  /**
   * Normalize the input text by replacing the pattern with the content.
   * @param {string} text The input text to be normalized.
   * @returns {string} The normalized text after replacing the pattern with the content.
   */
  normalize(text) {
    const pattern = createPattern(this.config.pattern);
    return pattern === null ? text : text.replaceAll(pattern, this.config.content);
  }
};
var UnicodeNormalizer = class extends Normalizer {
  /**
   * @type {string} The Unicode normalization form to apply.
   * Should be one of: 'NFC', 'NFD', 'NFKC', or 'NFKD'.
   */
  form = void 0;
  /**
   * Normalize the input text by applying Unicode normalization.
   * @param {string} text The input text to be normalized.
   * @returns {string} The normalized text.
   */
  normalize(text) {
    text = text.normalize(this.form);
    return text;
  }
};
var NFC = class extends UnicodeNormalizer {
  form = "NFC";
};
var NFD = class extends UnicodeNormalizer {
  form = "NFD";
};
var NFKC = class extends UnicodeNormalizer {
  form = "NFKC";
};
var NFKD = class extends UnicodeNormalizer {
  form = "NFKD";
};
var StripNormalizer = class extends Normalizer {
  /**
   * Strip leading and/or trailing whitespace from the input text.
   * @param {string} text The input text.
   * @returns {string} The normalized text.
   */
  normalize(text) {
    if (this.config.strip_left && this.config.strip_right) {
      text = text.trim();
    } else {
      if (this.config.strip_left) {
        text = text.trimStart();
      }
      if (this.config.strip_right) {
        text = text.trimEnd();
      }
    }
    return text;
  }
};
var StripAccents = class extends Normalizer {
  /**
   * Remove all accents from the text.
   * @param {string} text The input text.
   * @returns {string} The normalized text without accents.
   */
  normalize(text) {
    text = remove_accents(text);
    return text;
  }
};
var Lowercase = class extends Normalizer {
  /**
   * Lowercases the input string.
   * @param {string} text The text to normalize.
   * @returns {string} The normalized text.
   */
  normalize(text) {
    text = text.toLowerCase();
    return text;
  }
};
var Prepend = class extends Normalizer {
  /**
   * Prepends the input string.
   * @param {string} text The text to normalize.
   * @returns {string} The normalized text.
   */
  normalize(text) {
    text = this.config.prepend + text;
    return text;
  }
};
var NormalizerSequence = class extends Normalizer {
  /**
  * Create a new instance of NormalizerSequence.
  * @param {Object} config The configuration object.
  * @param {Object[]} config.normalizers An array of Normalizer configuration objects.
  */
  constructor(config) {
    super(config);
    this.normalizers = config.normalizers.map((x) => Normalizer.fromConfig(x));
  }
  /**
  * Apply a sequence of Normalizers to the input text.
  * @param {string} text The text to normalize.
  * @returns {string} The normalized text.
  */
  normalize(text) {
    return this.normalizers.reduce((t, normalizer) => {
      return normalizer.normalize(t);
    }, text);
  }
};
var BertNormalizer = class extends Normalizer {
  /**
   * Adds whitespace around any CJK (Chinese, Japanese, or Korean) character in the input text.
   *
   * @param {string} text The input text to tokenize.
   * @returns {string} The tokenized text with whitespace added around CJK characters.
   */
  _tokenize_chinese_chars(text) {
    const output = [];
    for (let i = 0; i < text.length; ++i) {
      const char = text[i];
      const cp = char.charCodeAt(0);
      if (is_chinese_char(cp)) {
        output.push(" ");
        output.push(char);
        output.push(" ");
      } else {
        output.push(char);
      }
    }
    return output.join("");
  }
  /**
   * Strips accents from the given text.
   * @param {string} text The text to strip accents from.
   * @returns {string} The text with accents removed.
   */
  stripAccents(text) {
    return text.normalize("NFD").replace(/\p{Mn}/gu, "");
  }
  /**
   * Checks whether `char` is a control character.
   * @param {string} char The character to check.
   * @returns {boolean} Whether `char` is a control character.
   * @private
   */
  _is_control(char) {
    switch (char) {
      case "	":
      case "\n":
      case "\r":
        return false;
      default:
        return /^\p{Cc}|\p{Cf}|\p{Co}|\p{Cs}$/u.test(char);
    }
  }
  /**
   * Performs invalid character removal and whitespace cleanup on text.
   * @param {string} text The text to clean.
   * @returns {string} The cleaned text.
   * @private
   */
  _clean_text(text) {
    const output = [];
    for (const char of text) {
      const cp = char.charCodeAt(0);
      if (cp === 0 || cp === 65533 || this._is_control(char)) {
        continue;
      }
      if (/^\s$/.test(char)) {
        output.push(" ");
      } else {
        output.push(char);
      }
    }
    return output.join("");
  }
  /**
   * Normalizes the given text based on the configuration.
   * @param {string} text The text to normalize.
   * @returns {string} The normalized text.
   */
  normalize(text) {
    if (this.config.clean_text) {
      text = this._clean_text(text);
    }
    if (this.config.handle_chinese_chars) {
      text = this._tokenize_chinese_chars(text);
    }
    if (this.config.lowercase) {
      text = text.toLowerCase();
      if (this.config.strip_accents !== false) {
        text = this.stripAccents(text);
      }
    } else if (this.config.strip_accents) {
      text = this.stripAccents(text);
    }
    return text;
  }
};
var PreTokenizer = class extends Callable {
  /**
  * Factory method that returns an instance of a subclass of `PreTokenizer` based on the provided configuration.
  *
  * @static
  * @param {Object} config A configuration object for the pre-tokenizer.
  * @returns {PreTokenizer} An instance of a subclass of `PreTokenizer`.
  * @throws {Error} If the provided configuration object does not correspond to any known pre-tokenizer.
  */
  static fromConfig(config) {
    if (config === null) return null;
    switch (config.type) {
      case "BertPreTokenizer":
        return new BertPreTokenizer(config);
      case "Sequence":
        return new PreTokenizerSequence(config);
      case "Whitespace":
        return new WhitespacePreTokenizer(config);
      case "WhitespaceSplit":
        return new WhitespaceSplit(config);
      case "Metaspace":
        return new MetaspacePreTokenizer(config);
      case "ByteLevel":
        return new ByteLevelPreTokenizer(config);
      case "Split":
        return new SplitPreTokenizer(config);
      case "Punctuation":
        return new PunctuationPreTokenizer(config);
      case "Digits":
        return new DigitsPreTokenizer(config);
      case "Replace":
        return new ReplacePreTokenizer(config);
      default:
        throw new Error(`Unknown PreTokenizer type: ${config.type}`);
    }
  }
  /**
   * Method that should be implemented by subclasses to define the specific pre-tokenization logic.
   *
   * @abstract
   * @param {string} text The text to pre-tokenize.
   * @param {Object} [options] Additional options for the pre-tokenization logic.
   * @returns {string[]} The pre-tokenized text.
   * @throws {Error} If the method is not implemented in the subclass.
   */
  pre_tokenize_text(text, options) {
    throw Error("pre_tokenize_text should be implemented in subclass.");
  }
  /**
   * Tokenizes the given text into pre-tokens.
   * @param {string|string[]} text The text or array of texts to pre-tokenize.
   * @param {Object} [options] Additional options for the pre-tokenization logic.
   * @returns {string[]} An array of pre-tokens.
   */
  pre_tokenize(text, options) {
    return (Array.isArray(text) ? text.map((x) => this.pre_tokenize_text(x, options)) : this.pre_tokenize_text(text, options)).flat();
  }
  /**
   * Alias for {@link PreTokenizer#pre_tokenize}.
   * @param {string|string[]} text The text or array of texts to pre-tokenize.
   * @param {Object} [options] Additional options for the pre-tokenization logic.
   * @returns {string[]} An array of pre-tokens.
   */
  _call(text, options) {
    return this.pre_tokenize(text, options);
  }
};
var BertPreTokenizer = class extends PreTokenizer {
  /**
   * A PreTokenizer that splits text into wordpieces using a basic tokenization scheme
   * similar to that used in the original implementation of BERT.
   * 
   * @param {Object} config The configuration object.
   */
  constructor(config) {
    super();
    this.pattern = new RegExp(`[^\\s${PUNCTUATION_REGEX}]+|[${PUNCTUATION_REGEX}]`, "gu");
  }
  /**
   * Tokenizes a single text using the BERT pre-tokenization scheme.
   * 
   * @param {string} text The text to tokenize.
   * @param {Object} [options] Additional options for the pre-tokenization logic.
   * @returns {string[]} An array of tokens.
   */
  pre_tokenize_text(text, options) {
    return text.trim().match(this.pattern) || [];
  }
};
var ByteLevelPreTokenizer = class extends PreTokenizer {
  /**
   * Creates a new instance of the `ByteLevelPreTokenizer` class.
   * @param {Object} config The configuration object.
   */
  constructor(config) {
    super();
    this.config = config;
    this.add_prefix_space = this.config.add_prefix_space;
    this.trim_offsets = this.config.trim_offsets;
    this.use_regex = this.config.use_regex ?? true;
    this.pattern = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
    this.byte_encoder = BYTES_TO_UNICODE;
    this.text_encoder = new TextEncoder();
  }
  /**
   * Tokenizes a single piece of text using byte-level tokenization.
   * @param {string} text The text to tokenize.
   * @param {Object} [options] Additional options for the pre-tokenization logic.
   * @returns {string[]} An array of tokens.
   */
  pre_tokenize_text(text, options) {
    if (this.add_prefix_space && !text.startsWith(" ")) {
      text = " " + text;
    }
    const tokens = this.use_regex ? text.match(this.pattern) || [] : [text];
    return tokens.map(
      (token) => Array.from(this.text_encoder.encode(token), (byte) => this.byte_encoder[byte]).join("")
    );
  }
};
var SplitPreTokenizer = class extends PreTokenizer {
  /**
   * @param {Object} config The configuration options for the pre-tokenizer.
   * @param {Object} config.pattern The pattern used to split the text. Can be a string or a regex object.
   * @param {string|undefined} config.pattern.String The string to use for splitting. Only defined if the pattern is a string.
   * @param {string|undefined} config.pattern.Regex The regex to use for splitting. Only defined if the pattern is a regex.
   * @param {SplitDelimiterBehavior} config.behavior The behavior to use when splitting.
   * @param {boolean} config.invert Whether to split (invert=false) or match (invert=true) the pattern.
   */
  constructor(config) {
    super();
    this.config = config;
    this.pattern = createPattern(this.config.pattern, this.config.invert);
  }
  /**
   * Tokenizes text by splitting it using the given pattern.
   * @param {string} text The text to tokenize.
   * @param {Object} [options] Additional options for the pre-tokenization logic.
   * @returns {string[]} An array of tokens.
   */
  pre_tokenize_text(text, options) {
    if (this.pattern === null) {
      return [];
    }
    if (this.config.invert) {
      return text.match(this.pattern) || [];
    } else if (this.config.behavior?.toLowerCase() === "removed") {
      return text.split(this.pattern).filter((x) => x);
    } else {
      return regexSplit(text, this.pattern);
    }
  }
};
var PunctuationPreTokenizer = class extends PreTokenizer {
  /**
   * @param {Object} config The configuration options for the pre-tokenizer.
   * @param {SplitDelimiterBehavior} config.behavior The behavior to use when splitting.
   */
  constructor(config) {
    super();
    this.config = config;
    this.pattern = new RegExp(`[^${PUNCTUATION_REGEX}]+|[${PUNCTUATION_REGEX}]+`, "gu");
  }
  /**
   * Tokenizes text by splitting it using the given pattern.
   * @param {string} text The text to tokenize.
   * @param {Object} [options] Additional options for the pre-tokenization logic.
   * @returns {string[]} An array of tokens.
   */
  pre_tokenize_text(text, options) {
    return text.match(this.pattern) || [];
  }
};
var DigitsPreTokenizer = class extends PreTokenizer {
  /**
   * @param {Object} config The configuration options for the pre-tokenizer.
   * @param {boolean} config.individual_digits Whether to split on individual digits.
   */
  constructor(config) {
    super();
    this.config = config;
    const digit_pattern = `[^\\d]+|\\d${this.config.individual_digits ? "" : "+"}`;
    this.pattern = new RegExp(digit_pattern, "gu");
  }
  /**
   * Tokenizes text by splitting it using the given pattern.
   * @param {string} text The text to tokenize.
   * @param {Object} [options] Additional options for the pre-tokenization logic.
   * @returns {string[]} An array of tokens.
   */
  pre_tokenize_text(text, options) {
    return text.match(this.pattern) || [];
  }
};
var PostProcessor = class extends Callable {
  /**
   * @param {Object} config The configuration for the post-processor.
   */
  constructor(config) {
    super();
    this.config = config;
  }
  /**
   * Factory method to create a PostProcessor object from a configuration object.
   *
   * @param {Object} config Configuration object representing a PostProcessor.
   * @returns {PostProcessor} A PostProcessor object created from the given configuration.
   * @throws {Error} If an unknown PostProcessor type is encountered.
   */
  static fromConfig(config) {
    if (config === null) return null;
    switch (config.type) {
      case "TemplateProcessing":
        return new TemplateProcessing(config);
      case "ByteLevel":
        return new ByteLevelPostProcessor(config);
      case "RobertaProcessing":
        return new RobertaProcessing(config);
      case "BertProcessing":
        return new BertProcessing(config);
      case "Sequence":
        return new PostProcessorSequence(config);
      default:
        throw new Error(`Unknown PostProcessor type: ${config.type}`);
    }
  }
  /**
   * Method to be implemented in subclass to apply post-processing on the given tokens.
   *
   * @param {Array} tokens The input tokens to be post-processed.
   * @param {...*} args Additional arguments required by the post-processing logic.
   * @returns {PostProcessedOutput} The post-processed tokens.
   * @throws {Error} If the method is not implemented in subclass.
   */
  post_process(tokens, ...args) {
    throw Error("post_process should be implemented in subclass.");
  }
  /**
   * Alias for {@link PostProcessor#post_process}.
   * @param {Array} tokens The text or array of texts to post-process.
   * @param {...*} args Additional arguments required by the post-processing logic.
   * @returns {PostProcessedOutput} The post-processed tokens.
   */
  _call(tokens, ...args) {
    return this.post_process(tokens, ...args);
  }
};
var BertProcessing = class extends PostProcessor {
  /**
   * @param {Object} config The configuration for the post-processor.
   * @param {string[]} config.cls The special tokens to add to the beginning of the input.
   * @param {string[]} config.sep The special tokens to add to the end of the input.
   */
  constructor(config) {
    super(config);
    this.cls = config.cls[0];
    this.sep = config.sep[0];
  }
  /**
   * Adds the special tokens to the beginning and end of the input.
   * @param {string[]} tokens The input tokens.
   * @param {string[]} [tokens_pair=null] An optional second set of input tokens.
   * @returns {PostProcessedOutput} The post-processed tokens with the special tokens added to the beginning and end.
   */
  post_process(tokens, tokens_pair = null, {
    add_special_tokens = true
  } = {}) {
    if (add_special_tokens) {
      tokens = mergeArrays([this.cls], tokens, [this.sep]);
    }
    let token_type_ids = new Array(tokens.length).fill(0);
    if (tokens_pair !== null) {
      const middle = add_special_tokens && this instanceof RobertaProcessing ? [this.sep] : [];
      const after = add_special_tokens ? [this.sep] : [];
      tokens = mergeArrays(tokens, middle, tokens_pair, after);
      token_type_ids = mergeArrays(token_type_ids, new Array(tokens_pair.length + middle.length + after.length).fill(1));
    }
    return { tokens, token_type_ids };
  }
};
var RobertaProcessing = class extends BertProcessing {
};
var TemplateProcessing = class extends PostProcessor {
  /**
   * Creates a new instance of `TemplateProcessing`.
   * @param {Object} config The configuration options for the post processor.
   * @param {Array} config.single The template for a single sequence of tokens.
   * @param {Array} config.pair The template for a pair of sequences of tokens.
   */
  constructor(config) {
    super(config);
    this.single = config.single;
    this.pair = config.pair;
  }
  /**
   * Replaces special tokens in the template with actual tokens.
   * @param {string[]} tokens The list of tokens for the first sequence.
   * @param {string[]} [tokens_pair=null] The list of tokens for the second sequence (optional).
   * @returns {PostProcessedOutput} An object containing the list of tokens with the special tokens replaced with actual tokens.
   */
  post_process(tokens, tokens_pair = null, {
    add_special_tokens = true
  } = {}) {
    const type = tokens_pair === null ? this.single : this.pair;
    let processedTokens = [];
    let types = [];
    for (const item of type) {
      if ("SpecialToken" in item) {
        if (add_special_tokens) {
          processedTokens.push(item.SpecialToken.id);
          types.push(item.SpecialToken.type_id);
        }
      } else if ("Sequence" in item) {
        if (item.Sequence.id === "A") {
          processedTokens = mergeArrays(processedTokens, tokens);
          types = mergeArrays(types, new Array(tokens.length).fill(item.Sequence.type_id));
        } else if (item.Sequence.id === "B") {
          processedTokens = mergeArrays(processedTokens, tokens_pair);
          types = mergeArrays(types, new Array(tokens_pair.length).fill(item.Sequence.type_id));
        }
      }
    }
    return { tokens: processedTokens, token_type_ids: types };
  }
};
var ByteLevelPostProcessor = class extends PostProcessor {
  /**
   * Post process the given tokens.
   * @param {string[]} tokens The list of tokens for the first sequence.
   * @param {string[]} [tokens_pair=null] The list of tokens for the second sequence (optional).
   * @returns {PostProcessedOutput} An object containing the post-processed tokens.
   */
  post_process(tokens, tokens_pair = null) {
    if (tokens_pair) {
      tokens = mergeArrays(tokens, tokens_pair);
    }
    return { tokens };
  }
};
var PostProcessorSequence = class extends PostProcessor {
  /**
   * Creates a new instance of PostProcessorSequence.
   * @param {Object} config The configuration object.
   * @param {Object[]} config.processors The list of post-processors to apply.
   */
  constructor(config) {
    super(config);
    this.processors = config.processors.map((x) => PostProcessor.fromConfig(x));
  }
  /**
   * Post process the given tokens.
   * @param {string[]} tokens The list of tokens for the first sequence.
   * @param {string[]} [tokens_pair=null] The list of tokens for the second sequence (optional).
   * @returns {PostProcessedOutput} An object containing the post-processed tokens.
   */
  post_process(tokens, tokens_pair = null, options = {}) {
    let token_type_ids;
    for (const processor of this.processors) {
      if (processor instanceof ByteLevelPostProcessor) {
        const output = processor.post_process(tokens);
        tokens = output.tokens;
        if (tokens_pair) {
          const pair_output = processor.post_process(tokens_pair);
          tokens_pair = pair_output.tokens;
        }
      } else {
        const output = processor.post_process(tokens, tokens_pair, options);
        tokens = output.tokens;
        token_type_ids = output.token_type_ids;
      }
    }
    return { tokens, token_type_ids };
  }
};
var Decoder = class extends Callable {
  /**
  * Creates an instance of `Decoder`.
  *
  * @param {Object} config The configuration object.
  */
  constructor(config) {
    super();
    this.config = config;
    this.added_tokens = [];
    this.end_of_word_suffix = null;
    this.trim_offsets = config.trim_offsets;
  }
  /**
  * Creates a decoder instance based on the provided configuration.
  *
  * @param {Object} config The configuration object.
  * @returns {Decoder} A decoder instance.
  * @throws {Error} If an unknown decoder type is provided.
  */
  static fromConfig(config) {
    if (config === null) return null;
    switch (config.type) {
      case "WordPiece":
        return new WordPieceDecoder(config);
      case "Metaspace":
        return new MetaspaceDecoder(config);
      case "ByteLevel":
        return new ByteLevelDecoder(config);
      case "Replace":
        return new ReplaceDecoder(config);
      case "ByteFallback":
        return new ByteFallback(config);
      case "Fuse":
        return new FuseDecoder(config);
      case "Strip":
        return new StripDecoder(config);
      case "Sequence":
        return new DecoderSequence(config);
      case "CTC":
        return new CTCDecoder(config);
      case "BPEDecoder":
        return new BPEDecoder(config);
      default:
        throw new Error(`Unknown Decoder type: ${config.type}`);
    }
  }
  /**
  * Calls the `decode` method.
  *
  * @param {string[]} tokens The list of tokens.
  * @returns {string} The decoded string.
  */
  _call(tokens) {
    return this.decode(tokens);
  }
  /**
  * Decodes a list of tokens.
  * @param {string[]} tokens The list of tokens.
  * @returns {string} The decoded string.
  */
  decode(tokens) {
    return this.decode_chain(tokens).join("");
  }
  /**
   * Apply the decoder to a list of tokens.
   * 
   * @param {string[]} tokens The list of tokens.
   * @returns {string[]} The decoded list of tokens.
   * @throws {Error} If the `decode_chain` method is not implemented in the subclass.
   */
  decode_chain(tokens) {
    throw Error("`decode_chain` should be implemented in subclass.");
  }
};
var ReplaceDecoder = class extends Decoder {
  /** @type {Decoder['decode_chain']} */
  decode_chain(tokens) {
    const pattern = createPattern(this.config.pattern);
    return pattern === null ? tokens : tokens.map((token) => token.replaceAll(pattern, this.config.content));
  }
};
var ByteFallback = class extends Decoder {
  constructor(config) {
    super(config);
    this.text_decoder = new TextDecoder();
  }
  /** @type {Decoder['decode_chain']} */
  decode_chain(tokens) {
    const new_tokens = [];
    let previous_byte_tokens = [];
    for (const token of tokens) {
      let bytes = null;
      if (token.length === 6 && token.startsWith("<0x") && token.endsWith(">")) {
        const byte = parseInt(token.slice(3, 5), 16);
        if (!isNaN(byte)) {
          bytes = byte;
        }
      }
      if (bytes !== null) {
        previous_byte_tokens.push(bytes);
      } else {
        if (previous_byte_tokens.length > 0) {
          const string = this.text_decoder.decode(Uint8Array.from(previous_byte_tokens));
          new_tokens.push(string);
          previous_byte_tokens = [];
        }
        new_tokens.push(token);
      }
    }
    if (previous_byte_tokens.length > 0) {
      const string = this.text_decoder.decode(Uint8Array.from(previous_byte_tokens));
      new_tokens.push(string);
      previous_byte_tokens = [];
    }
    return new_tokens;
  }
};
var FuseDecoder = class extends Decoder {
  /** @type {Decoder['decode_chain']} */
  decode_chain(tokens) {
    return [tokens.join("")];
  }
};
var StripDecoder = class extends Decoder {
  constructor(config) {
    super(config);
    this.content = this.config.content;
    this.start = this.config.start;
    this.stop = this.config.stop;
  }
  /** @type {Decoder['decode_chain']} */
  decode_chain(tokens) {
    return tokens.map((token) => {
      let start_cut = 0;
      for (let i = 0; i < this.start; ++i) {
        if (token[i] === this.content) {
          start_cut = i + 1;
          continue;
        } else {
          break;
        }
      }
      let stop_cut = token.length;
      for (let i = 0; i < this.stop; ++i) {
        const index = token.length - i - 1;
        if (token[index] === this.content) {
          stop_cut = index;
          continue;
        } else {
          break;
        }
      }
      return token.slice(start_cut, stop_cut);
    });
  }
};
var WordPieceDecoder = class extends Decoder {
  /**
   * Creates a new instance of WordPieceDecoder.
   * @param {Object} config The configuration object.
   * @param {string} config.prefix The prefix used for WordPiece encoding.
   * @param {boolean} config.cleanup Whether to cleanup the decoded string.
   */
  constructor(config) {
    super(config);
    this.cleanup = config.cleanup;
  }
  /** @type {Decoder['decode_chain']} */
  decode_chain(tokens) {
    return tokens.map((token, i) => {
      if (i !== 0) {
        if (token.startsWith(this.config.prefix)) {
          token = token.replace(this.config.prefix, "");
        } else {
          token = " " + token;
        }
      }
      if (this.cleanup) {
        token = clean_up_tokenization(token);
      }
      return token;
    });
  }
};
var ByteLevelDecoder = class extends Decoder {
  /**
   * Create a `ByteLevelDecoder` object.
   * @param {Object} config Configuration object.
   */
  constructor(config) {
    super(config);
    this.byte_decoder = UNICODE_TO_BYTES;
    this.text_decoder = new TextDecoder("utf-8", {
      fatal: false,
      ignoreBOM: true
    });
    this.end_of_word_suffix = null;
  }
  /**
   * Convert an array of tokens to string by decoding each byte.
   * @param {string[]} tokens Array of tokens to be decoded.
   * @returns {string} The decoded string.
   */
  convert_tokens_to_string(tokens) {
    const text = tokens.join("");
    const byteArray = new Uint8Array([...text].map((c) => this.byte_decoder[c]));
    const decoded_text = this.text_decoder.decode(byteArray);
    return decoded_text;
  }
  /** @type {Decoder['decode_chain']} */
  decode_chain(tokens) {
    const sub_texts = [];
    let current_sub_text = [];
    for (const token of tokens) {
      if (this.added_tokens.find((x) => x.content === token) !== void 0) {
        if (current_sub_text.length > 0) {
          sub_texts.push(this.convert_tokens_to_string(current_sub_text));
          current_sub_text = [];
        }
        sub_texts.push(token);
      } else {
        current_sub_text.push(token);
      }
    }
    if (current_sub_text.length > 0) {
      sub_texts.push(this.convert_tokens_to_string(current_sub_text));
    }
    return sub_texts;
  }
};
var CTCDecoder = class extends Decoder {
  constructor(config) {
    super(config);
    this.pad_token = this.config.pad_token;
    this.word_delimiter_token = this.config.word_delimiter_token;
    this.cleanup = this.config.cleanup;
  }
  /**
   * Converts a connectionist-temporal-classification (CTC) output tokens into a single string.
   * @param {string[]} tokens Array of tokens to be decoded.
   * @returns {string} The decoded string.
   */
  convert_tokens_to_string(tokens) {
    if (tokens.length === 0) return "";
    const grouped_tokens = [tokens[0]];
    for (let i = 1; i < tokens.length; ++i) {
      if (tokens[i] !== grouped_tokens.at(-1)) {
        grouped_tokens.push(tokens[i]);
      }
    }
    const filtered_tokens = grouped_tokens.filter((token) => token !== this.pad_token);
    let text = filtered_tokens.join("");
    if (this.cleanup) {
      text = clean_up_tokenization(text).replaceAll(this.word_delimiter_token, " ").trim();
    }
    return text;
  }
  /** @type {Decoder['decode_chain']} */
  decode_chain(tokens) {
    return [this.convert_tokens_to_string(tokens)];
  }
};
var DecoderSequence = class extends Decoder {
  /**
   * Creates a new instance of DecoderSequence.
   * @param {Object} config The configuration object.
   * @param {Object[]} config.decoders The list of decoders to apply.
   */
  constructor(config) {
    super(config);
    this.decoders = config.decoders.map((x) => Decoder.fromConfig(x));
  }
  /** @type {Decoder['decode_chain']} */
  decode_chain(tokens) {
    return this.decoders.reduce((toks, decoder) => {
      return decoder.decode_chain(toks);
    }, tokens);
  }
};
var BPEDecoder = class extends Decoder {
  constructor(config) {
    super(config);
    this.suffix = this.config.suffix;
  }
  /** @type {Decoder['decode_chain']} */
  decode_chain(tokens) {
    return tokens.map((token, i) => {
      return token.replaceAll(this.suffix, i === tokens.length - 1 ? "" : " ");
    });
  }
};
var VitsDecoder = class extends Decoder {
  /** @type {Decoder['decode_chain']} */
  decode_chain(tokens) {
    let decoded = "";
    for (let i = 1; i < tokens.length; i += 2) {
      decoded += tokens[i];
    }
    return [decoded];
  }
};
var MetaspacePreTokenizer = class extends PreTokenizer {
  /**
   * @param {Object} config The configuration object for the MetaspacePreTokenizer.
   * @param {string} config.replacement The character to replace spaces with.
   * @param {string} [config.str_rep=config.replacement] An optional string representation of the replacement character.
   * @param {'first'|'never'|'always'} [config.prepend_scheme='always'] The metaspace prepending scheme.
   */
  constructor(config) {
    super();
    this.replacement = config.replacement;
    this.strRep = config.str_rep || this.replacement;
    this.prepend_scheme = config.prepend_scheme ?? "always";
  }
  /**
   * This method takes a string, replaces spaces with the replacement character,
   * adds a prefix space if requested, and returns a new list of tokens.
   * @param {string} text The text to pre-tokenize.
   * @param {Object} [options] The options for the pre-tokenization.
   * @param {number} [options.section_index] The index of the section to pre-tokenize.
   * @returns {string[]} A new list of pre-tokenized tokens.
   */
  pre_tokenize_text(text, {
    section_index = void 0
  } = {}) {
    let normalized = text.replaceAll(" ", this.strRep);
    if (
      // We add a prefix space if:
      //  (1) The normalized token does not already start with the replacement character.
      !normalized.startsWith(this.replacement) && (this.prepend_scheme === "always" || this.prepend_scheme === "first" && section_index === 0)
    ) {
      normalized = this.strRep + normalized;
    }
    return [normalized];
  }
};
var MetaspaceDecoder = class extends Decoder {
  /**
   * Constructs a new MetaspaceDecoder object.
   * @param {Object} config The configuration object for the MetaspaceDecoder.
   * @param {string} config.replacement The string to replace spaces with.
   */
  constructor(config) {
    super(config);
    this.replacement = config.replacement;
  }
  /** @type {Decoder['decode_chain']} */
  decode_chain(tokens) {
    const result = [];
    for (let i = 0; i < tokens.length; ++i) {
      let normalized = tokens[i].replaceAll(this.replacement, " ");
      if (i == 0 && normalized.startsWith(" ")) {
        normalized = normalized.substring(1);
      }
      result.push(normalized);
    }
    return result;
  }
};
var Precompiled = class extends Normalizer {
  /**
   * Create a new instance of Precompiled normalizer.
   * @param {Object} config The configuration object.
   * @param {any} config.precompiled_charsmap Precompiled chars mapping.
   */
  constructor(config) {
    super(config);
    this.charsmap = config.precompiled_charsmap;
  }
  /**
   * Normalizes the given text by applying the precompiled charsmap.
   * @param {string} text The text to normalize.
   * @returns {string} The normalized text.
   */
  normalize(text) {
    text = text.replace(/[\u0001-\u0008\u000B\u000E-\u001F\u007F\u008F\u009F]/gm, "");
    text = text.replace(/[\u0009\u000A\u000C\u000D\u00A0\u1680\u2000-\u200F\u2028\u2029\u202F\u205F\u2581\u3000\uFEFF\uFFFD]/gm, " ");
    if (text.includes("\uFF5E")) {
      const parts = text.split("\uFF5E");
      text = parts.map((part) => part.normalize("NFKC")).join("\uFF5E");
    } else {
      text = text.normalize("NFKC");
    }
    return text;
  }
};
var PreTokenizerSequence = class extends PreTokenizer {
  /**
   * Creates an instance of PreTokenizerSequence.
   * @param {Object} config The configuration object for the pre-tokenizer sequence.
   * @param {Object[]} config.pretokenizers An array of pre-tokenizer configurations.
   */
  constructor(config) {
    super();
    this.tokenizers = config.pretokenizers.map((x) => PreTokenizer.fromConfig(x));
  }
  /**
   * Applies each pre-tokenizer in the sequence to the input text in turn.
   * @param {string} text The text to pre-tokenize.
   * @param {Object} [options] Additional options for the pre-tokenization logic.
   * @returns {string[]} The pre-tokenized text.
   */
  pre_tokenize_text(text, options) {
    return this.tokenizers.reduce((preTokenizedText, tokenizer) => {
      return tokenizer.pre_tokenize(preTokenizedText, options);
    }, [text]);
  }
};
var WhitespacePreTokenizer = class extends PreTokenizer {
  /**
   * Creates an instance of WhitespacePreTokenizer.
   * @param {Object} config The configuration object for the pre-tokenizer.
   */
  constructor(config) {
    super();
  }
  /**
   * Pre-tokenizes the input text by splitting it on word boundaries.
   * @param {string} text The text to be pre-tokenized.
   * @param {Object} [options] Additional options for the pre-tokenization logic.
   * @returns {string[]} An array of tokens produced by splitting the input text on whitespace.
   */
  pre_tokenize_text(text, options) {
    return text.match(/\w+|[^\w\s]+/g) || [];
  }
};
var WhitespaceSplit = class extends PreTokenizer {
  /**
   * Creates an instance of WhitespaceSplit.
   * @param {Object} config The configuration object for the pre-tokenizer.
   */
  constructor(config) {
    super();
  }
  /**
   * Pre-tokenizes the input text by splitting it on whitespace characters.
   * @param {string} text The text to be pre-tokenized.
   * @param {Object} [options] Additional options for the pre-tokenization logic.
   * @returns {string[]} An array of tokens produced by splitting the input text on whitespace.
   */
  pre_tokenize_text(text, options) {
    return whitespace_split(text);
  }
};
var ReplacePreTokenizer = class extends PreTokenizer {
  /**
   * @param {Object} config The configuration options for the pre-tokenizer.
   * @param {Object} config.pattern The pattern used to split the text. Can be a string or a regex object.
   * @param {string} config.content What to replace the pattern with.
   */
  constructor(config) {
    super();
    this.config = config;
    this.pattern = createPattern(this.config.pattern);
    this.content = this.config.content;
  }
  /**
   * Pre-tokenizes the input text by replacing certain characters.
   * @param {string} text The text to be pre-tokenized.
   * @param {Object} [options] Additional options for the pre-tokenization logic.
   * @returns {string[]} An array of tokens produced by replacing certain characters.
   */
  pre_tokenize_text(text, options) {
    if (this.pattern === null) {
      return [text];
    }
    return [text.replaceAll(this.pattern, this.config.content)];
  }
};
var SPECIAL_TOKEN_ATTRIBUTES = [
  "bos_token",
  "eos_token",
  "unk_token",
  "sep_token",
  "pad_token",
  "cls_token",
  "mask_token"
  // additional_special_tokens (TODO)
];
function padHelper(item, length, value_fn, side) {
  for (const key of Object.keys(item)) {
    const diff = length - item[key].length;
    const value = value_fn(key);
    const padData = new Array(diff).fill(value);
    item[key] = side === "right" ? mergeArrays(item[key], padData) : mergeArrays(padData, item[key]);
  }
}
function truncateHelper(item, length) {
  for (const key of Object.keys(item)) {
    item[key].length = length;
  }
}
var PreTrainedTokenizer = class extends Callable {
  return_token_type_ids = false;
  padding_side = "right";
  /**
   * Create a new PreTrainedTokenizer instance.
   * @param {Object} tokenizerJSON The JSON of the tokenizer.
   * @param {Object} tokenizerConfig The config of the tokenizer.
   */
  constructor(tokenizerJSON, tokenizerConfig) {
    super();
    this.config = tokenizerConfig;
    this.normalizer = Normalizer.fromConfig(tokenizerJSON.normalizer);
    this.pre_tokenizer = PreTokenizer.fromConfig(tokenizerJSON.pre_tokenizer);
    this.model = TokenizerModel.fromConfig(tokenizerJSON.model, tokenizerConfig);
    this.post_processor = PostProcessor.fromConfig(tokenizerJSON.post_processor);
    this.decoder = Decoder.fromConfig(tokenizerJSON.decoder);
    this.special_tokens = [];
    this.all_special_ids = [];
    this.added_tokens = [];
    for (const addedToken of tokenizerJSON.added_tokens) {
      const token = new AddedToken(addedToken);
      this.added_tokens.push(token);
      this.model.tokens_to_ids.set(token.content, token.id);
      this.model.vocab[token.id] = token.content;
      if (token.special) {
        this.special_tokens.push(token.content);
        this.all_special_ids.push(token.id);
      }
    }
    this.additional_special_tokens = tokenizerConfig.additional_special_tokens ?? [];
    this.special_tokens.push(...this.additional_special_tokens);
    this.special_tokens = [...new Set(this.special_tokens)];
    if (this.decoder) {
      this.decoder.added_tokens = this.added_tokens;
      this.decoder.end_of_word_suffix = this.model.end_of_word_suffix;
    }
    this.added_tokens_splitter = new DictionarySplitter(
      this.added_tokens.map((x) => x.content)
    );
    this.added_tokens_map = new Map(this.added_tokens.map((x) => [x.content, x]));
    this.mask_token = this.getToken("mask_token");
    this.mask_token_id = this.model.tokens_to_ids.get(this.mask_token);
    this.pad_token = this.getToken("pad_token", "eos_token");
    this.pad_token_id = this.model.tokens_to_ids.get(this.pad_token);
    this.sep_token = this.getToken("sep_token");
    this.sep_token_id = this.model.tokens_to_ids.get(this.sep_token);
    this.unk_token = this.getToken("unk_token");
    this.unk_token_id = this.model.tokens_to_ids.get(this.unk_token);
    this.bos_token = this.getToken("bos_token");
    this.bos_token_id = this.model.tokens_to_ids.get(this.bos_token);
    this.eos_token = this.getToken("eos_token");
    this.eos_token_id = this.model.tokens_to_ids.get(this.eos_token);
    this.model_max_length = tokenizerConfig.model_max_length;
    this.remove_space = tokenizerConfig.remove_space;
    this.clean_up_tokenization_spaces = tokenizerConfig.clean_up_tokenization_spaces ?? true;
    this.do_lowercase_and_remove_accent = tokenizerConfig.do_lowercase_and_remove_accent ?? false;
    if (tokenizerConfig.padding_side) {
      this.padding_side = tokenizerConfig.padding_side;
    }
    this.add_bos_token = tokenizerConfig.add_bos_token;
    this.add_eos_token = tokenizerConfig.add_eos_token;
    this.legacy = false;
    this.chat_template = tokenizerConfig.chat_template ?? null;
    if (Array.isArray(this.chat_template)) {
      const chat_template = /* @__PURE__ */ Object.create(null);
      for (const { name, template } of this.chat_template) {
        if (typeof name !== "string" || typeof template !== "string") {
          throw new Error('Chat template must be a list of objects with "name" and "template" properties');
        }
        chat_template[name] = template;
      }
      this.chat_template = chat_template;
    }
    this._compiled_template_cache = /* @__PURE__ */ new Map();
  }
  /**
   * Returns the value of the first matching key in the tokenizer config object.
   * @param {...string} keys One or more keys to search for in the tokenizer config object.
   * @returns {string|null} The value associated with the first matching key, or null if no match is found.
   * @throws {Error} If an object is found for a matching key and its __type property is not "AddedToken".
   * @private
   */
  getToken(...keys) {
    for (const key of keys) {
      const item = this.config[key];
      if (!item) continue;
      if (typeof item === "object") {
        if (item.__type === "AddedToken") {
          return item.content;
        } else {
          throw Error(`Unknown token: ${item}`);
        }
      } else {
        return item;
      }
    }
    return null;
  }
  /**
   * Loads a pre-trained tokenizer from the given `pretrained_model_name_or_path`. 
   * 
   * @param {string} pretrained_model_name_or_path The path to the pre-trained tokenizer.
   * @param {PretrainedTokenizerOptions} options Additional options for loading the tokenizer.
   * 
   * @throws {Error} Throws an error if the tokenizer.json or tokenizer_config.json files are not found in the `pretrained_model_name_or_path`.
   * @returns {Promise<PreTrainedTokenizer>} A new instance of the `PreTrainedTokenizer` class.
   */
  static async from_pretrained(pretrained_model_name_or_path, {
    progress_callback = null,
    config = null,
    cache_dir = null,
    local_files_only = false,
    revision = "main",
    legacy = null
  } = {}) {
    const info = await loadTokenizer(pretrained_model_name_or_path, {
      progress_callback,
      config,
      cache_dir,
      local_files_only,
      revision,
      legacy
    });
    return new this(...info);
  }
  /**
   * @typedef {number[]|number[][]|Tensor} BatchEncodingItem
   * 
   * @typedef {Object} BatchEncoding Holds the output of the tokenizer's call function.
   * @property {BatchEncodingItem} input_ids List of token ids to be fed to a model.
   * @property {BatchEncodingItem} attention_mask List of indices specifying which tokens should be attended to by the model.
   * @property {BatchEncodingItem} [token_type_ids] List of token type ids to be fed to a model.
   */
  /**
   * Encode/tokenize the given text(s).
   * @param {string|string[]} text The text to tokenize.
   * @param {Object} options An optional object containing the following properties:
   * @param {string|string[]} [options.text_pair=null] Optional second sequence to be encoded. If set, must be the same type as text.
   * @param {boolean|'max_length'} [options.padding=false] Whether to pad the input sequences.
   * @param {boolean} [options.add_special_tokens=true] Whether or not to add the special tokens associated with the corresponding model.
   * @param {boolean} [options.truncation=null] Whether to truncate the input sequences.
   * @param {number} [options.max_length=null] Maximum length of the returned list and optionally padding length.
   * @param {boolean} [options.return_tensor=true] Whether to return the results as Tensors or arrays.
   * @param {boolean} [options.return_token_type_ids=null] Whether to return the token type ids.
   * @returns {BatchEncoding} Object to be passed to the model.
   */
  _call(text, {
    text_pair = null,
    add_special_tokens = true,
    padding = false,
    truncation = null,
    max_length = null,
    return_tensor = true,
    // Different to HF
    return_token_type_ids = null
  } = {}) {
    const isBatched = Array.isArray(text);
    let encodedTokens;
    if (isBatched) {
      if (text.length === 0) {
        throw Error("text array must be non-empty");
      }
      if (text_pair !== null) {
        if (!Array.isArray(text_pair)) {
          throw Error("text_pair must also be an array");
        } else if (text.length !== text_pair.length) {
          throw Error("text and text_pair must have the same length");
        }
        encodedTokens = text.map(
          (t, i) => this._encode_plus(t, { text_pair: text_pair[i], add_special_tokens, return_token_type_ids })
        );
      } else {
        encodedTokens = text.map((x) => this._encode_plus(x, { add_special_tokens, return_token_type_ids }));
      }
    } else {
      if (text === null || text === void 0) {
        throw Error("text may not be null or undefined");
      }
      if (Array.isArray(text_pair)) {
        throw Error("When specifying `text_pair`, since `text` is a string, `text_pair` must also be a string (i.e., not an array).");
      }
      encodedTokens = [this._encode_plus(text, { text_pair, add_special_tokens, return_token_type_ids })];
    }
    if (max_length === null) {
      max_length = this.model_max_length;
    } else if (truncation === null) {
      if (padding === true) {
        console.warn(
          "`max_length` is ignored when `padding: true` and there is no truncation strategy. To pad to max length, use `padding: 'max_length'`."
        );
        max_length = this.model_max_length;
      } else if (padding === false) {
        console.warn("Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation: true` to explicitly truncate examples to max length.");
        truncation = true;
      }
    }
    if (padding === true) {
      max_length = Math.min(max(encodedTokens.map((x) => x.input_ids.length))[0], max_length ?? Infinity);
    }
    max_length = Math.min(max_length, this.model_max_length ?? Infinity);
    if (padding || truncation) {
      for (let i = 0; i < encodedTokens.length; ++i) {
        if (encodedTokens[i].input_ids.length === max_length) {
          continue;
        } else if (encodedTokens[i].input_ids.length > max_length) {
          if (truncation) {
            truncateHelper(encodedTokens[i], max_length);
          }
        } else {
          if (padding) {
            padHelper(
              encodedTokens[i],
              max_length,
              (key) => key === "input_ids" ? this.pad_token_id : 0,
              this.padding_side
            );
          }
        }
      }
    }
    const result = {};
    if (return_tensor) {
      if (!(padding && truncation)) {
        if (encodedTokens.some((x) => {
          for (const key of Object.keys(x)) {
            if (x[key].length !== encodedTokens[0][key]?.length) {
              return true;
            }
          }
          return false;
        })) {
          throw Error(
            "Unable to create tensor, you should probably activate truncation and/or padding with 'padding=true' and 'truncation=true' to have batched tensors with the same length."
          );
        }
      }
      const dims = [encodedTokens.length, encodedTokens[0].input_ids.length];
      for (const key of Object.keys(encodedTokens[0])) {
        result[key] = new Tensor2(
          "int64",
          BigInt64Array.from(encodedTokens.flatMap((x) => x[key]).map(BigInt)),
          dims
        );
      }
    } else {
      for (const key of Object.keys(encodedTokens[0])) {
        result[key] = encodedTokens.map((x) => x[key]);
      }
      if (!isBatched) {
        for (const key of Object.keys(result)) {
          result[key] = result[key][0];
        }
      }
    }
    return (
      /** @type {BatchEncoding} */
      result
    );
  }
  /**
   * Encodes a single text using the preprocessor pipeline of the tokenizer.
   *
   * @param {string|null} text The text to encode.
   * @returns {string[]|null} The encoded tokens.
   */
  _encode_text(text) {
    if (text === null) return null;
    const sections = this.added_tokens_splitter.split(text);
    for (let i = 0; i < sections.length; ++i) {
      const addedToken = this.added_tokens_map.get(sections[i]);
      if (addedToken) {
        if (addedToken.lstrip && i > 0) {
          sections[i - 1] = sections[i - 1].trimEnd();
        }
        if (addedToken.rstrip && i < sections.length - 1) {
          sections[i + 1] = sections[i + 1].trimStart();
        }
      }
    }
    const tokens = sections.flatMap((x, section_index) => {
      if (x.length === 0) return [];
      if (this.added_tokens_map.has(x)) return [x];
      if (this.remove_space === true) {
        x = x.trim().split(/\s+/).join(" ");
      }
      if (this.do_lowercase_and_remove_accent) {
        x = lowercase_and_remove_accent(x);
      }
      if (this.normalizer !== null) {
        x = this.normalizer(x);
      }
      if (x.length === 0) {
        return [];
      }
      const sectionTokens = this.pre_tokenizer !== null ? this.pre_tokenizer(x, {
        section_index
      }) : [x];
      const tokens2 = this.model(sectionTokens);
      return tokens2;
    });
    return tokens;
  }
  /**
   * Encodes a single text or a pair of texts using the model's tokenizer.
   *
   * @param {string} text The text to encode.
   * @param {Object} options An optional object containing the following properties:
   * @param {string} [options.text_pair=null] The optional second text to encode.
   * @param {boolean} [options.add_special_tokens=true] Whether or not to add the special tokens associated with the corresponding model.
   * @param {boolean} [options.return_token_type_ids=null] Whether to return token_type_ids.
   * @returns {EncodingSingle} An object containing the encoded text.
   * @private
   */
  _encode_plus(text, {
    text_pair = null,
    add_special_tokens = true,
    return_token_type_ids = null
  } = {}) {
    const { tokens, token_type_ids } = this._tokenize_helper(text, { pair: text_pair, add_special_tokens });
    const input_ids = this.model.convert_tokens_to_ids(tokens);
    const result = {
      input_ids,
      attention_mask: new Array(input_ids.length).fill(1)
    };
    if ((return_token_type_ids ?? this.return_token_type_ids) && token_type_ids) {
      result.token_type_ids = token_type_ids;
    }
    return result;
  }
  /**
   * Internal helper function to tokenize a text, and optionally a pair of texts.
   * @param {string} text The text to tokenize.
   * @param {Object} options An optional object containing the following properties:
   * @param {string} [options.pair=null] The optional second text to tokenize.
   * @param {boolean} [options.add_special_tokens=false] Whether or not to add the special tokens associated with the corresponding model.
   * @returns {{tokens: string[], token_type_ids?: number[]}} An object containing the tokens and optionally the token type IDs.
   */
  _tokenize_helper(text, {
    pair = null,
    add_special_tokens = false
  } = {}) {
    const tokens = this._encode_text(text);
    const tokens2 = this._encode_text(pair);
    return this.post_processor ? this.post_processor(tokens, tokens2, { add_special_tokens }) : { tokens: mergeArrays(tokens ?? [], tokens2 ?? []) };
  }
  /**
   * Converts a string into a sequence of tokens.
   * @param {string} text The sequence to be encoded.
   * @param {Object} options An optional object containing the following properties:
   * @param {string} [options.pair] A second sequence to be encoded with the first.
   * @param {boolean} [options.add_special_tokens=false] Whether or not to add the special tokens associated with the corresponding model.
   * @returns {string[]} The list of tokens.
   */
  tokenize(text, {
    pair = null,
    add_special_tokens = false
  } = {}) {
    return this._tokenize_helper(text, { pair, add_special_tokens }).tokens;
  }
  /**
   * Encodes a single text or a pair of texts using the model's tokenizer.
   *
   * @param {string} text The text to encode.
   * @param {Object} options An optional object containing the following properties:
   * @param {string} [options.text_pair=null] The optional second text to encode.
   * @param {boolean} [options.add_special_tokens=true] Whether or not to add the special tokens associated with the corresponding model.
   * @param {boolean} [options.return_token_type_ids=null] Whether to return token_type_ids.
   * @returns {number[]} An array of token IDs representing the encoded text(s).
   */
  encode(text, {
    text_pair = null,
    add_special_tokens = true,
    return_token_type_ids = null
  } = {}) {
    return this._encode_plus(text, {
      text_pair,
      add_special_tokens,
      return_token_type_ids
    }).input_ids;
  }
  /**
   * Decode a batch of tokenized sequences.
   * @param {number[][]|Tensor} batch List/Tensor of tokenized input sequences.
   * @param {Object} decode_args (Optional) Object with decoding arguments.
   * @returns {string[]} List of decoded sequences.
   */
  batch_decode(batch, decode_args = {}) {
    if (batch instanceof Tensor2) {
      batch = batch.tolist();
    }
    return batch.map((x) => this.decode(x, decode_args));
  }
  /**
   * Decodes a sequence of token IDs back to a string.
   *
   * @param {number[]|bigint[]|Tensor} token_ids List/Tensor of token IDs to decode.
   * @param {Object} [decode_args={}]
   * @param {boolean} [decode_args.skip_special_tokens=false] If true, special tokens are removed from the output string.
   * @param {boolean} [decode_args.clean_up_tokenization_spaces=true] If true, spaces before punctuations and abbreviated forms are removed.
   *
   * @returns {string} The decoded string.
   * @throws {Error} If `token_ids` is not a non-empty array of integers.
   */
  decode(token_ids, decode_args = {}) {
    if (token_ids instanceof Tensor2) {
      token_ids = prepareTensorForDecode(token_ids);
    }
    if (!Array.isArray(token_ids) || token_ids.length === 0 || !isIntegralNumber(token_ids[0])) {
      throw Error("token_ids must be a non-empty array of integers.");
    }
    return this.decode_single(token_ids, decode_args);
  }
  /**
   * Decode a single list of token ids to a string.
   * @param {number[]|bigint[]} token_ids List of token ids to decode
   * @param {Object} decode_args Optional arguments for decoding
   * @param {boolean} [decode_args.skip_special_tokens=false] Whether to skip special tokens during decoding
   * @param {boolean} [decode_args.clean_up_tokenization_spaces=null] Whether to clean up tokenization spaces during decoding.
   * If null, the value is set to `this.decoder.cleanup` if it exists, falling back to `this.clean_up_tokenization_spaces` if it exists, falling back to `true`.
   * @returns {string} The decoded string
   */
  decode_single(token_ids, {
    skip_special_tokens = false,
    clean_up_tokenization_spaces = null
  }) {
    let tokens = this.model.convert_ids_to_tokens(token_ids);
    if (skip_special_tokens) {
      tokens = tokens.filter((x) => !this.special_tokens.includes(x));
    }
    let decoded = this.decoder ? this.decoder(tokens) : tokens.join(" ");
    if (this.decoder && this.decoder.end_of_word_suffix) {
      decoded = decoded.replaceAll(this.decoder.end_of_word_suffix, " ");
      if (skip_special_tokens) {
        decoded = decoded.trim();
      }
    }
    if (clean_up_tokenization_spaces ?? this.clean_up_tokenization_spaces) {
      decoded = clean_up_tokenization(decoded);
    }
    return decoded;
  }
  /**
   * Retrieve the chat template string used for tokenizing chat messages. This template is used
   * internally by the `apply_chat_template` method and can also be used externally to retrieve the model's chat
   * template for better generation tracking.
   * 
   * @param {Object} options An optional object containing the following properties:
   * @param {string} [options.chat_template=null]
   * A Jinja template or the name of a template to use for this conversion.
   * It is usually not necessary to pass anything to this argument,
   * as the model's template will be used by default.
   * @param {Object[]} [options.tools=null]
   * A list of tools (callable functions) that will be accessible to the model. If the template does not
   * support function calling, this argument will have no effect. Each tool should be passed as a JSON Schema,
   * giving the name, description and argument types for the tool. See our
   * [chat templating guide](https://huggingface.co/docs/transformers/main/en/chat_templating#automated-function-conversion-for-tool-use)
   * for more information.
   * @returns {string} The chat template string.
   */
  get_chat_template({
    chat_template = null,
    tools = null
  } = {}) {
    if (this.chat_template && typeof this.chat_template === "object") {
      const template_dict = this.chat_template;
      if (chat_template !== null && Object.hasOwn(template_dict, chat_template)) {
        chat_template = template_dict[chat_template];
      } else if (chat_template === null) {
        if (tools !== null && "tool_use" in template_dict) {
          chat_template = template_dict["tool_use"];
        } else if ("default" in template_dict) {
          chat_template = template_dict["default"];
        } else {
          throw Error(
            `This model has multiple chat templates with no default specified! Please either pass a chat template or the name of the template you wish to use to the 'chat_template' argument. Available template names are ${Object.keys(template_dict).sort()}.`
          );
        }
      }
    } else if (chat_template === null) {
      if (this.chat_template) {
        chat_template = this.chat_template;
      } else {
        throw Error(
          "Cannot use apply_chat_template() because tokenizer.chat_template is not set and no template argument was passed! For information about writing templates and setting the tokenizer.chat_template attribute, please see the documentation at https://huggingface.co/docs/transformers/main/en/chat_templating"
        );
      }
    }
    return chat_template;
  }
  /**
   * Converts a list of message objects with `"role"` and `"content"` keys to a list of token
   * ids. This method is intended for use with chat models, and will read the tokenizer's chat_template attribute to
   * determine the format and control tokens to use when converting.
   * 
   * See [here](https://huggingface.co/docs/transformers/chat_templating) for more information.
   * 
   * **Example:** Applying a chat template to a conversation.
   * 
   * ```javascript
   * import { AutoTokenizer } from "@huggingface/transformers";
   * 
   * const tokenizer = await AutoTokenizer.from_pretrained("Xenova/mistral-tokenizer-v1");
   * 
   * const chat = [
   *   { "role": "user", "content": "Hello, how are you?" },
   *   { "role": "assistant", "content": "I'm doing great. How can I help you today?" },
   *   { "role": "user", "content": "I'd like to show off how chat templating works!" },
   * ]
   * 
   * const text = tokenizer.apply_chat_template(chat, { tokenize: false });
   * // "<s>[INST] Hello, how are you? [/INST]I'm doing great. How can I help you today?</s> [INST] I'd like to show off how chat templating works! [/INST]"
   * 
   * const input_ids = tokenizer.apply_chat_template(chat, { tokenize: true, return_tensor: false });
   * // [1, 733, 16289, 28793, 22557, 28725, 910, 460, 368, 28804, 733, 28748, 16289, 28793, 28737, 28742, 28719, 2548, 1598, 28723, 1602, 541, 315, 1316, 368, 3154, 28804, 2, 28705, 733, 16289, 28793, 315, 28742, 28715, 737, 298, 1347, 805, 910, 10706, 5752, 1077, 3791, 28808, 733, 28748, 16289, 28793]
   * ```
   * 
   * @param {Message[]} conversation A list of message objects with `"role"` and `"content"` keys,
   * representing the chat history so far.
   * @param {Object} options An optional object containing the following properties:
   * @param {string} [options.chat_template=null] A Jinja template to use for this conversion. If
   * this is not passed, the model's chat template will be used instead.
   * @param {Object[]} [options.tools=null]
   * A list of tools (callable functions) that will be accessible to the model. If the template does not
   * support function calling, this argument will have no effect. Each tool should be passed as a JSON Schema,
   * giving the name, description and argument types for the tool. See our
   * [chat templating guide](https://huggingface.co/docs/transformers/main/en/chat_templating#automated-function-conversion-for-tool-use)
   * for more information.
   * @param {Record<string, string>[]} [options.documents=null]
   * A list of dicts representing documents that will be accessible to the model if it is performing RAG
   * (retrieval-augmented generation). If the template does not support RAG, this argument will have no
   * effect. We recommend that each document should be a dict containing "title" and "text" keys. Please
   * see the RAG section of the [chat templating guide](https://huggingface.co/docs/transformers/main/en/chat_templating#arguments-for-RAG)
   * for examples of passing documents with chat templates.
   * @param {boolean} [options.add_generation_prompt=false] Whether to end the prompt with the token(s) that indicate
   * the start of an assistant message. This is useful when you want to generate a response from the model.
   * Note that this argument will be passed to the chat template, and so it must be supported in the
   * template for this argument to have any effect.
   * @param {boolean} [options.tokenize=true] Whether to tokenize the output. If false, the output will be a string.
   * @param {boolean} [options.padding=false] Whether to pad sequences to the maximum length. Has no effect if tokenize is false.
   * @param {boolean} [options.truncation=false] Whether to truncate sequences to the maximum length. Has no effect if tokenize is false.
   * @param {number} [options.max_length=null] Maximum length (in tokens) to use for padding or truncation. Has no effect if tokenize is false.
   * If not specified, the tokenizer's `max_length` attribute will be used as a default.
   * @param {boolean} [options.return_tensor=true] Whether to return the output as a Tensor or an Array. Has no effect if tokenize is false.
   * @param {boolean} [options.return_dict=true] Whether to return a dictionary with named outputs. Has no effect if tokenize is false.
   * @param {Object} [options.tokenizer_kwargs={}] Additional options to pass to the tokenizer.
   * @returns {string | Tensor | number[]| number[][]|BatchEncoding} The tokenized output.
   */
  apply_chat_template(conversation, {
    tools = null,
    documents = null,
    chat_template = null,
    add_generation_prompt = false,
    tokenize: tokenize2 = true,
    padding = false,
    truncation = false,
    max_length = null,
    return_tensor = true,
    return_dict = false,
    tokenizer_kwargs = {},
    ...kwargs
  } = {}) {
    chat_template = this.get_chat_template({ chat_template, tools });
    if (typeof chat_template !== "string") {
      throw Error(`chat_template must be a string, but got ${typeof chat_template}`);
    }
    let compiledTemplate = this._compiled_template_cache.get(chat_template);
    if (compiledTemplate === void 0) {
      compiledTemplate = new Template(chat_template);
      this._compiled_template_cache.set(chat_template, compiledTemplate);
    }
    const special_tokens_map = /* @__PURE__ */ Object.create(null);
    for (const key of SPECIAL_TOKEN_ATTRIBUTES) {
      const value = this.getToken(key);
      if (value) {
        special_tokens_map[key] = value;
      }
    }
    const rendered = compiledTemplate.render({
      messages: conversation,
      add_generation_prompt,
      tools,
      documents,
      ...special_tokens_map,
      ...kwargs
    });
    if (tokenize2) {
      const out = this._call(rendered, {
        add_special_tokens: false,
        padding,
        truncation,
        max_length,
        return_tensor,
        ...tokenizer_kwargs
      });
      return return_dict ? out : out.input_ids;
    }
    return rendered;
  }
};
var BertTokenizer = class extends PreTrainedTokenizer {
  return_token_type_ids = true;
};
var AlbertTokenizer = class extends PreTrainedTokenizer {
  return_token_type_ids = true;
};
var MobileBertTokenizer = class extends PreTrainedTokenizer {
  return_token_type_ids = true;
};
var SqueezeBertTokenizer = class extends PreTrainedTokenizer {
  return_token_type_ids = true;
};
var DebertaTokenizer = class extends PreTrainedTokenizer {
  return_token_type_ids = true;
};
var DebertaV2Tokenizer = class extends PreTrainedTokenizer {
  return_token_type_ids = true;
};
var HerbertTokenizer = class extends PreTrainedTokenizer {
  return_token_type_ids = true;
};
var ConvBertTokenizer = class extends PreTrainedTokenizer {
  return_token_type_ids = true;
};
var RoFormerTokenizer = class extends PreTrainedTokenizer {
  return_token_type_ids = true;
};
var DistilBertTokenizer = class extends PreTrainedTokenizer {
};
var CamembertTokenizer = class extends PreTrainedTokenizer {
};
var XLMTokenizer = class extends PreTrainedTokenizer {
  return_token_type_ids = true;
  constructor(tokenizerJSON, tokenizerConfig) {
    super(tokenizerJSON, tokenizerConfig);
    console.warn('WARNING: `XLMTokenizer` is not yet supported by Hugging Face\'s "fast" tokenizers library. Therefore, you may experience slightly inaccurate results.');
  }
};
var ElectraTokenizer = class extends PreTrainedTokenizer {
  return_token_type_ids = true;
};
var T5Tokenizer = class extends PreTrainedTokenizer {
};
var GPT2Tokenizer = class extends PreTrainedTokenizer {
};
var BartTokenizer = class extends PreTrainedTokenizer {
};
var MBartTokenizer = class extends PreTrainedTokenizer {
  constructor(tokenizerJSON, tokenizerConfig) {
    super(tokenizerJSON, tokenizerConfig);
    this.languageRegex = /^[a-z]{2}_[A-Z]{2}$/;
    this.language_codes = this.special_tokens.filter((x) => this.languageRegex.test(x));
    this.lang_to_token = (x) => x;
  }
  /**
   * Helper function to build translation inputs for an `MBartTokenizer`.
   * @param {string|string[]} raw_inputs The text to tokenize.
   * @param {Object} tokenizer_options Options to be sent to the tokenizer
   * @param {Object} generate_kwargs Generation options.
   * @returns {Object} Object to be passed to the model.
   */
  _build_translation_inputs(raw_inputs, tokenizer_options, generate_kwargs) {
    return _build_translation_inputs(this, raw_inputs, tokenizer_options, generate_kwargs);
  }
};
var MBart50Tokenizer = class extends MBartTokenizer {
};
var RobertaTokenizer = class extends PreTrainedTokenizer {
};
var BloomTokenizer = class extends PreTrainedTokenizer {
};
var SPIECE_UNDERLINE = "\u2581";
var LlamaTokenizer = class extends PreTrainedTokenizer {
  padding_side = "left";
  constructor(tokenizerJSON, tokenizerConfig) {
    super(tokenizerJSON, tokenizerConfig);
    this.legacy = tokenizerConfig.legacy ?? true;
    if (!this.legacy) {
      this.normalizer = null;
      this.pre_tokenizer = new MetaspacePreTokenizer({
        replacement: SPIECE_UNDERLINE,
        prepend_scheme: "first"
      });
    }
  }
  /**
   * Helper function to handle legacy encoding of SPM tokenizers.
   * Adapted from https://github.com/huggingface/transformers/blob/e6dcf8abd6f65bb4b6dfc1831b20d9ba49ce00e2/src/transformers/models/t5/tokenization_t5.py#L374-L387
   * @param {string} text The text to encode.
   * @returns {string[]} The encoded tokens.
   */
  _encode_text(text) {
    if (text === null) return null;
    if (this.legacy || text.length === 0) {
      return super._encode_text(text);
    }
    let tokens = super._encode_text(SPIECE_UNDERLINE + text.replaceAll(SPIECE_UNDERLINE, " "));
    if (tokens.length > 1 && tokens[0] === SPIECE_UNDERLINE && this.special_tokens.includes(tokens[1])) {
      tokens = tokens.slice(1);
    }
    return tokens;
  }
};
var CodeLlamaTokenizer = class extends PreTrainedTokenizer {
};
var XLMRobertaTokenizer = class extends PreTrainedTokenizer {
};
var MPNetTokenizer = class extends PreTrainedTokenizer {
};
var FalconTokenizer = class extends PreTrainedTokenizer {
};
var GPTNeoXTokenizer = class extends PreTrainedTokenizer {
};
var EsmTokenizer = class extends PreTrainedTokenizer {
};
var Qwen2Tokenizer = class extends PreTrainedTokenizer {
};
var GemmaTokenizer = class extends PreTrainedTokenizer {
};
var Grok1Tokenizer = class extends PreTrainedTokenizer {
};
function _build_translation_inputs(self2, raw_inputs, tokenizer_options, generate_kwargs) {
  if (!("language_codes" in self2) || !Array.isArray(self2.language_codes)) {
    throw new Error("Tokenizer must have `language_codes` attribute set and it should be an array of language ids.");
  }
  if (!("languageRegex" in self2) || !(self2.languageRegex instanceof RegExp)) {
    throw new Error("Tokenizer must have `languageRegex` attribute set and it should be a regular expression.");
  }
  if (!("lang_to_token" in self2) || typeof self2.lang_to_token !== "function") {
    throw new Error("Tokenizer must have `lang_to_token` attribute set and it should be a function.");
  }
  const src_lang_token = generate_kwargs.src_lang;
  const tgt_lang_token = generate_kwargs.tgt_lang;
  if (!self2.language_codes.includes(tgt_lang_token)) {
    throw new Error(`Target language code "${tgt_lang_token}" is not valid. Must be one of: {${self2.language_codes.join(", ")}}`);
  }
  if (src_lang_token !== void 0) {
    if (!self2.language_codes.includes(src_lang_token)) {
      throw new Error(`Source language code "${src_lang_token}" is not valid. Must be one of: {${self2.language_codes.join(", ")}}`);
    }
    for (const item of self2.post_processor.config.single) {
      if ("SpecialToken" in item && self2.languageRegex.test(item.SpecialToken.id)) {
        item.SpecialToken.id = self2.lang_to_token(src_lang_token);
        break;
      }
    }
  }
  generate_kwargs.forced_bos_token_id = self2.model.convert_tokens_to_ids([self2.lang_to_token(tgt_lang_token)])[0];
  return self2._call(raw_inputs, tokenizer_options);
}
var NllbTokenizer = class extends PreTrainedTokenizer {
  constructor(tokenizerJSON, tokenizerConfig) {
    super(tokenizerJSON, tokenizerConfig);
    this.languageRegex = /^[a-z]{3}_[A-Z][a-z]{3}$/;
    this.language_codes = this.special_tokens.filter((x) => this.languageRegex.test(x));
    this.lang_to_token = (x) => x;
  }
  /**
   * Helper function to build translation inputs for an `NllbTokenizer`.
   * @param {string|string[]} raw_inputs The text to tokenize.
   * @param {Object} tokenizer_options Options to be sent to the tokenizer
   * @param {Object} generate_kwargs Generation options.
   * @returns {Object} Object to be passed to the model.
   */
  _build_translation_inputs(raw_inputs, tokenizer_options, generate_kwargs) {
    return _build_translation_inputs(this, raw_inputs, tokenizer_options, generate_kwargs);
  }
};
var M2M100Tokenizer = class extends PreTrainedTokenizer {
  constructor(tokenizerJSON, tokenizerConfig) {
    super(tokenizerJSON, tokenizerConfig);
    this.languageRegex = /^__[a-z]{2,3}__$/;
    this.language_codes = this.special_tokens.filter((x) => this.languageRegex.test(x)).map((x) => x.slice(2, -2));
    this.lang_to_token = (x) => `__${x}__`;
  }
  /**
   * Helper function to build translation inputs for an `M2M100Tokenizer`.
   * @param {string|string[]} raw_inputs The text to tokenize.
   * @param {Object} tokenizer_options Options to be sent to the tokenizer
   * @param {Object} generate_kwargs Generation options.
   * @returns {Object} Object to be passed to the model.
   */
  _build_translation_inputs(raw_inputs, tokenizer_options, generate_kwargs) {
    return _build_translation_inputs(this, raw_inputs, tokenizer_options, generate_kwargs);
  }
};
var WhisperTokenizer = class extends PreTrainedTokenizer {
  get timestamp_begin() {
    return this.model.convert_tokens_to_ids(["<|notimestamps|>"])[0] + 1;
  }
  /**
   * Decodes automatic speech recognition (ASR) sequences.
   * @param {Array<{tokens: bigint[], token_timestamps?: number[], stride: number[]}>} sequences The sequences to decode.
   * @param {Object} options The options to use for decoding.
   * @returns {Array<string|{chunks?: undefined|Array<{language: string|null, timestamp: Array<number|null>, text: string}>}>} The decoded sequences.
   */
  _decode_asr(sequences, {
    return_timestamps = false,
    return_language = false,
    time_precision = null,
    force_full_sequences = true
  } = {}) {
    if (time_precision === null) {
      throw Error("Must specify time_precision");
    }
    let last_language = null;
    const returnWordTimestamps = return_timestamps === "word";
    function new_chunk() {
      return { "language": last_language, "timestamp": [null, null], "text": "" };
    }
    const chunks = [];
    let chunk2 = new_chunk();
    let time_offset = 0;
    const timestamp_begin = this.timestamp_begin;
    const total_timestamp_tokens = 1500;
    const timestamp_end = timestamp_begin + total_timestamp_tokens;
    let previous_tokens = [];
    let previous_token_timestamps = [];
    let skip = false;
    let right_stride_start = null;
    const all_special_ids = new Set(this.all_special_ids);
    for (const output of sequences) {
      const token_ids = output.tokens;
      const token_timestamps = returnWordTimestamps ? output.token_timestamps : null;
      let last_timestamp = null;
      let first_timestamp = timestamp_begin;
      if ("stride" in output) {
        const [chunk_len, stride_left, stride_right] = output.stride;
        time_offset -= stride_left;
        right_stride_start = chunk_len - stride_right;
        if (stride_left) {
          first_timestamp = stride_left / time_precision + timestamp_begin;
        }
        if (stride_right) {
          for (let i = token_ids.length - 1; i >= 0; --i) {
            const token = Number(token_ids[i]);
            if (token >= timestamp_begin) {
              if (last_timestamp !== null && (token - timestamp_begin) * time_precision < right_stride_start) {
                break;
              }
              last_timestamp = token;
            }
          }
        }
      }
      let current_tokens = [];
      let current_token_timestamps = [];
      for (let i = 0; i < token_ids.length; ++i) {
        const token = Number(token_ids[i]);
        if (all_special_ids.has(token)) {
          const text = this.decode([token]);
          const language = WHISPER_LANGUAGE_MAPPING.get(text.slice(2, -2));
          if (language !== void 0) {
            if (last_language !== null && language !== last_language && !return_timestamps) {
              previous_tokens.push(current_tokens);
              const resolved_tokens = this.findLongestCommonSequence(previous_tokens)[0];
              const resolved_text = this.decode(resolved_tokens);
              chunk2.text = resolved_text;
              chunks.push(chunk2);
              previous_tokens = [];
              current_tokens = [];
              chunk2 = new_chunk();
            }
            last_language = chunk2.language = language;
          } else {
          }
        } else if (token >= timestamp_begin && token <= timestamp_end) {
          const time = (token - timestamp_begin) * time_precision + time_offset;
          const rounded_time = round(time, 2);
          if (last_timestamp !== null && token >= last_timestamp) {
            skip = true;
          } else if (skip || previous_tokens.length > 0 && token < first_timestamp) {
            skip = false;
          } else if (chunk2.timestamp[0] === null) {
            chunk2.timestamp[0] = rounded_time;
          } else {
            if (rounded_time === chunk2.timestamp[0]) {
            } else {
              chunk2.timestamp[1] = rounded_time;
              previous_tokens.push(current_tokens);
              if (returnWordTimestamps) {
                previous_token_timestamps.push(current_token_timestamps);
              }
              const [resolved_tokens, resolved_token_timestamps] = this.findLongestCommonSequence(
                previous_tokens,
                previous_token_timestamps
              );
              const resolved_text = this.decode(resolved_tokens);
              chunk2.text = resolved_text;
              if (returnWordTimestamps) {
                chunk2.words = this.collateWordTimestamps(
                  resolved_tokens,
                  resolved_token_timestamps,
                  last_language
                );
              }
              chunks.push(chunk2);
              previous_tokens = [];
              current_tokens = [];
              previous_token_timestamps = [];
              current_token_timestamps = [];
              chunk2 = new_chunk();
            }
          }
        } else {
          current_tokens.push(token);
          if (returnWordTimestamps) {
            let start_time = round(token_timestamps[i] + time_offset, 2);
            let end_time;
            if (i + 1 < token_timestamps.length) {
              end_time = round(token_timestamps[i + 1] + time_offset, 2);
              const decoded_text = this.decode([token]);
              if (PUNCTUATION_ONLY_REGEX.test(decoded_text)) {
                end_time = round(Math.min(start_time + time_precision, end_time), 2);
              }
            } else {
              end_time = null;
            }
            current_token_timestamps.push([start_time, end_time]);
          }
        }
      }
      if ("stride" in output) {
        const [chunk_len, stride_left, stride_right] = output.stride;
        time_offset += chunk_len - stride_right;
      }
      if (current_tokens.length > 0) {
        previous_tokens.push(current_tokens);
        if (returnWordTimestamps) {
          previous_token_timestamps.push(current_token_timestamps);
        }
      } else if (previous_tokens.every((p) => p.length === 0)) {
        chunk2 = new_chunk();
        previous_tokens = [];
        current_tokens = [];
        previous_token_timestamps = [];
        current_token_timestamps = [];
      }
    }
    if (previous_tokens.length > 0) {
      if (force_full_sequences && return_timestamps) {
        throw new Error(
          "Whisper did not predict an ending timestamp, which can happen if audio is cut off in the middle of a word. Also make sure WhisperTimeStampLogitsProcessor was used during generation."
        );
      }
      const [resolved_tokens, resolved_token_timestamps] = this.findLongestCommonSequence(previous_tokens, previous_token_timestamps);
      const resolved_text = this.decode(resolved_tokens);
      chunk2.text = resolved_text;
      if (returnWordTimestamps) {
        chunk2.words = this.collateWordTimestamps(
          resolved_tokens,
          resolved_token_timestamps,
          last_language
        );
      }
      chunks.push(chunk2);
    }
    let optional = /* @__PURE__ */ Object.create(null);
    const full_text = chunks.map((chunk3) => chunk3.text).join("");
    if (return_timestamps || return_language) {
      for (let i = 0; i < chunks.length; ++i) {
        const chunk3 = chunks[i];
        if (!return_timestamps) {
          delete chunk3["timestamp"];
        }
        if (!return_language) {
          delete chunk3["language"];
        }
      }
      if (returnWordTimestamps) {
        const new_chunks = [];
        for (const chunk3 of chunks) {
          for (const word of chunk3.words) {
            new_chunks.push(word);
          }
        }
        optional = { "chunks": new_chunks };
      } else {
        optional = { "chunks": chunks };
      }
    }
    return [full_text, optional];
  }
  /**
   * Finds the longest common sequence among the provided sequences.
   * @param {number[][]} sequences An array of sequences of token ids to compare.
   * @returns {number[][]} The longest common sequence found.
   * @throws {Error} If there is a bug within the function.
   * @private
   */
  findLongestCommonSequence(sequences, token_timestamp_sequences = null) {
    let leftSequence = sequences[0];
    let leftLength = leftSequence.length;
    let totalSequence = [];
    const use_token_timestamp_sequences = Array.isArray(token_timestamp_sequences) && token_timestamp_sequences.length > 0;
    let total_token_timestamp_sequence = use_token_timestamp_sequences ? [] : null;
    let left_token_timestamp_sequence = use_token_timestamp_sequences ? token_timestamp_sequences[0] : null;
    for (let i = 1; i < sequences.length; ++i) {
      const rightSequence = sequences[i];
      let max2 = 0;
      let maxIndices = [leftLength, leftLength, 0, 0];
      const rightLength = rightSequence.length;
      for (let j = 1; j < leftLength + rightLength; ++j) {
        const leftStart2 = Math.max(0, leftLength - j);
        const leftStop2 = Math.min(leftLength, leftLength + rightLength - j);
        const left = leftSequence.slice(leftStart2, leftStop2);
        const rightStart2 = Math.max(0, j - leftLength);
        const rightStop2 = Math.min(rightLength, j);
        const right = rightSequence.slice(rightStart2, rightStop2);
        if (left.length !== right.length) {
          throw new Error("There is a bug within whisper `decode_asr` function, please report it. Dropping to prevent bad inference.");
        }
        let matches;
        if (use_token_timestamp_sequences) {
          matches = left.filter((elem, idx) => elem === right[idx] && left_token_timestamp_sequence[leftStart2 + idx] <= token_timestamp_sequences[i][rightStart2 + idx]).length;
        } else {
          matches = left.filter((elem, idx) => elem === right[idx]).length;
        }
        const eps = j / 1e4;
        const matching = matches / j + eps;
        if (matches > 1 && matching > max2) {
          max2 = matching;
          maxIndices = [leftStart2, leftStop2, rightStart2, rightStop2];
        }
      }
      const [leftStart, leftStop, rightStart, rightStop] = maxIndices;
      const leftMid = Math.floor((leftStop + leftStart) / 2);
      const rightMid = Math.floor((rightStop + rightStart) / 2);
      totalSequence.push(...leftSequence.slice(0, leftMid));
      leftSequence = rightSequence.slice(rightMid);
      leftLength = leftSequence.length;
      if (use_token_timestamp_sequences) {
        total_token_timestamp_sequence.push(...left_token_timestamp_sequence.slice(0, leftMid));
        left_token_timestamp_sequence = token_timestamp_sequences[i].slice(rightMid);
      }
    }
    totalSequence.push(...leftSequence);
    if (use_token_timestamp_sequences) {
      total_token_timestamp_sequence.push(...left_token_timestamp_sequence);
      return [totalSequence, total_token_timestamp_sequence];
    } else {
      return [totalSequence, []];
    }
  }
  /** @private */
  collateWordTimestamps(tokens, token_timestamps, language) {
    const [words, _, token_indices] = this.combineTokensIntoWords(tokens, language);
    const timings = [];
    for (let i = 0; i < words.length; ++i) {
      const indices = token_indices[i];
      timings.push({
        text: words[i],
        timestamp: [
          token_timestamps[indices.at(0)][0],
          token_timestamps[indices.at(-1)][1]
        ]
      });
    }
    return timings;
  }
  /**
   * Groups tokens by word. Returns a tuple containing a list of strings with the words,
   * and a list of `token_id` sequences with the tokens making up each word.
   * @param {number[]} tokens 
   * @param {string} [language] 
   * @param {string} prepend_punctionations 
   * @param {string} append_punctuations 
   * 
   * @private
   */
  combineTokensIntoWords(tokens, language, prepend_punctionations = `"'\u201C\xA1\xBF([{-`, append_punctuations = `"'.\u3002,\uFF0C!\uFF01?\uFF1F:\uFF1A\u201D)]}\u3001`) {
    language = language ?? "english";
    let words, word_tokens, token_indices;
    if (["chinese", "japanese", "thai", "lao", "myanmar"].includes(language)) {
      [words, word_tokens, token_indices] = this.splitTokensOnUnicode(tokens);
    } else {
      [words, word_tokens, token_indices] = this.splitTokensOnSpaces(tokens);
    }
    return this.mergePunctuations(words, word_tokens, token_indices, prepend_punctionations, append_punctuations);
  }
  /** @type {PreTrainedTokenizer['decode']} */
  decode(token_ids, decode_args) {
    let text;
    if (decode_args?.decode_with_timestamps) {
      if (token_ids instanceof Tensor2) {
        token_ids = prepareTensorForDecode(token_ids);
      }
      text = this.decodeWithTimestamps(token_ids, decode_args);
    } else {
      text = super.decode(token_ids, decode_args);
    }
    return text;
  }
  /**
   * @param {number[]|bigint[]} token_ids List of token IDs to decode.
   * @param {Object} decode_args Optional arguments for decoding
   * @private
   */
  decodeWithTimestamps(token_ids, decode_args) {
    const time_precision = decode_args?.time_precision ?? 0.02;
    const timestamp_begin = Array.from(this.all_special_ids).at(-1) + 1;
    let outputs = [[]];
    for (let token of token_ids) {
      token = Number(token);
      if (token >= timestamp_begin) {
        const timestamp = ((token - timestamp_begin) * time_precision).toFixed(2);
        outputs.push(`<|${timestamp}|>`);
        outputs.push([]);
      } else {
        outputs[outputs.length - 1].push(token);
      }
    }
    outputs = outputs.map(
      (s) => typeof s === "string" ? s : super.decode(s, decode_args)
    );
    return outputs.join("");
  }
  /**
   * Combine tokens into words by splitting at any position where the tokens are decoded as valid unicode points.
   * @param {number[]} tokens 
   * @returns {*}
   * @private
   */
  splitTokensOnUnicode(tokens) {
    const decoded_full = this.decode(tokens, {
      // @ts-ignore
      decode_with_timestamps: true
    });
    const replacement_char = "\uFFFD";
    const words = [];
    const word_tokens = [];
    const token_indices = [];
    let current_tokens = [];
    let current_indices = [];
    let unicode_offset = 0;
    for (let token_idx = 0; token_idx < tokens.length; ++token_idx) {
      const token = tokens[token_idx];
      current_tokens.push(token);
      current_indices.push(token_idx);
      const decoded = this.decode(current_tokens, {
        // @ts-ignore
        decode_with_timestamps: true
      });
      if (!decoded.includes(replacement_char) || decoded_full[unicode_offset + decoded.indexOf(replacement_char)] === replacement_char) {
        words.push(decoded);
        word_tokens.push(current_tokens);
        token_indices.push(current_indices);
        current_tokens = [];
        current_indices = [];
        unicode_offset += decoded.length;
      }
    }
    return [words, word_tokens, token_indices];
  }
  /**
   * Combine tokens into words by splitting at whitespace and punctuation tokens.
   * @param {number[]} tokens 
   * @private
   */
  splitTokensOnSpaces(tokens) {
    const [subwords, subword_tokens_list, subword_indices_list] = this.splitTokensOnUnicode(tokens);
    const words = [];
    const word_tokens = [];
    const token_indices = [];
    const punctuationRegex = new RegExp(`^[${PUNCTUATION_REGEX}]$`, "gu");
    for (let i = 0; i < subwords.length; ++i) {
      const subword = subwords[i];
      const subword_tokens = subword_tokens_list[i];
      const subword_indices = subword_indices_list[i];
      const special = subword_tokens[0] >= this.model.tokens_to_ids.get("<|endoftext|>");
      const with_space = subword.startsWith(" ");
      const trimmed = subword.trim();
      const punctuation = punctuationRegex.test(trimmed);
      if (special || with_space || punctuation || words.length === 0) {
        words.push(subword);
        word_tokens.push(subword_tokens);
        token_indices.push(subword_indices);
      } else {
        const ix = words.length - 1;
        words[ix] += subword;
        word_tokens[ix].push(...subword_tokens);
        token_indices[ix].push(...subword_indices);
      }
    }
    return [words, word_tokens, token_indices];
  }
  /**
   * Merges punctuation tokens with neighboring words.
   * @param {string[]} words 
   * @param {number[][]} tokens 
   * @param {number[][]} indices 
   * @param {string} prepended 
   * @param {string} appended 
   * @private
   */
  mergePunctuations(words, tokens, indices, prepended, appended) {
    const newWords = structuredClone(words);
    const newTokens = structuredClone(tokens);
    const newIndices = structuredClone(indices);
    let i = newWords.length - 2;
    let j = newWords.length - 1;
    while (i >= 0) {
      if (newWords[i].startsWith(" ") && prepended.includes(newWords[i].trim())) {
        newWords[j] = newWords[i] + newWords[j];
        newTokens[j] = mergeArrays(newTokens[i], newTokens[j]);
        newIndices[j] = mergeArrays(newIndices[i], newIndices[j]);
        newWords[i] = "";
        newTokens[i] = [];
        newIndices[i] = [];
      } else {
        j = i;
      }
      --i;
    }
    i = 0;
    j = 1;
    while (j < newWords.length) {
      if (!newWords[i].endsWith(" ") && appended.includes(newWords[j])) {
        newWords[i] += newWords[j];
        newTokens[i] = mergeArrays(newTokens[i], newTokens[j]);
        newIndices[i] = mergeArrays(newIndices[i], newIndices[j]);
        newWords[j] = "";
        newTokens[j] = [];
        newIndices[j] = [];
      } else {
        i = j;
      }
      ++j;
    }
    return [
      newWords.filter((x) => x),
      newTokens.filter((x) => x.length > 0),
      newIndices.filter((x) => x.length > 0)
    ];
  }
};
var CodeGenTokenizer = class extends PreTrainedTokenizer {
};
var CLIPTokenizer = class extends PreTrainedTokenizer {
};
var SiglipTokenizer = class extends PreTrainedTokenizer {
};
var MarianTokenizer = class extends PreTrainedTokenizer {
  /**
   * Create a new MarianTokenizer instance.
   * @param {Object} tokenizerJSON The JSON of the tokenizer.
   * @param {Object} tokenizerConfig The config of the tokenizer.
   */
  constructor(tokenizerJSON, tokenizerConfig) {
    super(tokenizerJSON, tokenizerConfig);
    this.languageRegex = /^(>>\w+<<)\s*/g;
    this.supported_language_codes = this.model.vocab.filter(
      (x) => this.languageRegex.test(x)
    );
    console.warn('WARNING: `MarianTokenizer` is not yet supported by Hugging Face\'s "fast" tokenizers library. Therefore, you may experience slightly inaccurate results.');
  }
  /**
   * Encodes a single text. Overriding this method is necessary since the language codes
   * must be removed before encoding with sentencepiece model.
   * @see https://github.com/huggingface/transformers/blob/12d51db243a00726a548a43cc333390ebae731e3/src/transformers/models/marian/tokenization_marian.py#L204-L213
   *
   * @param {string|null} text The text to encode.
   * @returns {Array} The encoded tokens.
   */
  _encode_text(text) {
    if (text === null) return null;
    const [matchInfo, ...remainder] = text.trim().split(this.languageRegex);
    if (remainder.length === 0) {
      return super._encode_text(matchInfo);
    } else if (remainder.length === 2) {
      const [language, text2] = remainder;
      if (!this.supported_language_codes.includes(language)) {
        console.warn(`Unsupported language code "${language}" detected, which may lead to unexpected behavior. Should be one of: ${JSON.stringify(this.supported_language_codes)}`);
      }
      return mergeArrays([language], super._encode_text(text2));
    }
  }
};
var Wav2Vec2CTCTokenizer = class extends PreTrainedTokenizer {
};
var BlenderbotTokenizer = class extends PreTrainedTokenizer {
};
var BlenderbotSmallTokenizer = class extends PreTrainedTokenizer {
};
var SpeechT5Tokenizer = class extends PreTrainedTokenizer {
};
var NougatTokenizer = class extends PreTrainedTokenizer {
};
var VitsTokenizer = class extends PreTrainedTokenizer {
  constructor(tokenizerJSON, tokenizerConfig) {
    super(tokenizerJSON, tokenizerConfig);
    this.decoder = new VitsDecoder({});
  }
};
var CohereTokenizer = class extends PreTrainedTokenizer {
};
var MgpstrTokenizer = class extends PreTrainedTokenizer {
};
var Ernie4_5_Tokenizer = class extends PreTrainedTokenizer {
};
var AutoTokenizer = class {
  static TOKENIZER_CLASS_MAPPING = {
    T5Tokenizer,
    DistilBertTokenizer,
    CamembertTokenizer,
    DebertaTokenizer,
    DebertaV2Tokenizer,
    BertTokenizer,
    HerbertTokenizer,
    ConvBertTokenizer,
    RoFormerTokenizer,
    XLMTokenizer,
    ElectraTokenizer,
    MobileBertTokenizer,
    SqueezeBertTokenizer,
    AlbertTokenizer,
    GPT2Tokenizer,
    BartTokenizer,
    MBartTokenizer,
    MBart50Tokenizer,
    RobertaTokenizer,
    WhisperTokenizer,
    CodeGenTokenizer,
    CLIPTokenizer,
    SiglipTokenizer,
    MarianTokenizer,
    BloomTokenizer,
    NllbTokenizer,
    M2M100Tokenizer,
    LlamaTokenizer,
    CodeLlamaTokenizer,
    XLMRobertaTokenizer,
    MPNetTokenizer,
    FalconTokenizer,
    GPTNeoXTokenizer,
    EsmTokenizer,
    Wav2Vec2CTCTokenizer,
    BlenderbotTokenizer,
    BlenderbotSmallTokenizer,
    SpeechT5Tokenizer,
    NougatTokenizer,
    VitsTokenizer,
    Qwen2Tokenizer,
    GemmaTokenizer,
    Grok1Tokenizer,
    CohereTokenizer,
    MgpstrTokenizer,
    Ernie4_5_Tokenizer,
    // Base case:
    PreTrainedTokenizer
  };
  /**
   * Instantiate one of the tokenizer classes of the library from a pretrained model.
   * 
   * The tokenizer class to instantiate is selected based on the `tokenizer_class` property of the config object
   * (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible)
   * 
   * @param {string} pretrained_model_name_or_path The name or path of the pretrained model. Can be either:
   * - A string, the *model id* of a pretrained tokenizer hosted inside a model repo on huggingface.co.
   *   Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
   *   user or organization name, like `dbmdz/bert-base-german-cased`.
   * - A path to a *directory* containing tokenizer files, e.g., `./my_model_directory/`.
   * @param {PretrainedTokenizerOptions} options Additional options for loading the tokenizer.
   * 
   * @returns {Promise<PreTrainedTokenizer>} A new instance of the PreTrainedTokenizer class.
   */
  static async from_pretrained(pretrained_model_name_or_path, {
    progress_callback = null,
    config = null,
    cache_dir = null,
    local_files_only = false,
    revision = "main",
    legacy = null
  } = {}) {
    const [tokenizerJSON, tokenizerConfig] = await loadTokenizer(pretrained_model_name_or_path, {
      progress_callback,
      config,
      cache_dir,
      local_files_only,
      revision,
      legacy
    });
    const tokenizerName = tokenizerConfig.tokenizer_class?.replace(/Fast$/, "") ?? "PreTrainedTokenizer";
    let cls = this.TOKENIZER_CLASS_MAPPING[tokenizerName];
    if (!cls) {
      console.warn(`Unknown tokenizer class "${tokenizerName}", attempting to construct from base class.`);
      cls = PreTrainedTokenizer;
    }
    return new cls(tokenizerJSON, tokenizerConfig);
  }
};

// src/configs.js
async function loadConfig(pretrained_model_name_or_path, options) {
  return await getModelJSON(pretrained_model_name_or_path, "config.json", true, options);
}
function getNormalizedConfig(config) {
  const mapping = {};
  let init_normalized_config = {};
  switch (config.model_type) {
    // Sub-configs
    case "llava":
    case "paligemma":
    case "gemma3":
    case "florence2":
    case "llava_onevision":
    case "idefics3":
    case "ultravox":
    case "voxtral":
    case "smolvlm":
    case "gemma3n":
      init_normalized_config = getNormalizedConfig(config.text_config);
      break;
    case "moondream1":
      init_normalized_config = getNormalizedConfig(config.phi_config);
      break;
    case "musicgen":
      init_normalized_config = getNormalizedConfig(config.decoder);
      break;
    case "multi_modality":
      init_normalized_config = getNormalizedConfig(config.language_config);
      break;
    // Decoder-only models
    case "gpt2":
    case "gptj":
    case "jais":
    case "codegen":
    case "gpt_bigcode":
      mapping["num_heads"] = "n_head";
      mapping["num_layers"] = "n_layer";
      mapping["hidden_size"] = "n_embd";
      break;
    case "gpt_neox":
    case "stablelm":
    case "opt":
    case "falcon":
    case "modernbert-decoder":
      mapping["num_heads"] = "num_attention_heads";
      mapping["num_layers"] = "num_hidden_layers";
      mapping["hidden_size"] = "hidden_size";
      break;
    case "llama":
    case "llama4_text":
    case "nanochat":
    case "arcee":
    case "lfm2":
    case "smollm3":
    case "olmo":
    case "olmo2":
    case "mobilellm":
    case "granite":
    case "granitemoehybrid":
    case "cohere":
    case "mistral":
    case "starcoder2":
    case "qwen2":
    case "qwen2_vl":
    case "phi":
    case "phi3":
    case "phi3_v":
    case "llava_qwen2":
      mapping["num_heads"] = "num_key_value_heads";
      mapping["num_layers"] = "num_hidden_layers";
      mapping["hidden_size"] = "hidden_size";
      mapping["num_attention_heads"] = "num_attention_heads";
      mapping["dim_kv"] = "head_dim";
      break;
    case "qwen3":
    case "gemma":
    case "gemma2":
    case "vaultgemma":
    case "gemma3_text":
    case "gemma3n_text":
    case "glm":
    case "helium":
    case "ernie4_5":
      mapping["num_heads"] = "num_key_value_heads";
      mapping["num_layers"] = "num_hidden_layers";
      mapping["dim_kv"] = "head_dim";
      break;
    case "openelm":
      mapping["num_heads"] = "num_kv_heads";
      mapping["num_layers"] = "num_transformer_layers";
      mapping["dim_kv"] = "head_dim";
      break;
    case "gpt_neo":
    case "donut-swin":
      mapping["num_heads"] = "num_heads";
      mapping["num_layers"] = "num_layers";
      mapping["hidden_size"] = "hidden_size";
      break;
    case "bloom":
      mapping["num_heads"] = "n_head";
      mapping["num_layers"] = "n_layer";
      mapping["hidden_size"] = "hidden_size";
      break;
    case "mpt":
      mapping["num_heads"] = "n_heads";
      mapping["num_layers"] = "n_layers";
      mapping["hidden_size"] = "d_model";
      break;
    case "exaone":
      mapping["num_heads"] = "num_key_value_heads";
      mapping["num_layers"] = "num_layers";
      mapping["dim_kv"] = "head_dim";
      mapping["num_attention_heads"] = "num_attention_heads";
      break;
    // Encoder-decoder models
    case "t5":
    case "mt5":
    case "longt5":
      mapping["num_decoder_layers"] = "num_decoder_layers";
      mapping["num_decoder_heads"] = "num_heads";
      mapping["decoder_dim_kv"] = "d_kv";
      mapping["num_encoder_layers"] = "num_layers";
      mapping["num_encoder_heads"] = "num_heads";
      mapping["encoder_dim_kv"] = "d_kv";
      break;
    case "bart":
    case "mbart":
    case "marian":
    case "whisper":
    case "lite-whisper":
    case "m2m_100":
    case "blenderbot":
    case "blenderbot-small":
    case "florence2_language":
      mapping["num_decoder_layers"] = "decoder_layers";
      mapping["num_decoder_heads"] = "decoder_attention_heads";
      mapping["decoder_hidden_size"] = "d_model";
      mapping["num_encoder_layers"] = "encoder_layers";
      mapping["num_encoder_heads"] = "encoder_attention_heads";
      mapping["encoder_hidden_size"] = "d_model";
      break;
    case "speecht5":
      mapping["num_decoder_layers"] = "decoder_layers";
      mapping["num_decoder_heads"] = "decoder_attention_heads";
      mapping["decoder_hidden_size"] = "hidden_size";
      mapping["num_encoder_layers"] = "encoder_layers";
      mapping["num_encoder_heads"] = "encoder_attention_heads";
      mapping["encoder_hidden_size"] = "hidden_size";
      break;
    case "trocr":
      mapping["num_encoder_layers"] = mapping["num_decoder_layers"] = "decoder_layers";
      mapping["num_encoder_heads"] = mapping["num_decoder_heads"] = "decoder_attention_heads";
      mapping["encoder_hidden_size"] = mapping["decoder_hidden_size"] = "d_model";
      break;
    case "musicgen_decoder":
      mapping["num_encoder_layers"] = mapping["num_decoder_layers"] = "num_hidden_layers";
      mapping["num_encoder_heads"] = mapping["num_decoder_heads"] = "num_attention_heads";
      mapping["encoder_hidden_size"] = mapping["decoder_hidden_size"] = "hidden_size";
      break;
    case "moonshine":
      mapping["num_decoder_layers"] = "decoder_num_hidden_layers";
      mapping["num_decoder_heads"] = "decoder_num_key_value_heads";
      mapping["num_encoder_layers"] = "encoder_num_hidden_layers";
      mapping["num_encoder_heads"] = "encoder_num_key_value_heads";
      mapping["encoder_hidden_size"] = mapping["decoder_hidden_size"] = "hidden_size";
      break;
    case "vision-encoder-decoder":
      const decoderConfig = getNormalizedConfig(config.decoder);
      const add_encoder_pkv = "num_decoder_layers" in decoderConfig;
      const result = pick(config, ["model_type", "is_encoder_decoder"]);
      if (add_encoder_pkv) {
        result.num_decoder_layers = decoderConfig.num_decoder_layers;
        result.num_decoder_heads = decoderConfig.num_decoder_heads;
        result.decoder_hidden_size = decoderConfig.decoder_hidden_size;
        result.num_encoder_layers = decoderConfig.num_encoder_layers;
        result.num_encoder_heads = decoderConfig.num_encoder_heads;
        result.encoder_hidden_size = decoderConfig.encoder_hidden_size;
      } else {
        result.num_layers = decoderConfig.num_layers;
        result.num_heads = decoderConfig.num_heads;
        result.hidden_size = decoderConfig.hidden_size;
      }
      return result;
  }
  const normalized_config = {
    ...init_normalized_config,
    ...pick(config, ["model_type", "multi_query", "is_encoder_decoder"])
  };
  for (const key in mapping) {
    normalized_config[key] = config[mapping[key]];
  }
  return normalized_config;
}
function getCacheShapes(config, options) {
  if (config.model_type === "lfm2") {
    const pkv_prefix = options?.prefix ?? "past_key_values";
    const conv_prefix = pkv_prefix === "present" ? "present" : "past";
    const cache_values = {};
    const { layer_types, num_attention_heads, num_key_value_heads, hidden_size, conv_L_cache } = config;
    const head_dim = hidden_size / num_attention_heads;
    const batch_size = options?.batch_size ?? 1;
    for (let i = 0; i < layer_types.length; ++i) {
      if (layer_types[i] === "full_attention") {
        for (const kv of ["key", "value"]) {
          cache_values[`${pkv_prefix}.${i}.${kv}`] = [batch_size, num_key_value_heads, 0, head_dim];
        }
      } else if (layer_types[i] === "conv") {
        cache_values[`${conv_prefix}_conv.${i}`] = [batch_size, hidden_size, conv_L_cache];
      } else {
        throw new Error(`Unsupported layer type: ${layer_types[i]}`);
      }
    }
    return cache_values;
  }
  return getKeyValueShapes(config, options);
}
function getKeyValueShapes(config, {
  prefix = "past_key_values",
  batch_size = 1
} = {}) {
  const decoderFeeds = {};
  const normalized_config = config.normalized_config;
  if (normalized_config.is_encoder_decoder && ("num_encoder_heads" in normalized_config && "num_decoder_heads" in normalized_config)) {
    const encoder_dim_kv = normalized_config.encoder_dim_kv ?? normalized_config.encoder_hidden_size / normalized_config.num_encoder_heads;
    const decoder_dim_kv = normalized_config.decoder_dim_kv ?? normalized_config.decoder_hidden_size / normalized_config.num_decoder_heads;
    const encoder_dims = [batch_size, normalized_config.num_encoder_heads, 0, encoder_dim_kv];
    const decoder_dims = [batch_size, normalized_config.num_decoder_heads, 0, decoder_dim_kv];
    for (let i = 0; i < normalized_config.num_decoder_layers; ++i) {
      decoderFeeds[`${prefix}.${i}.encoder.key`] = encoder_dims;
      decoderFeeds[`${prefix}.${i}.encoder.value`] = encoder_dims;
      decoderFeeds[`${prefix}.${i}.decoder.key`] = decoder_dims;
      decoderFeeds[`${prefix}.${i}.decoder.value`] = decoder_dims;
    }
  } else {
    const num_heads = normalized_config.num_heads;
    const num_layers = normalized_config.num_layers;
    const dim_kv = normalized_config.dim_kv ?? normalized_config.hidden_size / (normalized_config.num_attention_heads ?? num_heads);
    if (normalized_config.model_type === "falcon") {
      const dims = [batch_size * num_heads, 0, dim_kv];
      for (let i = 0; i < num_layers; ++i) {
        decoderFeeds[`${prefix}.${i}.key`] = dims;
        decoderFeeds[`${prefix}.${i}.value`] = dims;
      }
    } else if (normalized_config.multi_query) {
      const dims = [batch_size * num_heads, 0, 2 * dim_kv];
      for (let i = 0; i < num_layers; ++i) {
        decoderFeeds[`${prefix}.${i}.key_value`] = dims;
      }
    } else if (normalized_config.model_type === "bloom") {
      const keyDims = [batch_size * num_heads, dim_kv, 0];
      const valueDims = [batch_size * num_heads, 0, dim_kv];
      for (let i = 0; i < num_layers; ++i) {
        decoderFeeds[`${prefix}.${i}.key`] = keyDims;
        decoderFeeds[`${prefix}.${i}.value`] = valueDims;
      }
    } else if (normalized_config.model_type === "openelm") {
      for (let i = 0; i < num_layers; ++i) {
        const dims = [batch_size, num_heads[i], 0, dim_kv];
        decoderFeeds[`${prefix}.${i}.key`] = dims;
        decoderFeeds[`${prefix}.${i}.value`] = dims;
      }
    } else {
      const dims = [batch_size, num_heads, 0, dim_kv];
      for (let i = 0; i < num_layers; ++i) {
        decoderFeeds[`${prefix}.${i}.key`] = dims;
        decoderFeeds[`${prefix}.${i}.value`] = dims;
      }
    }
  }
  return decoderFeeds;
}
var PretrainedConfig = class _PretrainedConfig {
  // NOTE: Typo in original
  /** @type {string|null} */
  model_type = null;
  /** @type {boolean} */
  is_encoder_decoder = false;
  /** @type {number} */
  max_position_embeddings;
  /** @type {TransformersJSConfig} */
  "transformers.js_config";
  /**
   * Create a new PreTrainedTokenizer instance.
   * @param {Object} configJSON The JSON of the config.
   */
  constructor(configJSON) {
    Object.assign(this, configJSON);
    this.normalized_config = getNormalizedConfig(this);
  }
  /**
   * Loads a pre-trained config from the given `pretrained_model_name_or_path`. 
   * 
   * @param {string} pretrained_model_name_or_path The path to the pre-trained config.
   * @param {PretrainedOptions} options Additional options for loading the config.
   * @throws {Error} Throws an error if the config.json is not found in the `pretrained_model_name_or_path`.
   * 
   * @returns {Promise<PretrainedConfig>} A new instance of the `PretrainedConfig` class.
   */
  static async from_pretrained(pretrained_model_name_or_path, {
    progress_callback = null,
    config = null,
    cache_dir = null,
    local_files_only = false,
    revision = "main"
  } = {}) {
    if (config && !(config instanceof _PretrainedConfig)) {
      config = new _PretrainedConfig(config);
    }
    const data = config ?? await loadConfig(pretrained_model_name_or_path, {
      progress_callback,
      config,
      cache_dir,
      local_files_only,
      revision
    });
    return new this(data);
  }
};
var AutoConfig = class {
  /** @type {typeof PretrainedConfig.from_pretrained} */
  static async from_pretrained(...args) {
    return PretrainedConfig.from_pretrained(...args);
  }
};

// src/utils/devices.js
var DEVICE_TYPES = Object.freeze({
  auto: "auto",
  // Auto-detect based on device and environment
  gpu: "gpu",
  // Auto-detect GPU
  cpu: "cpu",
  // CPU
  wasm: "wasm",
  // WebAssembly
  webgpu: "webgpu",
  // WebGPU
  cuda: "cuda",
  // CUDA
  dml: "dml",
  // DirectML
  webnn: "webnn",
  // WebNN (default)
  "webnn-npu": "webnn-npu",
  // WebNN NPU
  "webnn-gpu": "webnn-gpu",
  // WebNN GPU
  "webnn-cpu": "webnn-cpu"
  // WebNN CPU
});

// src/utils/dtypes.js
var isWebGpuFp16Supported = /* @__PURE__ */ (function() {
  let cachedResult;
  return async function() {
    if (cachedResult === void 0) {
      if (!apis.IS_WEBGPU_AVAILABLE) {
        cachedResult = false;
      } else {
        try {
          const adapter = await navigator.gpu.requestAdapter();
          cachedResult = adapter.features.has("shader-f16");
        } catch (e) {
          cachedResult = false;
        }
      }
    }
    return cachedResult;
  };
})();
var DATA_TYPES = Object.freeze({
  auto: "auto",
  // Auto-detect based on environment
  fp32: "fp32",
  fp16: "fp16",
  q8: "q8",
  int8: "int8",
  uint8: "uint8",
  q4: "q4",
  bnb4: "bnb4",
  q4f16: "q4f16"
  // fp16 model with int4 block weight quantization
});
var DEFAULT_DEVICE_DTYPE_MAPPING = Object.freeze({
  // NOTE: If not specified, will default to fp32
  [DEVICE_TYPES.wasm]: DATA_TYPES.q8
});
var DEFAULT_DTYPE_SUFFIX_MAPPING = Object.freeze({
  [DATA_TYPES.fp32]: "",
  [DATA_TYPES.fp16]: "_fp16",
  [DATA_TYPES.int8]: "_int8",
  [DATA_TYPES.uint8]: "_uint8",
  [DATA_TYPES.q8]: "_quantized",
  [DATA_TYPES.q4]: "_q4",
  [DATA_TYPES.q4f16]: "_q4f16",
  [DATA_TYPES.bnb4]: "_bnb4"
});

// src/utils/constants.js
var GITHUB_ISSUE_URL = "https://github.com/huggingface/transformers.js/issues/new/choose";
var FEATURE_EXTRACTOR_NAME = "preprocessor_config.json";
var IMAGE_PROCESSOR_NAME = FEATURE_EXTRACTOR_NAME;
var PROCESSOR_NAME = "processor_config.json";
var CHAT_TEMPLATE_NAME = "chat_template.jinja";

// src/generation/logits_process.js
var LogitsProcessor = class extends Callable {
  /**
   * Apply the processor to the input logits.
   *
   * @abstract
   * @param {bigint[][]} input_ids The input ids.
   * @param {Tensor} logits The logits to process.
   * @throws {Error} Throws an error if `_call` is not implemented in the subclass.
   */
  _call(input_ids, logits) {
    throw Error("`_call` should be implemented in a subclass");
  }
};
var LogitsWarper = class extends Callable {
  /**
   * Apply the processor to the input logits.
   *
   * @abstract
   * @param {bigint[][]} input_ids The input ids.
   * @param {Tensor} logits The logits to process.
   * @throws {Error} Throws an error if `_call` is not implemented in the subclass.
   */
  _call(input_ids, logits) {
    throw Error("`_call` should be implemented in a subclass");
  }
};
var LogitsProcessorList = class extends Callable {
  /**
   * Constructs a new instance of `LogitsProcessorList`.
   */
  constructor() {
    super();
    this.processors = [];
  }
  /**
   * Adds a new logits processor to the list.
   *
   * @param {LogitsProcessor} item The logits processor function to add.
   */
  push(item) {
    this.processors.push(item);
  }
  /**
   * Adds multiple logits processors to the list.
   *
   * @param {LogitsProcessor[]} items The logits processor functions to add.
   */
  extend(items) {
    this.processors.push(...items);
  }
  /**
   * Applies all logits processors in the list to a batch of logits, modifying them in-place.
   *
   * @param {bigint[][]} input_ids The input IDs for the language model.
   * @param {Tensor} logits
   */
  _call(input_ids, logits) {
    let toReturn = logits;
    for (const processor of this.processors) {
      toReturn = processor(input_ids, toReturn);
    }
    return toReturn;
  }
  [Symbol.iterator]() {
    return this.processors.values();
  }
};
var ForcedBOSTokenLogitsProcessor = class extends LogitsProcessor {
  /**
   * Create a ForcedBOSTokenLogitsProcessor.
   * @param {number} bos_token_id The ID of the beginning-of-sequence token to be forced.
   */
  constructor(bos_token_id) {
    super();
    this.bos_token_id = bos_token_id;
  }
  /**
   * Apply the BOS token forcing to the logits.
   * @param {bigint[][]} input_ids The input IDs.
   * @param {Tensor} logits The logits.
   * @returns {Tensor} The logits with BOS token forcing.
   */
  _call(input_ids, logits) {
    for (let i = 0; i < input_ids.length; ++i) {
      if (input_ids[i].length === 1) {
        const batch_logits_data = (
          /** @type {Float32Array} */
          logits[i].data
        );
        batch_logits_data.fill(-Infinity);
        batch_logits_data[this.bos_token_id] = 0;
      }
    }
    return logits;
  }
};
var ForcedEOSTokenLogitsProcessor = class extends LogitsProcessor {
  /**
   * Create a ForcedEOSTokenLogitsProcessor.
   * @param {number} max_length The maximum length of the sequence to be generated.
   * @param {number|number[]} eos_token_id The id(s) of the *end-of-sequence* token.
   */
  constructor(max_length, eos_token_id) {
    super();
    this.max_length = max_length;
    this.eos_token_id = Array.isArray(eos_token_id) ? eos_token_id : [eos_token_id];
  }
  /**
   * Apply the processor to input_ids and logits.
   * 
   * @param {bigint[][]} input_ids The input ids.
   * @param {Tensor} logits The logits tensor.
   */
  _call(input_ids, logits) {
    for (let i = 0; i < input_ids.length; ++i) {
      if (input_ids[i].length === this.max_length - 1) {
        const batch_logits_data = (
          /** @type {Float32Array} */
          logits[i].data
        );
        batch_logits_data.fill(-Infinity);
        for (const eos_token of this.eos_token_id) {
          batch_logits_data[eos_token] = 0;
        }
      }
    }
    return logits;
  }
};
var SuppressTokensAtBeginLogitsProcessor = class extends LogitsProcessor {
  /**
   * Create a SuppressTokensAtBeginLogitsProcessor.
   * @param {number[]} begin_suppress_tokens The IDs of the tokens to suppress.
   * @param {number} begin_index The number of tokens to generate before suppressing tokens.
   */
  constructor(begin_suppress_tokens, begin_index) {
    super();
    this.begin_suppress_tokens = begin_suppress_tokens;
    this.begin_index = begin_index;
  }
  /**
   * Apply the BOS token forcing to the logits.
   * @param {bigint[][]} input_ids The input IDs.
   * @param {Tensor} logits The logits.
   * @returns {Tensor} The logits with BOS token forcing.
   */
  _call(input_ids, logits) {
    for (let i = 0; i < input_ids.length; ++i) {
      if (input_ids[i].length === this.begin_index) {
        const batch_logits_data = (
          /** @type {Float32Array} */
          logits[i].data
        );
        for (const token_id of this.begin_suppress_tokens) {
          batch_logits_data[token_id] = -Infinity;
        }
      }
    }
    return logits;
  }
};
var WhisperTimeStampLogitsProcessor = class extends LogitsProcessor {
  /**
   * Constructs a new WhisperTimeStampLogitsProcessor.
   * @param {import('../models/whisper/generation_whisper.js').WhisperGenerationConfig} generate_config The config object passed to the `generate()` method of a transformer model.
   * @param {number[]} init_tokens The initial tokens of the input sequence.
   */
  constructor(generate_config, init_tokens) {
    super();
    this.eos_token_id = Array.isArray(generate_config.eos_token_id) ? generate_config.eos_token_id[0] : generate_config.eos_token_id;
    this.no_timestamps_token_id = generate_config.no_timestamps_token_id;
    this.timestamp_begin = this.no_timestamps_token_id + 1;
    this.begin_index = init_tokens.length;
    if (init_tokens.at(-1) === this.no_timestamps_token_id) {
      this.begin_index -= 1;
    }
    this.max_initial_timestamp_index = generate_config.max_initial_timestamp_index;
  }
  /**
   * Modify the logits to handle timestamp tokens.
   * @param {bigint[][]} input_ids The input sequence of tokens.
   * @param {Tensor} logits The logits output by the model.
   * @returns {Tensor} The modified logits.
   */
  _call(input_ids, logits) {
    for (let i = 0; i < input_ids.length; ++i) {
      const batch_logits_data = (
        /** @type {Float32Array} */
        logits[i].data
      );
      batch_logits_data[this.no_timestamps_token_id] = -Infinity;
      if (input_ids[i].length === this.begin_index - 1) {
        batch_logits_data.fill(-Infinity);
        batch_logits_data[this.timestamp_begin] = 0;
        continue;
      }
      const seq = input_ids[i].slice(this.begin_index);
      const last_was_timestamp = seq.length >= 1 && seq[seq.length - 1] >= this.timestamp_begin;
      const penultimate_was_timestamp = seq.length < 2 || seq[seq.length - 2] >= this.timestamp_begin;
      if (last_was_timestamp) {
        if (penultimate_was_timestamp) {
          batch_logits_data.subarray(this.timestamp_begin).fill(-Infinity);
        } else {
          batch_logits_data.subarray(0, this.eos_token_id).fill(-Infinity);
        }
      }
      if (input_ids[i].length === this.begin_index && this.max_initial_timestamp_index !== null) {
        const last_allowed = this.timestamp_begin + this.max_initial_timestamp_index;
        batch_logits_data.subarray(last_allowed + 1).fill(-Infinity);
      }
      const logprobs = log_softmax(batch_logits_data);
      const timestamp_logprob = Math.log(logprobs.subarray(this.timestamp_begin).map(Math.exp).reduce((a, b) => a + b));
      const max_text_token_logprob = max(logprobs.subarray(0, this.timestamp_begin))[0];
      if (timestamp_logprob > max_text_token_logprob) {
        batch_logits_data.subarray(0, this.timestamp_begin).fill(-Infinity);
      }
    }
    return logits;
  }
};
var NoRepeatNGramLogitsProcessor = class extends LogitsProcessor {
  /**
   * Create a NoRepeatNGramLogitsProcessor.
   * @param {number} no_repeat_ngram_size The no-repeat-ngram size. All ngrams of this size can only occur once.
   */
  constructor(no_repeat_ngram_size) {
    super();
    this.no_repeat_ngram_size = no_repeat_ngram_size;
  }
  /**
   * Generate n-grams from a sequence of token ids.
   * @param {bigint[]} prevInputIds List of previous input ids
   * @returns {Map<string, number[]>} Map of generated n-grams
   */
  getNgrams(prevInputIds) {
    const curLen = prevInputIds.length;
    const ngrams = [];
    for (let j = 0; j < curLen + 1 - this.no_repeat_ngram_size; ++j) {
      const ngram = [];
      for (let k = 0; k < this.no_repeat_ngram_size; ++k) {
        ngram.push(prevInputIds[j + k]);
      }
      ngrams.push(ngram.map(Number));
    }
    const generatedNgram = /* @__PURE__ */ new Map();
    for (const ngram of ngrams) {
      const prevNgram = ngram.slice(0, ngram.length - 1);
      const prevNgramKey = JSON.stringify(prevNgram);
      const prevNgramValue = generatedNgram.get(prevNgramKey) ?? [];
      prevNgramValue.push(ngram[ngram.length - 1]);
      generatedNgram.set(prevNgramKey, prevNgramValue);
    }
    return generatedNgram;
  }
  /**
   * Generate n-grams from a sequence of token ids.
   * @param {Map<string, number[]>} bannedNgrams Map of banned n-grams
   * @param {bigint[]} prevInputIds List of previous input ids
   * @returns {number[]} Map of generated n-grams
   */
  getGeneratedNgrams(bannedNgrams, prevInputIds) {
    const ngramIdx = prevInputIds.slice(prevInputIds.length + 1 - this.no_repeat_ngram_size, prevInputIds.length);
    const banned = bannedNgrams.get(JSON.stringify(ngramIdx.map(Number))) ?? [];
    return banned;
  }
  /**
   * Calculate banned n-gram tokens
   * @param {bigint[]} prevInputIds List of previous input ids
   * @returns {number[]} Map of generated n-grams
   */
  calcBannedNgramTokens(prevInputIds) {
    const bannedTokens = [];
    if (prevInputIds.length + 1 < this.no_repeat_ngram_size) {
      return bannedTokens;
    } else {
      const generatedNgrams = this.getNgrams(prevInputIds);
      const bannedTokens2 = this.getGeneratedNgrams(generatedNgrams, prevInputIds);
      return bannedTokens2;
    }
  }
  /**
   * Apply the no-repeat-ngram processor to the logits.
   * @param {bigint[][]} input_ids The input IDs.
   * @param {Tensor} logits The logits.
   * @returns {Tensor} The logits with no-repeat-ngram processing.
   */
  _call(input_ids, logits) {
    for (let i = 0; i < input_ids.length; ++i) {
      const batch_logits_data = (
        /** @type {Float32Array} */
        logits[i].data
      );
      const bannedTokens = this.calcBannedNgramTokens(input_ids[i]);
      for (const token of bannedTokens) {
        batch_logits_data[token] = -Infinity;
      }
    }
    return logits;
  }
};
var RepetitionPenaltyLogitsProcessor = class extends LogitsProcessor {
  /**
   * Create a RepetitionPenaltyLogitsProcessor.
   * @param {number} penalty The parameter for repetition penalty.
   * - 1.0 means no penalty. Above 1.0 penalizes previously generated tokens.
   * - Between 0.0 and 1.0 rewards previously generated tokens.
   */
  constructor(penalty) {
    super();
    this.penalty = penalty;
  }
  /**
   * Apply the repetition penalty to the logits.
   * @param {bigint[][]} input_ids The input IDs.
   * @param {Tensor} logits The logits.
   * @returns {Tensor} The logits with repetition penalty processing.
   */
  _call(input_ids, logits) {
    for (let i = 0; i < input_ids.length; ++i) {
      const batch_logits_data = (
        /** @type {Float32Array} */
        logits[i].data
      );
      for (const input_id of new Set(input_ids[i])) {
        const token = Number(input_id);
        if (batch_logits_data[token] < 0) {
          batch_logits_data[token] *= this.penalty;
        } else {
          batch_logits_data[token] /= this.penalty;
        }
      }
    }
    return logits;
  }
};
var MinLengthLogitsProcessor = class extends LogitsProcessor {
  /**
   * Create a MinLengthLogitsProcessor.
   * @param {number} min_length The minimum length below which the score of `eos_token_id` is set to negative infinity.
   * @param {number|number[]} eos_token_id The ID/IDs of the end-of-sequence token.
   */
  constructor(min_length, eos_token_id) {
    super();
    this.min_length = min_length;
    this.eos_token_id = Array.isArray(eos_token_id) ? eos_token_id : [eos_token_id];
  }
  /**
   * Apply logit processor.
   * @param {bigint[][]} input_ids The input IDs.
   * @param {Tensor} logits The logits.
   * @returns {Tensor} The processed logits.
   */
  _call(input_ids, logits) {
    for (let i = 0; i < input_ids.length; ++i) {
      if (input_ids[i].length < this.min_length) {
        const batch_logits_data = (
          /** @type {Float32Array} */
          logits[i].data
        );
        for (const eos_token of this.eos_token_id) {
          batch_logits_data[eos_token] = -Infinity;
        }
      }
    }
    return logits;
  }
};
var MinNewTokensLengthLogitsProcessor = class extends LogitsProcessor {
  /**
   * Create a MinNewTokensLengthLogitsProcessor.
   * @param {number} prompt_length_to_skip The input tokens length.
   * @param {number} min_new_tokens The minimum *new* tokens length below which the score of `eos_token_id` is set to negative infinity.
   * @param {number|number[]} eos_token_id The ID/IDs of the end-of-sequence token.
   */
  constructor(prompt_length_to_skip, min_new_tokens, eos_token_id) {
    super();
    this.prompt_length_to_skip = prompt_length_to_skip;
    this.min_new_tokens = min_new_tokens;
    this.eos_token_id = Array.isArray(eos_token_id) ? eos_token_id : [eos_token_id];
  }
  /**
   * Apply logit processor.
   * @param {bigint[][]} input_ids The input IDs.
   * @param {Tensor} logits The logits.
   * @returns {Tensor} The processed logits.
   */
  _call(input_ids, logits) {
    for (let i = 0; i < input_ids.length; ++i) {
      const new_tokens_length = input_ids[i].length - this.prompt_length_to_skip;
      if (new_tokens_length < this.min_new_tokens) {
        const batch_logits_data = (
          /** @type {Float32Array} */
          logits[i].data
        );
        for (const eos_token of this.eos_token_id) {
          batch_logits_data[eos_token] = -Infinity;
        }
      }
    }
    return logits;
  }
};
var NoBadWordsLogitsProcessor = class extends LogitsProcessor {
  /**
   * Create a `NoBadWordsLogitsProcessor`.
   * @param {number[][]} bad_words_ids List of list of token ids that are not allowed to be generated.
   * @param {number|number[]} eos_token_id The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
   */
  constructor(bad_words_ids, eos_token_id) {
    super();
    this.bad_words_ids = bad_words_ids;
    this.eos_token_id = Array.isArray(eos_token_id) ? eos_token_id : [eos_token_id];
  }
  /**
   * Apply logit processor.
   * @param {bigint[][]} input_ids The input IDs.
   * @param {Tensor} logits The logits.
   * @returns {Tensor} The processed logits.
   */
  _call(input_ids, logits) {
    for (let i = 0; i < input_ids.length; ++i) {
      const batch_logits_data = (
        /** @type {Float32Array} */
        logits[i].data
      );
      const ids = input_ids[i];
      for (const bad_word_ids of this.bad_words_ids) {
        if (ids.length < bad_word_ids.length - 1) continue;
        let mark = true;
        for (let j = 1; j <= bad_word_ids.length - 1; ++j) {
          if (bad_word_ids.at(-j - 1) != ids.at(-j)) {
            mark = false;
            break;
          }
        }
        if (mark) {
          batch_logits_data[bad_word_ids.at(-1)] = -Infinity;
        }
      }
    }
    return logits;
  }
};
var ClassifierFreeGuidanceLogitsProcessor = class extends LogitsProcessor {
  /**
   * Create a `ClassifierFreeGuidanceLogitsProcessor`.
   * @param {number} guidance_scale The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale > 1`.
   * Higher guidance scale encourages the model to generate samples that are more closely linked to the input
   * prompt, usually at the expense of poorer quality.
   */
  constructor(guidance_scale) {
    super();
    if (guidance_scale <= 1) {
      throw new Error(
        `Require guidance scale >1 to use the classifier free guidance processor, got guidance scale ${guidance_scale}.`
      );
    }
    this.guidance_scale = guidance_scale;
  }
  /**
   * Apply logit processor.
   * @param {bigint[][]} input_ids The input IDs.
   * @param {Tensor} logits The logits.
   * @returns {Tensor} The processed logits.
   */
  _call(input_ids, logits) {
    if (logits.dims[0] !== 2 * input_ids.length) {
      throw new Error(
        `Logits should have twice the batch size of the input ids, the first half of batches corresponding to the conditional inputs, and the second half of batches corresponding to the unconditional inputs. Got batch size ${logits.dims[0]} for the logits and ${input_ids.length} for the input ids.`
      );
    }
    const unguided_bsz = input_ids.length;
    const cond_logits = logits.slice([0, unguided_bsz], null);
    const uncond_logits = logits.slice([unguided_bsz, logits.dims[0]], null);
    for (let i = 0; i < uncond_logits.data.length; ++i) {
      uncond_logits.data[i] += (cond_logits.data[i] - uncond_logits.data[i]) * this.guidance_scale;
    }
    return uncond_logits;
  }
};
var TemperatureLogitsWarper = class extends LogitsWarper {
  /**
   * Create a `TemperatureLogitsWarper`.
   * @param {number} temperature Strictly positive float value used to modulate the logits distribution.
   * A value smaller than `1` decreases randomness (and vice versa), with `0` being equivalent to shifting
   * all probability mass to the most likely token.
   */
  constructor(temperature) {
    super();
    if (typeof temperature !== "number" || temperature <= 0) {
      let errorMessage = `\`temperature\` (=${temperature}) must be a strictly positive float, otherwise your next token scores will be invalid.`;
      if (temperature === 0) {
        errorMessage += " If you're looking for greedy decoding strategies, set `do_sample=false`.";
      }
    }
    this.temperature = temperature;
  }
  /**
   * Apply logit warper.
   * @param {bigint[][]} input_ids The input IDs.
   * @param {Tensor} logits The logits.
   * @returns {Tensor} The processed logits.
   */
  _call(input_ids, logits) {
    const batch_logits_data = (
      /** @type {Float32Array} */
      logits.data
    );
    for (let i = 0; i < batch_logits_data.length; ++i) {
      batch_logits_data[i] /= this.temperature;
    }
    return logits;
  }
};
var TopPLogitsWarper = class extends LogitsWarper {
  /**
   * Create a `TopPLogitsWarper`.
   * @param {number} top_p If set to < 1, only the smallest set of most probable tokens with
   * probabilities that add up to `top_p` or higher are kept for generation.
   * @param {Object} options Additional options for the top-p sampling.
   * @param {number} [options.filter_value=-Infinity] All filtered values will be set to this float value.
   * @param {number} [options.min_tokens_to_keep=1] Minimum number of tokens that cannot be filtered.
   */
  constructor(top_p, {
    filter_value = -Infinity,
    min_tokens_to_keep = 1
  } = {}) {
    super();
    if (top_p < 0 || top_p > 1) {
      throw new Error(`\`top_p\` must be a float > 0 and < 1, but is ${top_p}`);
    }
    if (!Number.isInteger(min_tokens_to_keep) || min_tokens_to_keep < 1) {
      throw new Error(`\`min_tokens_to_keep\` must be a positive integer, but is ${min_tokens_to_keep}`);
    }
    this.top_p = top_p;
    this.filter_value = filter_value;
    this.min_tokens_to_keep = min_tokens_to_keep;
  }
};
var TopKLogitsWarper = class extends LogitsWarper {
  /**
   * Create a `TopKLogitsWarper`.
   * @param {number} top_k If set to > 0, only the top `top_k` tokens are kept for generation.
   * @param {Object} options Additional options for the top-k sampling.
   * @param {number} [options.filter_value=-Infinity] All filtered values will be set to this float value.
   * @param {number} [options.min_tokens_to_keep=1] Minimum number of tokens that cannot be filtered.
   */
  constructor(top_k, {
    filter_value = -Infinity,
    min_tokens_to_keep = 1
  } = {}) {
    super();
    if (!Number.isInteger(top_k) || top_k < 0) {
      throw new Error(`\`top_k\` must be a positive integer, but is ${top_k}`);
    }
    this.top_k = Math.max(top_k, min_tokens_to_keep);
    this.filter_value = filter_value;
  }
};

// src/generation/configuration_utils.js
var GenerationConfig = class {
  // Parameters that control the length of the output
  /**
   * The maximum length the generated tokens can have.
   * Corresponds to the length of the input prompt + `max_new_tokens`.
   * Its effect is overridden by `max_new_tokens`, if also set.
   * @type {number}
   * @default 20
   */
  max_length = 20;
  /**
   * The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
   * @type {number}
   * @default null
   */
  max_new_tokens = null;
  /**
   * The minimum length of the sequence to be generated.
   * Corresponds to the length of the input prompt + `min_new_tokens`.
   * Its effect is overridden by `min_new_tokens`, if also set.
   * @type {number}
   * @default 0
   */
  min_length = 0;
  /**
   * The minimum numbers of tokens to generate, ignoring the number of tokens in the prompt.
   * @type {number}
   * @default null
   */
  min_new_tokens = null;
  /**
   * Controls the stopping condition for beam-based methods, like beam-search. It accepts the following values:
   * - `true`, where the generation stops as soon as there are `num_beams` complete candidates;
   * - `false`, where an heuristic is applied and the generation stops when is it very unlikely to find better candidates;
   * - `"never"`, where the beam search procedure only stops when there cannot be better candidates (canonical beam search algorithm).
   * @type {boolean|"never"}
   * @default false
   */
  early_stopping = false;
  /**
   * The maximum amount of time you allow the computation to run for in seconds.
   * Generation will still finish the current pass after allocated time has been passed.
   * @type {number}
   * @default null
   */
  max_time = null;
  // Parameters that control the generation strategy used
  /**
   * Whether or not to use sampling; use greedy decoding otherwise.
   * @type {boolean}
   * @default false
   */
  do_sample = false;
  /**
   * Number of beams for beam search. 1 means no beam search.
   * @type {number}
   * @default 1
   */
  num_beams = 1;
  /**
   * Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
   * See [this paper](https://huggingface.co/papers/1610.02424) for more details.
   * @type {number}
   * @default 1
   */
  num_beam_groups = 1;
  /**
   * The values balance the model confidence and the degeneration penalty in contrastive search decoding.
   * @type {number}
   * @default null
   */
  penalty_alpha = null;
  /**
   * Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
   * @type {boolean}
   * @default true
   */
  use_cache = true;
  // Parameters for manipulation of the model output logits
  /**
   * The value used to modulate the next token probabilities.
   * @type {number}
   * @default 1.0
   */
  temperature = 1;
  /**
   * The number of highest probability vocabulary tokens to keep for top-k-filtering.
   * @type {number}
   * @default 50
   */
  top_k = 50;
  /**
   * If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or higher are kept for generation.
   * @type {number}
   * @default 1.0
   */
  top_p = 1;
  /**
   * Local typicality measures how similar the conditional probability of predicting a target token next is to the expected conditional probability of predicting a random token next, given the partial text already generated.
   * If set to float < 1, the smallest set of the most locally typical tokens with probabilities that add up to `typical_p` or higher are kept for generation.
   * See [this paper](https://huggingface.co/papers/2202.00666) for more details.
   * @type {number}
   * @default 1.0
   */
  typical_p = 1;
  /**
   * If set to float strictly between 0 and 1, only tokens with a conditional probability greater than `epsilon_cutoff` will be sampled.
   * In the paper, suggested values range from 3e-4 to 9e-4, depending on the size of the model.
   * See [Truncation Sampling as Language Model Desmoothing](https://huggingface.co/papers/2210.15191) for more details.
   * @type {number}
   * @default 0.0
   */
  epsilon_cutoff = 0;
  /**
   * Eta sampling is a hybrid of locally typical sampling and epsilon sampling.
   * If set to float strictly between 0 and 1, a token is only considered if it is greater than either `eta_cutoff` or `sqrt(eta_cutoff) * exp(-entropy(softmax(next_token_logits)))`.
   * The latter term is intuitively the expected next token probability, scaled by `sqrt(eta_cutoff)`. In the paper, suggested values range from 3e-4 to 2e-3, depending on the size of the model.
   * See [Truncation Sampling as Language Model Desmoothing](https://huggingface.co/papers/2210.15191) for more details.
   * @type {number}
   * @default 0.0
   */
  eta_cutoff = 0;
  /**
   * This value is subtracted from a beam's score if it generates a token same as any beam from other group at a particular time.
   * Note that `diversity_penalty` is only effective if `group beam search` is enabled.
   * @type {number}
   * @default 0.0
   */
  diversity_penalty = 0;
  /**
   * The parameter for repetition penalty. 1.0 means no penalty.
   * See [this paper](https://huggingface.co/papers/1909.05858) for more details.
   * @type {number}
   * @default 1.0
   */
  repetition_penalty = 1;
  /**
   * The paramater for encoder_repetition_penalty.
   * An exponential penalty on sequences that are not in the original input.
   * 1.0 means no penalty.
   * @type {number}
   * @default 1.0
   */
  encoder_repetition_penalty = 1;
  /**
   * Exponential penalty to the length that is used with beam-based generation.
   * It is applied as an exponent to the sequence length, which in turn is used to divide the score of the sequence.
   * Since the score is the log likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while `length_penalty` < 0.0 encourages shorter sequences.
   * @type {number}
   * @default 1.0
   */
  length_penalty = 1;
  /**
   * If set to int > 0, all ngrams of that size can only occur once.
   * @type {number}
   * @default 0
   */
  no_repeat_ngram_size = 0;
  /**
   * List of token ids that are not allowed to be generated.
   * In order to get the token ids of the words that should not appear in the generated text, use
   * `tokenizer(bad_words, { add_prefix_space: true, add_special_tokens: false }).input_ids`.
   * @type {number[][]}
   * @default null
   */
  bad_words_ids = null;
  /**
   * List of token ids that must be generated.
   * If given a `number[][]`, this is treated as a simple list of words that must be included, the opposite to `bad_words_ids`.
   * If given `number[][][]`, this triggers a [disjunctive constraint](https://github.com/huggingface/transformers/issues/14081), where one can allow different forms of each word.
   * @type {number[][]|number[][][]}
   * @default null
   */
  force_words_ids = null;
  /**
   * Whether to renormalize the logits after applying all the logits processors or warpers (including the custom ones).
   * It's highly recommended to set this flag to `true` as the search algorithms suppose the score logits are normalized but some logit processors or warpers break the normalization.
   * @type {boolean}
   * @default false
   */
  renormalize_logits = false;
  /**
   * Custom constraints that can be added to the generation to ensure that the output will contain the use of certain tokens as defined by `Constraint` objects, in the most sensible way possible.
   * @type {Object[]}
   * @default null
   */
  constraints = null;
  /**
   * The id of the token to force as the first generated token after the `decoder_start_token_id`.
   * Useful for multilingual models like mBART where the first generated token needs to be the target language token.
   * @type {number}
   * @default null
   */
  forced_bos_token_id = null;
  /**
   * The id of the token to force as the last generated token when `max_length` is reached.
   * Optionally, use a list to set multiple *end-of-sequence* tokens.
   * @type {number|number[]}
   * @default null
   */
  forced_eos_token_id = null;
  /**
   * Whether to remove possible *nan* and *inf* outputs of the model to prevent the generation method to crash. Note that using `remove_invalid_values` can slow down generation.
   * @type {boolean}
   */
  remove_invalid_values = false;
  /**
   * This Tuple adds an exponentially increasing length penalty, after a certain amount of tokens have been generated.
   * The tuple shall consist of: `(start_index, decay_factor)` where `start_index` indicates where penalty starts and `decay_factor` represents the factor of exponential decay.
   * @type {[number, number]}
   * @default null
   */
  exponential_decay_length_penalty = null;
  /**
   * A list of tokens that will be suppressed at generation.
   * The `SuppressTokens` logit processor will set their log probs to `-inf` so that they are not sampled.
   * @type {number[]}
   * @default null
   */
  suppress_tokens = null;
  /**
   * A streamer that will be used to stream the generation.
   * @type {import('./streamers.js').TextStreamer}
   * @default null
   */
  streamer = null;
  /**
   * A list of tokens that will be suppressed at the beginning of the generation.
   * The `SuppressBeginTokens` logit processor will set their log probs to `-inf` so that they are not sampled.
   * @type {number[]}
   * @default null
   */
  begin_suppress_tokens = null;
  /**
   * A list of pairs of integers which indicates a mapping from generation indices to token indices that will be forced before sampling.
   * For example, `[[1, 123]]` means the second generated token will always be a token of index 123.
   * @type {[number, number][]}
   * @default null
   */
  forced_decoder_ids = null;
  /**
   * The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale > 1`.
   * Higher guidance scale encourages the model to generate samples that are more closely linked to the input
   * prompt, usually at the expense of poorer quality.
   * @type {number}
   * @default null
   */
  guidance_scale = null;
  // Parameters that define the output variables of `generate`
  /**
   * The number of independently computed returned sequences for each element in the batch.
   * @type {number}
   * @default 1
   */
  num_return_sequences = 1;
  /**
   * Whether or not to return the attentions tensors of all attention layers.
   * See `attentions` under returned tensors for more details.
   * @type {boolean}
   * @default false
   */
  output_attentions = false;
  /**
   * Whether or not to return the hidden states of all layers.
   * See `hidden_states` under returned tensors for more details.
   * @type {boolean}
   * @default false
   */
  output_hidden_states = false;
  /**
   * Whether or not to return the prediction scores.
   * See `scores` under returned tensors for more details.
   * @type {boolean}
   * @default false
   */
  output_scores = false;
  /**
   * Whether or not to return a `ModelOutput` instead of a plain tuple.
   * @type {boolean}
   * @default false
   */
  return_dict_in_generate = false;
  // Special tokens that can be used at generation time
  /**
   * The id of the *padding* token.
   * @type {number}
   * @default null
   */
  pad_token_id = null;
  /**
   * The id of the *beginning-of-sequence* token.
   * @type {number}
   * @default null
   */
  bos_token_id = null;
  /**
   * The id of the *end-of-sequence* token.
   * Optionally, use a list to set multiple *end-of-sequence* tokens.
   * @type {number|number[]}
   * @default null
   */
  eos_token_id = null;
  // Generation parameters exclusive to encoder-decoder models
  /**
   * If set to int > 0, all ngrams of that size that occur in the `encoder_input_ids` cannot occur in the `decoder_input_ids`.
   * @type {number}
   * @default 0
   */
  encoder_no_repeat_ngram_size = 0;
  /**
   * If an encoder-decoder model starts decoding with a different token than *bos*, the id of that token.
   * @type {number}
   * @default null
   */
  decoder_start_token_id = null;
  // Wild card
  /**
   * Additional generation kwargs will be forwarded to the `generate` function of the model.
   * Kwargs that are not present in `generate`'s signature will be used in the model forward pass.
   * @type {Object}
   * @default {}
   */
  generation_kwargs = {};
  /**
   * 
   * @param {GenerationConfig|import('../configs.js').PretrainedConfig} config 
   */
  constructor(config) {
    Object.assign(this, pick(config, Object.getOwnPropertyNames(this)));
  }
};

// ignore-modules:sharp
var sharp_default = {};

// src/utils/image.js
var createCanvasFunction;
var ImageDataClass;
var loadImageFunction;
var IS_BROWSER_OR_WEBWORKER = apis.IS_BROWSER_ENV || apis.IS_WEBWORKER_ENV;
if (IS_BROWSER_OR_WEBWORKER) {
  createCanvasFunction = (width, height) => {
    if (!self.OffscreenCanvas) {
      throw new Error("OffscreenCanvas not supported by this browser.");
    }
    return new self.OffscreenCanvas(width, height);
  };
  loadImageFunction = self.createImageBitmap;
  ImageDataClass = self.ImageData;
} else if (sharp_default) {
  loadImageFunction = async (img) => {
    const metadata = await img.metadata();
    const rawChannels = metadata.channels;
    const { data, info } = await img.rotate().raw().toBuffer({ resolveWithObject: true });
    const newImage = new RawImage(new Uint8ClampedArray(data), info.width, info.height, info.channels);
    if (rawChannels !== void 0 && rawChannels !== info.channels) {
      newImage.convert(rawChannels);
    }
    return newImage;
  };
} else {
  throw new Error("Unable to load image processing library.");
}
var RESAMPLING_MAPPING = {
  0: "nearest",
  1: "lanczos",
  2: "bilinear",
  3: "bicubic",
  4: "box",
  5: "hamming"
};
var CONTENT_TYPE_MAP2 = /* @__PURE__ */ new Map([
  ["png", "image/png"],
  ["jpg", "image/jpeg"],
  ["jpeg", "image/jpeg"],
  ["gif", "image/gif"]
]);
var RawImage = class _RawImage {
  /**
   * Create a new `RawImage` object.
   * @param {Uint8ClampedArray|Uint8Array} data The pixel data.
   * @param {number} width The width of the image.
   * @param {number} height The height of the image.
   * @param {1|2|3|4} channels The number of channels.
   */
  constructor(data, width, height, channels) {
    this.data = data;
    this.width = width;
    this.height = height;
    this.channels = channels;
  }
  /**
   * Returns the size of the image (width, height).
   * @returns {[number, number]} The size of the image (width, height).
   */
  get size() {
    return [this.width, this.height];
  }
  /**
   * Helper method for reading an image from a variety of input types.
   * @param {RawImage|string|URL|Blob|HTMLCanvasElement|OffscreenCanvas} input
   * @returns The image object.
   *
   * **Example:** Read image from a URL.
   * ```javascript
   * let image = await RawImage.read('https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/football-match.jpg');
   * // RawImage {
   * //   "data": Uint8ClampedArray [ 25, 25, 25, 19, 19, 19, ... ],
   * //   "width": 800,
   * //   "height": 533,
   * //   "channels": 3
   * // }
   * ```
   */
  static async read(input) {
    if (input instanceof _RawImage) {
      return input;
    } else if (typeof input === "string" || input instanceof URL) {
      return await this.fromURL(input);
    } else if (input instanceof Blob) {
      return await this.fromBlob(input);
    } else if (typeof HTMLCanvasElement !== "undefined" && input instanceof HTMLCanvasElement || typeof OffscreenCanvas !== "undefined" && input instanceof OffscreenCanvas) {
      return this.fromCanvas(input);
    } else {
      throw new Error(`Unsupported input type: ${typeof input}`);
    }
  }
  /**
   * Read an image from a canvas.
   * @param {HTMLCanvasElement|OffscreenCanvas} canvas The canvas to read the image from.
   * @returns {RawImage} The image object.
   */
  static fromCanvas(canvas) {
    if (!IS_BROWSER_OR_WEBWORKER) {
      throw new Error("fromCanvas() is only supported in browser environments.");
    }
    const ctx = (
      /** @type {CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D} */
      canvas.getContext("2d")
    );
    const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    return new _RawImage(data, canvas.width, canvas.height, 4);
  }
  /**
   * Read an image from a URL or file path.
   * @param {string|URL} url The URL or file path to read the image from.
   * @returns {Promise<RawImage>} The image object.
   */
  static async fromURL(url2) {
    const response = await getFile(url2);
    if (response.status !== 200) {
      throw new Error(`Unable to read image from "${url2}" (${response.status} ${response.statusText})`);
    }
    const blob = await response.blob();
    return this.fromBlob(blob);
  }
  /**
   * Helper method to create a new Image from a blob.
   * @param {Blob} blob The blob to read the image from.
   * @returns {Promise<RawImage>} The image object.
   */
  static async fromBlob(blob) {
    if (IS_BROWSER_OR_WEBWORKER) {
      const img = await loadImageFunction(blob);
      const ctx = createCanvasFunction(img.width, img.height).getContext("2d");
      ctx.drawImage(img, 0, 0);
      return new this(ctx.getImageData(0, 0, img.width, img.height).data, img.width, img.height, 4);
    } else {
      const img = sharp_default(await blob.arrayBuffer());
      return await loadImageFunction(img);
    }
  }
  /**
   * Helper method to create a new Image from a tensor
   * @param {Tensor} tensor
   */
  static fromTensor(tensor, channel_format = "CHW") {
    if (tensor.dims.length !== 3) {
      throw new Error(`Tensor should have 3 dimensions, but has ${tensor.dims.length} dimensions.`);
    }
    if (channel_format === "CHW") {
      tensor = tensor.transpose(1, 2, 0);
    } else if (channel_format === "HWC") {
    } else {
      throw new Error(`Unsupported channel format: ${channel_format}`);
    }
    if (!(tensor.data instanceof Uint8ClampedArray || tensor.data instanceof Uint8Array)) {
      throw new Error(`Unsupported tensor type: ${tensor.type}`);
    }
    switch (tensor.dims[2]) {
      case 1:
      case 2:
      case 3:
      case 4:
        return new _RawImage(tensor.data, tensor.dims[1], tensor.dims[0], tensor.dims[2]);
      default:
        throw new Error(`Unsupported number of channels: ${tensor.dims[2]}`);
    }
  }
  /**
   * Convert the image to grayscale format.
   * @returns {RawImage} `this` to support chaining.
   */
  grayscale() {
    if (this.channels === 1) {
      return this;
    }
    const newData = new Uint8ClampedArray(this.width * this.height * 1);
    switch (this.channels) {
      case 3:
      // rgb to grayscale
      case 4:
        for (let i = 0, offset = 0; i < this.data.length; i += this.channels) {
          const red = this.data[i];
          const green = this.data[i + 1];
          const blue = this.data[i + 2];
          newData[offset++] = Math.round(0.2989 * red + 0.587 * green + 0.114 * blue);
        }
        break;
      default:
        throw new Error(`Conversion failed due to unsupported number of channels: ${this.channels}`);
    }
    return this._update(newData, this.width, this.height, 1);
  }
  /**
   * Convert the image to RGB format.
   * @returns {RawImage} `this` to support chaining.
   */
  rgb() {
    if (this.channels === 3) {
      return this;
    }
    const newData = new Uint8ClampedArray(this.width * this.height * 3);
    switch (this.channels) {
      case 1:
        for (let i = 0, offset = 0; i < this.data.length; ++i) {
          newData[offset++] = this.data[i];
          newData[offset++] = this.data[i];
          newData[offset++] = this.data[i];
        }
        break;
      case 4:
        for (let i = 0, offset = 0; i < this.data.length; i += 4) {
          newData[offset++] = this.data[i];
          newData[offset++] = this.data[i + 1];
          newData[offset++] = this.data[i + 2];
        }
        break;
      default:
        throw new Error(`Conversion failed due to unsupported number of channels: ${this.channels}`);
    }
    return this._update(newData, this.width, this.height, 3);
  }
  /**
   * Convert the image to RGBA format.
   * @returns {RawImage} `this` to support chaining.
   */
  rgba() {
    if (this.channels === 4) {
      return this;
    }
    const newData = new Uint8ClampedArray(this.width * this.height * 4);
    switch (this.channels) {
      case 1:
        for (let i = 0, offset = 0; i < this.data.length; ++i) {
          newData[offset++] = this.data[i];
          newData[offset++] = this.data[i];
          newData[offset++] = this.data[i];
          newData[offset++] = 255;
        }
        break;
      case 3:
        for (let i = 0, offset = 0; i < this.data.length; i += 3) {
          newData[offset++] = this.data[i];
          newData[offset++] = this.data[i + 1];
          newData[offset++] = this.data[i + 2];
          newData[offset++] = 255;
        }
        break;
      default:
        throw new Error(`Conversion failed due to unsupported number of channels: ${this.channels}`);
    }
    return this._update(newData, this.width, this.height, 4);
  }
  /**
   * Apply an alpha mask to the image. Operates in place.
   * @param {RawImage} mask The mask to apply. It should have a single channel.
   * @returns {RawImage} The masked image.
   * @throws {Error} If the mask is not the same size as the image.
   * @throws {Error} If the image does not have 4 channels.
   * @throws {Error} If the mask is not a single channel.
   */
  putAlpha(mask) {
    if (mask.width !== this.width || mask.height !== this.height) {
      throw new Error(`Expected mask size to be ${this.width}x${this.height}, but got ${mask.width}x${mask.height}`);
    }
    if (mask.channels !== 1) {
      throw new Error(`Expected mask to have 1 channel, but got ${mask.channels}`);
    }
    const this_data = this.data;
    const mask_data = mask.data;
    const num_pixels = this.width * this.height;
    if (this.channels === 3) {
      const newData = new Uint8ClampedArray(num_pixels * 4);
      for (let i = 0, in_offset = 0, out_offset = 0; i < num_pixels; ++i) {
        newData[out_offset++] = this_data[in_offset++];
        newData[out_offset++] = this_data[in_offset++];
        newData[out_offset++] = this_data[in_offset++];
        newData[out_offset++] = mask_data[i];
      }
      return this._update(newData, this.width, this.height, 4);
    } else if (this.channels === 4) {
      for (let i = 0; i < num_pixels; ++i) {
        this_data[4 * i + 3] = mask_data[i];
      }
      return this;
    }
    throw new Error(`Expected image to have 3 or 4 channels, but got ${this.channels}`);
  }
  /**
   * Resize the image to the given dimensions. This method uses the canvas API to perform the resizing.
   * @param {number} width The width of the new image. `null` or `-1` will preserve the aspect ratio.
   * @param {number} height The height of the new image. `null` or `-1` will preserve the aspect ratio.
   * @param {Object} options Additional options for resizing.
   * @param {0|1|2|3|4|5|string} [options.resample] The resampling method to use.
   * @returns {Promise<RawImage>} `this` to support chaining.
   */
  async resize(width, height, {
    resample = 2
  } = {}) {
    if (this.width === width && this.height === height) {
      return this;
    }
    let resampleMethod = RESAMPLING_MAPPING[resample] ?? resample;
    const nullish_width = isNullishDimension(width);
    const nullish_height = isNullishDimension(height);
    if (nullish_width && nullish_height) {
      return this;
    } else if (nullish_width) {
      width = height / this.height * this.width;
    } else if (nullish_height) {
      height = width / this.width * this.height;
    }
    if (IS_BROWSER_OR_WEBWORKER) {
      const numChannels = this.channels;
      const canvas = this.toCanvas();
      const ctx = createCanvasFunction(width, height).getContext("2d");
      ctx.drawImage(canvas, 0, 0, width, height);
      const resizedImage = new _RawImage(ctx.getImageData(0, 0, width, height).data, width, height, 4);
      return resizedImage.convert(numChannels);
    } else {
      let img = this.toSharp();
      switch (resampleMethod) {
        case "box":
        case "hamming":
          if (resampleMethod === "box" || resampleMethod === "hamming") {
            console.warn(`Resampling method ${resampleMethod} is not yet supported. Using bilinear instead.`);
            resampleMethod = "bilinear";
          }
        case "nearest":
        case "bilinear":
        case "bicubic":
          img = img.affine([width / this.width, 0, 0, height / this.height], {
            interpolator: resampleMethod
          });
          break;
        case "lanczos":
          img = img.resize({
            width,
            height,
            fit: "fill",
            kernel: "lanczos3"
            // PIL Lanczos uses a kernel size of 3
          });
          break;
        default:
          throw new Error(`Resampling method ${resampleMethod} is not supported.`);
      }
      return await loadImageFunction(img);
    }
  }
  async pad([left, right, top, bottom]) {
    left = Math.max(left, 0);
    right = Math.max(right, 0);
    top = Math.max(top, 0);
    bottom = Math.max(bottom, 0);
    if (left === 0 && right === 0 && top === 0 && bottom === 0) {
      return this;
    }
    if (IS_BROWSER_OR_WEBWORKER) {
      const numChannels = this.channels;
      const canvas = this.toCanvas();
      const newWidth = this.width + left + right;
      const newHeight = this.height + top + bottom;
      const ctx = createCanvasFunction(newWidth, newHeight).getContext("2d");
      ctx.drawImage(
        canvas,
        0,
        0,
        this.width,
        this.height,
        left,
        top,
        this.width,
        this.height
      );
      const paddedImage = new _RawImage(
        ctx.getImageData(0, 0, newWidth, newHeight).data,
        newWidth,
        newHeight,
        4
      );
      return paddedImage.convert(numChannels);
    } else {
      const img = this.toSharp().extend({ left, right, top, bottom });
      return await loadImageFunction(img);
    }
  }
  async crop([x_min, y_min, x_max, y_max]) {
    x_min = Math.max(x_min, 0);
    y_min = Math.max(y_min, 0);
    x_max = Math.min(x_max, this.width - 1);
    y_max = Math.min(y_max, this.height - 1);
    if (x_min === 0 && y_min === 0 && x_max === this.width - 1 && y_max === this.height - 1) {
      return this;
    }
    const crop_width = x_max - x_min + 1;
    const crop_height = y_max - y_min + 1;
    if (IS_BROWSER_OR_WEBWORKER) {
      const numChannels = this.channels;
      const canvas = this.toCanvas();
      const ctx = createCanvasFunction(crop_width, crop_height).getContext("2d");
      ctx.drawImage(
        canvas,
        x_min,
        y_min,
        crop_width,
        crop_height,
        0,
        0,
        crop_width,
        crop_height
      );
      const resizedImage = new _RawImage(ctx.getImageData(0, 0, crop_width, crop_height).data, crop_width, crop_height, 4);
      return resizedImage.convert(numChannels);
    } else {
      const img = this.toSharp().extract({
        left: x_min,
        top: y_min,
        width: crop_width,
        height: crop_height
      });
      return await loadImageFunction(img);
    }
  }
  async center_crop(crop_width, crop_height) {
    if (this.width === crop_width && this.height === crop_height) {
      return this;
    }
    const width_offset = (this.width - crop_width) / 2;
    const height_offset = (this.height - crop_height) / 2;
    if (IS_BROWSER_OR_WEBWORKER) {
      const numChannels = this.channels;
      const canvas = this.toCanvas();
      const ctx = createCanvasFunction(crop_width, crop_height).getContext("2d");
      let sourceX = 0;
      let sourceY = 0;
      let destX = 0;
      let destY = 0;
      if (width_offset >= 0) {
        sourceX = width_offset;
      } else {
        destX = -width_offset;
      }
      if (height_offset >= 0) {
        sourceY = height_offset;
      } else {
        destY = -height_offset;
      }
      ctx.drawImage(
        canvas,
        sourceX,
        sourceY,
        crop_width,
        crop_height,
        destX,
        destY,
        crop_width,
        crop_height
      );
      const resizedImage = new _RawImage(ctx.getImageData(0, 0, crop_width, crop_height).data, crop_width, crop_height, 4);
      return resizedImage.convert(numChannels);
    } else {
      let img = this.toSharp();
      if (width_offset >= 0 && height_offset >= 0) {
        img = img.extract({
          left: Math.floor(width_offset),
          top: Math.floor(height_offset),
          width: crop_width,
          height: crop_height
        });
      } else if (width_offset <= 0 && height_offset <= 0) {
        const top = Math.floor(-height_offset);
        const left = Math.floor(-width_offset);
        img = img.extend({
          top,
          left,
          // Ensures the resulting image has the desired dimensions
          right: crop_width - this.width - left,
          bottom: crop_height - this.height - top
        });
      } else {
        let y_padding = [0, 0];
        let y_extract = 0;
        if (height_offset < 0) {
          y_padding[0] = Math.floor(-height_offset);
          y_padding[1] = crop_height - this.height - y_padding[0];
        } else {
          y_extract = Math.floor(height_offset);
        }
        let x_padding = [0, 0];
        let x_extract = 0;
        if (width_offset < 0) {
          x_padding[0] = Math.floor(-width_offset);
          x_padding[1] = crop_width - this.width - x_padding[0];
        } else {
          x_extract = Math.floor(width_offset);
        }
        img = img.extend({
          top: y_padding[0],
          bottom: y_padding[1],
          left: x_padding[0],
          right: x_padding[1]
        }).extract({
          left: x_extract,
          top: y_extract,
          width: crop_width,
          height: crop_height
        });
      }
      return await loadImageFunction(img);
    }
  }
  async toBlob(type = "image/png", quality = 1) {
    if (!IS_BROWSER_OR_WEBWORKER) {
      throw new Error("toBlob() is only supported in browser environments.");
    }
    const canvas = this.toCanvas();
    return await canvas.convertToBlob({ type, quality });
  }
  toTensor(channel_format = "CHW") {
    let tensor = new Tensor2(
      "uint8",
      new Uint8Array(this.data),
      [this.height, this.width, this.channels]
    );
    if (channel_format === "HWC") {
    } else if (channel_format === "CHW") {
      tensor = tensor.permute(2, 0, 1);
    } else {
      throw new Error(`Unsupported channel format: ${channel_format}`);
    }
    return tensor;
  }
  toCanvas() {
    if (!IS_BROWSER_OR_WEBWORKER) {
      throw new Error("toCanvas() is only supported in browser environments.");
    }
    const cloned = this.clone().rgba();
    const clonedCanvas = createCanvasFunction(cloned.width, cloned.height);
    const data = new ImageDataClass(cloned.data, cloned.width, cloned.height);
    clonedCanvas.getContext("2d").putImageData(data, 0, 0);
    return clonedCanvas;
  }
  /**
   * Split this image into individual bands. This method returns an array of individual image bands from an image.
   * For example, splitting an "RGB" image creates three new images each containing a copy of one of the original bands (red, green, blue).
   * 
   * Inspired by PIL's `Image.split()` [function](https://pillow.readthedocs.io/en/latest/reference/Image.html#PIL.Image.Image.split).
   * @returns {RawImage[]} An array containing bands.
   */
  split() {
    const { data, width, height, channels } = this;
    const data_type = (
      /** @type {any} */
      data.constructor
    );
    const per_channel_length = data.length / channels;
    const split_data = Array.from(
      { length: channels },
      () => new data_type(per_channel_length)
    );
    for (let i = 0; i < per_channel_length; ++i) {
      const data_offset = channels * i;
      for (let j = 0; j < channels; ++j) {
        split_data[j][i] = data[data_offset + j];
      }
    }
    return split_data.map((data2) => new _RawImage(data2, width, height, 1));
  }
  /**
   * Helper method to update the image data.
   * @param {Uint8ClampedArray} data The new image data.
   * @param {number} width The new width of the image.
   * @param {number} height The new height of the image.
   * @param {1|2|3|4|null} [channels] The new number of channels of the image.
   * @private
   */
  _update(data, width, height, channels = null) {
    this.data = data;
    this.width = width;
    this.height = height;
    if (channels !== null) {
      this.channels = channels;
    }
    return this;
  }
  /**
   * Clone the image
   * @returns {RawImage} The cloned image
   */
  clone() {
    return new _RawImage(this.data.slice(), this.width, this.height, this.channels);
  }
  /**
   * Helper method for converting image to have a certain number of channels
   * @param {number} numChannels The number of channels. Must be 1, 3, or 4.
   * @returns {RawImage} `this` to support chaining.
   */
  convert(numChannels) {
    if (this.channels === numChannels) return this;
    switch (numChannels) {
      case 1:
        this.grayscale();
        break;
      case 3:
        this.rgb();
        break;
      case 4:
        this.rgba();
        break;
      default:
        throw new Error(`Conversion failed due to unsupported number of channels: ${this.channels}`);
    }
    return this;
  }
  /**
   * Save the image to the given path.
   * @param {string} path The path to save the image to.
   */
  async save(path3) {
    if (IS_BROWSER_OR_WEBWORKER) {
      if (apis.IS_WEBWORKER_ENV) {
        throw new Error("Unable to save an image from a Web Worker.");
      }
      const extension = path3.split(".").pop().toLowerCase();
      const mime = CONTENT_TYPE_MAP2.get(extension) ?? "image/png";
      const blob = await this.toBlob(mime);
      saveBlob(path3, blob);
    } else if (!apis.IS_FS_AVAILABLE) {
      throw new Error("Unable to save the image because filesystem is disabled in this environment.");
    } else {
      const img = this.toSharp();
      return await img.toFile(path3);
    }
  }
  toSharp() {
    if (IS_BROWSER_OR_WEBWORKER) {
      throw new Error("toSharp() is only supported in server-side environments.");
    }
    return sharp_default(this.data, {
      raw: {
        width: this.width,
        height: this.height,
        channels: this.channels
      }
    });
  }
};
var load_image = RawImage.read.bind(RawImage);

// src/generation/stopping_criteria.js
var StoppingCriteria = class extends Callable {
  /**
   * 
   * @param {number[][]} input_ids (`number[][]` of shape `(batch_size, sequence_length)`):
   * Indices of input sequence tokens in the vocabulary.
   * @param {number[][]} scores scores (`number[][]` of shape `(batch_size, config.vocab_size)`):
   * Prediction scores of a language modeling head. These can be scores for each vocabulary token before SoftMax
   * or scores for each vocabulary token after SoftMax.
   * @returns {boolean[]} A list of booleans indicating whether each sequence should be stopped.
   */
  _call(input_ids, scores) {
    throw Error("StoppingCriteria needs to be subclassed");
  }
};
var StoppingCriteriaList = class _StoppingCriteriaList extends Callable {
  /**
   * Constructs a new instance of `StoppingCriteriaList`.
   */
  constructor() {
    super();
    this.criteria = [];
  }
  /**
   * Adds a new stopping criterion to the list.
   *
   * @param {StoppingCriteria} item The stopping criterion to add.
   */
  push(item) {
    this.criteria.push(item);
  }
  /**
   * Adds multiple stopping criteria to the list.
   *
   * @param {StoppingCriteria|StoppingCriteriaList|StoppingCriteria[]} items The stopping criteria to add.
   */
  extend(items) {
    if (items instanceof _StoppingCriteriaList) {
      items = items.criteria;
    } else if (items instanceof StoppingCriteria) {
      items = [items];
    }
    this.criteria.push(...items);
  }
  _call(input_ids, scores) {
    const is_done = new Array(input_ids.length).fill(false);
    for (const criterion of this.criteria) {
      const criterion_done = criterion(input_ids, scores);
      for (let i = 0; i < is_done.length; ++i) {
        is_done[i] ||= criterion_done[i];
      }
    }
    return is_done;
  }
  [Symbol.iterator]() {
    return this.criteria.values();
  }
};
var MaxLengthCriteria = class extends StoppingCriteria {
  /**
   * 
   * @param {number} max_length The maximum length that the output sequence can have in number of tokens.
   * @param {number} [max_position_embeddings=null] The maximum model length, as defined by the model's `config.max_position_embeddings` attribute.
   */
  constructor(max_length, max_position_embeddings = null) {
    super();
    this.max_length = max_length;
    this.max_position_embeddings = max_position_embeddings;
  }
  _call(input_ids) {
    return input_ids.map((ids) => ids.length >= this.max_length);
  }
};
var EosTokenCriteria = class extends StoppingCriteria {
  /**
   * 
   * @param {number|number[]} eos_token_id The id of the *end-of-sequence* token.
   * Optionally, use a list to set multiple *end-of-sequence* tokens.
   */
  constructor(eos_token_id) {
    super();
    if (!Array.isArray(eos_token_id)) {
      eos_token_id = [eos_token_id];
    }
    this.eos_token_id = eos_token_id;
  }
  /**
   * 
   * @param {number[][]} input_ids 
   * @param {number[][]} scores 
   * @returns {boolean[]}
   */
  _call(input_ids, scores) {
    return input_ids.map((ids) => {
      const last = ids.at(-1);
      return this.eos_token_id.some((eos_id) => last == eos_id);
    });
  }
};
var InterruptableStoppingCriteria = class extends StoppingCriteria {
  constructor() {
    super();
    this.interrupted = false;
  }
  interrupt() {
    this.interrupted = true;
  }
  reset() {
    this.interrupted = false;
  }
  _call(input_ids, scores) {
    return new Array(input_ids.length).fill(this.interrupted);
  }
};

// src/generation/logits_sampler.js
var LogitsSampler = class extends Callable {
  /**
   * Creates a new Sampler object with the specified generation config.
   * @param {GenerationConfig} generation_config The generation config.
   */
  constructor(generation_config) {
    super();
    this.generation_config = generation_config;
  }
  /**
   * Executes the sampler, using the specified logits.
   * @param {Tensor} logits
   * @returns {Promise<[bigint, number][]>}
   */
  async _call(logits) {
    return this.sample(logits);
  }
  /**
   * Abstract method for sampling the logits.
   * @param {Tensor} logits
   * @throws {Error} If not implemented in subclass.
   * @returns {Promise<[bigint, number][]>}
   */
  async sample(logits) {
    throw Error("sample should be implemented in subclasses.");
  }
  /**
   * Returns the specified logits as an array, with temperature applied.
   * @param {Tensor} logits
   * @param {number} index
   * @returns {Float32Array}
   */
  getLogits(logits, index) {
    let vocabSize = logits.dims.at(-1);
    let logs = (
      /** @type {Float32Array} */
      logits.data
    );
    if (index === -1) {
      logs = logs.slice(-vocabSize);
    } else {
      let startIndex = index * vocabSize;
      logs = logs.slice(startIndex, startIndex + vocabSize);
    }
    return logs;
  }
  /**
   * Selects an item randomly based on the specified probabilities.
   * @param {import("../transformers.js").DataArray} probabilities An array of probabilities to use for selection.
   * @returns {number} The index of the selected item.
   */
  randomSelect(probabilities) {
    let sumProbabilities = 0;
    for (let i = 0; i < probabilities.length; ++i) {
      sumProbabilities += probabilities[i];
    }
    let r = Math.random() * sumProbabilities;
    for (let i = 0; i < probabilities.length; ++i) {
      r -= probabilities[i];
      if (r <= 0) {
        return i;
      }
    }
    return 0;
  }
  /**
   * Returns a Sampler object based on the specified options.
   * @param {GenerationConfig} generation_config An object containing options for the sampler.
   * @returns {LogitsSampler} A Sampler object.
   */
  static getSampler(generation_config) {
    if (generation_config.do_sample) {
      return new MultinomialSampler(generation_config);
    } else if (generation_config.num_beams > 1) {
      return new BeamSearchSampler(generation_config);
    } else {
      if (generation_config.num_return_sequences > 1) {
        throw Error(`num_return_sequences has to be 1 when doing greedy search, but is ${generation_config.num_return_sequences}.`);
      }
      return new GreedySampler(generation_config);
    }
  }
};
var GreedySampler = class extends LogitsSampler {
  /**
   * Sample the maximum probability of a given logits tensor.
   * @param {Tensor} logits
   * @returns {Promise<[bigint, number][]>} An array with a single tuple, containing the index of the maximum value and a meaningless score (since this is a greedy search).
   */
  async sample(logits) {
    const argmax = max(logits.data)[1];
    return [
      [BigInt(argmax), 0]
    ];
  }
};
var MultinomialSampler = class extends LogitsSampler {
  /**
   * Sample from the logits.
   * @param {Tensor} logits
   * @returns {Promise<[bigint, number][]>}
   */
  async sample(logits) {
    let k = logits.dims.at(-1);
    if (this.generation_config.top_k > 0) {
      k = Math.min(this.generation_config.top_k, k);
    }
    const [v, i] = await topk(logits, k);
    const probabilities = softmax(
      /** @type {Float32Array} */
      v.data
    );
    return Array.from({ length: this.generation_config.num_beams }, () => {
      const sampledIndex = this.randomSelect(probabilities);
      return [
        i.data[sampledIndex],
        // token id
        Math.log(probabilities[sampledIndex])
        // score
      ];
    });
  }
};
var BeamSearchSampler = class extends LogitsSampler {
  /**
   * Sample from the logits.
   * @param {Tensor} logits
   * @returns {Promise<[bigint, number][]>}
   */
  async sample(logits) {
    let k = logits.dims.at(-1);
    if (this.generation_config.top_k > 0) {
      k = Math.min(this.generation_config.top_k, k);
    }
    const [v, i] = await topk(logits, k);
    const probabilities = softmax(
      /** @type {Float32Array} */
      v.data
    );
    return Array.from({ length: this.generation_config.num_beams }, (_, x) => {
      return [
        i.data[x],
        // token id
        Math.log(probabilities[x])
        // score
      ];
    });
  }
};

// src/models/whisper/generation_whisper.js
var WhisperGenerationConfig = class extends GenerationConfig {
  /**
   * Whether to return the timestamps with the text. This enables the `WhisperTimestampsLogitsProcessor`.
   * @type {boolean}
   */
  return_timestamps = null;
  /**
   * Whether to return token-level timestamps
   * with the text. This can be used with or without the `return_timestamps` option. To get word-level
   * timestamps, use the tokenizer to group the tokens into words.
   * @type {boolean}
   */
  return_token_timestamps = null;
  /**
   * The number of audio frames available in this chunk. This is only used generating word-level timestamps.
   * @type {number}
   */
  num_frames = null;
  /**
   * Alignment heads to predict word-level timestamps. This is a list of [layer, head] pairs that
   * select the cross-attention heads that are highly correlated to word-level timing.
   * @type {[number, number][]}
   */
  alignment_heads = null;
  /**
   * Task to use for generation, either "translate" or "transcribe".
   * @type {string}
   */
  task = null;
  /**
   * Language token to use for generation, can be either in the form of `<|en|>`, `en` or `english`.
   * You can find all the possible language tokens in the `model.generation_config.lang_to_id` dictionary.
   * @type {string}
   */
  language = null;
  /**
   * The id of the `"<|notimestamps|>"` token.
   * @type {number}
   */
  no_timestamps_token_id = null;
  /**
   * Rank-1 list of token IDs created by passing text to [`~WhisperProcessor.get_prompt_ids`] that is
   * provided as a prompt to each chunk. This can be used to provide or "prompt-engineer" a context for
   * transcription, e.g. custom vocabularies or proper nouns to make it more likely to predict those words
   * correctly. It cannot be used in conjunction with `decoder_start_token_id` as it overwrites this value.
   * @type {number[]}
   */
  prompt_ids = null;
  /**
   * Whether the model is multilingual or not.
   * @type {boolean}
   */
  is_multilingual = null;
  /**
   * (Optional) A mapping from language tokens to their corresponding IDs.
   * Only required if the model is multilingual.
   * @type {Record<string, number>|null}
   */
  lang_to_id = null;
  /**
   * (Optional) A mapping from task tokens to their corresponding IDs.
   * @type {Record<string, number>|null}
   */
  task_to_id = null;
  /**
   * Used to set the maximum value of the initial timestamp. This is used to prevent the model from
   * predicting timestamps that are too far in the future.
   * @type {number}
   */
  max_initial_timestamp_index = 1;
};

// src/models.js
var MODEL_TYPES = {
  EncoderOnly: 0,
  EncoderDecoder: 1,
  Seq2Seq: 2,
  Vision2Seq: 3,
  DecoderOnly: 4,
  MaskGeneration: 5,
  ImageTextToText: 6,
  Musicgen: 7,
  MultiModality: 8,
  Phi3V: 9,
  AudioTextToText: 10,
  AutoEncoder: 11,
  ImageAudioTextToText: 12
};
var MODEL_TYPE_MAPPING = /* @__PURE__ */ new Map();
var MODEL_NAME_TO_CLASS_MAPPING = /* @__PURE__ */ new Map();
var MODEL_CLASS_TO_NAME_MAPPING = /* @__PURE__ */ new Map();
async function getSession(pretrained_model_name_or_path, fileName, options) {
  let custom_config = options.config?.["transformers.js_config"] ?? {};
  let device = options.device ?? custom_config.device;
  if (device && typeof device !== "string") {
    if (device.hasOwnProperty(fileName)) {
      device = device[fileName];
    } else {
      console.warn(`device not specified for "${fileName}". Using the default device.`);
      device = null;
    }
  }
  const selectedDevice = (
    /** @type {import("./utils/devices.js").DeviceType} */
    device ?? (apis.IS_NODE_ENV ? "cpu" : "wasm")
  );
  const executionProviders = deviceToExecutionProviders(selectedDevice);
  const device_config = custom_config.device_config ?? {};
  if (device_config.hasOwnProperty(selectedDevice)) {
    custom_config = {
      ...custom_config,
      ...device_config[selectedDevice]
    };
  }
  let dtype = options.dtype ?? custom_config.dtype;
  if (typeof dtype !== "string") {
    if (dtype && dtype.hasOwnProperty(fileName)) {
      dtype = dtype[fileName];
    } else {
      dtype = DEFAULT_DEVICE_DTYPE_MAPPING[selectedDevice] ?? DATA_TYPES.fp32;
      console.warn(`dtype not specified for "${fileName}". Using the default dtype (${dtype}) for this device (${selectedDevice}).`);
    }
  }
  if (dtype === DATA_TYPES.auto) {
    let config_dtype = custom_config.dtype;
    if (typeof config_dtype !== "string") {
      config_dtype = config_dtype?.[fileName];
    }
    if (config_dtype && config_dtype !== DATA_TYPES.auto && DATA_TYPES.hasOwnProperty(config_dtype)) {
      dtype = config_dtype;
    } else {
      dtype = DEFAULT_DEVICE_DTYPE_MAPPING[selectedDevice] ?? DATA_TYPES.fp32;
    }
  }
  const selectedDtype = (
    /** @type {import("./utils/dtypes.js").DataType} */
    dtype
  );
  if (!DEFAULT_DTYPE_SUFFIX_MAPPING.hasOwnProperty(selectedDtype)) {
    throw new Error(`Invalid dtype: ${selectedDtype}. Should be one of: ${Object.keys(DATA_TYPES).join(", ")}`);
  } else if (selectedDtype === DATA_TYPES.fp16 && selectedDevice === "webgpu" && !await isWebGpuFp16Supported()) {
    throw new Error(`The device (${selectedDevice}) does not support fp16.`);
  }
  const kv_cache_dtype_config = custom_config.kv_cache_dtype;
  const kv_cache_dtype = kv_cache_dtype_config ? typeof kv_cache_dtype_config === "string" ? kv_cache_dtype_config : kv_cache_dtype_config[selectedDtype] ?? "float32" : void 0;
  if (kv_cache_dtype && !["float32", "float16"].includes(kv_cache_dtype)) {
    throw new Error(`Invalid kv_cache_dtype: ${kv_cache_dtype}. Should be one of: float32, float16`);
  }
  const session_config = {
    dtype: selectedDtype,
    kv_cache_dtype,
    device: selectedDevice
  };
  const suffix = DEFAULT_DTYPE_SUFFIX_MAPPING[selectedDtype];
  const baseName = `${fileName}${suffix}.onnx`;
  const modelFileName = `${options.subfolder ?? ""}/${baseName}`;
  const session_options = { ...options.session_options };
  session_options.executionProviders ??= executionProviders;
  const free_dimension_overrides = custom_config.free_dimension_overrides;
  if (free_dimension_overrides) {
    session_options.freeDimensionOverrides ??= free_dimension_overrides;
  } else if (selectedDevice.startsWith("webnn") && !session_options.freeDimensionOverrides) {
    console.warn(
      `WebNN does not currently support dynamic shapes and requires 'free_dimension_overrides' to be set in config.json, preferably as a field within config["transformers.js_config"]["device_config"]["${selectedDevice}"]. When 'free_dimension_overrides' is not set, you may experience significant performance degradation.`
    );
  }
  const return_path = apis.IS_NODE_ENV && env.useFSCache;
  const bufferOrPathPromise = getModelFile(pretrained_model_name_or_path, modelFileName, true, options, return_path);
  const use_external_data_format = options.use_external_data_format ?? custom_config.use_external_data_format;
  let externalDataPromises = [];
  if (use_external_data_format) {
    let external_data_format;
    if (typeof use_external_data_format === "object") {
      if (use_external_data_format.hasOwnProperty(baseName)) {
        external_data_format = use_external_data_format[baseName];
      } else if (use_external_data_format.hasOwnProperty(fileName)) {
        external_data_format = use_external_data_format[fileName];
      } else {
        external_data_format = false;
      }
    } else {
      external_data_format = use_external_data_format;
    }
    const num_chunks = +external_data_format;
    if (num_chunks > MAX_EXTERNAL_DATA_CHUNKS) {
      throw new Error(`The number of external data chunks (${num_chunks}) exceeds the maximum allowed value (${MAX_EXTERNAL_DATA_CHUNKS}).`);
    }
    for (let i = 0; i < num_chunks; ++i) {
      const path3 = `${baseName}_data${i === 0 ? "" : "_" + i}`;
      const fullPath = `${options.subfolder ?? ""}/${path3}`;
      externalDataPromises.push(new Promise(async (resolve, reject) => {
        const data = await getModelFile(pretrained_model_name_or_path, fullPath, true, options, return_path);
        resolve(data instanceof Uint8Array ? { path: path3, data } : path3);
      }));
    }
  } else if (session_options.externalData !== void 0) {
    externalDataPromises = session_options.externalData.map(async (ext) => {
      if (typeof ext.data === "string") {
        const ext_buffer = await getModelFile(pretrained_model_name_or_path, ext.data, true, options);
        return { ...ext, data: ext_buffer };
      }
      return ext;
    });
  }
  if (externalDataPromises.length > 0) {
    const externalData = await Promise.all(externalDataPromises);
    if (!apis.IS_NODE_ENV) {
      session_options.externalData = externalData;
    }
  }
  if (selectedDevice === "webgpu") {
    const shapes = getCacheShapes(options.config, {
      prefix: "present"
    });
    if (Object.keys(shapes).length > 0 && !isONNXProxy()) {
      const preferredOutputLocation = {};
      for (const key in shapes) {
        preferredOutputLocation[key] = "gpu-buffer";
      }
      session_options.preferredOutputLocation = preferredOutputLocation;
    }
  }
  const buffer_or_path = await bufferOrPathPromise;
  return { buffer_or_path, session_options, session_config };
}
async function constructSessions(pretrained_model_name_or_path, names, options) {
  return Object.fromEntries(await Promise.all(
    Object.keys(names).map(async (name) => {
      const { buffer_or_path, session_options, session_config } = await getSession(pretrained_model_name_or_path, names[name], options);
      const session = await createInferenceSession(buffer_or_path, session_options, session_config);
      return [name, session];
    })
  ));
}
async function getOptionalConfigs(pretrained_model_name_or_path, names, options) {
  return Object.fromEntries(await Promise.all(
    Object.keys(names).map(async (name) => {
      const config = await getModelJSON(pretrained_model_name_or_path, names[name], false, options);
      return [name, config];
    })
  ));
}
function validateInputs(session, inputs) {
  const checkedInputs = /* @__PURE__ */ Object.create(null);
  const missingInputs = [];
  for (const inputName of session.inputNames) {
    const tensor = inputs[inputName];
    if (!(tensor instanceof Tensor2)) {
      missingInputs.push(inputName);
      continue;
    }
    checkedInputs[inputName] = isONNXProxy() ? tensor.clone() : tensor;
  }
  if (missingInputs.length > 0) {
    throw new Error(
      `An error occurred during model execution: "Missing the following inputs: ${missingInputs.join(", ")}.`
    );
  }
  const numInputsProvided = Object.keys(inputs).length;
  const numInputsNeeded = session.inputNames.length;
  if (numInputsProvided > numInputsNeeded) {
    let ignored = Object.keys(inputs).filter((inputName) => !session.inputNames.includes(inputName));
    console.warn(`WARNING: Too many inputs were provided (${numInputsProvided} > ${numInputsNeeded}). The following inputs will be ignored: "${ignored.join(", ")}".`);
  }
  return checkedInputs;
}
async function sessionRun(session, inputs) {
  const checkedInputs = validateInputs(session, inputs);
  try {
    const ortFeed = Object.fromEntries(Object.entries(checkedInputs).map(([k, v]) => [k, v.ort_tensor]));
    const output = await runInferenceSession(session, ortFeed);
    return replaceTensors(output);
  } catch (e) {
    const formatted = Object.fromEntries(Object.entries(checkedInputs).map(([k, tensor]) => {
      const unpacked = {
        type: tensor.type,
        dims: tensor.dims,
        location: tensor.location
      };
      if (unpacked.location !== "gpu-buffer") {
        unpacked.data = tensor.data;
      }
      return [k, unpacked];
    }));
    console.error(`An error occurred during model execution: "${e}".`);
    console.error("Inputs given to model:", formatted);
    throw e;
  }
}
function replaceTensors(obj) {
  for (let prop in obj) {
    if (isONNXTensor(obj[prop])) {
      obj[prop] = new Tensor2(obj[prop]);
    } else if (typeof obj[prop] === "object") {
      replaceTensors(obj[prop]);
    }
  }
  return obj;
}
function toI64Tensor(items) {
  if (items instanceof Tensor2) {
    return items;
  }
  if (items.length === 0) {
    throw Error("items must be non-empty");
  }
  if (Array.isArray(items[0])) {
    if (items.some((x) => x.length !== items[0].length)) {
      throw Error("Unable to create tensor, you should probably activate truncation and/or padding with 'padding=True' and/or 'truncation=True' to have batched tensors with the same length.");
    }
    return new Tensor2(
      "int64",
      BigInt64Array.from(items.flat().map((x) => BigInt(x))),
      [items.length, items[0].length]
    );
  } else {
    return new Tensor2(
      "int64",
      BigInt64Array.from(items.map((x) => BigInt(x))),
      [1, items.length]
    );
  }
}
function boolTensor(value) {
  return new Tensor2("bool", [value], [1]);
}
async function seq2seqForward(self2, model_inputs) {
  let { encoder_outputs, input_ids, decoder_input_ids, ...other_decoder_inputs } = model_inputs;
  if (!encoder_outputs) {
    const encoder_inputs = pick(model_inputs, self2.sessions["model"].inputNames);
    encoder_outputs = (await encoderForward(self2, encoder_inputs)).last_hidden_state;
  }
  other_decoder_inputs.input_ids = decoder_input_ids;
  other_decoder_inputs.encoder_hidden_states = encoder_outputs;
  if (self2.sessions["decoder_model_merged"].inputNames.includes("encoder_attention_mask")) {
    other_decoder_inputs.encoder_attention_mask = model_inputs.attention_mask;
  }
  const decoderResults = await decoderForward(self2, other_decoder_inputs, true);
  return decoderResults;
}
async function encoderForward(self2, model_inputs) {
  const session = self2.sessions["model"];
  const encoderFeeds = pick(model_inputs, session.inputNames);
  if (session.inputNames.includes("inputs_embeds") && !encoderFeeds.inputs_embeds) {
    if (!model_inputs.input_ids) {
      throw new Error("Both `input_ids` and `inputs_embeds` are missing in the model inputs.");
    }
    encoderFeeds.inputs_embeds = await self2.encode_text({ input_ids: model_inputs.input_ids });
  }
  if (session.inputNames.includes("token_type_ids") && !encoderFeeds.token_type_ids) {
    if (!encoderFeeds.input_ids) {
      throw new Error("Both `input_ids` and `token_type_ids` are missing in the model inputs.");
    }
    encoderFeeds.token_type_ids = zeros_like(encoderFeeds.input_ids);
  }
  if (session.inputNames.includes("pixel_mask") && !encoderFeeds.pixel_mask) {
    if (!encoderFeeds.pixel_values) {
      throw new Error("Both `pixel_values` and `pixel_mask` are missing in the model inputs.");
    }
    const dims = encoderFeeds.pixel_values.dims;
    encoderFeeds.pixel_mask = ones([dims[0], dims[2], dims[3]]);
  }
  return await sessionRun(session, encoderFeeds);
}
async function autoEncoderForward(self2, model_inputs) {
  const encoded = await self2.encode(model_inputs);
  const decoded = await self2.decode(encoded);
  return decoded;
}
async function decoderForward(self2, model_inputs, is_encoder_decoder = false) {
  const session = self2.sessions[is_encoder_decoder ? "decoder_model_merged" : "model"];
  const { past_key_values, ...new_model_inputs } = model_inputs;
  if (session.inputNames.includes("use_cache_branch")) {
    new_model_inputs.use_cache_branch = boolTensor(!!past_key_values);
  }
  if (session.inputNames.includes("position_ids") && new_model_inputs.attention_mask && !new_model_inputs.position_ids) {
    const start_index = ["paligemma", "gemma3_text", "gemma3"].includes(self2.config.model_type) ? 1 : 0;
    new_model_inputs.position_ids = createPositionIds(new_model_inputs, past_key_values, start_index);
  }
  self2.addPastKeyValues(new_model_inputs, past_key_values);
  const fixed = pick(new_model_inputs, session.inputNames);
  return await sessionRun(session, fixed);
}
function default_merge_input_ids_with_features({
  modality_token_id,
  inputs_embeds,
  modality_features,
  input_ids,
  attention_mask
}) {
  const token_positions = input_ids.tolist().map(
    (ids) => ids.reduce((acc, x, idx) => {
      if (x == modality_token_id) acc.push(idx);
      return acc;
    }, [])
  );
  const n_tokens = token_positions.reduce((acc, x) => acc + x.length, 0);
  const n_features = modality_features.dims[0];
  if (n_tokens !== n_features) {
    throw new Error(`Number of tokens and features do not match: tokens: ${n_tokens}, features ${n_features}`);
  }
  let img = 0;
  for (let i = 0; i < token_positions.length; ++i) {
    const tokens = token_positions[i];
    const embeds = inputs_embeds[i];
    for (let j = 0; j < tokens.length; ++j) {
      embeds[tokens[j]].data.set(modality_features[img++].data);
    }
  }
  return { inputs_embeds, attention_mask };
}
function default_merge_input_ids_with_image_features({
  image_token_id,
  inputs_embeds,
  image_features,
  input_ids,
  attention_mask
}) {
  return default_merge_input_ids_with_features({
    modality_token_id: image_token_id,
    inputs_embeds,
    modality_features: image_features,
    input_ids,
    attention_mask
  });
}
function default_merge_input_ids_with_audio_features({
  audio_token_id,
  inputs_embeds,
  audio_features,
  input_ids,
  attention_mask
}) {
  return default_merge_input_ids_with_features({
    modality_token_id: audio_token_id,
    inputs_embeds,
    modality_features: audio_features,
    input_ids,
    attention_mask
  });
}
async function genericTextToTextForward(self2, {
  // Generic parameters:
  encode_function,
  merge_function,
  modality_input_name,
  modality_output_name,
  // Produced by the tokenizer/processor:
  input_ids = null,
  attention_mask = null,
  // Used during generation:
  position_ids = null,
  inputs_embeds = null,
  past_key_values = null,
  // Generic generation parameters
  generation_config = null,
  logits_processor = null,
  // Additional parameters
  ...kwargs
}) {
  const modality_values = kwargs[modality_input_name];
  if (!inputs_embeds) {
    inputs_embeds = await self2.encode_text({ input_ids, ...kwargs });
    if (modality_values && input_ids.dims[1] !== 1) {
      const modality_features = await encode_function({
        // Pass the modality values under its expected key.
        // The caller knows whether this is audio or image.
        [modality_input_name]: modality_values,
        ...kwargs
      });
      ({ inputs_embeds, attention_mask } = merge_function({
        [modality_output_name]: modality_features,
        inputs_embeds,
        input_ids,
        attention_mask
      }));
    } else if (past_key_values && modality_values && input_ids.dims[1] === 1) {
      const target_length = input_ids.dims[1];
      const past_length = Object.values(past_key_values)[0].dims.at(-2);
      attention_mask = cat([
        ones([input_ids.dims[0], past_length]),
        attention_mask.slice(null, [attention_mask.dims[1] - target_length, attention_mask.dims[1]])
      ], 1);
    }
  }
  if (!position_ids) {
    if (self2.config.model_type === "qwen2_vl") {
      const { image_grid_thw, video_grid_thw } = kwargs;
      [position_ids] = self2.get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask);
    }
  }
  const outputs = await decoderForward(self2, {
    inputs_embeds,
    past_key_values,
    attention_mask,
    position_ids,
    generation_config,
    logits_processor
  }, true);
  return outputs;
}
async function audioTextToTextForward(self2, params) {
  return await genericTextToTextForward(self2, {
    ...params,
    modality_input_name: "audio_values",
    modality_output_name: "audio_features",
    encode_function: self2.encode_audio.bind(self2),
    merge_function: self2._merge_input_ids_with_audio_features.bind(self2)
  });
}
async function imageTextToTextForward(self2, params) {
  return await genericTextToTextForward(self2, {
    ...params,
    modality_input_name: "pixel_values",
    modality_output_name: "image_features",
    encode_function: self2.encode_image.bind(self2),
    merge_function: self2._merge_input_ids_with_image_features.bind(self2)
  });
}
function cumsum_masked_fill(attention_mask, start_index = 0) {
  const [bz, seq_len] = attention_mask.dims;
  const attn_mask_data = attention_mask.data;
  const data = new BigInt64Array(attn_mask_data.length);
  for (let i = 0; i < bz; ++i) {
    const start = i * seq_len;
    let sum = BigInt(start_index);
    for (let j = 0; j < seq_len; ++j) {
      const index = start + j;
      if (attn_mask_data[index] === 0n) {
        data[index] = BigInt(1);
      } else {
        data[index] = sum;
        sum += attn_mask_data[index];
      }
    }
  }
  return { data, dims: attention_mask.dims };
}
function createPositionIds(model_inputs, past_key_values = null, start_index = 0) {
  const { input_ids, inputs_embeds, attention_mask } = model_inputs;
  const { data, dims } = cumsum_masked_fill(attention_mask, start_index);
  let position_ids = new Tensor2("int64", data, dims);
  if (past_key_values) {
    const offset = -(input_ids ?? inputs_embeds).dims.at(1);
    position_ids = position_ids.slice(null, [offset, null]);
  }
  return position_ids;
}
function decoder_prepare_inputs_for_generation(self2, input_ids, model_inputs, generation_config) {
  const past_length = model_inputs.past_key_values ? Object.values(model_inputs.past_key_values)[0].dims.at(-2) : 0;
  if (!model_inputs.attention_mask) {
    let dims;
    for (const key of ["input_ids", "inputs_embeds", "position_ids"]) {
      if (model_inputs[key]) {
        dims = model_inputs[key].dims;
        break;
      }
    }
    if (!dims) {
      throw new Error("attention_mask is not provided, and unable to infer its shape from model inputs.");
    }
    model_inputs.attention_mask = ones([dims[0], past_length + dims[1]]);
  }
  if (model_inputs.past_key_values) {
    const { input_ids: input_ids2, attention_mask } = model_inputs;
    if (attention_mask && attention_mask.dims[1] > input_ids2.dims[1]) {
    } else if (past_length < input_ids2.dims[1]) {
      model_inputs.input_ids = input_ids2.slice(null, [past_length, null]);
    } else {
    }
  }
  return model_inputs;
}
function encoder_decoder_prepare_inputs_for_generation(self2, input_ids, model_inputs, generation_config) {
  if (model_inputs.past_key_values) {
    input_ids = input_ids.map((x) => [x.at(-1)]);
  }
  return {
    ...model_inputs,
    decoder_input_ids: toI64Tensor(input_ids)
  };
}
function multimodal_text_to_text_prepare_inputs_for_generation(self2, ...args) {
  if (self2.config.is_encoder_decoder) {
    return encoder_decoder_prepare_inputs_for_generation(self2, ...args);
  } else {
    return decoder_prepare_inputs_for_generation(self2, ...args);
  }
}
function multimodality_prepare_inputs_for_generation(self2, input_ids, model_inputs, generation_config) {
  const has_past_key_values = !!model_inputs.past_key_values;
  if (generation_config.guidance_scale !== null && generation_config.guidance_scale > 1) {
    if (has_past_key_values) {
      model_inputs.input_ids = cat([
        model_inputs.input_ids,
        model_inputs.input_ids
      ], 0);
    } else {
      model_inputs.input_ids = cat([
        model_inputs.input_ids,
        full_like(model_inputs.input_ids, BigInt(generation_config.pad_token_id))
      ], 0);
      model_inputs.attention_mask = cat([
        model_inputs.attention_mask,
        full_like(model_inputs.attention_mask, 0n)
      ], 0);
    }
  }
  if (has_past_key_values || !model_inputs.pixel_values) {
    model_inputs.pixel_values = full([0, 0, 3, 384, 384], 1);
  }
  if (has_past_key_values) {
    const num_img_tokens = 0;
    const num_text_tokens = 1;
    const has_image = num_img_tokens > 0 ? 1 : 0;
    const batch_size = 1;
    model_inputs.images_seq_mask = new Tensor2(
      "bool",
      new Array(num_img_tokens + num_text_tokens).fill(true).fill(false, 0, num_text_tokens),
      [batch_size, num_img_tokens + num_text_tokens]
    );
    model_inputs.images_emb_mask = new Tensor2(
      "bool",
      new Array(num_img_tokens).fill(!!has_image),
      [batch_size, 1, num_img_tokens]
    );
  }
  return model_inputs;
}
var PreTrainedModel = class extends Callable {
  main_input_name = "input_ids";
  forward_params = ["input_ids", "attention_mask"];
  /**
   * Creates a new instance of the `PreTrainedModel` class.
   * @param {import('./configs.js').PretrainedConfig} config The model configuration.
   * @param {Record<string, any>} sessions The inference sessions for the model.
   * @param {Record<string, Object>} configs Additional configuration files (e.g., generation_config.json).
   */
  constructor(config, sessions, configs) {
    super();
    this.config = config;
    this.sessions = sessions;
    this.configs = configs;
    const modelName = MODEL_CLASS_TO_NAME_MAPPING.get(this.constructor);
    const modelType = MODEL_TYPE_MAPPING.get(modelName);
    this.can_generate = false;
    this._forward = null;
    this._prepare_inputs_for_generation = null;
    switch (modelType) {
      case MODEL_TYPES.DecoderOnly:
        this.can_generate = true;
        this._forward = decoderForward;
        this._prepare_inputs_for_generation = decoder_prepare_inputs_for_generation;
        break;
      case MODEL_TYPES.Seq2Seq:
      case MODEL_TYPES.Vision2Seq:
      case MODEL_TYPES.Musicgen:
        this.can_generate = true;
        this._forward = seq2seqForward;
        this._prepare_inputs_for_generation = encoder_decoder_prepare_inputs_for_generation;
        break;
      case MODEL_TYPES.EncoderDecoder:
        this._forward = seq2seqForward;
        break;
      case MODEL_TYPES.ImageTextToText:
        this.can_generate = true;
        this._forward = imageTextToTextForward;
        this._prepare_inputs_for_generation = multimodal_text_to_text_prepare_inputs_for_generation;
        break;
      case MODEL_TYPES.AudioTextToText:
        this.can_generate = true;
        this._forward = audioTextToTextForward;
        this._prepare_inputs_for_generation = multimodal_text_to_text_prepare_inputs_for_generation;
        break;
      case MODEL_TYPES.Phi3V:
      case MODEL_TYPES.ImageAudioTextToText:
        this.can_generate = true;
        this._prepare_inputs_for_generation = multimodal_text_to_text_prepare_inputs_for_generation;
        break;
      case MODEL_TYPES.MultiModality:
        this.can_generate = true;
        this._prepare_inputs_for_generation = multimodality_prepare_inputs_for_generation;
        break;
      case MODEL_TYPES.AutoEncoder:
        this._forward = autoEncoderForward;
        break;
      default:
        this._forward = encoderForward;
        break;
    }
    if (this.can_generate) {
      this.forward_params.push("past_key_values");
    }
    this.custom_config = this.config["transformers.js_config"] ?? {};
  }
  /**
  * Disposes of all the ONNX sessions that were created during inference.
  * @returns {Promise<unknown[]>} An array of promises, one for each ONNX session that is being disposed.
  * @todo Use https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/FinalizationRegistry
  */
  async dispose() {
    const promises = [];
    for (const session of Object.values(this.sessions)) {
      if (session?.handler?.dispose) {
        promises.push(session.handler.dispose());
      }
    }
    return await Promise.all(promises);
  }
  /**
   * Instantiate one of the model classes of the library from a pretrained model.
   * 
   * The model class to instantiate is selected based on the `model_type` property of the config object
   * (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible)
   * 
   * @param {string} pretrained_model_name_or_path The name or path of the pretrained model. Can be either:
   * - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
   *   Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
   *   user or organization name, like `dbmdz/bert-base-german-cased`.
   * - A path to a *directory* containing model weights, e.g., `./my_model_directory/`.
   * @param {import('./utils/hub.js').PretrainedModelOptions} options Additional options for loading the model.
   * 
   * @returns {Promise<PreTrainedModel>} A new instance of the `PreTrainedModel` class.
   */
  static async from_pretrained(pretrained_model_name_or_path, {
    progress_callback = null,
    config = null,
    cache_dir = null,
    local_files_only = false,
    revision = "main",
    model_file_name = null,
    subfolder = "onnx",
    device = null,
    dtype = null,
    use_external_data_format = null,
    session_options = {}
  } = {}) {
    let options = {
      progress_callback,
      config,
      cache_dir,
      local_files_only,
      revision,
      model_file_name,
      subfolder,
      device,
      dtype,
      use_external_data_format,
      session_options
    };
    const modelName = MODEL_CLASS_TO_NAME_MAPPING.get(this);
    const modelType = MODEL_TYPE_MAPPING.get(modelName);
    config = options.config = await AutoConfig.from_pretrained(pretrained_model_name_or_path, options);
    let info;
    if (modelType === MODEL_TYPES.DecoderOnly) {
      info = await Promise.all([
        constructSessions(pretrained_model_name_or_path, {
          model: options.model_file_name ?? "model"
        }, options),
        getOptionalConfigs(pretrained_model_name_or_path, {
          generation_config: "generation_config.json"
        }, options)
      ]);
    } else if (modelType === MODEL_TYPES.Seq2Seq || modelType === MODEL_TYPES.Vision2Seq) {
      info = await Promise.all([
        constructSessions(pretrained_model_name_or_path, {
          model: "encoder_model",
          decoder_model_merged: "decoder_model_merged"
        }, options),
        getOptionalConfigs(pretrained_model_name_or_path, {
          generation_config: "generation_config.json"
        }, options)
      ]);
    } else if (modelType === MODEL_TYPES.MaskGeneration) {
      info = await Promise.all([
        constructSessions(pretrained_model_name_or_path, {
          model: "vision_encoder",
          prompt_encoder_mask_decoder: "prompt_encoder_mask_decoder"
        }, options)
      ]);
    } else if (modelType === MODEL_TYPES.EncoderDecoder) {
      info = await Promise.all([
        constructSessions(pretrained_model_name_or_path, {
          model: "encoder_model",
          decoder_model_merged: "decoder_model_merged"
        }, options)
      ]);
    } else if (modelType === MODEL_TYPES.ImageTextToText) {
      const sessions = {
        embed_tokens: "embed_tokens",
        vision_encoder: "vision_encoder",
        decoder_model_merged: "decoder_model_merged"
      };
      if (config.is_encoder_decoder) {
        sessions["model"] = "encoder_model";
      }
      info = await Promise.all([
        constructSessions(pretrained_model_name_or_path, sessions, options),
        getOptionalConfigs(pretrained_model_name_or_path, {
          generation_config: "generation_config.json"
        }, options)
      ]);
    } else if (modelType === MODEL_TYPES.AudioTextToText) {
      const sessions = {
        embed_tokens: "embed_tokens",
        audio_encoder: "audio_encoder",
        decoder_model_merged: "decoder_model_merged"
      };
      info = await Promise.all([
        constructSessions(pretrained_model_name_or_path, sessions, options),
        getOptionalConfigs(pretrained_model_name_or_path, {
          generation_config: "generation_config.json"
        }, options)
      ]);
    } else if (modelType === MODEL_TYPES.ImageAudioTextToText) {
      const sessions = {
        embed_tokens: "embed_tokens",
        audio_encoder: "audio_encoder",
        vision_encoder: "vision_encoder",
        decoder_model_merged: "decoder_model_merged"
      };
      info = await Promise.all([
        constructSessions(pretrained_model_name_or_path, sessions, options),
        getOptionalConfigs(pretrained_model_name_or_path, {
          generation_config: "generation_config.json"
        }, options)
      ]);
    } else if (modelType === MODEL_TYPES.Musicgen) {
      info = await Promise.all([
        constructSessions(pretrained_model_name_or_path, {
          model: "text_encoder",
          decoder_model_merged: "decoder_model_merged",
          encodec_decode: "encodec_decode"
        }, options),
        getOptionalConfigs(pretrained_model_name_or_path, {
          generation_config: "generation_config.json"
        }, options)
      ]);
    } else if (modelType === MODEL_TYPES.MultiModality) {
      info = await Promise.all([
        constructSessions(pretrained_model_name_or_path, {
          prepare_inputs_embeds: "prepare_inputs_embeds",
          model: "language_model",
          lm_head: "lm_head",
          gen_head: "gen_head",
          gen_img_embeds: "gen_img_embeds",
          image_decode: "image_decode"
        }, options),
        getOptionalConfigs(pretrained_model_name_or_path, {
          generation_config: "generation_config.json"
        }, options)
      ]);
    } else if (modelType === MODEL_TYPES.Phi3V) {
      info = await Promise.all([
        constructSessions(pretrained_model_name_or_path, {
          prepare_inputs_embeds: "prepare_inputs_embeds",
          model: "model",
          vision_encoder: "vision_encoder"
        }, options),
        getOptionalConfigs(pretrained_model_name_or_path, {
          generation_config: "generation_config.json"
        }, options)
      ]);
    } else if (modelType === MODEL_TYPES.AutoEncoder) {
      info = await Promise.all([
        constructSessions(pretrained_model_name_or_path, {
          encoder_model: "encoder_model",
          decoder_model: "decoder_model"
        }, options)
      ]);
    } else {
      if (modelType !== MODEL_TYPES.EncoderOnly) {
        const type = modelName ?? config?.model_type;
        if (type !== "custom") {
          console.warn(`Model type for '${type}' not found, assuming encoder-only architecture. Please report this at ${GITHUB_ISSUE_URL}.`);
        }
      }
      info = await Promise.all([
        constructSessions(pretrained_model_name_or_path, {
          model: options.model_file_name ?? "model"
        }, options)
      ]);
    }
    return new this(config, ...info);
  }
  /**
   * Runs the model with the provided inputs
   * @param {Object} model_inputs Object containing input tensors
   * @returns {Promise<Object>} Object containing output tensors
   */
  async _call(model_inputs) {
    return await this.forward(model_inputs);
  }
  /**
   * Forward method for a pretrained model. If not overridden by a subclass, the correct forward method
   * will be chosen based on the model type.
   * @param {Object} model_inputs The input data to the model in the format specified in the ONNX model.
   * @returns {Promise<Object>} The output data from the model in the format specified in the ONNX model.
   * @throws {Error} This method must be implemented in subclasses.
   */
  async forward(model_inputs) {
    return await this._forward(this, model_inputs);
  }
  /**
   * Get the model's generation config, if it exists.
   * @returns {GenerationConfig|null} The model's generation config if it exists, otherwise `null`.
   */
  get generation_config() {
    return this.configs?.generation_config ?? null;
  }
  /**
   * @param {GenerationConfig} generation_config 
   * @param {number} input_ids_seq_length The starting sequence length for the input ids.
   * @returns {LogitsProcessorList}
   * @private
   */
  _get_logits_processor(generation_config, input_ids_seq_length, logits_processor = null) {
    const processors = new LogitsProcessorList();
    if (generation_config.repetition_penalty !== null && generation_config.repetition_penalty !== 1) {
      processors.push(new RepetitionPenaltyLogitsProcessor(generation_config.repetition_penalty));
    }
    if (generation_config.no_repeat_ngram_size !== null && generation_config.no_repeat_ngram_size > 0) {
      processors.push(new NoRepeatNGramLogitsProcessor(generation_config.no_repeat_ngram_size));
    }
    if (generation_config.bad_words_ids !== null) {
      processors.push(new NoBadWordsLogitsProcessor(generation_config.bad_words_ids, generation_config.eos_token_id));
    }
    if (generation_config.min_length !== null && generation_config.eos_token_id !== null && generation_config.min_length > 0) {
      processors.push(new MinLengthLogitsProcessor(generation_config.min_length, generation_config.eos_token_id));
    }
    if (generation_config.min_new_tokens !== null && generation_config.eos_token_id !== null && generation_config.min_new_tokens > 0) {
      processors.push(new MinNewTokensLengthLogitsProcessor(
        input_ids_seq_length,
        generation_config.min_new_tokens,
        generation_config.eos_token_id
      ));
    }
    if (generation_config.forced_bos_token_id !== null) {
      processors.push(new ForcedBOSTokenLogitsProcessor(generation_config.forced_bos_token_id));
    }
    if (generation_config.forced_eos_token_id !== null) {
      processors.push(new ForcedEOSTokenLogitsProcessor(
        generation_config.max_length,
        generation_config.forced_eos_token_id
      ));
    }
    if (generation_config.begin_suppress_tokens !== null) {
      const begin_index = input_ids_seq_length > 1 || generation_config.forced_bos_token_id === null ? input_ids_seq_length : input_ids_seq_length + 1;
      processors.push(new SuppressTokensAtBeginLogitsProcessor(generation_config.begin_suppress_tokens, begin_index));
    }
    if (generation_config.guidance_scale !== null && generation_config.guidance_scale > 1) {
      processors.push(new ClassifierFreeGuidanceLogitsProcessor(generation_config.guidance_scale));
    }
    if (generation_config.temperature === 0 && generation_config.do_sample) {
      console.warn("`do_sample` changed to false because `temperature: 0` implies greedy sampling (always selecting the most likely token), which is incompatible with `do_sample: true`.");
      generation_config.do_sample = false;
    }
    if (generation_config.do_sample) {
      if (generation_config.temperature !== null && generation_config.temperature !== 1) {
        processors.push(new TemperatureLogitsWarper(generation_config.temperature));
      }
    }
    if (logits_processor !== null) {
      processors.extend(logits_processor);
    }
    return processors;
  }
  /**
   * This function merges multiple generation configs together to form a final generation config to be used by the model for text generation.
   * It first creates an empty `GenerationConfig` object, then it applies the model's own `generation_config` property to it. Finally, if a `generation_config` object was passed in the arguments, it overwrites the corresponding properties in the final config with those of the passed config object.
   * @param {GenerationConfig|null} generation_config A `GenerationConfig` object containing generation parameters.
   * @param {Object} kwargs Additional generation parameters to be used in place of those in the `generation_config` object.
   * @returns {GenerationConfig} The final generation config object to be used by the model for text generation.
   */
  _prepare_generation_config(generation_config, kwargs, cls = GenerationConfig) {
    const config = { ...this.config };
    for (const key of ["decoder", "generator", "text_config"]) {
      if (key in config) {
        Object.assign(config, config[key]);
      }
    }
    const gen_config = new cls(config);
    Object.assign(gen_config, this.generation_config ?? {});
    if (generation_config) {
      Object.assign(gen_config, generation_config);
    }
    if (kwargs) {
      Object.assign(gen_config, pick(kwargs, Object.getOwnPropertyNames(gen_config)));
    }
    return gen_config;
  }
  /**
   * 
   * @param {GenerationConfig} generation_config 
   * @param {StoppingCriteriaList} [stopping_criteria=null] 
   */
  _get_stopping_criteria(generation_config, stopping_criteria = null) {
    const criteria = new StoppingCriteriaList();
    if (generation_config.max_length !== null) {
      criteria.push(new MaxLengthCriteria(
        generation_config.max_length,
        this.config.max_position_embeddings ?? null
      ));
    }
    if (generation_config.eos_token_id !== null) {
      criteria.push(new EosTokenCriteria(generation_config.eos_token_id));
    }
    if (stopping_criteria) {
      criteria.extend(stopping_criteria);
    }
    return criteria;
  }
  /**
   * Confirms that the model class is compatible with generation.
   * If not, raises an exception that points to the right class to use.
   */
  _validate_model_class() {
    if (!this.can_generate) {
      const generate_compatible_mappings = [
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
        // MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING, // TODO
        MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES,
        MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
        MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES
      ];
      const modelName = MODEL_CLASS_TO_NAME_MAPPING.get(this.constructor);
      const generate_compatible_classes = /* @__PURE__ */ new Set();
      const modelType = this.config.model_type;
      for (const model_mapping of generate_compatible_mappings) {
        const supported_models = model_mapping.get(modelType);
        if (supported_models) {
          generate_compatible_classes.add(supported_models[0]);
        }
      }
      let errorMessage = `The current model class (${modelName}) is not compatible with \`.generate()\`, as it doesn't have a language model head.`;
      if (generate_compatible_classes.size > 0) {
        errorMessage += ` Please use the following class instead: ${[...generate_compatible_classes].join(", ")}`;
      }
      throw Error(errorMessage);
    }
  }
  prepare_inputs_for_generation(...args) {
    return this._prepare_inputs_for_generation(this, ...args);
  }
  /**
   * 
   * @param {Object} inputs
   * @param {bigint[][]} inputs.generated_input_ids
   * @param {Object} inputs.outputs
   * @param {Object} inputs.model_inputs
   * @param {boolean} inputs.is_encoder_decoder
   * @returns {Object} The updated model inputs for the next generation iteration.
   */
  _update_model_kwargs_for_generation({ generated_input_ids, outputs, model_inputs, is_encoder_decoder }) {
    model_inputs["past_key_values"] = this.getPastKeyValues(outputs, model_inputs.past_key_values);
    model_inputs["input_ids"] = new Tensor2("int64", generated_input_ids.flat(), [generated_input_ids.length, 1]);
    if (!is_encoder_decoder) {
      model_inputs.attention_mask = cat(
        [
          model_inputs.attention_mask,
          ones([model_inputs.attention_mask.dims[0], 1])
        ],
        1
      );
    } else if ("decoder_attention_mask" in model_inputs) {
    }
    model_inputs["position_ids"] = null;
    return model_inputs;
  }
  /**
   * This function extracts the model-specific `inputs` for generation.
   * @param {Object} params
   * @param {Tensor} [params.inputs=null]
   * @param {number} [params.bos_token_id=null]
   * @param {Record<string, Tensor|number[]>} [params.model_kwargs]
   * @returns {{inputs_tensor: Tensor, model_inputs: Record<string, Tensor>, model_input_name: string}} The model-specific inputs for generation.
   */
  _prepare_model_inputs({ inputs, bos_token_id, model_kwargs }) {
    const model_inputs = pick(model_kwargs, this.forward_params);
    const input_name = this.main_input_name;
    if (input_name in model_inputs) {
      if (inputs) {
        throw new Error(
          "`inputs`: {inputs}` were passed alongside {input_name} which is not allowed. Make sure to either pass {inputs} or {input_name}=..."
        );
      }
    } else {
      model_inputs[input_name] = inputs;
    }
    const inputs_tensor = model_inputs[input_name];
    return { inputs_tensor, model_inputs, model_input_name: input_name };
  }
  async _prepare_encoder_decoder_kwargs_for_generation({ inputs_tensor, model_inputs, model_input_name, generation_config }) {
    if (this.sessions["model"].inputNames.includes("inputs_embeds") && !model_inputs.inputs_embeds && "_prepare_inputs_embeds" in this) {
      const { input_ids, pixel_values, attention_mask, ...kwargs } = model_inputs;
      const prepared_inputs = await this._prepare_inputs_embeds(model_inputs);
      model_inputs = {
        ...kwargs,
        ...pick(prepared_inputs, ["inputs_embeds", "attention_mask"])
      };
    }
    let { last_hidden_state } = await encoderForward(this, model_inputs);
    if (generation_config.guidance_scale !== null && generation_config.guidance_scale > 1) {
      last_hidden_state = cat([
        last_hidden_state,
        full_like(last_hidden_state, 0)
      ], 0);
      if ("attention_mask" in model_inputs) {
        model_inputs["attention_mask"] = cat([
          model_inputs["attention_mask"],
          zeros_like(model_inputs["attention_mask"])
        ], 0);
      }
    } else if (model_inputs.decoder_input_ids) {
      const decoder_input_ids_batch_size = toI64Tensor(model_inputs.decoder_input_ids).dims[0];
      if (decoder_input_ids_batch_size !== last_hidden_state.dims[0]) {
        if (last_hidden_state.dims[0] !== 1) {
          throw new Error(
            `The encoder outputs have a different batch size (${last_hidden_state.dims[0]}) than the decoder inputs (${decoder_input_ids_batch_size}).`
          );
        }
        last_hidden_state = cat(Array.from({ length: decoder_input_ids_batch_size }, () => last_hidden_state), 0);
      }
    }
    model_inputs["encoder_outputs"] = last_hidden_state;
    return model_inputs;
  }
  /**
   * Prepares `decoder_input_ids` for generation with encoder-decoder models
   * @param {*} param0 
   */
  _prepare_decoder_input_ids_for_generation({ batch_size, model_input_name, model_kwargs, decoder_start_token_id, bos_token_id, generation_config }) {
    let { decoder_input_ids, ...model_inputs } = model_kwargs;
    if (!(decoder_input_ids instanceof Tensor2)) {
      if (!decoder_input_ids) {
        decoder_start_token_id ??= bos_token_id;
        if (this.config.model_type === "musicgen") {
          decoder_input_ids = Array.from({
            // @ts-expect-error TS2339
            length: batch_size * this.config.decoder.num_codebooks
          }, () => [decoder_start_token_id]);
        } else if (Array.isArray(decoder_start_token_id)) {
          if (decoder_start_token_id.length !== batch_size) {
            throw new Error(
              `\`decoder_start_token_id\` expcted to have length ${batch_size} but got ${decoder_start_token_id.length}`
            );
          }
          decoder_input_ids = decoder_start_token_id;
        } else {
          decoder_input_ids = Array.from({
            length: batch_size
          }, () => [decoder_start_token_id]);
        }
      } else if (!Array.isArray(decoder_input_ids[0])) {
        decoder_input_ids = Array.from({
          length: batch_size
        }, () => decoder_input_ids);
      }
      decoder_input_ids = toI64Tensor(decoder_input_ids);
    }
    model_kwargs["decoder_attention_mask"] = ones_like(decoder_input_ids);
    return { input_ids: decoder_input_ids, model_inputs };
  }
  /**
   * Generates sequences of token ids for models with a language modeling head.
   * @param {import('./generation/parameters.js').GenerationFunctionParameters} options
   * @returns {Promise<ModelOutput|Tensor>} The output of the model, which can contain the generated token ids, attentions, and scores.
   */
  async generate({
    inputs = null,
    generation_config = null,
    logits_processor = null,
    stopping_criteria = null,
    streamer = null,
    // inputs_attention_mask = null,
    ...kwargs
  }) {
    this._validate_model_class();
    generation_config = this._prepare_generation_config(generation_config, kwargs);
    let { inputs_tensor, model_inputs, model_input_name } = this._prepare_model_inputs({
      inputs,
      model_kwargs: kwargs
    });
    const is_encoder_decoder = this.config.is_encoder_decoder;
    if (!is_encoder_decoder) {
    } else if (!("encoder_outputs" in model_inputs)) {
      model_inputs = await this._prepare_encoder_decoder_kwargs_for_generation(
        { inputs_tensor, model_inputs, model_input_name, generation_config }
      );
    }
    let input_ids;
    if (is_encoder_decoder) {
      ({ input_ids, model_inputs } = this._prepare_decoder_input_ids_for_generation({
        batch_size: model_inputs[model_input_name].dims.at(0),
        model_input_name,
        model_kwargs: model_inputs,
        decoder_start_token_id: generation_config.decoder_start_token_id,
        bos_token_id: generation_config.bos_token_id,
        generation_config
      }));
    } else {
      input_ids = model_inputs[model_input_name];
    }
    let input_ids_length = input_ids.dims.at(-1);
    if (generation_config.max_new_tokens !== null) {
      generation_config.max_length = input_ids_length + generation_config.max_new_tokens;
    }
    const prepared_logits_processor = this._get_logits_processor(
      generation_config,
      input_ids_length,
      logits_processor
    );
    const prepared_stopping_criteria = this._get_stopping_criteria(
      generation_config,
      stopping_criteria
    );
    const numInputs = model_inputs[model_input_name].dims.at(0);
    const sampler = LogitsSampler.getSampler(generation_config);
    const scores = new Array(numInputs).fill(0);
    const all_input_ids = input_ids.tolist();
    if (streamer) {
      streamer.put(all_input_ids);
    }
    let outputs;
    let attentions = {};
    while (true) {
      model_inputs = this.prepare_inputs_for_generation(all_input_ids, model_inputs, generation_config);
      outputs = await this.forward(model_inputs);
      if (generation_config.output_attentions && generation_config.return_dict_in_generate) {
        const token_attentions = this.getAttentions(outputs);
        for (const key in token_attentions) {
          if (!(key in attentions)) {
            attentions[key] = [];
          }
          attentions[key].push(token_attentions[key]);
        }
      }
      const logits = outputs.logits.slice(null, -1, null);
      const next_tokens_scores = prepared_logits_processor(all_input_ids, logits);
      const generated_input_ids = [];
      for (let batch_idx = 0; batch_idx < next_tokens_scores.dims.at(0); ++batch_idx) {
        const logs = next_tokens_scores[batch_idx];
        const sampledTokens = await sampler(logs);
        for (const [newTokenId, logProb] of sampledTokens) {
          const bigint = BigInt(newTokenId);
          scores[batch_idx] += logProb;
          all_input_ids[batch_idx].push(bigint);
          generated_input_ids.push([bigint]);
          break;
        }
      }
      if (streamer) {
        streamer.put(generated_input_ids);
      }
      const stop = prepared_stopping_criteria(all_input_ids);
      if (stop.every((x) => x)) {
        break;
      }
      model_inputs = this._update_model_kwargs_for_generation({
        generated_input_ids,
        outputs,
        model_inputs,
        is_encoder_decoder
      });
    }
    if (streamer) {
      streamer.end();
    }
    const past_key_values = this.getPastKeyValues(outputs, model_inputs.past_key_values, true);
    const sequences = new Tensor2("int64", all_input_ids.flat(), [all_input_ids.length, all_input_ids[0].length]);
    if (generation_config.return_dict_in_generate) {
      return {
        sequences,
        past_key_values,
        ...attentions
        // TODO:
        // scores,
        // logits,
      };
    } else {
      for (const tensor of Object.values(outputs)) {
        if (tensor.location === "gpu-buffer") {
          tensor.dispose();
        }
      }
      return sequences;
    }
  }
  /**
   * Returns an object containing past key values from the given decoder results object.
   *
   * @param {Object} decoderResults The decoder results object.
   * @param {Object} pastKeyValues The previous past key values.
   * @returns {Object} An object containing past key values.
   */
  getPastKeyValues(decoderResults, pastKeyValues, disposeEncoderPKVs = false) {
    const pkvs = /* @__PURE__ */ Object.create(null);
    for (const name in decoderResults) {
      if (name.startsWith("present")) {
        const newName = name.replace("present_conv", "past_conv").replace("present", "past_key_values");
        const is_encoder_pkv = name.includes("encoder");
        if (is_encoder_pkv && pastKeyValues) {
          pkvs[newName] = pastKeyValues[newName];
        } else {
          pkvs[newName] = decoderResults[name];
        }
        if (pastKeyValues && (!is_encoder_pkv || disposeEncoderPKVs)) {
          const t = pastKeyValues[newName];
          if (t.location === "gpu-buffer") {
            t.dispose();
          }
        }
      }
    }
    return pkvs;
  }
  /**
   * Returns an object containing attentions from the given model output object.
   *
   * @param {Object} model_output The output of the model.
   * @returns {{cross_attentions?: Tensor[]}} An object containing attentions.
   */
  getAttentions(model_output) {
    const attentions = {};
    for (const attnName of ["cross_attentions", "encoder_attentions", "decoder_attentions"]) {
      for (const name in model_output) {
        if (name.startsWith(attnName)) {
          if (!(attnName in attentions)) {
            attentions[attnName] = [];
          }
          attentions[attnName].push(model_output[name]);
        }
      }
    }
    return attentions;
  }
  /**
   * Adds past key values to the decoder feeds object. If pastKeyValues is null, creates new tensors for past key values.
   *
   * @param {Object} decoderFeeds The decoder feeds object to add past key values to.
   * @param {Object} pastKeyValues An object containing past key values.
   */
  addPastKeyValues(decoderFeeds, pastKeyValues) {
    if (pastKeyValues) {
      Object.assign(decoderFeeds, pastKeyValues);
    } else {
      const session = this.sessions["decoder_model_merged"] ?? this.sessions["model"];
      const batch_size = (decoderFeeds[this.main_input_name] ?? decoderFeeds.attention_mask)?.dims?.[0] ?? 1;
      const dtype = session?.config?.kv_cache_dtype ?? "float32";
      const cls = dtype === "float16" ? DataTypeMap.float16 : DataTypeMap.float32;
      const shapes = getCacheShapes(this.config, { batch_size });
      for (const name in shapes) {
        const size = shapes[name].reduce((a, b) => a * b, 1);
        decoderFeeds[name] = new Tensor2(dtype, new cls(size), shapes[name]);
      }
    }
  }
  async encode_image({ pixel_values }) {
    return (await sessionRun(this.sessions["vision_encoder"], { pixel_values })).image_features;
  }
  async encode_text({ input_ids }) {
    return (await sessionRun(this.sessions["embed_tokens"], { input_ids })).inputs_embeds;
  }
  async encode_audio({ audio_values }) {
    return (await sessionRun(this.sessions["audio_encoder"], { audio_values })).audio_features;
  }
};
var ModelOutput = class {
};
var BaseModelOutput = class extends ModelOutput {
  /**
   * @param {Object} output The output of the model.
   * @param {Tensor} output.last_hidden_state Sequence of hidden-states at the output of the last layer of the model.
   * @param {Tensor} [output.hidden_states] Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
   * @param {Tensor} [output.attentions] Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
   */
  constructor({ last_hidden_state, hidden_states = null, attentions = null }) {
    super();
    this.last_hidden_state = last_hidden_state;
    this.hidden_states = hidden_states;
    this.attentions = attentions;
  }
};
var BertPreTrainedModel = class extends PreTrainedModel {
};
var BertModel = class extends BertPreTrainedModel {
};
var BertForMaskedLM = class extends BertPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<MaskedLMOutput>} An object containing the model's output logits for masked language modeling.
   */
  async _call(model_inputs) {
    return new MaskedLMOutput(await super._call(model_inputs));
  }
};
var BertForSequenceClassification = class extends BertPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<SequenceClassifierOutput>} An object containing the model's output logits for sequence classification.
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var BertForTokenClassification = class extends BertPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<TokenClassifierOutput>} An object containing the model's output logits for token classification.
   */
  async _call(model_inputs) {
    return new TokenClassifierOutput(await super._call(model_inputs));
  }
};
var BertForQuestionAnswering = class extends BertPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<QuestionAnsweringModelOutput>} An object containing the model's output logits for question answering.
   */
  async _call(model_inputs) {
    return new QuestionAnsweringModelOutput(await super._call(model_inputs));
  }
};
var NeoBertPreTrainedModel = class extends PreTrainedModel {
};
var NeoBertModel = class extends NeoBertPreTrainedModel {
};
var NeoBertForMaskedLM = class extends NeoBertPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<MaskedLMOutput>} An object containing the model's output logits for masked language modeling.
   */
  async _call(model_inputs) {
    return new MaskedLMOutput(await super._call(model_inputs));
  }
};
var NeoBertForSequenceClassification = class extends NeoBertPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<SequenceClassifierOutput>} An object containing the model's output logits for sequence classification.
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var NeoBertForTokenClassification = class extends NeoBertPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<TokenClassifierOutput>} An object containing the model's output logits for token classification.
   */
  async _call(model_inputs) {
    return new TokenClassifierOutput(await super._call(model_inputs));
  }
};
var NeoBertForQuestionAnswering = class extends NeoBertPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<QuestionAnsweringModelOutput>} An object containing the model's output logits for question answering.
   */
  async _call(model_inputs) {
    return new QuestionAnsweringModelOutput(await super._call(model_inputs));
  }
};
var ModernBertPreTrainedModel = class extends PreTrainedModel {
};
var ModernBertModel = class extends ModernBertPreTrainedModel {
};
var ModernBertForMaskedLM = class extends ModernBertPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<MaskedLMOutput>} An object containing the model's output logits for masked language modeling.
   */
  async _call(model_inputs) {
    return new MaskedLMOutput(await super._call(model_inputs));
  }
};
var ModernBertForSequenceClassification = class extends ModernBertPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<SequenceClassifierOutput>} An object containing the model's output logits for sequence classification.
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var ModernBertForTokenClassification = class extends ModernBertPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<TokenClassifierOutput>} An object containing the model's output logits for token classification.
   */
  async _call(model_inputs) {
    return new TokenClassifierOutput(await super._call(model_inputs));
  }
};
var ModernBertDecoderPreTrainedModel = class extends PreTrainedModel {
};
var ModernBertDecoderModel = class extends ModernBertDecoderPreTrainedModel {
};
var ModernBertDecoderForCausalLM = class extends ModernBertDecoderPreTrainedModel {
};
var NomicBertPreTrainedModel = class extends PreTrainedModel {
};
var NomicBertModel = class extends NomicBertPreTrainedModel {
};
var RoFormerPreTrainedModel = class extends PreTrainedModel {
};
var RoFormerModel = class extends RoFormerPreTrainedModel {
};
var RoFormerForMaskedLM = class extends RoFormerPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<MaskedLMOutput>} An object containing the model's output logits for masked language modeling.
   */
  async _call(model_inputs) {
    return new MaskedLMOutput(await super._call(model_inputs));
  }
};
var RoFormerForSequenceClassification = class extends RoFormerPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<SequenceClassifierOutput>} An object containing the model's output logits for sequence classification.
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var RoFormerForTokenClassification = class extends RoFormerPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<TokenClassifierOutput>} An object containing the model's output logits for token classification.
   */
  async _call(model_inputs) {
    return new TokenClassifierOutput(await super._call(model_inputs));
  }
};
var RoFormerForQuestionAnswering = class extends RoFormerPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<QuestionAnsweringModelOutput>} An object containing the model's output logits for question answering.
   */
  async _call(model_inputs) {
    return new QuestionAnsweringModelOutput(await super._call(model_inputs));
  }
};
var ConvBertPreTrainedModel = class extends PreTrainedModel {
};
var ConvBertModel = class extends ConvBertPreTrainedModel {
};
var ConvBertForMaskedLM = class extends ConvBertPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<MaskedLMOutput>} An object containing the model's output logits for masked language modeling.
   */
  async _call(model_inputs) {
    return new MaskedLMOutput(await super._call(model_inputs));
  }
};
var ConvBertForSequenceClassification = class extends ConvBertPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<SequenceClassifierOutput>} An object containing the model's output logits for sequence classification.
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var ConvBertForTokenClassification = class extends ConvBertPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<TokenClassifierOutput>} An object containing the model's output logits for token classification.
   */
  async _call(model_inputs) {
    return new TokenClassifierOutput(await super._call(model_inputs));
  }
};
var ConvBertForQuestionAnswering = class extends ConvBertPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<QuestionAnsweringModelOutput>} An object containing the model's output logits for question answering.
   */
  async _call(model_inputs) {
    return new QuestionAnsweringModelOutput(await super._call(model_inputs));
  }
};
var ElectraPreTrainedModel = class extends PreTrainedModel {
};
var ElectraModel = class extends ElectraPreTrainedModel {
};
var ElectraForMaskedLM = class extends ElectraPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<MaskedLMOutput>} An object containing the model's output logits for masked language modeling.
   */
  async _call(model_inputs) {
    return new MaskedLMOutput(await super._call(model_inputs));
  }
};
var ElectraForSequenceClassification = class extends ElectraPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<SequenceClassifierOutput>} An object containing the model's output logits for sequence classification.
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var ElectraForTokenClassification = class extends ElectraPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<TokenClassifierOutput>} An object containing the model's output logits for token classification.
   */
  async _call(model_inputs) {
    return new TokenClassifierOutput(await super._call(model_inputs));
  }
};
var ElectraForQuestionAnswering = class extends ElectraPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<QuestionAnsweringModelOutput>} An object containing the model's output logits for question answering.
   */
  async _call(model_inputs) {
    return new QuestionAnsweringModelOutput(await super._call(model_inputs));
  }
};
var CamembertPreTrainedModel = class extends PreTrainedModel {
};
var CamembertModel = class extends CamembertPreTrainedModel {
};
var CamembertForMaskedLM = class extends CamembertPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<MaskedLMOutput>} An object containing the model's output logits for masked language modeling.
   */
  async _call(model_inputs) {
    return new MaskedLMOutput(await super._call(model_inputs));
  }
};
var CamembertForSequenceClassification = class extends CamembertPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<SequenceClassifierOutput>} An object containing the model's output logits for sequence classification.
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var CamembertForTokenClassification = class extends CamembertPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<TokenClassifierOutput>} An object containing the model's output logits for token classification.
   */
  async _call(model_inputs) {
    return new TokenClassifierOutput(await super._call(model_inputs));
  }
};
var CamembertForQuestionAnswering = class extends CamembertPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<QuestionAnsweringModelOutput>} An object containing the model's output logits for question answering.
   */
  async _call(model_inputs) {
    return new QuestionAnsweringModelOutput(await super._call(model_inputs));
  }
};
var DebertaPreTrainedModel = class extends PreTrainedModel {
};
var DebertaModel = class extends DebertaPreTrainedModel {
};
var DebertaForMaskedLM = class extends DebertaPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<MaskedLMOutput>} An object containing the model's output logits for masked language modeling.
   */
  async _call(model_inputs) {
    return new MaskedLMOutput(await super._call(model_inputs));
  }
};
var DebertaForSequenceClassification = class extends DebertaPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<SequenceClassifierOutput>} An object containing the model's output logits for sequence classification.
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var DebertaForTokenClassification = class extends DebertaPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<TokenClassifierOutput>} An object containing the model's output logits for token classification.
   */
  async _call(model_inputs) {
    return new TokenClassifierOutput(await super._call(model_inputs));
  }
};
var DebertaForQuestionAnswering = class extends DebertaPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<QuestionAnsweringModelOutput>} An object containing the model's output logits for question answering.
   */
  async _call(model_inputs) {
    return new QuestionAnsweringModelOutput(await super._call(model_inputs));
  }
};
var DebertaV2PreTrainedModel = class extends PreTrainedModel {
};
var DebertaV2Model = class extends DebertaV2PreTrainedModel {
};
var DebertaV2ForMaskedLM = class extends DebertaV2PreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<MaskedLMOutput>} An object containing the model's output logits for masked language modeling.
   */
  async _call(model_inputs) {
    return new MaskedLMOutput(await super._call(model_inputs));
  }
};
var DebertaV2ForSequenceClassification = class extends DebertaV2PreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<SequenceClassifierOutput>} An object containing the model's output logits for sequence classification.
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var DebertaV2ForTokenClassification = class extends DebertaV2PreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<TokenClassifierOutput>} An object containing the model's output logits for token classification.
   */
  async _call(model_inputs) {
    return new TokenClassifierOutput(await super._call(model_inputs));
  }
};
var DebertaV2ForQuestionAnswering = class extends DebertaV2PreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<QuestionAnsweringModelOutput>} An object containing the model's output logits for question answering.
   */
  async _call(model_inputs) {
    return new QuestionAnsweringModelOutput(await super._call(model_inputs));
  }
};
var DistilBertPreTrainedModel = class extends PreTrainedModel {
};
var DistilBertModel = class extends DistilBertPreTrainedModel {
};
var DistilBertForSequenceClassification = class extends DistilBertPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<SequenceClassifierOutput>} An object containing the model's output logits for sequence classification.
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var DistilBertForTokenClassification = class extends DistilBertPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<TokenClassifierOutput>} An object containing the model's output logits for token classification.
   */
  async _call(model_inputs) {
    return new TokenClassifierOutput(await super._call(model_inputs));
  }
};
var DistilBertForQuestionAnswering = class extends DistilBertPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<QuestionAnsweringModelOutput>} An object containing the model's output logits for question answering.
   */
  async _call(model_inputs) {
    return new QuestionAnsweringModelOutput(await super._call(model_inputs));
  }
};
var DistilBertForMaskedLM = class extends DistilBertPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<MaskedLMOutput>} returned object
   */
  async _call(model_inputs) {
    return new MaskedLMOutput(await super._call(model_inputs));
  }
};
var EsmPreTrainedModel = class extends PreTrainedModel {
};
var EsmModel = class extends EsmPreTrainedModel {
};
var EsmForMaskedLM = class extends EsmPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<MaskedLMOutput>} An object containing the model's output logits for masked language modeling.
   */
  async _call(model_inputs) {
    return new MaskedLMOutput(await super._call(model_inputs));
  }
};
var EsmForSequenceClassification = class extends EsmPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<SequenceClassifierOutput>} An object containing the model's output logits for sequence classification.
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var EsmForTokenClassification = class extends EsmPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<TokenClassifierOutput>} An object containing the model's output logits for token classification.
   */
  async _call(model_inputs) {
    return new TokenClassifierOutput(await super._call(model_inputs));
  }
};
var MobileBertPreTrainedModel = class extends PreTrainedModel {
};
var MobileBertModel = class extends MobileBertPreTrainedModel {
};
var MobileBertForMaskedLM = class extends MobileBertPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<MaskedLMOutput>} returned object
   */
  async _call(model_inputs) {
    return new MaskedLMOutput(await super._call(model_inputs));
  }
};
var MobileBertForSequenceClassification = class extends MobileBertPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<SequenceClassifierOutput>} returned object
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var MobileBertForQuestionAnswering = class extends MobileBertPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<QuestionAnsweringModelOutput>} returned object
   */
  async _call(model_inputs) {
    return new QuestionAnsweringModelOutput(await super._call(model_inputs));
  }
};
var MPNetPreTrainedModel = class extends PreTrainedModel {
};
var MPNetModel = class extends MPNetPreTrainedModel {
};
var MPNetForMaskedLM = class extends MPNetPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<MaskedLMOutput>} An object containing the model's output logits for masked language modeling.
   */
  async _call(model_inputs) {
    return new MaskedLMOutput(await super._call(model_inputs));
  }
};
var MPNetForSequenceClassification = class extends MPNetPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<SequenceClassifierOutput>} An object containing the model's output logits for sequence classification.
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var MPNetForTokenClassification = class extends MPNetPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<TokenClassifierOutput>} An object containing the model's output logits for token classification.
   */
  async _call(model_inputs) {
    return new TokenClassifierOutput(await super._call(model_inputs));
  }
};
var MPNetForQuestionAnswering = class extends MPNetPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<QuestionAnsweringModelOutput>} An object containing the model's output logits for question answering.
   */
  async _call(model_inputs) {
    return new QuestionAnsweringModelOutput(await super._call(model_inputs));
  }
};
var SqueezeBertPreTrainedModel = class extends PreTrainedModel {
};
var SqueezeBertModel = class extends SqueezeBertPreTrainedModel {
};
var SqueezeBertForMaskedLM = class extends SqueezeBertPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<MaskedLMOutput>} returned object
   */
  async _call(model_inputs) {
    return new MaskedLMOutput(await super._call(model_inputs));
  }
};
var SqueezeBertForSequenceClassification = class extends SqueezeBertPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<SequenceClassifierOutput>} returned object
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var SqueezeBertForQuestionAnswering = class extends SqueezeBertPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<QuestionAnsweringModelOutput>} returned object
   */
  async _call(model_inputs) {
    return new QuestionAnsweringModelOutput(await super._call(model_inputs));
  }
};
var AlbertPreTrainedModel = class extends PreTrainedModel {
};
var AlbertModel = class extends AlbertPreTrainedModel {
};
var AlbertForSequenceClassification = class extends AlbertPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<SequenceClassifierOutput>} returned object
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var AlbertForQuestionAnswering = class extends AlbertPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<QuestionAnsweringModelOutput>} returned object
   */
  async _call(model_inputs) {
    return new QuestionAnsweringModelOutput(await super._call(model_inputs));
  }
};
var AlbertForMaskedLM = class extends AlbertPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<MaskedLMOutput>} returned object
   */
  async _call(model_inputs) {
    return new MaskedLMOutput(await super._call(model_inputs));
  }
};
var T5PreTrainedModel = class extends PreTrainedModel {
  forward_params = [
    "input_ids",
    "attention_mask",
    "encoder_outputs",
    "decoder_input_ids",
    "decoder_attention_mask",
    "past_key_values"
  ];
};
var T5Model = class extends T5PreTrainedModel {
};
var T5ForConditionalGeneration = class extends T5PreTrainedModel {
};
var LongT5PreTrainedModel = class extends PreTrainedModel {
};
var LongT5Model = class extends LongT5PreTrainedModel {
};
var LongT5ForConditionalGeneration = class extends LongT5PreTrainedModel {
};
var MT5PreTrainedModel = class extends PreTrainedModel {
};
var MT5Model = class extends MT5PreTrainedModel {
};
var MT5ForConditionalGeneration = class extends MT5PreTrainedModel {
};
var BartPretrainedModel = class extends PreTrainedModel {
};
var BartModel = class extends BartPretrainedModel {
};
var BartForConditionalGeneration = class extends BartPretrainedModel {
};
var BartForSequenceClassification = class extends BartPretrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<SequenceClassifierOutput>} An object containing the model's output logits for sequence classification.
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var MBartPreTrainedModel = class extends PreTrainedModel {
};
var MBartModel = class extends MBartPreTrainedModel {
};
var MBartForConditionalGeneration = class extends MBartPreTrainedModel {
};
var MBartForSequenceClassification = class extends MBartPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<SequenceClassifierOutput>} An object containing the model's output logits for sequence classification.
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var MBartForCausalLM = class extends MBartPreTrainedModel {
};
var BlenderbotPreTrainedModel = class extends PreTrainedModel {
};
var BlenderbotModel = class extends BlenderbotPreTrainedModel {
};
var BlenderbotForConditionalGeneration = class extends BlenderbotPreTrainedModel {
};
var BlenderbotSmallPreTrainedModel = class extends PreTrainedModel {
};
var BlenderbotSmallModel = class extends BlenderbotSmallPreTrainedModel {
};
var BlenderbotSmallForConditionalGeneration = class extends BlenderbotSmallPreTrainedModel {
};
var RobertaPreTrainedModel = class extends PreTrainedModel {
};
var RobertaModel = class extends RobertaPreTrainedModel {
};
var RobertaForMaskedLM = class extends RobertaPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<MaskedLMOutput>} returned object
   */
  async _call(model_inputs) {
    return new MaskedLMOutput(await super._call(model_inputs));
  }
};
var RobertaForSequenceClassification = class extends RobertaPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<SequenceClassifierOutput>} returned object
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var RobertaForTokenClassification = class extends RobertaPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<TokenClassifierOutput>} An object containing the model's output logits for token classification.
   */
  async _call(model_inputs) {
    return new TokenClassifierOutput(await super._call(model_inputs));
  }
};
var RobertaForQuestionAnswering = class extends RobertaPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<QuestionAnsweringModelOutput>} returned object
   */
  async _call(model_inputs) {
    return new QuestionAnsweringModelOutput(await super._call(model_inputs));
  }
};
var XLMPreTrainedModel = class extends PreTrainedModel {
};
var XLMModel = class extends XLMPreTrainedModel {
};
var XLMWithLMHeadModel = class extends XLMPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<MaskedLMOutput>} returned object
   */
  async _call(model_inputs) {
    return new MaskedLMOutput(await super._call(model_inputs));
  }
};
var XLMForSequenceClassification = class extends XLMPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<SequenceClassifierOutput>} returned object
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var XLMForTokenClassification = class extends XLMPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<TokenClassifierOutput>} An object containing the model's output logits for token classification.
   */
  async _call(model_inputs) {
    return new TokenClassifierOutput(await super._call(model_inputs));
  }
};
var XLMForQuestionAnswering = class extends XLMPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<QuestionAnsweringModelOutput>} returned object
   */
  async _call(model_inputs) {
    return new QuestionAnsweringModelOutput(await super._call(model_inputs));
  }
};
var XLMRobertaPreTrainedModel = class extends PreTrainedModel {
};
var XLMRobertaModel = class extends XLMRobertaPreTrainedModel {
};
var XLMRobertaForMaskedLM = class extends XLMRobertaPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<MaskedLMOutput>} returned object
   */
  async _call(model_inputs) {
    return new MaskedLMOutput(await super._call(model_inputs));
  }
};
var XLMRobertaForSequenceClassification = class extends XLMRobertaPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<SequenceClassifierOutput>} returned object
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var XLMRobertaForTokenClassification = class extends XLMRobertaPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<TokenClassifierOutput>} An object containing the model's output logits for token classification.
   */
  async _call(model_inputs) {
    return new TokenClassifierOutput(await super._call(model_inputs));
  }
};
var XLMRobertaForQuestionAnswering = class extends XLMRobertaPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<QuestionAnsweringModelOutput>} returned object
   */
  async _call(model_inputs) {
    return new QuestionAnsweringModelOutput(await super._call(model_inputs));
  }
};
var ASTPreTrainedModel = class extends PreTrainedModel {
};
var ASTModel = class extends ASTPreTrainedModel {
};
var ASTForAudioClassification = class extends ASTPreTrainedModel {
};
var WhisperPreTrainedModel = class extends PreTrainedModel {
  requires_attention_mask = false;
  main_input_name = "input_features";
  forward_params = [
    "input_features",
    "attention_mask",
    "decoder_input_ids",
    "decoder_attention_mask",
    "past_key_values"
  ];
};
var WhisperModel = class extends WhisperPreTrainedModel {
};
var WhisperForConditionalGeneration = class extends WhisperPreTrainedModel {
  _prepare_generation_config(generation_config, kwargs) {
    return (
      /** @type {WhisperGenerationConfig} */
      super._prepare_generation_config(generation_config, kwargs, WhisperGenerationConfig)
    );
  }
  /**
   * 
   * @param {WhisperGenerationConfig} generation_config 
   */
  _retrieve_init_tokens(generation_config) {
    const init_tokens = [generation_config.decoder_start_token_id];
    let language = generation_config.language;
    const task = generation_config.task;
    if (generation_config.is_multilingual) {
      if (!language) {
        console.warn("No language specified - defaulting to English (en).");
        language = "en";
      }
      const language_code = whisper_language_to_code(language);
      const language_token = `<|${language_code}|>`;
      init_tokens.push(generation_config.lang_to_id[language_token]);
      init_tokens.push(generation_config.task_to_id[task ?? "transcribe"]);
    } else if (language || task) {
      throw new Error(
        "Cannot specify `task` or `language` for an English-only model. If the model is intended to be multilingual, pass `is_multilingual=true` to generate, or update the generation config."
      );
    }
    if (!generation_config.return_timestamps && generation_config.no_timestamps_token_id && init_tokens.at(-1) !== generation_config.no_timestamps_token_id) {
      init_tokens.push(generation_config.no_timestamps_token_id);
    } else if (generation_config.return_timestamps && init_tokens.at(-1) === generation_config.no_timestamps_token_id) {
      console.warn("<|notimestamps|> prompt token is removed from generation_config since `return_timestamps` is set to `true`.");
      init_tokens.pop();
    }
    return init_tokens.filter((token) => token != null);
  }
  /**
   * Transcribes or translates log-mel input features to a sequence of auto-regressively generated token ids.
   * @param {import('./models/whisper/generation_whisper.js').WhisperGenerationFunctionParameters} options
   * @returns {Promise<ModelOutput|Tensor>} The output of the model, which can contain the generated token ids, attentions, and scores.
   */
  async generate({
    inputs = null,
    generation_config = null,
    logits_processor = null,
    stopping_criteria = null,
    // Whisper-specific options (passed to kwargs)
    // prompt_ids = null,
    // language = null,
    // task = null,
    ...kwargs
  }) {
    generation_config = this._prepare_generation_config(generation_config, kwargs);
    const init_tokens = kwargs.decoder_input_ids ?? this._retrieve_init_tokens(generation_config);
    if (generation_config.return_timestamps) {
      logits_processor ??= new LogitsProcessorList();
      logits_processor.push(
        new WhisperTimeStampLogitsProcessor(generation_config, init_tokens)
      );
    }
    if (generation_config.begin_suppress_tokens) {
      logits_processor ??= new LogitsProcessorList();
      logits_processor.push(
        new SuppressTokensAtBeginLogitsProcessor(generation_config.begin_suppress_tokens, init_tokens.length)
      );
    }
    if (generation_config.return_token_timestamps) {
      if (!generation_config.alignment_heads) {
        throw new Error(
          "Model generation config has no `alignment_heads`, token-level timestamps not available. See https://gist.github.com/hollance/42e32852f24243b748ae6bc1f985b13a on how to add this property to the generation config."
        );
      }
      if (generation_config.task === "translate") {
        console.warn("Token-level timestamps may not be reliable for task 'translate'.");
      }
      generation_config.output_attentions = true;
      generation_config.return_dict_in_generate = true;
    }
    const outputs = await super.generate({
      inputs,
      generation_config,
      logits_processor,
      decoder_input_ids: init_tokens,
      ...kwargs
    });
    if (generation_config.return_token_timestamps) {
      outputs["token_timestamps"] = this._extract_token_timestamps(
        // @ts-expect-error TS2345
        outputs,
        generation_config.alignment_heads,
        generation_config.num_frames
      );
    }
    return outputs;
  }
  /**
   * Calculates token-level timestamps using the encoder-decoder cross-attentions and
   * dynamic time-warping (DTW) to map each output token to a position in the input audio.
   * If `num_frames` is specified, the encoder-decoder cross-attentions will be cropped before applying DTW.
   * @param {Object} generate_outputs Outputs generated by the model
   * @param {Tensor[][]} generate_outputs.cross_attentions The cross attentions output by the model
   * @param {Tensor} generate_outputs.sequences The sequences output by the model
   * @param {number[][]} alignment_heads Alignment heads of the model
   * @param {number} [num_frames=null] Number of frames in the input audio.
   * @param {number} [time_precision=0.02] Precision of the timestamps in seconds
   * @returns {Tensor} tensor containing the timestamps in seconds for each predicted token
   */
  _extract_token_timestamps(generate_outputs, alignment_heads, num_frames = null, time_precision = 0.02) {
    if (!generate_outputs.cross_attentions) {
      throw new Error(
        "Model outputs must contain cross attentions to extract timestamps. This is most likely because the model was not exported with `output_attentions=True`."
      );
    }
    if (num_frames == null) {
      console.warn(
        "`num_frames` has not been set, meaning the entire audio will be analyzed. This may lead to inaccurate token-level timestamps for short audios (< 30 seconds)."
      );
    }
    let median_filter_width = this.config.median_filter_width;
    if (median_filter_width === void 0) {
      console.warn("Model config has no `median_filter_width`, using default value of 7.");
      median_filter_width = 7;
    }
    const batch = generate_outputs.cross_attentions;
    const cross_attentions = Array.from(
      { length: this.config.decoder_layers },
      // Concatenate the cross attentions for each layer across sequence length dimension.
      (_, i) => cat(batch.map((x) => x[i]), 2)
    );
    const weights = stack(alignment_heads.map(([l, h]) => {
      if (l >= cross_attentions.length) {
        throw new Error(`Layer index ${l} is out of bounds for cross attentions (length ${cross_attentions.length}).`);
      }
      return num_frames ? cross_attentions[l].slice(null, h, null, [0, num_frames]) : cross_attentions[l].slice(null, h);
    })).transpose(1, 0, 2, 3);
    const [std, calculatedMean] = std_mean(weights, -2, 0, true);
    const smoothedWeights = weights.clone();
    for (let a = 0; a < smoothedWeights.dims[0]; ++a) {
      const aTensor = smoothedWeights[a];
      for (let b = 0; b < aTensor.dims[0]; ++b) {
        const bTensor = aTensor[b];
        const stdTensorData = std[a][b][0].data;
        const meanTensorData = calculatedMean[a][b][0].data;
        for (let c = 0; c < bTensor.dims[0]; ++c) {
          let cTensorData = bTensor[c].data;
          for (let d = 0; d < cTensorData.length; ++d) {
            cTensorData[d] = (cTensorData[d] - meanTensorData[d]) / stdTensorData[d];
          }
          cTensorData.set(medianFilter(cTensorData, median_filter_width));
        }
      }
    }
    const batchedMatrices = [mean(smoothedWeights, 1)];
    const timestampsShape = generate_outputs.sequences.dims;
    const timestamps = new Tensor2(
      "float32",
      new Float32Array(timestampsShape[0] * timestampsShape[1]),
      timestampsShape
    );
    for (let batch_idx = 0; batch_idx < timestampsShape[0]; ++batch_idx) {
      const matrix = batchedMatrices[batch_idx].neg().squeeze_(0);
      const [text_indices, time_indices] = dynamic_time_warping(matrix.tolist());
      const diffs = Array.from({ length: text_indices.length - 1 }, (v, i) => text_indices[i + 1] - text_indices[i]);
      const jumps = mergeArrays([1], diffs).map((x) => !!x);
      const jump_times = [];
      for (let i = 0; i < jumps.length; ++i) {
        if (jumps[i]) {
          jump_times.push(time_indices[i] * time_precision);
        }
      }
      timestamps[batch_idx].data.set(jump_times, 1);
    }
    return timestamps;
  }
};
var LiteWhisperForConditionalGeneration = class extends WhisperForConditionalGeneration {
};
var MoonshinePreTrainedModel = class extends PreTrainedModel {
  requires_attention_mask = false;
  main_input_name = "input_values";
  forward_params = [
    "input_values",
    "decoder_input_ids",
    "past_key_values"
  ];
};
var MoonshineModel = class extends MoonshinePreTrainedModel {
};
var MoonshineForConditionalGeneration = class extends MoonshinePreTrainedModel {
};
var VisionEncoderDecoderModel = class extends PreTrainedModel {
  main_input_name = "pixel_values";
  forward_params = [
    // Encoder inputs
    "pixel_values",
    // Decoder inpputs
    "decoder_input_ids",
    "encoder_hidden_states",
    "past_key_values"
  ];
};
var LlavaPreTrainedModel = class extends PreTrainedModel {
  forward_params = [
    "input_ids",
    "attention_mask",
    "pixel_values",
    "position_ids",
    "past_key_values"
  ];
};
var LlavaForConditionalGeneration = class extends LlavaPreTrainedModel {
  _merge_input_ids_with_image_features(kwargs) {
    const vision_hidden_size = kwargs.image_features.dims.at(-1);
    const reshaped_image_hidden_states = kwargs.image_features.view(-1, vision_hidden_size);
    return default_merge_input_ids_with_image_features({
      // @ts-ignore
      image_token_id: this.config.image_token_index,
      ...kwargs,
      image_features: reshaped_image_hidden_states
    });
  }
};
var LlavaOnevisionForConditionalGeneration = class extends LlavaForConditionalGeneration {
};
var Moondream1ForConditionalGeneration = class extends LlavaForConditionalGeneration {
};
var Florence2PreTrainedModel = class extends PreTrainedModel {
  forward_params = [
    // Encoder inputs
    "input_ids",
    "inputs_embeds",
    "attention_mask",
    "pixel_values",
    // Decoder inputs
    "encoder_outputs",
    "decoder_input_ids",
    "decoder_inputs_embeds",
    "decoder_attention_mask",
    "past_key_values"
  ];
  main_input_name = "inputs_embeds";
};
var Florence2ForConditionalGeneration = class extends Florence2PreTrainedModel {
  _merge_input_ids_with_image_features({
    inputs_embeds,
    image_features,
    input_ids,
    attention_mask
  }) {
    return {
      inputs_embeds: cat([
        image_features,
        // image embeds
        inputs_embeds
        // task prefix embeds
      ], 1),
      attention_mask: cat([
        ones(image_features.dims.slice(0, 2)),
        // image attention mask
        attention_mask
        // task prefix attention mask
      ], 1)
    };
  }
  async _prepare_inputs_embeds({ input_ids, pixel_values, inputs_embeds, attention_mask }) {
    if (!input_ids && !pixel_values) {
      throw new Error("Either `input_ids` or `pixel_values` should be provided.");
    }
    let text_features, image_features;
    if (input_ids) {
      text_features = await this.encode_text({ input_ids });
    }
    if (pixel_values) {
      image_features = await this.encode_image({ pixel_values });
    }
    if (text_features && image_features) {
      ({ inputs_embeds, attention_mask } = this._merge_input_ids_with_image_features({
        inputs_embeds: text_features,
        image_features,
        input_ids,
        attention_mask
      }));
    } else {
      inputs_embeds = text_features || image_features;
    }
    return { inputs_embeds, attention_mask };
  }
  async forward({
    input_ids,
    pixel_values,
    attention_mask,
    decoder_input_ids,
    decoder_attention_mask,
    encoder_outputs,
    past_key_values,
    inputs_embeds,
    decoder_inputs_embeds
  }) {
    if (!inputs_embeds) {
      ({ inputs_embeds, attention_mask } = await this._prepare_inputs_embeds({ input_ids, pixel_values, inputs_embeds, attention_mask }));
    }
    if (!encoder_outputs) {
      let { last_hidden_state } = await encoderForward(this, { inputs_embeds, attention_mask });
      encoder_outputs = last_hidden_state;
    }
    if (!decoder_inputs_embeds) {
      if (!decoder_input_ids) {
        throw new Error("Either `decoder_input_ids` or `decoder_inputs_embeds` should be provided.");
      }
      decoder_inputs_embeds = await this.encode_text({ input_ids: decoder_input_ids });
    }
    const decoderFeeds = {
      inputs_embeds: decoder_inputs_embeds,
      attention_mask: decoder_attention_mask,
      encoder_attention_mask: attention_mask,
      encoder_hidden_states: encoder_outputs,
      past_key_values
    };
    const decoder_outputs = await decoderForward(this, decoderFeeds, true);
    return decoder_outputs;
  }
};
var PaliGemmaPreTrainedModel = class extends PreTrainedModel {
  forward_params = [
    "input_ids",
    // 'inputs_embeds',
    "attention_mask",
    "pixel_values",
    "position_ids",
    "past_key_values"
  ];
};
var PaliGemmaForConditionalGeneration = class extends PaliGemmaPreTrainedModel {
  _merge_input_ids_with_image_features(kwargs) {
    const vision_hidden_size = kwargs.image_features.dims.at(-1);
    const reshaped_image_hidden_states = kwargs.image_features.view(-1, vision_hidden_size);
    return default_merge_input_ids_with_image_features({
      // @ts-ignore
      image_token_id: this.config.image_token_index,
      ...kwargs,
      image_features: reshaped_image_hidden_states
    });
  }
};
var LlavaQwen2ForCausalLM = class extends LlavaPreTrainedModel {
  _merge_input_ids_with_image_features(kwargs) {
    const vision_hidden_size = kwargs.image_features.dims.at(-1);
    const reshaped_image_hidden_states = kwargs.image_features.view(-1, vision_hidden_size);
    return default_merge_input_ids_with_image_features({
      // @ts-ignore
      image_token_id: this.config.image_token_index,
      ...kwargs,
      image_features: reshaped_image_hidden_states
    });
  }
};
var Gemma3nPreTrainedModel = class extends PreTrainedModel {
  forward_params = [
    "input_ids",
    "attention_mask",
    "inputs_embeds",
    "per_layer_inputs",
    "position_ids",
    "pixel_values",
    "input_features",
    "input_features_mask",
    "past_key_values"
  ];
};
var Gemma3nForConditionalGeneration = class extends Gemma3nPreTrainedModel {
  async forward({
    // Produced by the tokenizer/processor:
    input_ids = null,
    attention_mask = null,
    pixel_values = null,
    input_features = null,
    input_features_mask = null,
    // Used during generation:
    position_ids = null,
    inputs_embeds = null,
    per_layer_inputs = null,
    past_key_values = null,
    // Generic generation parameters
    generation_config = null,
    logits_processor = null,
    // TODO: needed?
    ...kwargs
  }) {
    if (!inputs_embeds || !per_layer_inputs) {
      ({ inputs_embeds, per_layer_inputs } = await sessionRun(this.sessions["embed_tokens"], {
        input_ids
      }));
      if (input_ids.dims[1] !== 1) {
        if (pixel_values) {
          const { image_features } = await sessionRun(this.sessions["vision_encoder"], {
            pixel_values
          });
          ({ inputs_embeds, attention_mask } = this._merge_input_ids_with_image_features({
            image_features,
            inputs_embeds,
            input_ids,
            attention_mask
          }));
        }
        if (input_features) {
          const { audio_features } = await sessionRun(this.sessions["audio_encoder"], {
            input_features,
            input_features_mask
          });
          ({ inputs_embeds, attention_mask } = this._merge_input_ids_with_audio_features({
            audio_features,
            inputs_embeds,
            input_ids,
            attention_mask
          }));
        }
      }
    }
    const outputs = await decoderForward(this, {
      inputs_embeds,
      per_layer_inputs,
      past_key_values,
      attention_mask,
      position_ids,
      generation_config,
      logits_processor
    }, true);
    return outputs;
  }
  _merge_input_ids_with_image_features(kwargs) {
    const vision_hidden_size = kwargs.image_features.dims.at(-1);
    const reshaped_image_hidden_states = kwargs.image_features.view(-1, vision_hidden_size);
    return default_merge_input_ids_with_image_features({
      // @ts-ignore
      image_token_id: this.config.image_token_id,
      ...kwargs,
      image_features: reshaped_image_hidden_states
    });
  }
  _merge_input_ids_with_audio_features(kwargs) {
    const audio_hidden_size = kwargs.audio_features.dims.at(-1);
    const reshaped_audio_features = kwargs.audio_features.view(-1, audio_hidden_size);
    return default_merge_input_ids_with_audio_features({
      // @ts-ignore
      audio_token_id: this.config.audio_token_id,
      ...kwargs,
      audio_features: reshaped_audio_features
    });
  }
};
var Idefics3PreTrainedModel = class extends PreTrainedModel {
  forward_params = [
    "input_ids",
    "attention_mask",
    "pixel_values",
    "pixel_attention_mask",
    "position_ids",
    "past_key_values"
  ];
};
var Idefics3ForConditionalGeneration = class extends Idefics3PreTrainedModel {
  async encode_image({ pixel_values, pixel_attention_mask }) {
    const features = (await sessionRun(this.sessions["vision_encoder"], { pixel_values, pixel_attention_mask })).image_features;
    return features;
  }
  _merge_input_ids_with_image_features(kwargs) {
    const vision_hidden_size = kwargs.image_features.dims.at(-1);
    const reshaped_image_hidden_states = kwargs.image_features.view(-1, vision_hidden_size);
    return default_merge_input_ids_with_image_features({
      // @ts-ignore
      image_token_id: this.config.image_token_id,
      ...kwargs,
      image_features: reshaped_image_hidden_states
    });
  }
};
var SmolVLMForConditionalGeneration = class extends Idefics3ForConditionalGeneration {
};
var Phi3VPreTrainedModel = class extends PreTrainedModel {
  forward_params = [
    "input_ids",
    "inputs_embeds",
    "attention_mask",
    "position_ids",
    "pixel_values",
    "image_sizes",
    "past_key_values"
  ];
};
var Phi3VForCausalLM = class extends Phi3VPreTrainedModel {
  async forward({
    // Produced by the tokenizer/processor:
    input_ids = null,
    attention_mask = null,
    pixel_values = null,
    image_sizes = null,
    // Used during generation:
    position_ids = null,
    inputs_embeds = null,
    past_key_values = null,
    // Generic generation parameters
    generation_config = null,
    logits_processor = null,
    // TODO: needed?
    ...kwargs
  }) {
    if (!inputs_embeds) {
      let image_features;
      if (pixel_values && input_ids.dims[1] !== 1) {
        if (!image_sizes) {
          throw new Error("`image_sizes` must be provided when `pixel_values` is provided.");
        }
        ({ image_features } = await sessionRun(this.sessions["vision_encoder"], {
          pixel_values,
          image_sizes
        }));
      } else {
        const hidden_size = this.config.normalized_config.hidden_size;
        image_features = new Tensor2(
          "float32",
          [],
          [0, hidden_size]
        );
      }
      ({ inputs_embeds } = await sessionRun(this.sessions["prepare_inputs_embeds"], {
        input_ids,
        image_features
      }));
    }
    const outputs = await decoderForward(this, {
      inputs_embeds,
      past_key_values,
      attention_mask,
      position_ids,
      generation_config,
      logits_processor
    }, false);
    return outputs;
  }
};
var CLIPPreTrainedModel = class extends PreTrainedModel {
};
var CLIPModel = class extends CLIPPreTrainedModel {
};
var CLIPTextModel = class extends CLIPPreTrainedModel {
  /** @type {typeof PreTrainedModel.from_pretrained} */
  static async from_pretrained(pretrained_model_name_or_path, options = {}) {
    return super.from_pretrained(pretrained_model_name_or_path, {
      ...options,
      // Update default model file name if not provided
      model_file_name: options.model_file_name ?? "text_model"
    });
  }
};
var CLIPTextModelWithProjection = class extends CLIPPreTrainedModel {
  /** @type {typeof PreTrainedModel.from_pretrained} */
  static async from_pretrained(pretrained_model_name_or_path, options = {}) {
    return super.from_pretrained(pretrained_model_name_or_path, {
      ...options,
      // Update default model file name if not provided
      model_file_name: options.model_file_name ?? "text_model"
    });
  }
};
var CLIPVisionModel = class extends CLIPPreTrainedModel {
  /** @type {typeof PreTrainedModel.from_pretrained} */
  static async from_pretrained(pretrained_model_name_or_path, options = {}) {
    return super.from_pretrained(pretrained_model_name_or_path, {
      ...options,
      // Update default model file name if not provided
      model_file_name: options.model_file_name ?? "vision_model"
    });
  }
};
var CLIPVisionModelWithProjection = class extends CLIPPreTrainedModel {
  /** @type {typeof PreTrainedModel.from_pretrained} */
  static async from_pretrained(pretrained_model_name_or_path, options = {}) {
    return super.from_pretrained(pretrained_model_name_or_path, {
      ...options,
      // Update default model file name if not provided
      model_file_name: options.model_file_name ?? "vision_model"
    });
  }
};
var SiglipPreTrainedModel = class extends PreTrainedModel {
};
var SiglipModel = class extends SiglipPreTrainedModel {
};
var SiglipTextModel = class extends SiglipPreTrainedModel {
  /** @type {typeof PreTrainedModel.from_pretrained} */
  static async from_pretrained(pretrained_model_name_or_path, options = {}) {
    return super.from_pretrained(pretrained_model_name_or_path, {
      ...options,
      // Update default model file name if not provided
      model_file_name: options.model_file_name ?? "text_model"
    });
  }
};
var SiglipVisionModel = class extends CLIPPreTrainedModel {
  /** @type {typeof PreTrainedModel.from_pretrained} */
  static async from_pretrained(pretrained_model_name_or_path, options = {}) {
    return super.from_pretrained(pretrained_model_name_or_path, {
      ...options,
      // Update default model file name if not provided
      model_file_name: options.model_file_name ?? "vision_model"
    });
  }
};
var ChineseCLIPPreTrainedModel = class extends PreTrainedModel {
};
var ChineseCLIPModel = class extends ChineseCLIPPreTrainedModel {
};
var JinaCLIPPreTrainedModel = class extends PreTrainedModel {
};
var JinaCLIPModel = class extends JinaCLIPPreTrainedModel {
  async forward(model_inputs) {
    const missing_text_inputs = !model_inputs.input_ids;
    const missing_image_inputs = !model_inputs.pixel_values;
    if (missing_text_inputs && missing_image_inputs) {
      throw new Error("Either `input_ids` or `pixel_values` should be provided.");
    }
    if (missing_text_inputs) {
      model_inputs.input_ids = ones([model_inputs.pixel_values.dims[0], 1]);
    }
    if (missing_image_inputs) {
      const { image_size } = this.config.vision_config;
      model_inputs.pixel_values = full([0, 3, image_size, image_size], 0);
    }
    const { text_embeddings, image_embeddings, l2norm_text_embeddings, l2norm_image_embeddings } = await super.forward(model_inputs);
    const result = {};
    if (!missing_text_inputs) {
      result.text_embeddings = text_embeddings;
      result.l2norm_text_embeddings = l2norm_text_embeddings;
    }
    if (!missing_image_inputs) {
      result.image_embeddings = image_embeddings;
      result.l2norm_image_embeddings = l2norm_image_embeddings;
    }
    return result;
  }
};
var JinaCLIPTextModel = class extends JinaCLIPPreTrainedModel {
  /** @type {typeof PreTrainedModel.from_pretrained} */
  static async from_pretrained(pretrained_model_name_or_path, options = {}) {
    return super.from_pretrained(pretrained_model_name_or_path, {
      ...options,
      // Update default model file name if not provided
      model_file_name: options.model_file_name ?? "text_model"
    });
  }
};
var JinaCLIPVisionModel = class extends JinaCLIPPreTrainedModel {
  /** @type {typeof PreTrainedModel.from_pretrained} */
  static async from_pretrained(pretrained_model_name_or_path, options = {}) {
    return super.from_pretrained(pretrained_model_name_or_path, {
      ...options,
      // Update default model file name if not provided
      model_file_name: options.model_file_name ?? "vision_model"
    });
  }
};
var CLIPSegPreTrainedModel = class extends PreTrainedModel {
};
var CLIPSegModel = class extends CLIPSegPreTrainedModel {
};
var CLIPSegForImageSegmentation = class extends CLIPSegPreTrainedModel {
};
var GPT2PreTrainedModel = class extends PreTrainedModel {
};
var GPT2Model = class extends GPT2PreTrainedModel {
};
var GPT2LMHeadModel = class extends GPT2PreTrainedModel {
};
var JAISPreTrainedModel = class extends PreTrainedModel {
};
var JAISModel = class extends JAISPreTrainedModel {
};
var JAISLMHeadModel = class extends JAISPreTrainedModel {
};
var GPTNeoPreTrainedModel = class extends PreTrainedModel {
};
var GPTNeoModel = class extends GPTNeoPreTrainedModel {
};
var GPTNeoForCausalLM = class extends GPTNeoPreTrainedModel {
};
var GPTNeoXPreTrainedModel = class extends PreTrainedModel {
};
var GPTNeoXModel = class extends GPTNeoXPreTrainedModel {
};
var GPTNeoXForCausalLM = class extends GPTNeoXPreTrainedModel {
};
var GPTJPreTrainedModel = class extends PreTrainedModel {
};
var GPTJModel = class extends GPTJPreTrainedModel {
};
var GPTJForCausalLM = class extends GPTJPreTrainedModel {
};
var GPTBigCodePreTrainedModel = class extends PreTrainedModel {
};
var GPTBigCodeModel = class extends GPTBigCodePreTrainedModel {
};
var GPTBigCodeForCausalLM = class extends GPTBigCodePreTrainedModel {
};
var CodeGenPreTrainedModel = class extends PreTrainedModel {
};
var CodeGenModel = class extends CodeGenPreTrainedModel {
};
var CodeGenForCausalLM = class extends CodeGenPreTrainedModel {
};
var LlamaPreTrainedModel = class extends PreTrainedModel {
};
var LlamaModel = class extends LlamaPreTrainedModel {
};
var LlamaForCausalLM = class extends LlamaPreTrainedModel {
};
var Llama4PreTrainedModel = class extends PreTrainedModel {
};
var Llama4ForCausalLM = class extends Llama4PreTrainedModel {
};
var NanoChatPreTrainedModel = class extends PreTrainedModel {
};
var NanoChatModel = class extends NanoChatPreTrainedModel {
};
var NanoChatForCausalLM = class extends NanoChatPreTrainedModel {
};
var ArceePreTrainedModel = class extends PreTrainedModel {
};
var ArceeModel = class extends ArceePreTrainedModel {
};
var ArceeForCausalLM = class extends ArceePreTrainedModel {
};
var Lfm2PreTrainedModel = class extends PreTrainedModel {
};
var Lfm2Model = class extends Lfm2PreTrainedModel {
};
var Lfm2ForCausalLM = class extends Lfm2PreTrainedModel {
};
var SmolLM3PreTrainedModel = class extends PreTrainedModel {
};
var SmolLM3Model = class extends SmolLM3PreTrainedModel {
};
var SmolLM3ForCausalLM = class extends SmolLM3PreTrainedModel {
};
var HeliumPreTrainedModel = class extends PreTrainedModel {
};
var HeliumModel = class extends HeliumPreTrainedModel {
};
var HeliumForCausalLM = class extends HeliumPreTrainedModel {
};
var GlmPreTrainedModel = class extends PreTrainedModel {
};
var GlmModel = class extends GlmPreTrainedModel {
};
var GlmForCausalLM = class extends GlmPreTrainedModel {
};
var ExaonePreTrainedModel = class extends PreTrainedModel {
};
var ExaoneModel = class extends ExaonePreTrainedModel {
};
var ExaoneForCausalLM = class extends ExaonePreTrainedModel {
};
var MobileLLMPreTrainedModel = class extends PreTrainedModel {
};
var MobileLLMModel = class extends MobileLLMPreTrainedModel {
};
var MobileLLMForCausalLM = class extends MobileLLMPreTrainedModel {
};
var OlmoPreTrainedModel = class extends PreTrainedModel {
};
var OlmoModel = class extends OlmoPreTrainedModel {
};
var OlmoForCausalLM = class extends OlmoPreTrainedModel {
};
var Olmo2PreTrainedModel = class extends PreTrainedModel {
};
var Olmo2Model = class extends Olmo2PreTrainedModel {
};
var Olmo2ForCausalLM = class extends Olmo2PreTrainedModel {
};
var GranitePreTrainedModel = class extends PreTrainedModel {
};
var GraniteModel = class extends GranitePreTrainedModel {
};
var GraniteForCausalLM = class extends GranitePreTrainedModel {
};
var GraniteMoeHybridPreTrainedModel = class extends PreTrainedModel {
};
var GraniteMoeHybridModel = class extends GraniteMoeHybridPreTrainedModel {
};
var GraniteMoeHybridForCausalLM = class extends GraniteMoeHybridPreTrainedModel {
};
var CoherePreTrainedModel = class extends PreTrainedModel {
};
var CohereModel = class extends CoherePreTrainedModel {
};
var CohereForCausalLM = class extends CoherePreTrainedModel {
};
var GemmaPreTrainedModel = class extends PreTrainedModel {
};
var GemmaModel = class extends GemmaPreTrainedModel {
};
var GemmaForCausalLM = class extends GemmaPreTrainedModel {
};
var Gemma2PreTrainedModel = class extends PreTrainedModel {
};
var Gemma2Model = class extends Gemma2PreTrainedModel {
};
var Gemma2ForCausalLM = class extends Gemma2PreTrainedModel {
};
var VaultGemmaPreTrainedModel = class extends PreTrainedModel {
};
var VaultGemmaModel = class extends VaultGemmaPreTrainedModel {
};
var VaultGemmaForCausalLM = class extends VaultGemmaPreTrainedModel {
};
var Gemma3PreTrainedModel = class extends PreTrainedModel {
};
var Gemma3Model = class extends Gemma3PreTrainedModel {
};
var Gemma3ForCausalLM = class extends Gemma3PreTrainedModel {
};
var OpenELMPreTrainedModel = class extends PreTrainedModel {
};
var OpenELMModel = class extends OpenELMPreTrainedModel {
};
var OpenELMForCausalLM = class extends OpenELMPreTrainedModel {
};
var Qwen2PreTrainedModel = class extends PreTrainedModel {
};
var Qwen2Model = class extends Qwen2PreTrainedModel {
};
var Qwen2ForCausalLM = class extends Qwen2PreTrainedModel {
};
var Qwen3PreTrainedModel = class extends PreTrainedModel {
};
var Qwen3Model = class extends Qwen3PreTrainedModel {
};
var Qwen3ForCausalLM = class extends Qwen3PreTrainedModel {
};
var Qwen2VLPreTrainedModel = class extends PreTrainedModel {
  forward_params = [
    // Text inputs
    "input_ids",
    "attention_mask",
    "position_ids",
    "past_key_values",
    // Vision inputs
    "pixel_values",
    "image_grid_thw"
  ];
};
var Qwen2VLForConditionalGeneration = class extends Qwen2VLPreTrainedModel {
  /**
   * Calculate the 3D rope index based on image and video's temporal, height and width in LLM.
   *
   * Explanation:
   *     Each embedding sequence contains vision embedding and text embedding or just contains text embedding.
   *
   *     For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.
   *     Examples:
   *         input_ids: [T T T T T], here T is for text.
   *         temporal position_ids: [0, 1, 2, 3, 4]
   *         height position_ids: [0, 1, 2, 3, 4]
   *         width position_ids: [0, 1, 2, 3, 4]
   *
   *     For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
   *     and 1D rotary position embeddin for text part.
   *     Examples:
   *         Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
   *         input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
   *         vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
   *         vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
   *         vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
   *         text temporal position_ids: [3, 4, 5, 6, 7]
   *         text height position_ids: [3, 4, 5, 6, 7]
   *         text width position_ids: [3, 4, 5, 6, 7]
   *         Here we calculate the text start position_ids as the max vision position_ids plus 1.
   * 
   * @param {Tensor} input_ids Indices of input sequence tokens in the vocabulary. Tensor of shape `(batch_size, sequence_length)`.
   * @param {Tensor} image_grid_thw (Optional) The temporal, height and width of feature shape of each image in LLM. Tensor of shape `(num_images, 3)`.
   * @param {Tensor} video_grid_thw (Optional) The temporal, height and width of feature shape of each video in LLM. Tensor of shape `(num_videos, 3)`.
   * @param {Tensor} attention_mask (Optional) Mask to avoid performing attention on padding token indices. Tensor of shape `(batch_size, sequence_length)`. Mask values selected in `[0, 1]`:
   * - 1 for tokens that are **not masked**,
   * - 0 for tokens that are **masked**.
   * @returns {[Tensor, Tensor]} [position_ids, mrope_position_deltas] with:
   * - position_ids: Tensor of shape `(3, batch_size, sequence_length)`.
   * - mrope_position_deltas: Tensor of shape `(batch_size)`.
   */
  get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask) {
    const { vision_config, image_token_id, video_token_id, vision_start_token_id } = this.config;
    const spatial_merge_size = vision_config.spatial_merge_size ?? 2;
    const mrope_position_deltas = [];
    if (image_grid_thw || video_grid_thw) {
      let total_input_ids = input_ids.tolist();
      if (!attention_mask) {
        attention_mask = ones_like(input_ids);
      }
      const attention_mask_list = attention_mask.tolist();
      const position_ids_list = Array.from({ length: 3 }, (_) => Array.from({ length: input_ids.dims[0] }, (_2) => Array.from({ length: input_ids.dims[1] }, (_3) => 1)));
      const image_grid_thw_list = image_grid_thw ? image_grid_thw.tolist() : [];
      const video_grid_thw_list = video_grid_thw ? video_grid_thw.tolist() : [];
      let image_index = 0;
      let video_index = 0;
      for (let i = 0; i < total_input_ids.length; ++i) {
        const ids = total_input_ids[i].filter((_, j) => attention_mask_list[i][j] == 1);
        const vision_start_indices = ids.reduce((acc, x, idx) => {
          if (x == vision_start_token_id) acc.push(idx);
          return acc;
        }, []);
        const vision_tokens = vision_start_indices.map((x) => ids[x + 1]);
        const image_nums = vision_tokens.filter((x) => x == image_token_id).length;
        const video_nums = vision_tokens.filter((x) => x == video_token_id).length;
        let llm_pos_ids_list = [];
        let st = 0;
        let remain_images = image_nums;
        let remain_videos = video_nums;
        for (let j = 0; j < vision_tokens.length; ++j) {
          const next_image_token = ids.findIndex((x, i2) => i2 > st && x == image_token_id);
          const next_video_token = ids.findIndex((x, i2) => i2 > st && x == video_token_id);
          const ed_image = remain_images > 0 && next_image_token !== -1 ? next_image_token : ids.length + 1;
          const ed_video = remain_videos > 0 && next_video_token !== -1 ? next_video_token : ids.length + 1;
          let ed;
          let t, h, w;
          if (ed_image < ed_video) {
            [t, h, w] = image_grid_thw_list[image_index];
            ++image_index;
            --remain_images;
            ed = ed_image;
          } else {
            [t, h, w] = video_grid_thw_list[video_index];
            ++video_index;
            --remain_videos;
            ed = ed_video;
          }
          const [llm_grid_t, llm_grid_h, llm_grid_w] = [
            Number(t),
            Math.floor(Number(h) / spatial_merge_size),
            Math.floor(Number(w) / spatial_merge_size)
          ];
          const text_len = ed - st;
          const st_idx = llm_pos_ids_list.length > 0 ? max(llm_pos_ids_list.at(-1))[0] + 1 : 0;
          llm_pos_ids_list.push(
            Array.from({ length: 3 * text_len }, (_, i2) => st_idx + i2 % text_len)
          );
          const offset = text_len + st_idx;
          const grid_size = llm_grid_t * llm_grid_h * llm_grid_w;
          const t_index = Array.from({ length: grid_size }, (_, i2) => offset + Math.floor(i2 / (llm_grid_h * llm_grid_w)));
          const h_index = Array.from({ length: grid_size }, (_, i2) => offset + Math.floor(i2 / llm_grid_w) % llm_grid_h);
          const w_index = Array.from({ length: grid_size }, (_, i2) => offset + i2 % llm_grid_w);
          llm_pos_ids_list.push([t_index, h_index, w_index].flat());
          st = ed + grid_size;
        }
        if (st < ids.length) {
          const st_idx = llm_pos_ids_list.length > 0 ? max(llm_pos_ids_list.at(-1))[0] + 1 : 0;
          const text_len = ids.length - st;
          llm_pos_ids_list.push(
            Array.from({ length: 3 * text_len }, (_, i2) => st_idx + i2 % text_len)
          );
        }
        const num_items = llm_pos_ids_list.reduce((acc, x) => acc + x.length, 0);
        const llm_positions = new Array(num_items);
        let index = 0;
        for (let x = 0; x < 3; ++x) {
          for (let y = 0; y < llm_pos_ids_list.length; ++y) {
            const val = llm_pos_ids_list[y];
            const text_len = val.length / 3;
            for (let z = x * text_len; z < (x + 1) * text_len; ++z) {
              llm_positions[index++] = val[z];
            }
          }
        }
        let count2 = 0;
        const attn_mask = attention_mask_list[i];
        for (let y = 0; y < attn_mask.length; ++y) {
          if (attn_mask[y] == 1) {
            for (let x = 0; x < 3; ++x) {
              position_ids_list[x][i][y] = llm_positions[x * num_items / 3 + count2];
            }
            ++count2;
          }
        }
        const max_llm_positions = max(llm_positions)[0];
        mrope_position_deltas.push(max_llm_positions + 1 - total_input_ids[i].length);
      }
      return [
        new Tensor2("int64", position_ids_list.flat(Infinity), [3, input_ids.dims[0], input_ids.dims[1]]),
        new Tensor2("int64", mrope_position_deltas, [mrope_position_deltas.length, 1])
      ];
    } else {
      if (attention_mask) {
        const { data, dims } = cumsum_masked_fill(attention_mask);
        const position_ids = BigInt64Array.from(
          { length: 3 * data.length },
          (_, i) => data[i % data.length]
        );
        const mrope_position_deltas2 = Array.from(
          { length: dims[0] },
          (_, i) => max(data.subarray(dims[1] * i, dims[1] * (i + 1)))[0] + 1n + BigInt(dims[1])
        );
        return [
          new Tensor2("int64", position_ids, [3, ...dims]),
          new Tensor2("int64", mrope_position_deltas2, [mrope_position_deltas2.length, 1])
        ];
      } else {
        const [batch_size, seq_length] = input_ids.dims;
        const position_ids = BigInt64Array.from(
          { length: 3 * batch_size * seq_length },
          (_, i) => BigInt(Math.floor(i % seq_length / batch_size))
        );
        return [
          new Tensor2("int64", position_ids, [3, ...input_ids.dims]),
          zeros([batch_size, 1])
        ];
      }
    }
  }
  async encode_image({ pixel_values, image_grid_thw }) {
    const features = (await sessionRun(this.sessions["vision_encoder"], { pixel_values, grid_thw: image_grid_thw })).image_features;
    return features;
  }
  _merge_input_ids_with_image_features(kwargs) {
    return default_merge_input_ids_with_image_features({
      // @ts-ignore
      image_token_id: this.config.image_token_id,
      ...kwargs
    });
  }
  prepare_inputs_for_generation(input_ids, model_inputs, generation_config) {
    if (model_inputs.attention_mask && !model_inputs.position_ids) {
      if (!model_inputs.past_key_values) {
        [model_inputs.position_ids, model_inputs.rope_deltas] = this.get_rope_index(
          model_inputs.input_ids,
          model_inputs.image_grid_thw,
          model_inputs.video_grid_thw,
          model_inputs.attention_mask
        );
      } else {
        model_inputs.pixel_values = null;
        const delta = BigInt(Object.values(model_inputs.past_key_values)[0].dims.at(-2));
        const rope_deltas_list = model_inputs.rope_deltas.map((x) => delta + x);
        model_inputs.position_ids = stack([rope_deltas_list, rope_deltas_list, rope_deltas_list], 0);
      }
    }
    return model_inputs;
  }
};
var PhiPreTrainedModel = class extends PreTrainedModel {
};
var PhiModel = class extends PhiPreTrainedModel {
};
var PhiForCausalLM = class extends PhiPreTrainedModel {
};
var Phi3PreTrainedModel = class extends PreTrainedModel {
};
var Phi3Model = class extends Phi3PreTrainedModel {
};
var Phi3ForCausalLM = class extends Phi3PreTrainedModel {
};
var BloomPreTrainedModel = class extends PreTrainedModel {
};
var BloomModel = class extends BloomPreTrainedModel {
};
var BloomForCausalLM = class extends BloomPreTrainedModel {
};
var MptPreTrainedModel = class extends PreTrainedModel {
};
var MptModel = class extends MptPreTrainedModel {
};
var MptForCausalLM = class extends MptPreTrainedModel {
};
var OPTPreTrainedModel = class extends PreTrainedModel {
};
var OPTModel = class extends OPTPreTrainedModel {
};
var OPTForCausalLM = class extends OPTPreTrainedModel {
};
var ViTPreTrainedModel = class extends PreTrainedModel {
};
var ViTModel = class extends ViTPreTrainedModel {
};
var ViTForImageClassification = class extends ViTPreTrainedModel {
  /**
   * @param {any} model_inputs
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var IJepaPreTrainedModel = class extends PreTrainedModel {
};
var IJepaModel = class extends IJepaPreTrainedModel {
};
var IJepaForImageClassification = class extends IJepaPreTrainedModel {
  /**
   * @param {any} model_inputs
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var VitPosePreTrainedModel = class extends PreTrainedModel {
};
var VitPoseForPoseEstimation = class extends VitPosePreTrainedModel {
};
var PvtPreTrainedModel = class extends PreTrainedModel {
};
var PvtModel = class extends PvtPreTrainedModel {
};
var PvtForImageClassification = class extends PvtPreTrainedModel {
  /**
   * @param {any} model_inputs
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var ViTMAEPreTrainedModel = class extends PreTrainedModel {
};
var ViTMAEModel = class extends ViTMAEPreTrainedModel {
};
var ViTMSNPreTrainedModel = class extends PreTrainedModel {
};
var ViTMSNModel = class extends ViTMSNPreTrainedModel {
};
var ViTMSNForImageClassification = class extends ViTMSNPreTrainedModel {
  /**
   * @param {any} model_inputs
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var GroupViTPreTrainedModel = class extends PreTrainedModel {
};
var GroupViTModel = class extends GroupViTPreTrainedModel {
};
var FastViTPreTrainedModel = class extends PreTrainedModel {
};
var FastViTModel = class extends FastViTPreTrainedModel {
};
var FastViTForImageClassification = class extends FastViTPreTrainedModel {
  /**
   * @param {any} model_inputs
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var VitMattePreTrainedModel = class extends PreTrainedModel {
};
var VitMatteForImageMatting = class extends VitMattePreTrainedModel {
  /**
   * @param {any} model_inputs
   */
  async _call(model_inputs) {
    return new ImageMattingOutput(await super._call(model_inputs));
  }
};
var MobileViTPreTrainedModel = class extends PreTrainedModel {
};
var MobileViTModel = class extends MobileViTPreTrainedModel {
};
var MobileViTForImageClassification = class extends MobileViTPreTrainedModel {
  /**
   * @param {any} model_inputs
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var MobileViTV2PreTrainedModel = class extends PreTrainedModel {
};
var MobileViTV2Model = class extends MobileViTV2PreTrainedModel {
};
var MobileViTV2ForImageClassification = class extends MobileViTV2PreTrainedModel {
  /**
   * @param {any} model_inputs
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var OwlViTPreTrainedModel = class extends PreTrainedModel {
};
var OwlViTModel = class extends OwlViTPreTrainedModel {
};
var OwlViTForObjectDetection = class extends OwlViTPreTrainedModel {
};
var Owlv2PreTrainedModel = class extends PreTrainedModel {
};
var Owlv2Model = class extends Owlv2PreTrainedModel {
};
var Owlv2ForObjectDetection = class extends Owlv2PreTrainedModel {
};
var BeitPreTrainedModel = class extends PreTrainedModel {
};
var BeitModel = class extends BeitPreTrainedModel {
};
var BeitForImageClassification = class extends BeitPreTrainedModel {
  /**
   * @param {any} model_inputs
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var DetrPreTrainedModel = class extends PreTrainedModel {
};
var DetrModel = class extends DetrPreTrainedModel {
};
var DetrForObjectDetection = class extends DetrPreTrainedModel {
  /**
   * @param {any} model_inputs
   */
  async _call(model_inputs) {
    return new DetrObjectDetectionOutput(await super._call(model_inputs));
  }
};
var DetrForSegmentation = class extends DetrPreTrainedModel {
  /**
   * Runs the model with the provided inputs
   * @param {Object} model_inputs Model inputs
   * @returns {Promise<DetrSegmentationOutput>} Object containing segmentation outputs
   */
  async _call(model_inputs) {
    return new DetrSegmentationOutput(await super._call(model_inputs));
  }
};
var DetrObjectDetectionOutput = class extends ModelOutput {
  /**
   * @param {Object} output The output of the model.
   * @param {Tensor} output.logits Classification logits (including no-object) for all queries.
   * @param {Tensor} output.pred_boxes Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height).
   * These values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding possible padding).
   */
  constructor({ logits, pred_boxes }) {
    super();
    this.logits = logits;
    this.pred_boxes = pred_boxes;
  }
};
var DetrSegmentationOutput = class extends ModelOutput {
  /**
   * @param {Object} output The output of the model.
   * @param {Tensor} output.logits The output logits of the model.
   * @param {Tensor} output.pred_boxes Predicted boxes.
   * @param {Tensor} output.pred_masks Predicted masks.
   */
  constructor({ logits, pred_boxes, pred_masks }) {
    super();
    this.logits = logits;
    this.pred_boxes = pred_boxes;
    this.pred_masks = pred_masks;
  }
};
var RTDetrPreTrainedModel = class extends PreTrainedModel {
};
var RTDetrModel = class extends RTDetrPreTrainedModel {
};
var RTDetrForObjectDetection = class extends RTDetrPreTrainedModel {
  /**
   * @param {any} model_inputs
   */
  async _call(model_inputs) {
    return new RTDetrObjectDetectionOutput(await super._call(model_inputs));
  }
};
var RTDetrObjectDetectionOutput = class extends ModelOutput {
  /**
   * @param {Object} output The output of the model.
   * @param {Tensor} output.logits Classification logits (including no-object) for all queries.
   * @param {Tensor} output.pred_boxes Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height).
   * These values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding possible padding).
   */
  constructor({ logits, pred_boxes }) {
    super();
    this.logits = logits;
    this.pred_boxes = pred_boxes;
  }
};
var RTDetrV2PreTrainedModel = class extends PreTrainedModel {
};
var RTDetrV2Model = class extends RTDetrV2PreTrainedModel {
};
var RTDetrV2ForObjectDetection = class extends RTDetrV2PreTrainedModel {
  /**
   * @param {any} model_inputs
   */
  async _call(model_inputs) {
    return new RTDetrV2ObjectDetectionOutput(await super._call(model_inputs));
  }
};
var RTDetrV2ObjectDetectionOutput = class extends RTDetrObjectDetectionOutput {
};
var RFDetrPreTrainedModel = class extends PreTrainedModel {
};
var RFDetrModel = class extends RFDetrPreTrainedModel {
};
var RFDetrForObjectDetection = class extends RFDetrPreTrainedModel {
  /**
   * @param {any} model_inputs
   */
  async _call(model_inputs) {
    return new RFDetrObjectDetectionOutput(await super._call(model_inputs));
  }
};
var RFDetrObjectDetectionOutput = class extends RTDetrObjectDetectionOutput {
};
var DFinePreTrainedModel = class extends PreTrainedModel {
};
var DFineModel = class extends DFinePreTrainedModel {
};
var DFineForObjectDetection = class extends DFinePreTrainedModel {
  /**
   * @param {any} model_inputs
   */
  async _call(model_inputs) {
    return new RTDetrObjectDetectionOutput(await super._call(model_inputs));
  }
};
var TableTransformerPreTrainedModel = class extends PreTrainedModel {
};
var TableTransformerModel = class extends TableTransformerPreTrainedModel {
};
var TableTransformerForObjectDetection = class extends TableTransformerPreTrainedModel {
  /**
   * @param {any} model_inputs
   */
  async _call(model_inputs) {
    return new TableTransformerObjectDetectionOutput(await super._call(model_inputs));
  }
};
var TableTransformerObjectDetectionOutput = class extends DetrObjectDetectionOutput {
};
var DeiTPreTrainedModel = class extends PreTrainedModel {
};
var DeiTModel = class extends DeiTPreTrainedModel {
};
var DeiTForImageClassification = class extends DeiTPreTrainedModel {
  /**
   * @param {any} model_inputs
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var HieraPreTrainedModel = class extends PreTrainedModel {
};
var HieraModel = class extends HieraPreTrainedModel {
};
var HieraForImageClassification = class extends HieraPreTrainedModel {
  /**
   * @param {any} model_inputs
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var ResNetPreTrainedModel = class extends PreTrainedModel {
};
var ResNetModel = class extends ResNetPreTrainedModel {
};
var ResNetForImageClassification = class extends ResNetPreTrainedModel {
  /**
   * @param {any} model_inputs
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var SwinPreTrainedModel = class extends PreTrainedModel {
};
var SwinModel = class extends SwinPreTrainedModel {
};
var SwinForImageClassification = class extends SwinPreTrainedModel {
  /**
   * @param {any} model_inputs
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var SwinForSemanticSegmentation = class extends SwinPreTrainedModel {
};
var Swin2SRPreTrainedModel = class extends PreTrainedModel {
};
var Swin2SRModel = class extends Swin2SRPreTrainedModel {
};
var Swin2SRForImageSuperResolution = class extends Swin2SRPreTrainedModel {
};
var DPTPreTrainedModel = class extends PreTrainedModel {
};
var DPTModel = class extends DPTPreTrainedModel {
};
var DPTForDepthEstimation = class extends DPTPreTrainedModel {
};
var DepthAnythingPreTrainedModel = class extends PreTrainedModel {
};
var DepthAnythingForDepthEstimation = class extends DepthAnythingPreTrainedModel {
};
var SapiensPreTrainedModel = class extends PreTrainedModel {
};
var SapiensForSemanticSegmentation = class extends SapiensPreTrainedModel {
};
var SapiensForDepthEstimation = class extends SapiensPreTrainedModel {
};
var SapiensForNormalEstimation = class extends SapiensPreTrainedModel {
};
var DepthProPreTrainedModel = class extends PreTrainedModel {
};
var DepthProForDepthEstimation = class extends DepthProPreTrainedModel {
};
var Metric3DPreTrainedModel = class extends PreTrainedModel {
};
var Metric3DForDepthEstimation = class extends Metric3DPreTrainedModel {
};
var Metric3Dv2PreTrainedModel = class extends PreTrainedModel {
};
var Metric3Dv2ForDepthEstimation = class extends Metric3Dv2PreTrainedModel {
};
var MaskFormerPreTrainedModel = class extends PreTrainedModel {
};
var MaskFormerModel = class extends MaskFormerPreTrainedModel {
};
var MaskFormerForInstanceSegmentation = class extends MaskFormerPreTrainedModel {
};
var GLPNPreTrainedModel = class extends PreTrainedModel {
};
var GLPNModel = class extends GLPNPreTrainedModel {
};
var GLPNForDepthEstimation = class extends GLPNPreTrainedModel {
};
var DonutSwinPreTrainedModel = class extends PreTrainedModel {
};
var DonutSwinModel = class extends DonutSwinPreTrainedModel {
};
var ConvNextPreTrainedModel = class extends PreTrainedModel {
};
var ConvNextModel = class extends ConvNextPreTrainedModel {
};
var ConvNextForImageClassification = class extends ConvNextPreTrainedModel {
  /**
   * @param {any} model_inputs
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var ConvNextV2PreTrainedModel = class extends PreTrainedModel {
};
var ConvNextV2Model = class extends ConvNextV2PreTrainedModel {
};
var ConvNextV2ForImageClassification = class extends ConvNextV2PreTrainedModel {
  /**
   * @param {any} model_inputs
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var Dinov2PreTrainedModel = class extends PreTrainedModel {
};
var Dinov2Model = class extends Dinov2PreTrainedModel {
};
var Dinov2ForImageClassification = class extends Dinov2PreTrainedModel {
  /**
   * @param {any} model_inputs
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var Dinov2WithRegistersPreTrainedModel = class extends PreTrainedModel {
};
var Dinov2WithRegistersModel = class extends Dinov2WithRegistersPreTrainedModel {
};
var Dinov2WithRegistersForImageClassification = class extends Dinov2WithRegistersPreTrainedModel {
  /**
   * @param {any} model_inputs
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var DINOv3ViTPreTrainedModel = class extends PreTrainedModel {
};
var DINOv3ViTModel = class extends DINOv3ViTPreTrainedModel {
};
var DINOv3ConvNextPreTrainedModel = class extends PreTrainedModel {
};
var DINOv3ConvNextModel = class extends DINOv3ConvNextPreTrainedModel {
};
var GroundingDinoPreTrainedModel = class extends PreTrainedModel {
};
var GroundingDinoForObjectDetection = class extends GroundingDinoPreTrainedModel {
};
var YolosPreTrainedModel = class extends PreTrainedModel {
};
var YolosModel = class extends YolosPreTrainedModel {
};
var YolosForObjectDetection = class extends YolosPreTrainedModel {
  /**
   * @param {any} model_inputs
   */
  async _call(model_inputs) {
    return new YolosObjectDetectionOutput(await super._call(model_inputs));
  }
};
var YolosObjectDetectionOutput = class extends ModelOutput {
  /**
   * @param {Object} output The output of the model.
   * @param {Tensor} output.logits Classification logits (including no-object) for all queries.
   * @param {Tensor} output.pred_boxes Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height).
   * These values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding possible padding).
   */
  constructor({ logits, pred_boxes }) {
    super();
    this.logits = logits;
    this.pred_boxes = pred_boxes;
  }
};
var SamPreTrainedModel = class extends PreTrainedModel {
};
var SamModel = class extends SamPreTrainedModel {
  /**
   * Compute image embeddings and positional image embeddings, given the pixel values of an image.
   * @param {Object} model_inputs Object containing the model inputs.
   * @param {Tensor} model_inputs.pixel_values Pixel values obtained using a `SamProcessor`.
   * @returns {Promise<{ image_embeddings: Tensor, image_positional_embeddings: Tensor }>} The image embeddings and positional image embeddings.
   */
  async get_image_embeddings({ pixel_values }) {
    return await encoderForward(this, { pixel_values });
  }
  /**
   * @typedef {Object} SamModelInputs Object containing the model inputs.
   * @property {Tensor} pixel_values Pixel values as a Tensor with shape `(batch_size, num_channels, height, width)`.
   * These can be obtained using a `SamProcessor`.
   * @property {Tensor} [input_points] Input 2D spatial points with shape `(batch_size, num_points, 2)`.
   * This is used by the prompt encoder to encode the prompt.
   * @property {Tensor} [input_labels] Input labels for the points, as a Tensor of shape `(batch_size, point_batch_size, num_points)`.
   * This is used by the prompt encoder to encode the prompt. There are 4 types of labels:
   *  - `1`: the point is a point that contains the object of interest
   *  - `0`: the point is a point that does not contain the object of interest
   *  - `-1`: the point corresponds to the background
   *  - `-10`: the point is a padding point, thus should be ignored by the prompt encoder
   * @property {Tensor} [input_boxes] Input bounding boxes with shape `(batch_size, num_boxes, 4)`.
   * @property {Tensor} [image_embeddings] Image embeddings used by the mask decoder.
   * @property {Tensor} [image_positional_embeddings] Image positional embeddings used by the mask decoder.
   */
  /**
   * @param {SamModelInputs} model_inputs Object containing the model inputs.
   * @returns {Promise<Object>} The output of the model.
   */
  async forward(model_inputs) {
    if (!model_inputs.image_embeddings || !model_inputs.image_positional_embeddings) {
      model_inputs = {
        ...model_inputs,
        ...await this.get_image_embeddings(model_inputs)
      };
    } else {
      model_inputs = { ...model_inputs };
    }
    model_inputs.input_labels ??= ones(model_inputs.input_points.dims.slice(0, -1));
    const decoder_inputs = {
      image_embeddings: model_inputs.image_embeddings,
      image_positional_embeddings: model_inputs.image_positional_embeddings
    };
    if (model_inputs.input_points) {
      decoder_inputs.input_points = model_inputs.input_points;
    }
    if (model_inputs.input_labels) {
      decoder_inputs.input_labels = model_inputs.input_labels;
    }
    if (model_inputs.input_boxes) {
      decoder_inputs.input_boxes = model_inputs.input_boxes;
    }
    return await sessionRun(this.sessions["prompt_encoder_mask_decoder"], decoder_inputs);
  }
  /**
   * Runs the model with the provided inputs
   * @param {Object} model_inputs Model inputs
   * @returns {Promise<SamImageSegmentationOutput>} Object containing segmentation outputs
   */
  async _call(model_inputs) {
    return new SamImageSegmentationOutput(await super._call(model_inputs));
  }
};
var SamImageSegmentationOutput = class extends ModelOutput {
  /**
   * @param {Object} output The output of the model.
   * @param {Tensor} output.iou_scores The output logits of the model.
   * @param {Tensor} output.pred_masks Predicted boxes.
   */
  constructor({ iou_scores, pred_masks }) {
    super();
    this.iou_scores = iou_scores;
    this.pred_masks = pred_masks;
  }
};
var Sam2ImageSegmentationOutput = class extends ModelOutput {
  /**
   * @param {Object} output The output of the model.
   * @param {Tensor} output.iou_scores The output logits of the model.
   * @param {Tensor} output.pred_masks Predicted boxes.
   * @param {Tensor} output.object_score_logits Logits for the object score, indicating if an object is present.
   */
  constructor({ iou_scores, pred_masks, object_score_logits }) {
    super();
    this.iou_scores = iou_scores;
    this.pred_masks = pred_masks;
    this.object_score_logits = object_score_logits;
  }
};
var EdgeTamPreTrainedModel = class extends PreTrainedModel {
};
var EdgeTamModel = class extends EdgeTamPreTrainedModel {
  /**
   * Compute image embeddings and positional image embeddings, given the pixel values of an image.
   * @param {Object} model_inputs Object containing the model inputs.
   * @param {Tensor} model_inputs.pixel_values Pixel values obtained using a `Sam2Processor`.
   * @returns {Promise<Record<String, Tensor>>} The image embeddings.
   */
  async get_image_embeddings({ pixel_values }) {
    return await encoderForward(this, { pixel_values });
  }
  async forward(model_inputs) {
    const { num_feature_levels } = this.config.vision_config;
    const image_embeddings_name = Array.from({ length: num_feature_levels }, (_, i) => `image_embeddings.${i}`);
    if (image_embeddings_name.some((name) => !model_inputs[name])) {
      model_inputs = {
        ...model_inputs,
        ...await this.get_image_embeddings(model_inputs)
      };
    } else {
      model_inputs = { ...model_inputs };
    }
    if (model_inputs.input_points) {
      if (model_inputs.input_boxes && model_inputs.input_boxes.dims[1] !== 1) {
        throw new Error("When both `input_points` and `input_boxes` are provided, the number of boxes per image must be 1.");
      }
      const shape = model_inputs.input_points.dims;
      model_inputs.input_labels ??= ones(shape.slice(0, -1));
      model_inputs.input_boxes ??= full([shape[0], 0, 4], 0);
    } else if (model_inputs.input_boxes) {
      const shape = model_inputs.input_boxes.dims;
      model_inputs.input_labels = full([shape[0], shape[1], 0], -1n);
      model_inputs.input_points = full([shape[0], 1, 0, 2], 0);
    } else {
      throw new Error("At least one of `input_points` or `input_boxes` must be provided.");
    }
    const prompt_encoder_mask_decoder_session = this.sessions["prompt_encoder_mask_decoder"];
    const decoder_inputs = pick(model_inputs, prompt_encoder_mask_decoder_session.inputNames);
    return await sessionRun(prompt_encoder_mask_decoder_session, decoder_inputs);
  }
  /**
   * Runs the model with the provided inputs
   * @param {Object} model_inputs Model inputs
   * @returns {Promise<Sam2ImageSegmentationOutput>} Object containing segmentation outputs
   */
  async _call(model_inputs) {
    return new Sam2ImageSegmentationOutput(await super._call(model_inputs));
  }
};
var MarianPreTrainedModel = class extends PreTrainedModel {
};
var MarianModel = class extends MarianPreTrainedModel {
};
var MarianMTModel = class extends MarianPreTrainedModel {
};
var M2M100PreTrainedModel = class extends PreTrainedModel {
};
var M2M100Model = class extends M2M100PreTrainedModel {
};
var M2M100ForConditionalGeneration = class extends M2M100PreTrainedModel {
};
var Wav2Vec2PreTrainedModel = class extends PreTrainedModel {
};
var Wav2Vec2Model = class extends Wav2Vec2PreTrainedModel {
};
var Wav2Vec2ForCTC = class extends Wav2Vec2PreTrainedModel {
  /**
   * @param {Object} model_inputs
   * @param {Tensor} model_inputs.input_values Float values of input raw speech waveform.
   * @param {Tensor} model_inputs.attention_mask Mask to avoid performing convolution and attention on padding token indices. Mask values selected in [0, 1]
   */
  async _call(model_inputs) {
    return new CausalLMOutput(await super._call(model_inputs));
  }
};
var Wav2Vec2ForSequenceClassification = class extends Wav2Vec2PreTrainedModel {
  /**
   * Calls the model on new inputs.
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<SequenceClassifierOutput>} An object containing the model's output logits for sequence classification.
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var Wav2Vec2ForAudioFrameClassification = class extends Wav2Vec2PreTrainedModel {
  /**
   * Calls the model on new inputs.
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<TokenClassifierOutput>} An object containing the model's output logits for sequence classification.
   */
  async _call(model_inputs) {
    return new TokenClassifierOutput(await super._call(model_inputs));
  }
};
var ParakeetPreTrainedModel = class extends PreTrainedModel {
};
var ParakeetForCTC = class extends ParakeetPreTrainedModel {
  /**
   * @param {Object} model_inputs
   * @param {Tensor} model_inputs.input_values Float values of input raw speech waveform.
   * @param {Tensor} model_inputs.attention_mask Mask to avoid performing convolution and attention on padding token indices. Mask values selected in [0, 1]
   */
  async _call(model_inputs) {
    return new CausalLMOutput(await super._call(model_inputs));
  }
};
var PyAnnotePreTrainedModel = class extends PreTrainedModel {
};
var PyAnnoteModel = class extends PyAnnotePreTrainedModel {
};
var PyAnnoteForAudioFrameClassification = class extends PyAnnotePreTrainedModel {
  /**
   * Calls the model on new inputs.
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<TokenClassifierOutput>} An object containing the model's output logits for sequence classification.
   */
  async _call(model_inputs) {
    return new TokenClassifierOutput(await super._call(model_inputs));
  }
};
var WeSpeakerResNetPreTrainedModel = class extends PreTrainedModel {
};
var WeSpeakerResNetModel = class extends WeSpeakerResNetPreTrainedModel {
};
var UniSpeechPreTrainedModel = class extends PreTrainedModel {
};
var UniSpeechModel = class extends UniSpeechPreTrainedModel {
};
var UniSpeechForCTC = class extends UniSpeechPreTrainedModel {
  /**
   * @param {Object} model_inputs
   * @param {Tensor} model_inputs.input_values Float values of input raw speech waveform.
   * @param {Tensor} model_inputs.attention_mask Mask to avoid performing convolution and attention on padding token indices. Mask values selected in [0, 1]
   */
  async _call(model_inputs) {
    return new CausalLMOutput(await super._call(model_inputs));
  }
};
var UniSpeechForSequenceClassification = class extends UniSpeechPreTrainedModel {
  /**
   * Calls the model on new inputs.
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<SequenceClassifierOutput>} An object containing the model's output logits for sequence classification.
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var UniSpeechSatPreTrainedModel = class extends PreTrainedModel {
};
var UniSpeechSatModel = class extends UniSpeechSatPreTrainedModel {
};
var UniSpeechSatForCTC = class extends UniSpeechSatPreTrainedModel {
  /**
   * @param {Object} model_inputs
   * @param {Tensor} model_inputs.input_values Float values of input raw speech waveform.
   * @param {Tensor} model_inputs.attention_mask Mask to avoid performing convolution and attention on padding token indices. Mask values selected in [0, 1]
   */
  async _call(model_inputs) {
    return new CausalLMOutput(await super._call(model_inputs));
  }
};
var UniSpeechSatForSequenceClassification = class extends UniSpeechSatPreTrainedModel {
  /**
   * Calls the model on new inputs.
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<SequenceClassifierOutput>} An object containing the model's output logits for sequence classification.
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var UniSpeechSatForAudioFrameClassification = class extends UniSpeechSatPreTrainedModel {
  /**
   * Calls the model on new inputs.
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<TokenClassifierOutput>} An object containing the model's output logits for sequence classification.
   */
  async _call(model_inputs) {
    return new TokenClassifierOutput(await super._call(model_inputs));
  }
};
var Wav2Vec2BertPreTrainedModel = class extends PreTrainedModel {
};
var Wav2Vec2BertModel = class extends Wav2Vec2BertPreTrainedModel {
};
var Wav2Vec2BertForCTC = class extends Wav2Vec2BertPreTrainedModel {
  /**
   * @param {Object} model_inputs
   * @param {Tensor} model_inputs.input_features Float values of input mel-spectrogram.
   * @param {Tensor} model_inputs.attention_mask Mask to avoid performing convolution and attention on padding token indices. Mask values selected in [0, 1]
   */
  async _call(model_inputs) {
    return new CausalLMOutput(await super._call(model_inputs));
  }
};
var Wav2Vec2BertForSequenceClassification = class extends Wav2Vec2BertPreTrainedModel {
  /**
   * Calls the model on new inputs.
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<SequenceClassifierOutput>} An object containing the model's output logits for sequence classification.
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var HubertPreTrainedModel = class extends PreTrainedModel {
};
var HubertModel = class extends Wav2Vec2PreTrainedModel {
};
var HubertForCTC = class extends Wav2Vec2PreTrainedModel {
  /**
   * @param {Object} model_inputs
   * @param {Tensor} model_inputs.input_values Float values of input raw speech waveform.
   * @param {Tensor} model_inputs.attention_mask Mask to avoid performing convolution and attention on padding token indices. Mask values selected in [0, 1]
   */
  async _call(model_inputs) {
    return new CausalLMOutput(await super._call(model_inputs));
  }
};
var HubertForSequenceClassification = class extends Wav2Vec2PreTrainedModel {
  /**
   * Calls the model on new inputs.
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<SequenceClassifierOutput>} An object containing the model's output logits for sequence classification.
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var WavLMPreTrainedModel = class extends PreTrainedModel {
};
var WavLMModel = class extends WavLMPreTrainedModel {
};
var WavLMForCTC = class extends WavLMPreTrainedModel {
  /**
   * @param {Object} model_inputs
   * @param {Tensor} model_inputs.input_values Float values of input raw speech waveform.
   * @param {Tensor} model_inputs.attention_mask Mask to avoid performing convolution and attention on padding token indices. Mask values selected in [0, 1]
   */
  async _call(model_inputs) {
    return new CausalLMOutput(await super._call(model_inputs));
  }
};
var WavLMForSequenceClassification = class extends WavLMPreTrainedModel {
  /**
   * Calls the model on new inputs.
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<SequenceClassifierOutput>} An object containing the model's output logits for sequence classification.
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var WavLMForXVector = class extends WavLMPreTrainedModel {
  /**
   * Calls the model on new inputs.
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<XVectorOutput>} An object containing the model's output logits and speaker embeddings.
   */
  async _call(model_inputs) {
    return new XVectorOutput(await super._call(model_inputs));
  }
};
var WavLMForAudioFrameClassification = class extends WavLMPreTrainedModel {
  /**
   * Calls the model on new inputs.
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<TokenClassifierOutput>} An object containing the model's output logits for sequence classification.
   */
  async _call(model_inputs) {
    return new TokenClassifierOutput(await super._call(model_inputs));
  }
};
var StyleTextToSpeech2PreTrainedModel = class extends PreTrainedModel {
};
var StyleTextToSpeech2Model = class extends StyleTextToSpeech2PreTrainedModel {
};
var SpeechT5PreTrainedModel = class extends PreTrainedModel {
};
var SpeechT5Model = class extends SpeechT5PreTrainedModel {
};
var SpeechT5ForSpeechToText = class extends SpeechT5PreTrainedModel {
};
var SpeechT5ForTextToSpeech = class extends SpeechT5PreTrainedModel {
  /**
   * @typedef {Object} SpeechOutput
   * @property {Tensor} [spectrogram] The predicted log-mel spectrogram of shape
   * `(output_sequence_length, config.num_mel_bins)`. Returned when no `vocoder` is provided
   * @property {Tensor} [waveform] The predicted waveform of shape `(num_frames,)`. Returned when a `vocoder` is provided.
   * @property {Tensor} [cross_attentions] The outputs of the decoder's cross-attention layers of shape
   * `(config.decoder_layers, config.decoder_attention_heads, output_sequence_length, input_sequence_length)`. returned when `output_cross_attentions` is `true`.
   */
  /**
   * Converts a sequence of input tokens into a sequence of mel spectrograms, which are subsequently turned into a speech waveform using a vocoder.
   * @param {Tensor} input_values Indices of input sequence tokens in the vocabulary.
   * @param {Tensor} speaker_embeddings Tensor containing the speaker embeddings.
   * @param {Object} options Optional parameters for generating speech.
   * @param {number} [options.threshold=0.5] The generated sequence ends when the predicted stop token probability exceeds this value.
   * @param {number} [options.minlenratio=0.0] Used to calculate the minimum required length for the output sequence.
   * @param {number} [options.maxlenratio=20.0] Used to calculate the maximum allowed length for the output sequence.
   * @param {Object} [options.vocoder=null] The vocoder that converts the mel spectrogram into a speech waveform. If `null`, the output is the mel spectrogram.
   * @param {boolean} [options.output_cross_attentions=false] Whether or not to return the attentions tensors of the decoder's cross-attention layers.
   * @returns {Promise<SpeechOutput>} A promise which resolves to an object containing the spectrogram, waveform, and cross-attention tensors.
   */
  async generate_speech(input_values, speaker_embeddings, {
    threshold = 0.5,
    minlenratio = 0,
    maxlenratio = 20,
    vocoder = null
    // output_cross_attentions = false, // TODO add
  } = {}) {
    const model_inputs = {
      input_ids: input_values
    };
    const { encoder_outputs, encoder_attention_mask } = await encoderForward(this, model_inputs);
    const r = encoder_outputs.dims[1] / this.config.reduction_factor;
    const maxlen = Math.floor(r * maxlenratio);
    const minlen = Math.floor(r * minlenratio);
    const num_mel_bins = this.config.num_mel_bins;
    let spectrogramParts = [];
    let past_key_values = null;
    let decoder_outputs = null;
    let idx = 0;
    while (true) {
      ++idx;
      const use_cache_branch = boolTensor(!!decoder_outputs);
      let output_sequence;
      if (decoder_outputs) {
        output_sequence = decoder_outputs.output_sequence_out;
      } else {
        output_sequence = new Tensor2(
          "float32",
          new Float32Array(num_mel_bins),
          [1, 1, num_mel_bins]
        );
      }
      let decoderFeeds = {
        use_cache_branch,
        output_sequence,
        encoder_attention_mask,
        speaker_embeddings,
        encoder_hidden_states: encoder_outputs
      };
      this.addPastKeyValues(decoderFeeds, past_key_values);
      decoder_outputs = await sessionRun(this.sessions["decoder_model_merged"], decoderFeeds);
      past_key_values = this.getPastKeyValues(decoder_outputs, past_key_values);
      const { prob, spectrum } = decoder_outputs;
      spectrogramParts.push(spectrum);
      if (idx >= minlen && // Finished when stop token or maximum length is reached.
      (Array.from(prob.data).filter((p) => p >= threshold).length > 0 || idx >= maxlen)) {
        break;
      }
    }
    const spectrogram2 = cat(spectrogramParts);
    const { waveform } = await sessionRun(vocoder.sessions["model"], { spectrogram: spectrogram2 });
    return {
      spectrogram: spectrogram2,
      waveform
      // cross_attentions: null, // TODO add
    };
  }
};
var SpeechT5HifiGan = class extends PreTrainedModel {
  main_input_name = "spectrogram";
};
var TrOCRPreTrainedModel = class extends PreTrainedModel {
};
var TrOCRForCausalLM = class extends TrOCRPreTrainedModel {
};
var MistralPreTrainedModel = class extends PreTrainedModel {
};
var MistralModel = class extends MistralPreTrainedModel {
};
var MistralForCausalLM = class extends MistralPreTrainedModel {
};
var Ernie4_5_PretrainedModel = class extends PreTrainedModel {
};
var Ernie4_5_Model = class extends Ernie4_5_PretrainedModel {
};
var Ernie4_5_ForCausalLM = class extends Ernie4_5_PretrainedModel {
};
var Starcoder2PreTrainedModel = class extends PreTrainedModel {
};
var Starcoder2Model = class extends Starcoder2PreTrainedModel {
};
var Starcoder2ForCausalLM = class extends Starcoder2PreTrainedModel {
};
var FalconPreTrainedModel = class extends PreTrainedModel {
};
var FalconModel = class extends FalconPreTrainedModel {
};
var FalconForCausalLM = class extends FalconPreTrainedModel {
};
var ClapPreTrainedModel = class extends PreTrainedModel {
};
var ClapModel = class extends ClapPreTrainedModel {
};
var ClapTextModelWithProjection = class extends ClapPreTrainedModel {
  /** @type {typeof PreTrainedModel.from_pretrained} */
  static async from_pretrained(pretrained_model_name_or_path, options = {}) {
    return super.from_pretrained(pretrained_model_name_or_path, {
      ...options,
      // Update default model file name if not provided
      model_file_name: options.model_file_name ?? "text_model"
    });
  }
};
var ClapAudioModelWithProjection = class extends ClapPreTrainedModel {
  /** @type {typeof PreTrainedModel.from_pretrained} */
  static async from_pretrained(pretrained_model_name_or_path, options = {}) {
    return super.from_pretrained(pretrained_model_name_or_path, {
      ...options,
      // Update default model file name if not provided
      model_file_name: options.model_file_name ?? "audio_model"
    });
  }
};
var VitsPreTrainedModel = class extends PreTrainedModel {
};
var VitsModel = class extends VitsPreTrainedModel {
  /**
   * Calls the model on new inputs.
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<VitsModelOutput>} The outputs for the VITS model.
   */
  async _call(model_inputs) {
    return new VitsModelOutput(await super._call(model_inputs));
  }
};
var SegformerPreTrainedModel = class extends PreTrainedModel {
};
var SegformerModel = class extends SegformerPreTrainedModel {
};
var SegformerForImageClassification = class extends SegformerPreTrainedModel {
};
var SegformerForSemanticSegmentation = class extends SegformerPreTrainedModel {
};
var StableLmPreTrainedModel = class extends PreTrainedModel {
};
var StableLmModel = class extends StableLmPreTrainedModel {
};
var StableLmForCausalLM = class extends StableLmPreTrainedModel {
};
var EfficientNetPreTrainedModel = class extends PreTrainedModel {
};
var EfficientNetModel = class extends EfficientNetPreTrainedModel {
};
var EfficientNetForImageClassification = class extends EfficientNetPreTrainedModel {
  /**
   * @param {any} model_inputs
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var MusicgenPreTrainedModel = class extends PreTrainedModel {
};
var MusicgenModel = class extends MusicgenPreTrainedModel {
};
var MusicgenForCausalLM = class extends MusicgenPreTrainedModel {
};
var MusicgenForConditionalGeneration = class extends PreTrainedModel {
  // NOTE: not MusicgenPreTrainedModel
  forward_params = [
    "input_ids",
    "attention_mask",
    "encoder_outputs",
    "decoder_input_ids",
    "decoder_attention_mask",
    "past_key_values"
  ];
  /**
   * Apply the pattern mask to the final ids,
   * then revert the pattern delay mask by filtering the pad token id in a single step.
   * @param {Tensor} outputs The output tensor from the model.
   * @returns {Tensor} The filtered output tensor.
   */
  _apply_and_filter_by_delay_pattern_mask(outputs) {
    const [bs_x_codebooks, seqLength] = outputs.dims;
    const num_codebooks = this.config.decoder.num_codebooks;
    const upperBound = seqLength - num_codebooks;
    let newDataSize = 0;
    for (let i = 0; i < outputs.size; ++i) {
      if (outputs.data[i] === this.config.decoder.pad_token_id) {
        continue;
      }
      const row = i % seqLength;
      const col = Math.floor(i / seqLength) % num_codebooks;
      const diff = row - col;
      if (diff > 0 && diff <= upperBound) {
        outputs.data[newDataSize++] = outputs.data[i];
      }
    }
    const batch_size = Math.floor(bs_x_codebooks / num_codebooks);
    const inferred = newDataSize / (batch_size * num_codebooks);
    return new Tensor2(
      outputs.type,
      outputs.data.slice(0, newDataSize),
      [batch_size, num_codebooks, inferred]
    );
  }
  prepare_inputs_for_generation(input_ids, model_inputs, generation_config) {
    let clonedInputIds = structuredClone(input_ids);
    for (let i = 0; i < clonedInputIds.length; ++i) {
      for (let j = 0; j < clonedInputIds[i].length; ++j) {
        if (i % this.config.decoder.num_codebooks >= j) {
          clonedInputIds[i][j] = BigInt(this.config.decoder.pad_token_id);
        }
      }
    }
    if (generation_config.guidance_scale !== null && generation_config.guidance_scale > 1) {
      clonedInputIds = clonedInputIds.concat(clonedInputIds);
    }
    const prepped = super.prepare_inputs_for_generation(clonedInputIds, model_inputs, generation_config);
    return prepped;
  }
  /**
   * Generates sequences of token ids for models with a language modeling head.
   * @param {import('./generation/parameters.js').GenerationFunctionParameters} options
   * @returns {Promise<ModelOutput|Tensor>} The output of the model, which can contain the generated token ids, attentions, and scores.
   */
  async generate(options) {
    const output_ids = await super.generate(options);
    const audio_codes = this._apply_and_filter_by_delay_pattern_mask(
      /** @type {Tensor} */
      output_ids
    ).unsqueeze_(0);
    const { audio_values } = await sessionRun(this.sessions["encodec_decode"], { audio_codes });
    return audio_values;
  }
};
var MobileNetV1PreTrainedModel = class extends PreTrainedModel {
};
var MobileNetV1Model = class extends MobileNetV1PreTrainedModel {
};
var MobileNetV1ForImageClassification = class extends MobileNetV1PreTrainedModel {
  /**
   * @param {any} model_inputs
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var MobileNetV1ForSemanticSegmentation = class extends MobileNetV1PreTrainedModel {
};
var MobileNetV2PreTrainedModel = class extends PreTrainedModel {
};
var MobileNetV2Model = class extends MobileNetV2PreTrainedModel {
};
var MobileNetV2ForImageClassification = class extends MobileNetV2PreTrainedModel {
  /**
   * @param {any} model_inputs
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var MobileNetV2ForSemanticSegmentation = class extends MobileNetV2PreTrainedModel {
};
var MobileNetV3PreTrainedModel = class extends PreTrainedModel {
};
var MobileNetV3Model = class extends MobileNetV3PreTrainedModel {
};
var MobileNetV3ForImageClassification = class extends MobileNetV3PreTrainedModel {
  /**
   * @param {any} model_inputs
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var MobileNetV3ForSemanticSegmentation = class extends MobileNetV3PreTrainedModel {
};
var MobileNetV4PreTrainedModel = class extends PreTrainedModel {
};
var MobileNetV4Model = class extends MobileNetV4PreTrainedModel {
};
var MobileNetV4ForImageClassification = class extends MobileNetV4PreTrainedModel {
  /**
   * @param {any} model_inputs
   */
  async _call(model_inputs) {
    return new SequenceClassifierOutput(await super._call(model_inputs));
  }
};
var MobileNetV4ForSemanticSegmentation = class extends MobileNetV4PreTrainedModel {
};
var DecisionTransformerPreTrainedModel = class extends PreTrainedModel {
};
var DecisionTransformerModel = class extends DecisionTransformerPreTrainedModel {
};
var MultiModalityPreTrainedModel = class extends PreTrainedModel {
};
var MultiModalityCausalLM = class extends MultiModalityPreTrainedModel {
  forward_params = [
    // prepare_inputs_embeds
    "input_ids",
    "pixel_values",
    "images_seq_mask",
    "images_emb_mask",
    // language_model
    "attention_mask",
    "position_ids",
    "past_key_values"
  ];
  /**
   * @param {ConstructorParameters<typeof MultiModalityPreTrainedModel>} args
   */
  constructor(...args) {
    super(...args);
    this._generation_mode = "text";
  }
  async forward(model_inputs) {
    const mode = this._generation_mode ?? "text";
    let output_1;
    if (mode === "text" || !model_inputs.past_key_values) {
      const session = this.sessions["prepare_inputs_embeds"];
      const prep_inputs = pick(model_inputs, session.inputNames);
      output_1 = await sessionRun(session, prep_inputs);
    } else {
      const session = this.sessions["gen_img_embeds"];
      const prep_inputs = pick({
        image_ids: model_inputs.input_ids
      }, session.inputNames);
      output_1 = await sessionRun(session, prep_inputs);
    }
    const input_2 = { ...model_inputs, ...output_1 };
    const output_2 = await decoderForward(this, input_2);
    const head = this.sessions[mode === "text" ? "lm_head" : "gen_head"];
    if (!head) {
      throw new Error(`Unable to find "${head}" generation head`);
    }
    const output_3 = await sessionRun(head, pick(output_2, head.inputNames));
    return {
      ...output_1,
      ...output_2,
      ...output_3
    };
  }
  /**
   * @param {import('./generation/parameters.js').GenerationFunctionParameters} options
   */
  async generate(options) {
    this._generation_mode = "text";
    return super.generate(options);
  }
  /**
   * @param {import('./generation/parameters.js').GenerationFunctionParameters} options
   */
  async generate_images(options) {
    this._generation_mode = "image";
    const start_num_tokens = (options.inputs ?? options[this.main_input_name]).dims[1];
    const all_tokens = await super.generate(options);
    const generated_tokens = (
      /** @type {Tensor} */
      all_tokens.slice(null, [start_num_tokens, null])
    );
    const image_decode = this.sessions["image_decode"];
    const { decoded_image } = await sessionRun(image_decode, {
      generated_tokens
    });
    const clamped = decoded_image.add_(1).mul_(255 / 2).clamp_(0, 255).to("uint8");
    const images = [];
    for (const tensor of clamped) {
      const img = RawImage.fromTensor(tensor);
      images.push(img);
    }
    return images;
  }
};
var MgpstrModelOutput = class extends ModelOutput {
  constructor({ char_logits, bpe_logits, wp_logits }) {
    super();
    this.char_logits = char_logits;
    this.bpe_logits = bpe_logits;
    this.wp_logits = wp_logits;
  }
  get logits() {
    return [this.char_logits, this.bpe_logits, this.wp_logits];
  }
};
var MgpstrPreTrainedModel = class extends PreTrainedModel {
};
var MgpstrForSceneTextRecognition = class extends MgpstrPreTrainedModel {
  /**
   * @param {any} model_inputs
   */
  async _call(model_inputs) {
    return new MgpstrModelOutput(await super._call(model_inputs));
  }
};
var PatchTSTPreTrainedModel = class extends PreTrainedModel {
};
var PatchTSTModel = class extends PatchTSTPreTrainedModel {
};
var PatchTSTForPrediction = class extends PatchTSTPreTrainedModel {
};
var PatchTSMixerPreTrainedModel = class extends PreTrainedModel {
};
var PatchTSMixerModel = class extends PatchTSMixerPreTrainedModel {
};
var PatchTSMixerForPrediction = class extends PatchTSMixerPreTrainedModel {
};
var UltravoxPreTrainedModel = class extends PreTrainedModel {
  forward_params = [
    "input_ids",
    "attention_mask",
    "position_ids",
    "audio_values",
    "past_key_values"
  ];
};
var UltravoxModel = class extends UltravoxPreTrainedModel {
  _merge_input_ids_with_audio_features(kwargs) {
    const audio_hidden_size = kwargs.audio_features.dims.at(-1);
    const reshaped_audio_features = kwargs.audio_features.view(-1, audio_hidden_size);
    return default_merge_input_ids_with_audio_features({
      // @ts-ignore
      audio_token_id: this.config.ignore_index ?? this.config.audio_token_id,
      ...kwargs,
      audio_features: reshaped_audio_features
    });
  }
};
var VoxtralForConditionalGeneration = class extends UltravoxModel {
};
var MimiPreTrainedModel = class extends PreTrainedModel {
  main_input_name = "input_values";
  forward_params = ["input_values"];
};
var MimiEncoderOutput = class extends ModelOutput {
  /**
   * @param {Object} output The output of the model.
   * @param {Tensor} output.audio_codes Discrete code embeddings, of shape `(batch_size, num_quantizers, codes_length)`.
   */
  constructor({ audio_codes }) {
    super();
    this.audio_codes = audio_codes;
  }
};
var MimiDecoderOutput = class extends ModelOutput {
  /**
   * @param {Object} output The output of the model.
   * @param {Tensor} output.audio_values Decoded audio values, of shape `(batch_size, num_channels, sequence_length)`.
   */
  constructor({ audio_values }) {
    super();
    this.audio_values = audio_values;
  }
};
var MimiModel = class extends MimiPreTrainedModel {
  /**
   * Encodes the input audio waveform into discrete codes.
   * @param {Object} inputs Model inputs
   * @param {Tensor} [inputs.input_values] Float values of the input audio waveform, of shape `(batch_size, channels, sequence_length)`).
   * @returns {Promise<MimiEncoderOutput>} The output tensor of shape `(batch_size, num_codebooks, sequence_length)`.
   */
  async encode(inputs) {
    return new MimiEncoderOutput(await sessionRun(this.sessions["encoder_model"], inputs));
  }
  /**
   * Decodes the given frames into an output audio waveform.
   * @param {MimiEncoderOutput} inputs The encoded audio codes.
   * @returns {Promise<MimiDecoderOutput>} The output tensor of shape `(batch_size, num_channels, sequence_length)`.
   */
  async decode(inputs) {
    return new MimiDecoderOutput(await sessionRun(this.sessions["decoder_model"], inputs));
  }
};
var MimiEncoderModel = class extends MimiPreTrainedModel {
  /** @type {typeof PreTrainedModel.from_pretrained} */
  static async from_pretrained(pretrained_model_name_or_path, options = {}) {
    return super.from_pretrained(pretrained_model_name_or_path, {
      ...options,
      // Update default model file name if not provided
      model_file_name: options.model_file_name ?? "encoder_model"
    });
  }
};
var MimiDecoderModel = class extends MimiPreTrainedModel {
  /** @type {typeof PreTrainedModel.from_pretrained} */
  static async from_pretrained(pretrained_model_name_or_path, options = {}) {
    return super.from_pretrained(pretrained_model_name_or_path, {
      ...options,
      // Update default model file name if not provided
      model_file_name: options.model_file_name ?? "decoder_model"
    });
  }
};
var DacPreTrainedModel = class extends PreTrainedModel {
  main_input_name = "input_values";
  forward_params = ["input_values"];
};
var DacEncoderOutput = class extends ModelOutput {
  /**
   * @param {Object} output The output of the model.
   * @param {Tensor} output.audio_codes Discrete code embeddings, of shape `(batch_size, num_quantizers, codes_length)`.
   */
  constructor({ audio_codes }) {
    super();
    this.audio_codes = audio_codes;
  }
};
var DacDecoderOutput = class extends ModelOutput {
  /**
   * @param {Object} output The output of the model.
   * @param {Tensor} output.audio_values Decoded audio values, of shape `(batch_size, num_channels, sequence_length)`.
   */
  constructor({ audio_values }) {
    super();
    this.audio_values = audio_values;
  }
};
var DacModel = class extends DacPreTrainedModel {
  /**
   * Encodes the input audio waveform into discrete codes.
   * @param {Object} inputs Model inputs
   * @param {Tensor} [inputs.input_values] Float values of the input audio waveform, of shape `(batch_size, channels, sequence_length)`).
   * @returns {Promise<DacEncoderOutput>} The output tensor of shape `(batch_size, num_codebooks, sequence_length)`.
   */
  async encode(inputs) {
    return new DacEncoderOutput(await sessionRun(this.sessions["encoder_model"], inputs));
  }
  /**
   * Decodes the given frames into an output audio waveform.
   * @param {DacEncoderOutput} inputs The encoded audio codes.
   * @returns {Promise<DacDecoderOutput>} The output tensor of shape `(batch_size, num_channels, sequence_length)`.
   */
  async decode(inputs) {
    return new DacDecoderOutput(await sessionRun(this.sessions["decoder_model"], inputs));
  }
};
var DacEncoderModel = class extends DacPreTrainedModel {
  /** @type {typeof PreTrainedModel.from_pretrained} */
  static async from_pretrained(pretrained_model_name_or_path, options = {}) {
    return super.from_pretrained(pretrained_model_name_or_path, {
      ...options,
      // Update default model file name if not provided
      model_file_name: options.model_file_name ?? "encoder_model"
    });
  }
};
var DacDecoderModel = class extends DacPreTrainedModel {
  /** @type {typeof PreTrainedModel.from_pretrained} */
  static async from_pretrained(pretrained_model_name_or_path, options = {}) {
    return super.from_pretrained(pretrained_model_name_or_path, {
      ...options,
      // Update default model file name if not provided
      model_file_name: options.model_file_name ?? "decoder_model"
    });
  }
};
var SnacPreTrainedModel = class extends PreTrainedModel {
  main_input_name = "input_values";
  forward_params = ["input_values"];
};
var SnacModel = class extends SnacPreTrainedModel {
  /**
   * Encodes the input audio waveform into discrete codes.
   * @param {Object} inputs Model inputs
   * @param {Tensor} [inputs.input_values] Float values of the input audio waveform, of shape `(batch_size, channels, sequence_length)`).
   * @returns {Promise<Record<string, Tensor>>} The output tensors of shape `(batch_size, num_codebooks, sequence_length)`.
   */
  async encode(inputs) {
    return await sessionRun(this.sessions["encoder_model"], inputs);
  }
  /**
   * Decodes the given frames into an output audio waveform.
   * @param {Record<string, Tensor>} inputs The encoded audio codes.
   * @returns {Promise<{audio_values: Tensor}>} The output tensor of shape `(batch_size, num_channels, sequence_length)`.
   */
  async decode(inputs) {
    return await sessionRun(this.sessions["decoder_model"], inputs);
  }
};
var SnacEncoderModel = class extends SnacPreTrainedModel {
  /** @type {typeof PreTrainedModel.from_pretrained} */
  static async from_pretrained(pretrained_model_name_or_path, options = {}) {
    return super.from_pretrained(pretrained_model_name_or_path, {
      ...options,
      // Update default model file name if not provided
      model_file_name: options.model_file_name ?? "encoder_model"
    });
  }
};
var SnacDecoderModel = class extends SnacPreTrainedModel {
  /** @type {typeof PreTrainedModel.from_pretrained} */
  static async from_pretrained(pretrained_model_name_or_path, options = {}) {
    return super.from_pretrained(pretrained_model_name_or_path, {
      ...options,
      // Update default model file name if not provided
      model_file_name: options.model_file_name ?? "decoder_model"
    });
  }
};
var PretrainedMixin = class {
  /**
   * Mapping from model type to model class.
   * @type {Map<string, Object>[]}
   */
  static MODEL_CLASS_MAPPINGS = null;
  /**
   * Whether to attempt to instantiate the base class (`PretrainedModel`) if 
   * the model type is not found in the mapping.
   */
  static BASE_IF_FAIL = false;
  /** @type {typeof PreTrainedModel.from_pretrained} */
  static async from_pretrained(pretrained_model_name_or_path, {
    progress_callback = null,
    config = null,
    cache_dir = null,
    local_files_only = false,
    revision = "main",
    model_file_name = null,
    subfolder = "onnx",
    device = null,
    dtype = null,
    use_external_data_format = null,
    session_options = {}
  } = {}) {
    const options = {
      progress_callback,
      config,
      cache_dir,
      local_files_only,
      revision,
      model_file_name,
      subfolder,
      device,
      dtype,
      use_external_data_format,
      session_options
    };
    options.config = await AutoConfig.from_pretrained(pretrained_model_name_or_path, options);
    if (!this.MODEL_CLASS_MAPPINGS) {
      throw new Error("`MODEL_CLASS_MAPPINGS` not implemented for this type of `AutoClass`: " + this.name);
    }
    const model_type = options.config.model_type;
    for (const MODEL_CLASS_MAPPING of this.MODEL_CLASS_MAPPINGS) {
      let modelInfo = MODEL_CLASS_MAPPING.get(model_type);
      if (!modelInfo) {
        for (const cls of MODEL_CLASS_MAPPING.values()) {
          if (cls[0] === model_type) {
            modelInfo = cls;
            break;
          }
        }
        if (!modelInfo) continue;
      }
      return await modelInfo[1].from_pretrained(pretrained_model_name_or_path, options);
    }
    if (this.BASE_IF_FAIL) {
      if (!CUSTOM_ARCHITECTURES.has(model_type)) {
        console.warn(`Unknown model class "${model_type}", attempting to construct from base class.`);
      }
      return await PreTrainedModel.from_pretrained(pretrained_model_name_or_path, options);
    } else {
      throw Error(`Unsupported model type: ${model_type}`);
    }
  }
};
var MODEL_MAPPING_NAMES_ENCODER_ONLY = /* @__PURE__ */ new Map([
  ["bert", ["BertModel", BertModel]],
  ["neobert", ["NeoBertModel", NeoBertModel]],
  ["modernbert", ["ModernBertModel", ModernBertModel]],
  ["nomic_bert", ["NomicBertModel", NomicBertModel]],
  ["roformer", ["RoFormerModel", RoFormerModel]],
  ["electra", ["ElectraModel", ElectraModel]],
  ["esm", ["EsmModel", EsmModel]],
  ["convbert", ["ConvBertModel", ConvBertModel]],
  ["camembert", ["CamembertModel", CamembertModel]],
  ["deberta", ["DebertaModel", DebertaModel]],
  ["deberta-v2", ["DebertaV2Model", DebertaV2Model]],
  ["mpnet", ["MPNetModel", MPNetModel]],
  ["albert", ["AlbertModel", AlbertModel]],
  ["distilbert", ["DistilBertModel", DistilBertModel]],
  ["roberta", ["RobertaModel", RobertaModel]],
  ["xlm", ["XLMModel", XLMModel]],
  ["xlm-roberta", ["XLMRobertaModel", XLMRobertaModel]],
  ["clap", ["ClapModel", ClapModel]],
  ["clip", ["CLIPModel", CLIPModel]],
  ["clipseg", ["CLIPSegModel", CLIPSegModel]],
  ["chinese_clip", ["ChineseCLIPModel", ChineseCLIPModel]],
  ["siglip", ["SiglipModel", SiglipModel]],
  ["jina_clip", ["JinaCLIPModel", JinaCLIPModel]],
  ["mobilebert", ["MobileBertModel", MobileBertModel]],
  ["squeezebert", ["SqueezeBertModel", SqueezeBertModel]],
  ["wav2vec2", ["Wav2Vec2Model", Wav2Vec2Model]],
  ["wav2vec2-bert", ["Wav2Vec2BertModel", Wav2Vec2BertModel]],
  ["unispeech", ["UniSpeechModel", UniSpeechModel]],
  ["unispeech-sat", ["UniSpeechSatModel", UniSpeechSatModel]],
  ["hubert", ["HubertModel", HubertModel]],
  ["wavlm", ["WavLMModel", WavLMModel]],
  ["audio-spectrogram-transformer", ["ASTModel", ASTModel]],
  ["vits", ["VitsModel", VitsModel]],
  ["pyannote", ["PyAnnoteModel", PyAnnoteModel]],
  ["wespeaker-resnet", ["WeSpeakerResNetModel", WeSpeakerResNetModel]],
  ["detr", ["DetrModel", DetrModel]],
  ["rt_detr", ["RTDetrModel", RTDetrModel]],
  ["rt_detr_v2", ["RTDetrV2Model", RTDetrV2Model]],
  ["rf_detr", ["RFDetrModel", RFDetrModel]],
  ["d_fine", ["DFineModel", DFineModel]],
  ["table-transformer", ["TableTransformerModel", TableTransformerModel]],
  ["vit", ["ViTModel", ViTModel]],
  ["ijepa", ["IJepaModel", IJepaModel]],
  ["pvt", ["PvtModel", PvtModel]],
  ["vit_msn", ["ViTMSNModel", ViTMSNModel]],
  ["vit_mae", ["ViTMAEModel", ViTMAEModel]],
  ["groupvit", ["GroupViTModel", GroupViTModel]],
  ["fastvit", ["FastViTModel", FastViTModel]],
  ["mobilevit", ["MobileViTModel", MobileViTModel]],
  ["mobilevitv2", ["MobileViTV2Model", MobileViTV2Model]],
  ["owlvit", ["OwlViTModel", OwlViTModel]],
  ["owlv2", ["Owlv2Model", Owlv2Model]],
  ["beit", ["BeitModel", BeitModel]],
  ["deit", ["DeiTModel", DeiTModel]],
  ["hiera", ["HieraModel", HieraModel]],
  ["convnext", ["ConvNextModel", ConvNextModel]],
  ["convnextv2", ["ConvNextV2Model", ConvNextV2Model]],
  ["dinov2", ["Dinov2Model", Dinov2Model]],
  ["dinov2_with_registers", ["Dinov2WithRegistersModel", Dinov2WithRegistersModel]],
  ["dinov3_vit", ["DINOv3ViTModel", DINOv3ViTModel]],
  ["dinov3_convnext", ["DINOv3ConvNextModel", DINOv3ConvNextModel]],
  ["resnet", ["ResNetModel", ResNetModel]],
  ["swin", ["SwinModel", SwinModel]],
  ["swin2sr", ["Swin2SRModel", Swin2SRModel]],
  ["donut-swin", ["DonutSwinModel", DonutSwinModel]],
  ["yolos", ["YolosModel", YolosModel]],
  ["dpt", ["DPTModel", DPTModel]],
  ["glpn", ["GLPNModel", GLPNModel]],
  ["hifigan", ["SpeechT5HifiGan", SpeechT5HifiGan]],
  ["efficientnet", ["EfficientNetModel", EfficientNetModel]],
  ["decision_transformer", ["DecisionTransformerModel", DecisionTransformerModel]],
  ["patchtst", ["PatchTSTForPrediction", PatchTSTModel]],
  ["patchtsmixer", ["PatchTSMixerForPrediction", PatchTSMixerModel]],
  ["mobilenet_v1", ["MobileNetV1Model", MobileNetV1Model]],
  ["mobilenet_v2", ["MobileNetV2Model", MobileNetV2Model]],
  ["mobilenet_v3", ["MobileNetV3Model", MobileNetV3Model]],
  ["mobilenet_v4", ["MobileNetV4Model", MobileNetV4Model]],
  ["maskformer", ["MaskFormerModel", MaskFormerModel]],
  ["mgp-str", ["MgpstrForSceneTextRecognition", MgpstrForSceneTextRecognition]],
  ["style_text_to_speech_2", ["StyleTextToSpeech2Model", StyleTextToSpeech2Model]]
]);
var MODEL_MAPPING_NAMES_ENCODER_DECODER = /* @__PURE__ */ new Map([
  ["t5", ["T5Model", T5Model]],
  ["longt5", ["LongT5Model", LongT5Model]],
  ["mt5", ["MT5Model", MT5Model]],
  ["bart", ["BartModel", BartModel]],
  ["mbart", ["MBartModel", MBartModel]],
  ["marian", ["MarianModel", MarianModel]],
  ["whisper", ["WhisperModel", WhisperModel]],
  ["m2m_100", ["M2M100Model", M2M100Model]],
  ["blenderbot", ["BlenderbotModel", BlenderbotModel]],
  ["blenderbot-small", ["BlenderbotSmallModel", BlenderbotSmallModel]]
]);
var MODEL_MAPPING_NAMES_AUTO_ENCODER = /* @__PURE__ */ new Map([
  ["mimi", ["MimiModel", MimiModel]],
  ["dac", ["DacModel", DacModel]],
  ["snac", ["SnacModel", SnacModel]]
]);
var MODEL_MAPPING_NAMES_DECODER_ONLY = /* @__PURE__ */ new Map([
  ["bloom", ["BloomModel", BloomModel]],
  ["jais", ["JAISModel", JAISModel]],
  ["gpt2", ["GPT2Model", GPT2Model]],
  ["gptj", ["GPTJModel", GPTJModel]],
  ["gpt_bigcode", ["GPTBigCodeModel", GPTBigCodeModel]],
  ["gpt_neo", ["GPTNeoModel", GPTNeoModel]],
  ["gpt_neox", ["GPTNeoXModel", GPTNeoXModel]],
  ["codegen", ["CodeGenModel", CodeGenModel]],
  ["llama", ["LlamaModel", LlamaModel]],
  ["nanochat", ["NanoChatModel", NanoChatModel]],
  ["arcee", ["ArceeModel", ArceeModel]],
  ["lfm2", ["Lfm2Model", Lfm2Model]],
  ["smollm3", ["SmolLM3Model", SmolLM3Model]],
  ["exaone", ["ExaoneModel", ExaoneModel]],
  ["olmo", ["OlmoModel", OlmoModel]],
  ["olmo2", ["Olmo2Model", Olmo2Model]],
  ["mobilellm", ["MobileLLMModel", MobileLLMModel]],
  ["granite", ["GraniteModel", GraniteModel]],
  ["granitemoehybrid", ["GraniteMoeHybridModel", GraniteMoeHybridModel]],
  ["cohere", ["CohereModel", CohereModel]],
  ["gemma", ["GemmaModel", GemmaModel]],
  ["gemma2", ["Gemma2Model", Gemma2Model]],
  ["vaultgemma", ["VaultGemmaModel", VaultGemmaModel]],
  ["gemma3_text", ["Gemma3Model", Gemma3Model]],
  ["helium", ["HeliumModel", HeliumModel]],
  ["glm", ["GlmModel", GlmModel]],
  ["openelm", ["OpenELMModel", OpenELMModel]],
  ["qwen2", ["Qwen2Model", Qwen2Model]],
  ["qwen3", ["Qwen3Model", Qwen3Model]],
  ["phi", ["PhiModel", PhiModel]],
  ["phi3", ["Phi3Model", Phi3Model]],
  ["mpt", ["MptModel", MptModel]],
  ["opt", ["OPTModel", OPTModel]],
  ["mistral", ["MistralModel", MistralModel]],
  ["ernie4_5", ["Ernie4_5_Model", Ernie4_5_Model]],
  ["starcoder2", ["Starcoder2Model", Starcoder2Model]],
  ["falcon", ["FalconModel", FalconModel]],
  ["stablelm", ["StableLmModel", StableLmModel]],
  ["modernbert-decoder", ["ModernBertDecoderModel", ModernBertDecoderModel]]
]);
var MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES = /* @__PURE__ */ new Map([
  ["speecht5", ["SpeechT5ForSpeechToText", SpeechT5ForSpeechToText]],
  ["whisper", ["WhisperForConditionalGeneration", WhisperForConditionalGeneration]],
  ["lite-whisper", ["LiteWhisperForConditionalGeneration", LiteWhisperForConditionalGeneration]],
  ["moonshine", ["MoonshineForConditionalGeneration", MoonshineForConditionalGeneration]]
]);
var MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES = /* @__PURE__ */ new Map([
  ["speecht5", ["SpeechT5ForTextToSpeech", SpeechT5ForTextToSpeech]]
]);
var MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES = /* @__PURE__ */ new Map([
  ["vits", ["VitsModel", VitsModel]],
  ["musicgen", ["MusicgenForConditionalGeneration", MusicgenForConditionalGeneration]]
]);
var MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES = /* @__PURE__ */ new Map([
  ["bert", ["BertForSequenceClassification", BertForSequenceClassification]],
  ["neobert", ["NeoBertForSequenceClassification", NeoBertForSequenceClassification]],
  ["modernbert", ["ModernBertForSequenceClassification", ModernBertForSequenceClassification]],
  ["roformer", ["RoFormerForSequenceClassification", RoFormerForSequenceClassification]],
  ["electra", ["ElectraForSequenceClassification", ElectraForSequenceClassification]],
  ["esm", ["EsmForSequenceClassification", EsmForSequenceClassification]],
  ["convbert", ["ConvBertForSequenceClassification", ConvBertForSequenceClassification]],
  ["camembert", ["CamembertForSequenceClassification", CamembertForSequenceClassification]],
  ["deberta", ["DebertaForSequenceClassification", DebertaForSequenceClassification]],
  ["deberta-v2", ["DebertaV2ForSequenceClassification", DebertaV2ForSequenceClassification]],
  ["mpnet", ["MPNetForSequenceClassification", MPNetForSequenceClassification]],
  ["albert", ["AlbertForSequenceClassification", AlbertForSequenceClassification]],
  ["distilbert", ["DistilBertForSequenceClassification", DistilBertForSequenceClassification]],
  ["roberta", ["RobertaForSequenceClassification", RobertaForSequenceClassification]],
  ["xlm", ["XLMForSequenceClassification", XLMForSequenceClassification]],
  ["xlm-roberta", ["XLMRobertaForSequenceClassification", XLMRobertaForSequenceClassification]],
  ["bart", ["BartForSequenceClassification", BartForSequenceClassification]],
  ["mbart", ["MBartForSequenceClassification", MBartForSequenceClassification]],
  ["mobilebert", ["MobileBertForSequenceClassification", MobileBertForSequenceClassification]],
  ["squeezebert", ["SqueezeBertForSequenceClassification", SqueezeBertForSequenceClassification]]
]);
var MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES = /* @__PURE__ */ new Map([
  ["bert", ["BertForTokenClassification", BertForTokenClassification]],
  ["neobert", ["NeoBertForTokenClassification", NeoBertForTokenClassification]],
  ["modernbert", ["ModernBertForTokenClassification", ModernBertForTokenClassification]],
  ["roformer", ["RoFormerForTokenClassification", RoFormerForTokenClassification]],
  ["electra", ["ElectraForTokenClassification", ElectraForTokenClassification]],
  ["esm", ["EsmForTokenClassification", EsmForTokenClassification]],
  ["convbert", ["ConvBertForTokenClassification", ConvBertForTokenClassification]],
  ["camembert", ["CamembertForTokenClassification", CamembertForTokenClassification]],
  ["deberta", ["DebertaForTokenClassification", DebertaForTokenClassification]],
  ["deberta-v2", ["DebertaV2ForTokenClassification", DebertaV2ForTokenClassification]],
  ["mpnet", ["MPNetForTokenClassification", MPNetForTokenClassification]],
  ["distilbert", ["DistilBertForTokenClassification", DistilBertForTokenClassification]],
  ["roberta", ["RobertaForTokenClassification", RobertaForTokenClassification]],
  ["xlm", ["XLMForTokenClassification", XLMForTokenClassification]],
  ["xlm-roberta", ["XLMRobertaForTokenClassification", XLMRobertaForTokenClassification]]
]);
var MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES = /* @__PURE__ */ new Map([
  ["t5", ["T5ForConditionalGeneration", T5ForConditionalGeneration]],
  ["longt5", ["LongT5ForConditionalGeneration", LongT5ForConditionalGeneration]],
  ["mt5", ["MT5ForConditionalGeneration", MT5ForConditionalGeneration]],
  ["bart", ["BartForConditionalGeneration", BartForConditionalGeneration]],
  ["mbart", ["MBartForConditionalGeneration", MBartForConditionalGeneration]],
  ["marian", ["MarianMTModel", MarianMTModel]],
  ["m2m_100", ["M2M100ForConditionalGeneration", M2M100ForConditionalGeneration]],
  ["blenderbot", ["BlenderbotForConditionalGeneration", BlenderbotForConditionalGeneration]],
  ["blenderbot-small", ["BlenderbotSmallForConditionalGeneration", BlenderbotSmallForConditionalGeneration]]
]);
var MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = /* @__PURE__ */ new Map([
  ["bloom", ["BloomForCausalLM", BloomForCausalLM]],
  ["gpt2", ["GPT2LMHeadModel", GPT2LMHeadModel]],
  ["jais", ["JAISLMHeadModel", JAISLMHeadModel]],
  ["gptj", ["GPTJForCausalLM", GPTJForCausalLM]],
  ["gpt_bigcode", ["GPTBigCodeForCausalLM", GPTBigCodeForCausalLM]],
  ["gpt_neo", ["GPTNeoForCausalLM", GPTNeoForCausalLM]],
  ["gpt_neox", ["GPTNeoXForCausalLM", GPTNeoXForCausalLM]],
  ["codegen", ["CodeGenForCausalLM", CodeGenForCausalLM]],
  ["llama", ["LlamaForCausalLM", LlamaForCausalLM]],
  ["nanochat", ["NanoChatForCausalLM", NanoChatForCausalLM]],
  ["llama4_text", ["Llama4ForCausalLM", Llama4ForCausalLM]],
  ["arcee", ["ArceeForCausalLM", ArceeForCausalLM]],
  ["lfm2", ["Lfm2ForCausalLM", Lfm2ForCausalLM]],
  ["smollm3", ["SmolLM3ForCausalLM", SmolLM3ForCausalLM]],
  ["exaone", ["ExaoneForCausalLM", ExaoneForCausalLM]],
  ["olmo", ["OlmoForCausalLM", OlmoForCausalLM]],
  ["olmo2", ["Olmo2ForCausalLM", Olmo2ForCausalLM]],
  ["mobilellm", ["MobileLLMForCausalLM", MobileLLMForCausalLM]],
  ["granite", ["GraniteForCausalLM", GraniteForCausalLM]],
  ["granitemoehybrid", ["GraniteMoeHybridForCausalLM", GraniteMoeHybridForCausalLM]],
  ["cohere", ["CohereForCausalLM", CohereForCausalLM]],
  ["gemma", ["GemmaForCausalLM", GemmaForCausalLM]],
  ["gemma2", ["Gemma2ForCausalLM", Gemma2ForCausalLM]],
  ["vaultgemma", ["VaultGemmaForCausalLM", VaultGemmaForCausalLM]],
  ["gemma3_text", ["Gemma3ForCausalLM", Gemma3ForCausalLM]],
  ["helium", ["HeliumForCausalLM", HeliumForCausalLM]],
  ["glm", ["GlmForCausalLM", GlmForCausalLM]],
  ["openelm", ["OpenELMForCausalLM", OpenELMForCausalLM]],
  ["qwen2", ["Qwen2ForCausalLM", Qwen2ForCausalLM]],
  ["qwen3", ["Qwen3ForCausalLM", Qwen3ForCausalLM]],
  ["phi", ["PhiForCausalLM", PhiForCausalLM]],
  ["phi3", ["Phi3ForCausalLM", Phi3ForCausalLM]],
  ["mpt", ["MptForCausalLM", MptForCausalLM]],
  ["opt", ["OPTForCausalLM", OPTForCausalLM]],
  ["mbart", ["MBartForCausalLM", MBartForCausalLM]],
  ["mistral", ["MistralForCausalLM", MistralForCausalLM]],
  ["ernie4_5", ["Ernie4_5_ForCausalLM", Ernie4_5_ForCausalLM]],
  ["starcoder2", ["Starcoder2ForCausalLM", Starcoder2ForCausalLM]],
  ["falcon", ["FalconForCausalLM", FalconForCausalLM]],
  ["trocr", ["TrOCRForCausalLM", TrOCRForCausalLM]],
  ["stablelm", ["StableLmForCausalLM", StableLmForCausalLM]],
  ["modernbert-decoder", ["ModernBertDecoderForCausalLM", ModernBertDecoderForCausalLM]],
  // Also image-text-to-text
  ["phi3_v", ["Phi3VForCausalLM", Phi3VForCausalLM]]
]);
var MODEL_FOR_MULTIMODALITY_MAPPING_NAMES = /* @__PURE__ */ new Map([
  ["multi_modality", ["MultiModalityCausalLM", MultiModalityCausalLM]]
]);
var MODEL_FOR_MASKED_LM_MAPPING_NAMES = /* @__PURE__ */ new Map([
  ["bert", ["BertForMaskedLM", BertForMaskedLM]],
  ["neobert", ["NeoBertForMaskedLM", NeoBertForMaskedLM]],
  ["modernbert", ["ModernBertForMaskedLM", ModernBertForMaskedLM]],
  ["roformer", ["RoFormerForMaskedLM", RoFormerForMaskedLM]],
  ["electra", ["ElectraForMaskedLM", ElectraForMaskedLM]],
  ["esm", ["EsmForMaskedLM", EsmForMaskedLM]],
  ["convbert", ["ConvBertForMaskedLM", ConvBertForMaskedLM]],
  ["camembert", ["CamembertForMaskedLM", CamembertForMaskedLM]],
  ["deberta", ["DebertaForMaskedLM", DebertaForMaskedLM]],
  ["deberta-v2", ["DebertaV2ForMaskedLM", DebertaV2ForMaskedLM]],
  ["mpnet", ["MPNetForMaskedLM", MPNetForMaskedLM]],
  ["albert", ["AlbertForMaskedLM", AlbertForMaskedLM]],
  ["distilbert", ["DistilBertForMaskedLM", DistilBertForMaskedLM]],
  ["roberta", ["RobertaForMaskedLM", RobertaForMaskedLM]],
  ["xlm", ["XLMWithLMHeadModel", XLMWithLMHeadModel]],
  ["xlm-roberta", ["XLMRobertaForMaskedLM", XLMRobertaForMaskedLM]],
  ["mobilebert", ["MobileBertForMaskedLM", MobileBertForMaskedLM]],
  ["squeezebert", ["SqueezeBertForMaskedLM", SqueezeBertForMaskedLM]]
]);
var MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = /* @__PURE__ */ new Map([
  ["bert", ["BertForQuestionAnswering", BertForQuestionAnswering]],
  ["neobert", ["NeoBertForQuestionAnswering", NeoBertForQuestionAnswering]],
  ["roformer", ["RoFormerForQuestionAnswering", RoFormerForQuestionAnswering]],
  ["electra", ["ElectraForQuestionAnswering", ElectraForQuestionAnswering]],
  ["convbert", ["ConvBertForQuestionAnswering", ConvBertForQuestionAnswering]],
  ["camembert", ["CamembertForQuestionAnswering", CamembertForQuestionAnswering]],
  ["deberta", ["DebertaForQuestionAnswering", DebertaForQuestionAnswering]],
  ["deberta-v2", ["DebertaV2ForQuestionAnswering", DebertaV2ForQuestionAnswering]],
  ["mpnet", ["MPNetForQuestionAnswering", MPNetForQuestionAnswering]],
  ["albert", ["AlbertForQuestionAnswering", AlbertForQuestionAnswering]],
  ["distilbert", ["DistilBertForQuestionAnswering", DistilBertForQuestionAnswering]],
  ["roberta", ["RobertaForQuestionAnswering", RobertaForQuestionAnswering]],
  ["xlm", ["XLMForQuestionAnswering", XLMForQuestionAnswering]],
  ["xlm-roberta", ["XLMRobertaForQuestionAnswering", XLMRobertaForQuestionAnswering]],
  ["mobilebert", ["MobileBertForQuestionAnswering", MobileBertForQuestionAnswering]],
  ["squeezebert", ["SqueezeBertForQuestionAnswering", SqueezeBertForQuestionAnswering]]
]);
var MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = /* @__PURE__ */ new Map([
  ["vision-encoder-decoder", ["VisionEncoderDecoderModel", VisionEncoderDecoderModel]],
  ["idefics3", ["Idefics3ForConditionalGeneration", Idefics3ForConditionalGeneration]],
  ["smolvlm", ["SmolVLMForConditionalGeneration", SmolVLMForConditionalGeneration]]
]);
var MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES = /* @__PURE__ */ new Map([
  ["llava", ["LlavaForConditionalGeneration", LlavaForConditionalGeneration]],
  ["llava_onevision", ["LlavaOnevisionForConditionalGeneration", LlavaOnevisionForConditionalGeneration]],
  ["moondream1", ["Moondream1ForConditionalGeneration", Moondream1ForConditionalGeneration]],
  ["florence2", ["Florence2ForConditionalGeneration", Florence2ForConditionalGeneration]],
  ["qwen2-vl", ["Qwen2VLForConditionalGeneration", Qwen2VLForConditionalGeneration]],
  ["idefics3", ["Idefics3ForConditionalGeneration", Idefics3ForConditionalGeneration]],
  ["smolvlm", ["SmolVLMForConditionalGeneration", SmolVLMForConditionalGeneration]],
  ["paligemma", ["PaliGemmaForConditionalGeneration", PaliGemmaForConditionalGeneration]],
  ["llava_qwen2", ["LlavaQwen2ForCausalLM", LlavaQwen2ForCausalLM]],
  ["gemma3n", ["Gemma3nForConditionalGeneration", Gemma3nForConditionalGeneration]]
]);
var MODEL_FOR_AUDIO_TEXT_TO_TEXT_MAPPING_NAMES = /* @__PURE__ */ new Map([
  ["ultravox", ["UltravoxModel", UltravoxModel]],
  ["voxtral", ["VoxtralForConditionalGeneration", VoxtralForConditionalGeneration]]
]);
var MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES = /* @__PURE__ */ new Map([
  ["vision-encoder-decoder", ["VisionEncoderDecoderModel", VisionEncoderDecoderModel]]
]);
var MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES = /* @__PURE__ */ new Map([
  ["vit", ["ViTForImageClassification", ViTForImageClassification]],
  ["ijepa", ["IJepaForImageClassification", IJepaForImageClassification]],
  ["pvt", ["PvtForImageClassification", PvtForImageClassification]],
  ["vit_msn", ["ViTMSNForImageClassification", ViTMSNForImageClassification]],
  ["fastvit", ["FastViTForImageClassification", FastViTForImageClassification]],
  ["mobilevit", ["MobileViTForImageClassification", MobileViTForImageClassification]],
  ["mobilevitv2", ["MobileViTV2ForImageClassification", MobileViTV2ForImageClassification]],
  ["beit", ["BeitForImageClassification", BeitForImageClassification]],
  ["deit", ["DeiTForImageClassification", DeiTForImageClassification]],
  ["hiera", ["HieraForImageClassification", HieraForImageClassification]],
  ["convnext", ["ConvNextForImageClassification", ConvNextForImageClassification]],
  ["convnextv2", ["ConvNextV2ForImageClassification", ConvNextV2ForImageClassification]],
  ["dinov2", ["Dinov2ForImageClassification", Dinov2ForImageClassification]],
  ["dinov2_with_registers", ["Dinov2WithRegistersForImageClassification", Dinov2WithRegistersForImageClassification]],
  ["resnet", ["ResNetForImageClassification", ResNetForImageClassification]],
  ["swin", ["SwinForImageClassification", SwinForImageClassification]],
  ["segformer", ["SegformerForImageClassification", SegformerForImageClassification]],
  ["efficientnet", ["EfficientNetForImageClassification", EfficientNetForImageClassification]],
  ["mobilenet_v1", ["MobileNetV1ForImageClassification", MobileNetV1ForImageClassification]],
  ["mobilenet_v2", ["MobileNetV2ForImageClassification", MobileNetV2ForImageClassification]],
  ["mobilenet_v3", ["MobileNetV3ForImageClassification", MobileNetV3ForImageClassification]],
  ["mobilenet_v4", ["MobileNetV4ForImageClassification", MobileNetV4ForImageClassification]]
]);
var MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES = /* @__PURE__ */ new Map([
  ["detr", ["DetrForObjectDetection", DetrForObjectDetection]],
  ["rt_detr", ["RTDetrForObjectDetection", RTDetrForObjectDetection]],
  ["rt_detr_v2", ["RTDetrV2ForObjectDetection", RTDetrV2ForObjectDetection]],
  ["rf_detr", ["RFDetrForObjectDetection", RFDetrForObjectDetection]],
  ["d_fine", ["DFineForObjectDetection", DFineForObjectDetection]],
  ["table-transformer", ["TableTransformerForObjectDetection", TableTransformerForObjectDetection]],
  ["yolos", ["YolosForObjectDetection", YolosForObjectDetection]]
]);
var MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES = /* @__PURE__ */ new Map([
  ["owlvit", ["OwlViTForObjectDetection", OwlViTForObjectDetection]],
  ["owlv2", ["Owlv2ForObjectDetection", Owlv2ForObjectDetection]],
  ["grounding-dino", ["GroundingDinoForObjectDetection", GroundingDinoForObjectDetection]]
]);
var MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES = /* @__PURE__ */ new Map([
  // TODO: Do not add new models here
  ["detr", ["DetrForSegmentation", DetrForSegmentation]],
  ["clipseg", ["CLIPSegForImageSegmentation", CLIPSegForImageSegmentation]]
]);
var MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES = /* @__PURE__ */ new Map([
  ["segformer", ["SegformerForSemanticSegmentation", SegformerForSemanticSegmentation]],
  ["sapiens", ["SapiensForSemanticSegmentation", SapiensForSemanticSegmentation]],
  ["swin", ["SwinForSemanticSegmentation", SwinForSemanticSegmentation]],
  ["mobilenet_v1", ["MobileNetV1ForSemanticSegmentation", MobileNetV1ForSemanticSegmentation]],
  ["mobilenet_v2", ["MobileNetV2ForSemanticSegmentation", MobileNetV2ForSemanticSegmentation]],
  ["mobilenet_v3", ["MobileNetV3ForSemanticSegmentation", MobileNetV3ForSemanticSegmentation]],
  ["mobilenet_v4", ["MobileNetV4ForSemanticSegmentation", MobileNetV4ForSemanticSegmentation]]
]);
var MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES = /* @__PURE__ */ new Map([
  ["detr", ["DetrForSegmentation", DetrForSegmentation]],
  ["maskformer", ["MaskFormerForInstanceSegmentation", MaskFormerForInstanceSegmentation]]
]);
var MODEL_FOR_MASK_GENERATION_MAPPING_NAMES = /* @__PURE__ */ new Map([
  ["sam", ["SamModel", SamModel]],
  ["edgetam", ["EdgeTamModel", EdgeTamModel]]
]);
var MODEL_FOR_CTC_MAPPING_NAMES = /* @__PURE__ */ new Map([
  ["wav2vec2", ["Wav2Vec2ForCTC", Wav2Vec2ForCTC]],
  ["wav2vec2-bert", ["Wav2Vec2BertForCTC", Wav2Vec2BertForCTC]],
  ["unispeech", ["UniSpeechForCTC", UniSpeechForCTC]],
  ["unispeech-sat", ["UniSpeechSatForCTC", UniSpeechSatForCTC]],
  ["wavlm", ["WavLMForCTC", WavLMForCTC]],
  ["hubert", ["HubertForCTC", HubertForCTC]],
  ["parakeet_ctc", ["ParakeetForCTC", ParakeetForCTC]]
]);
var MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES = /* @__PURE__ */ new Map([
  ["wav2vec2", ["Wav2Vec2ForSequenceClassification", Wav2Vec2ForSequenceClassification]],
  ["wav2vec2-bert", ["Wav2Vec2BertForSequenceClassification", Wav2Vec2BertForSequenceClassification]],
  ["unispeech", ["UniSpeechForSequenceClassification", UniSpeechForSequenceClassification]],
  ["unispeech-sat", ["UniSpeechSatForSequenceClassification", UniSpeechSatForSequenceClassification]],
  ["wavlm", ["WavLMForSequenceClassification", WavLMForSequenceClassification]],
  ["hubert", ["HubertForSequenceClassification", HubertForSequenceClassification]],
  ["audio-spectrogram-transformer", ["ASTForAudioClassification", ASTForAudioClassification]]
]);
var MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES = /* @__PURE__ */ new Map([
  ["wavlm", ["WavLMForXVector", WavLMForXVector]]
]);
var MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING_NAMES = /* @__PURE__ */ new Map([
  ["unispeech-sat", ["UniSpeechSatForAudioFrameClassification", UniSpeechSatForAudioFrameClassification]],
  ["wavlm", ["WavLMForAudioFrameClassification", WavLMForAudioFrameClassification]],
  ["wav2vec2", ["Wav2Vec2ForAudioFrameClassification", Wav2Vec2ForAudioFrameClassification]],
  ["pyannote", ["PyAnnoteForAudioFrameClassification", PyAnnoteForAudioFrameClassification]]
]);
var MODEL_FOR_IMAGE_MATTING_MAPPING_NAMES = /* @__PURE__ */ new Map([
  ["vitmatte", ["VitMatteForImageMatting", VitMatteForImageMatting]]
]);
var MODEL_FOR_TIME_SERIES_PREDICTION_MAPPING_NAMES = /* @__PURE__ */ new Map([
  ["patchtst", ["PatchTSTForPrediction", PatchTSTForPrediction]],
  ["patchtsmixer", ["PatchTSMixerForPrediction", PatchTSMixerForPrediction]]
]);
var MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES = /* @__PURE__ */ new Map([
  ["swin2sr", ["Swin2SRForImageSuperResolution", Swin2SRForImageSuperResolution]]
]);
var MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES = /* @__PURE__ */ new Map([
  ["dpt", ["DPTForDepthEstimation", DPTForDepthEstimation]],
  ["depth_anything", ["DepthAnythingForDepthEstimation", DepthAnythingForDepthEstimation]],
  ["glpn", ["GLPNForDepthEstimation", GLPNForDepthEstimation]],
  ["sapiens", ["SapiensForDepthEstimation", SapiensForDepthEstimation]],
  ["depth_pro", ["DepthProForDepthEstimation", DepthProForDepthEstimation]],
  ["metric3d", ["Metric3DForDepthEstimation", Metric3DForDepthEstimation]],
  ["metric3dv2", ["Metric3Dv2ForDepthEstimation", Metric3Dv2ForDepthEstimation]]
]);
var MODEL_FOR_NORMAL_ESTIMATION_MAPPING_NAMES = /* @__PURE__ */ new Map([
  ["sapiens", ["SapiensForNormalEstimation", SapiensForNormalEstimation]]
]);
var MODEL_FOR_POSE_ESTIMATION_MAPPING_NAMES = /* @__PURE__ */ new Map([
  ["vitpose", ["VitPoseForPoseEstimation", VitPoseForPoseEstimation]]
]);
var MODEL_FOR_IMAGE_FEATURE_EXTRACTION_MAPPING_NAMES = /* @__PURE__ */ new Map([
  ["clip", ["CLIPVisionModelWithProjection", CLIPVisionModelWithProjection]],
  ["siglip", ["SiglipVisionModel", SiglipVisionModel]],
  ["jina_clip", ["JinaCLIPVisionModel", JinaCLIPVisionModel]]
]);
var MODEL_CLASS_TYPE_MAPPING = [
  // MODEL_MAPPING_NAMES:
  [MODEL_MAPPING_NAMES_ENCODER_ONLY, MODEL_TYPES.EncoderOnly],
  [MODEL_MAPPING_NAMES_ENCODER_DECODER, MODEL_TYPES.EncoderDecoder],
  [MODEL_MAPPING_NAMES_DECODER_ONLY, MODEL_TYPES.DecoderOnly],
  [MODEL_MAPPING_NAMES_AUTO_ENCODER, MODEL_TYPES.AutoEncoder],
  [MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
  [MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
  [MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES, MODEL_TYPES.Seq2Seq],
  [MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES, MODEL_TYPES.Seq2Seq],
  [MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_TYPES.DecoderOnly],
  [MODEL_FOR_MULTIMODALITY_MAPPING_NAMES, MODEL_TYPES.MultiModality],
  [MODEL_FOR_MASKED_LM_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
  [MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
  [MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES, MODEL_TYPES.Vision2Seq],
  [MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES, MODEL_TYPES.ImageTextToText],
  [MODEL_FOR_AUDIO_TEXT_TO_TEXT_MAPPING_NAMES, MODEL_TYPES.AudioTextToText],
  [MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
  [MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
  [MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
  [MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
  [MODEL_FOR_IMAGE_MATTING_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
  [MODEL_FOR_TIME_SERIES_PREDICTION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
  [MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
  [MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
  [MODEL_FOR_NORMAL_ESTIMATION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
  [MODEL_FOR_POSE_ESTIMATION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
  [MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
  [MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
  [MODEL_FOR_MASK_GENERATION_MAPPING_NAMES, MODEL_TYPES.MaskGeneration],
  [MODEL_FOR_CTC_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
  [MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
  [MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES, MODEL_TYPES.Seq2Seq],
  [MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
  [MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
  [MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
  // Custom:
  [MODEL_FOR_IMAGE_FEATURE_EXTRACTION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly]
];
for (const [mappings, type] of MODEL_CLASS_TYPE_MAPPING) {
  for (const [name, model] of mappings.values()) {
    MODEL_TYPE_MAPPING.set(name, type);
    MODEL_CLASS_TO_NAME_MAPPING.set(model, name);
    MODEL_NAME_TO_CLASS_MAPPING.set(name, model);
  }
}
var CUSTOM_MAPPING = [
  // OVERRIDE:
  // TODO: Refactor to allow class to specify model
  ["MusicgenForConditionalGeneration", MusicgenForConditionalGeneration, MODEL_TYPES.Musicgen],
  ["Phi3VForCausalLM", Phi3VForCausalLM, MODEL_TYPES.Phi3V],
  ["CLIPTextModelWithProjection", CLIPTextModelWithProjection, MODEL_TYPES.EncoderOnly],
  ["SiglipTextModel", SiglipTextModel, MODEL_TYPES.EncoderOnly],
  ["JinaCLIPTextModel", JinaCLIPTextModel, MODEL_TYPES.EncoderOnly],
  ["ClapTextModelWithProjection", ClapTextModelWithProjection, MODEL_TYPES.EncoderOnly],
  ["ClapAudioModelWithProjection", ClapAudioModelWithProjection, MODEL_TYPES.EncoderOnly],
  ["DacEncoderModel", DacEncoderModel, MODEL_TYPES.EncoderOnly],
  ["DacDecoderModel", DacDecoderModel, MODEL_TYPES.EncoderOnly],
  ["MimiEncoderModel", MimiEncoderModel, MODEL_TYPES.EncoderOnly],
  ["MimiDecoderModel", MimiDecoderModel, MODEL_TYPES.EncoderOnly],
  ["SnacEncoderModel", SnacEncoderModel, MODEL_TYPES.EncoderOnly],
  ["SnacDecoderModel", SnacDecoderModel, MODEL_TYPES.EncoderOnly],
  ["Gemma3nForConditionalGeneration", Gemma3nForConditionalGeneration, MODEL_TYPES.ImageAudioTextToText]
];
for (const [name, model, type] of CUSTOM_MAPPING) {
  MODEL_TYPE_MAPPING.set(name, type);
  MODEL_CLASS_TO_NAME_MAPPING.set(model, name);
  MODEL_NAME_TO_CLASS_MAPPING.set(name, model);
}
var CUSTOM_ARCHITECTURES = /* @__PURE__ */ new Map([
  ["modnet", MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES],
  ["birefnet", MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES],
  ["isnet", MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES],
  ["ben", MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES]
]);
for (const [name, mapping] of CUSTOM_ARCHITECTURES.entries()) {
  mapping.set(name, ["PreTrainedModel", PreTrainedModel]);
  MODEL_TYPE_MAPPING.set(name, MODEL_TYPES.EncoderOnly);
  MODEL_CLASS_TO_NAME_MAPPING.set(PreTrainedModel, name);
  MODEL_NAME_TO_CLASS_MAPPING.set(name, PreTrainedModel);
}
var AutoModel = class extends PretrainedMixin {
  /** @type {Map<string, Object>[]} */
  // @ts-ignore
  static MODEL_CLASS_MAPPINGS = MODEL_CLASS_TYPE_MAPPING.map((x) => x[0]);
  static BASE_IF_FAIL = true;
};
var AutoModelForSequenceClassification = class extends PretrainedMixin {
  static MODEL_CLASS_MAPPINGS = [MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES];
};
var AutoModelForTokenClassification = class extends PretrainedMixin {
  static MODEL_CLASS_MAPPINGS = [MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES];
};
var AutoModelForSeq2SeqLM = class extends PretrainedMixin {
  static MODEL_CLASS_MAPPINGS = [MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES];
};
var AutoModelForSpeechSeq2Seq = class extends PretrainedMixin {
  static MODEL_CLASS_MAPPINGS = [MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES];
};
var AutoModelForTextToSpectrogram = class extends PretrainedMixin {
  static MODEL_CLASS_MAPPINGS = [MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES];
};
var AutoModelForTextToWaveform = class extends PretrainedMixin {
  static MODEL_CLASS_MAPPINGS = [MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES];
};
var AutoModelForCausalLM = class extends PretrainedMixin {
  static MODEL_CLASS_MAPPINGS = [MODEL_FOR_CAUSAL_LM_MAPPING_NAMES];
};
var AutoModelForMaskedLM = class extends PretrainedMixin {
  static MODEL_CLASS_MAPPINGS = [MODEL_FOR_MASKED_LM_MAPPING_NAMES];
};
var AutoModelForQuestionAnswering = class extends PretrainedMixin {
  static MODEL_CLASS_MAPPINGS = [MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES];
};
var AutoModelForVision2Seq = class extends PretrainedMixin {
  static MODEL_CLASS_MAPPINGS = [MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES];
};
var AutoModelForImageClassification = class extends PretrainedMixin {
  static MODEL_CLASS_MAPPINGS = [MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES];
};
var AutoModelForImageSegmentation = class extends PretrainedMixin {
  static MODEL_CLASS_MAPPINGS = [MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES];
};
var AutoModelForSemanticSegmentation = class extends PretrainedMixin {
  static MODEL_CLASS_MAPPINGS = [MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES];
};
var AutoModelForUniversalSegmentation = class extends PretrainedMixin {
  static MODEL_CLASS_MAPPINGS = [MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES];
};
var AutoModelForObjectDetection = class extends PretrainedMixin {
  static MODEL_CLASS_MAPPINGS = [MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES];
};
var AutoModelForZeroShotObjectDetection = class extends PretrainedMixin {
  static MODEL_CLASS_MAPPINGS = [MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES];
};
var AutoModelForMaskGeneration = class extends PretrainedMixin {
  static MODEL_CLASS_MAPPINGS = [MODEL_FOR_MASK_GENERATION_MAPPING_NAMES];
};
var AutoModelForCTC = class extends PretrainedMixin {
  static MODEL_CLASS_MAPPINGS = [MODEL_FOR_CTC_MAPPING_NAMES];
};
var AutoModelForAudioClassification = class extends PretrainedMixin {
  static MODEL_CLASS_MAPPINGS = [MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES];
};
var AutoModelForXVector = class extends PretrainedMixin {
  static MODEL_CLASS_MAPPINGS = [MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES];
};
var AutoModelForAudioFrameClassification = class extends PretrainedMixin {
  static MODEL_CLASS_MAPPINGS = [MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING_NAMES];
};
var AutoModelForDocumentQuestionAnswering = class extends PretrainedMixin {
  static MODEL_CLASS_MAPPINGS = [MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES];
};
var AutoModelForImageMatting = class extends PretrainedMixin {
  static MODEL_CLASS_MAPPINGS = [MODEL_FOR_IMAGE_MATTING_MAPPING_NAMES];
};
var AutoModelForImageToImage = class extends PretrainedMixin {
  static MODEL_CLASS_MAPPINGS = [MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES];
};
var AutoModelForDepthEstimation = class extends PretrainedMixin {
  static MODEL_CLASS_MAPPINGS = [MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES];
};
var AutoModelForNormalEstimation = class extends PretrainedMixin {
  static MODEL_CLASS_MAPPINGS = [MODEL_FOR_NORMAL_ESTIMATION_MAPPING_NAMES];
};
var AutoModelForPoseEstimation = class extends PretrainedMixin {
  static MODEL_CLASS_MAPPINGS = [MODEL_FOR_POSE_ESTIMATION_MAPPING_NAMES];
};
var AutoModelForImageFeatureExtraction = class extends PretrainedMixin {
  static MODEL_CLASS_MAPPINGS = [MODEL_FOR_IMAGE_FEATURE_EXTRACTION_MAPPING_NAMES];
};
var AutoModelForImageTextToText = class extends PretrainedMixin {
  static MODEL_CLASS_MAPPINGS = [MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES];
};
var AutoModelForAudioTextToText = class extends PretrainedMixin {
  static MODEL_CLASS_MAPPINGS = [MODEL_FOR_AUDIO_TEXT_TO_TEXT_MAPPING_NAMES];
};
var Seq2SeqLMOutput = class extends ModelOutput {
  /**
   * @param {Object} output The output of the model.
   * @param {Tensor} output.logits The output logits of the model.
   * @param {Tensor} output.past_key_values An tensor of key/value pairs that represent the previous state of the model.
   * @param {Tensor} output.encoder_outputs The output of the encoder in a sequence-to-sequence model.
   * @param {Tensor} [output.decoder_attentions] Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the self-attention heads.
   * @param {Tensor} [output.cross_attentions] Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.
   */
  constructor({ logits, past_key_values, encoder_outputs, decoder_attentions = null, cross_attentions = null }) {
    super();
    this.logits = logits;
    this.past_key_values = past_key_values;
    this.encoder_outputs = encoder_outputs;
    this.decoder_attentions = decoder_attentions;
    this.cross_attentions = cross_attentions;
  }
};
var SequenceClassifierOutput = class extends ModelOutput {
  /**
   * @param {Object} output The output of the model.
   * @param {Tensor} output.logits classification (or regression if config.num_labels==1) scores (before SoftMax).
   * @param {Record<string, Tensor>} [output.attentions] Object of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
   * Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
   */
  constructor({ logits, ...attentions }) {
    super();
    this.logits = logits;
    const attentions_list = Object.values(attentions);
    if (attentions_list.length > 0) {
      this.attentions = attentions_list;
    }
  }
};
var XVectorOutput = class extends ModelOutput {
  /**
   * @param {Object} output The output of the model.
   * @param {Tensor} output.logits Classification hidden states before AMSoftmax, of shape `(batch_size, config.xvector_output_dim)`.
   * @param {Tensor} output.embeddings Utterance embeddings used for vector similarity-based retrieval, of shape `(batch_size, config.xvector_output_dim)`.
   */
  constructor({ logits, embeddings }) {
    super();
    this.logits = logits;
    this.embeddings = embeddings;
  }
};
var TokenClassifierOutput = class extends ModelOutput {
  /**
   * @param {Object} output The output of the model.
   * @param {Tensor} output.logits Classification scores (before SoftMax).
   */
  constructor({ logits }) {
    super();
    this.logits = logits;
  }
};
var MaskedLMOutput = class extends ModelOutput {
  /**
   * @param {Object} output The output of the model.
   * @param {Tensor} output.logits Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
   */
  constructor({ logits }) {
    super();
    this.logits = logits;
  }
};
var QuestionAnsweringModelOutput = class extends ModelOutput {
  /**
   * @param {Object} output The output of the model.
   * @param {Tensor} output.start_logits Span-start scores (before SoftMax).
   * @param {Tensor} output.end_logits Span-end scores (before SoftMax).
   */
  constructor({ start_logits, end_logits }) {
    super();
    this.start_logits = start_logits;
    this.end_logits = end_logits;
  }
};
var CausalLMOutput = class extends ModelOutput {
  /**
   * @param {Object} output The output of the model.
   * @param {Tensor} output.logits Prediction scores of the language modeling head (scores for each vocabulary token before softmax).
   */
  constructor({ logits }) {
    super();
    this.logits = logits;
  }
};
var CausalLMOutputWithPast = class extends ModelOutput {
  /**
   * @param {Object} output The output of the model.
   * @param {Tensor} output.logits Prediction scores of the language modeling head (scores for each vocabulary token before softmax).
   * @param {Tensor} output.past_key_values Contains pre-computed hidden-states (key and values in the self-attention blocks)
   * that can be used (see `past_key_values` input) to speed up sequential decoding.
   */
  constructor({ logits, past_key_values }) {
    super();
    this.logits = logits;
    this.past_key_values = past_key_values;
  }
};
var ImageMattingOutput = class extends ModelOutput {
  /**
   * @param {Object} output The output of the model.
   * @param {Tensor} output.alphas Estimated alpha values, of shape `(batch_size, num_channels, height, width)`.
   */
  constructor({ alphas }) {
    super();
    this.alphas = alphas;
  }
};
var VitsModelOutput = class extends ModelOutput {
  /**
   * @param {Object} output The output of the model.
   * @param {Tensor} output.waveform The final audio waveform predicted by the model, of shape `(batch_size, sequence_length)`.
   * @param {Tensor} output.spectrogram The log-mel spectrogram predicted at the output of the flow model.
   * This spectrogram is passed to the Hi-Fi GAN decoder model to obtain the final audio waveform.
   */
  constructor({ waveform, spectrogram: spectrogram2 }) {
    super();
    this.waveform = waveform;
    this.spectrogram = spectrogram2;
  }
};

// src/base/processing_utils.js
var Processor = class extends Callable {
  static classes = [
    "image_processor_class",
    "tokenizer_class",
    "feature_extractor_class"
  ];
  static uses_processor_config = false;
  static uses_chat_template_file = false;
  /**
   * Creates a new Processor with the given components
   * @param {Object} config 
   * @param {Record<string, Object>} components 
   * @param {string} chat_template
   */
  constructor(config, components, chat_template) {
    super();
    this.config = config;
    this.components = components;
    this.chat_template = chat_template;
  }
  /**
   * @returns {import('./image_processors_utils.js').ImageProcessor|undefined} The image processor of the processor, if it exists.
   */
  get image_processor() {
    return this.components.image_processor;
  }
  /**
   * @returns {PreTrainedTokenizer|undefined} The tokenizer of the processor, if it exists.
   */
  get tokenizer() {
    return this.components.tokenizer;
  }
  /**
   * @returns {import('./feature_extraction_utils.js').FeatureExtractor|undefined} The feature extractor of the processor, if it exists.
   */
  get feature_extractor() {
    return this.components.feature_extractor;
  }
  /**
   * @param {Parameters<PreTrainedTokenizer['apply_chat_template']>[0]} messages
   * @param {Parameters<PreTrainedTokenizer['apply_chat_template']>[1]} options
   * @returns {ReturnType<PreTrainedTokenizer['apply_chat_template']>}
   */
  apply_chat_template(messages, options = {}) {
    if (!this.tokenizer) {
      throw new Error("Unable to apply chat template without a tokenizer.");
    }
    return this.tokenizer.apply_chat_template(messages, {
      tokenize: false,
      // default to false
      chat_template: this.chat_template ?? void 0,
      ...options
    });
  }
  /**
   * @param {Parameters<PreTrainedTokenizer['batch_decode']>} args
   * @returns {ReturnType<PreTrainedTokenizer['batch_decode']>}
   */
  batch_decode(...args) {
    if (!this.tokenizer) {
      throw new Error("Unable to decode without a tokenizer.");
    }
    return this.tokenizer.batch_decode(...args);
  }
  /**
   * @param {Parameters<PreTrainedTokenizer['decode']>} args
   * @returns {ReturnType<PreTrainedTokenizer['decode']>}
   */
  decode(...args) {
    if (!this.tokenizer) {
      throw new Error("Unable to decode without a tokenizer.");
    }
    return this.tokenizer.decode(...args);
  }
  /**
   * Calls the feature_extractor function with the given input.
   * @param {any} input The input to extract features from.
   * @param {...any} args Additional arguments.
   * @returns {Promise<any>} A Promise that resolves with the extracted features.
   */
  async _call(input, ...args) {
    for (const item of [this.image_processor, this.feature_extractor, this.tokenizer]) {
      if (item) {
        return item(input, ...args);
      }
    }
    throw new Error("No image processor, feature extractor, or tokenizer found.");
  }
  /**
   * Instantiate one of the processor classes of the library from a pretrained model.
   * 
   * The processor class to instantiate is selected based on the `image_processor_type` (or `feature_extractor_type`; legacy)
   * property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible)
   * 
   * @param {string} pretrained_model_name_or_path The name or path of the pretrained model. Can be either:
   * - A string, the *model id* of a pretrained processor hosted inside a model repo on huggingface.co.
   *   Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
   *   user or organization name, like `dbmdz/bert-base-german-cased`.
   * - A path to a *directory* containing processor files, e.g., `./my_model_directory/`.
   * @param {PretrainedProcessorOptions} options Additional options for loading the processor.
   * 
   * @returns {Promise<Processor>} A new instance of the Processor class.
   */
  static async from_pretrained(pretrained_model_name_or_path, options = {}) {
    const [config, components, chat_template] = await Promise.all([
      // TODO:
      this.uses_processor_config ? getModelJSON(pretrained_model_name_or_path, PROCESSOR_NAME, true, options) : {},
      Promise.all(
        this.classes.filter((cls) => cls in this).map(async (cls) => {
          const component = await this[cls].from_pretrained(pretrained_model_name_or_path, options);
          return [cls.replace(/_class$/, ""), component];
        })
      ).then(Object.fromEntries),
      this.uses_chat_template_file ? getModelText(pretrained_model_name_or_path, CHAT_TEMPLATE_NAME, true, options) : null
    ]);
    return new this(config, components, chat_template);
  }
};

// src/models/processors.js
var processors_exports = {};
__export(processors_exports, {
  Florence2Processor: () => Florence2Processor,
  Gemma3nProcessor: () => Gemma3nProcessor,
  GroundingDinoProcessor: () => GroundingDinoProcessor,
  Idefics3Processor: () => Idefics3Processor,
  JinaCLIPProcessor: () => JinaCLIPProcessor,
  LlavaProcessor: () => LlavaProcessor,
  MgpstrProcessor: () => MgpstrProcessor,
  MoonshineProcessor: () => MoonshineProcessor,
  OwlViTProcessor: () => OwlViTProcessor,
  PaliGemmaProcessor: () => PaliGemmaProcessor,
  Phi3VProcessor: () => Phi3VProcessor,
  PyAnnoteProcessor: () => PyAnnoteProcessor,
  Qwen2VLProcessor: () => Qwen2VLProcessor,
  Sam2VideoProcessor: () => Sam2VideoProcessor,
  SamProcessor: () => SamProcessor,
  SmolVLMProcessor: () => Idefics3Processor,
  SpeechT5Processor: () => SpeechT5Processor,
  UltravoxProcessor: () => UltravoxProcessor,
  VLChatProcessor: () => VLChatProcessor,
  VoxtralProcessor: () => VoxtralProcessor,
  Wav2Vec2Processor: () => Wav2Vec2Processor,
  Wav2Vec2ProcessorWithLM: () => Wav2Vec2ProcessorWithLM,
  WhisperProcessor: () => WhisperProcessor
});

// src/base/image_processors_utils.js
function constraint_to_multiple_of(val, multiple, minVal = 0, maxVal = null) {
  const a = val / multiple;
  let x = bankers_round(a) * multiple;
  if (maxVal !== null && x > maxVal) {
    x = Math.floor(a) * multiple;
  }
  if (x < minVal) {
    x = Math.ceil(a) * multiple;
  }
  return x;
}
function enforce_size_divisibility([width, height], divisor) {
  return [
    Math.max(Math.floor(width / divisor), 1) * divisor,
    Math.max(Math.floor(height / divisor), 1) * divisor
  ];
}
function center_to_corners_format([centerX, centerY, width, height]) {
  return [
    centerX - width / 2,
    centerY - height / 2,
    centerX + width / 2,
    centerY + height / 2
  ];
}
function post_process_object_detection(outputs, threshold = 0.5, target_sizes = null, is_zero_shot = false) {
  const out_logits = outputs.logits;
  const out_bbox = outputs.pred_boxes;
  const [batch_size, num_boxes, num_classes] = out_logits.dims;
  if (target_sizes !== null && target_sizes.length !== batch_size) {
    throw Error("Make sure that you pass in as many target sizes as the batch dimension of the logits");
  }
  let toReturn = [];
  for (let i = 0; i < batch_size; ++i) {
    let target_size = target_sizes !== null ? target_sizes[i] : null;
    let info = {
      boxes: [],
      classes: [],
      scores: []
    };
    let logits = out_logits[i];
    let bbox = out_bbox[i];
    for (let j = 0; j < num_boxes; ++j) {
      let logit = logits[j];
      let indices = [];
      let probs;
      if (is_zero_shot) {
        probs = logit.sigmoid().data;
        for (let k = 0; k < probs.length; ++k) {
          if (probs[k] > threshold) {
            indices.push(k);
          }
        }
      } else {
        let maxIndex = max(logit.data)[1];
        if (maxIndex === num_classes - 1) {
          continue;
        }
        probs = softmax(logit.data);
        if (probs[maxIndex] < threshold) {
          continue;
        }
        indices.push(maxIndex);
      }
      for (const index of indices) {
        let box = bbox[j].data;
        box = center_to_corners_format(box);
        if (target_size !== null) {
          box = box.map((x, i2) => x * target_size[(i2 + 1) % 2]);
        }
        info.boxes.push(box);
        info.classes.push(index);
        info.scores.push(probs[index]);
      }
    }
    toReturn.push(info);
  }
  return toReturn;
}
function post_process_semantic_segmentation(outputs, target_sizes = null) {
  const logits = outputs.logits;
  const batch_size = logits.dims[0];
  if (target_sizes !== null && target_sizes.length !== batch_size) {
    throw Error("Make sure that you pass in as many target sizes as the batch dimension of the logits");
  }
  const toReturn = [];
  for (let i = 0; i < batch_size; ++i) {
    const target_size = target_sizes !== null ? target_sizes[i] : null;
    let data = logits[i];
    if (target_size !== null) {
      data = interpolate(data, target_size, "bilinear", false);
    }
    const [height, width] = target_size ?? data.dims.slice(-2);
    const segmentation = new Tensor2(
      "int32",
      new Int32Array(height * width),
      [height, width]
    );
    const buffer = data[0].data;
    const segmentation_data = segmentation.data;
    for (let j = 1; j < data.dims[0]; ++j) {
      const row = data[j].data;
      for (let k = 0; k < row.length; ++k) {
        if (row[k] > buffer[k]) {
          buffer[k] = row[k];
          segmentation_data[k] = j;
        }
      }
    }
    const hasLabel = new Array(data.dims[0]);
    for (let j = 0; j < segmentation_data.length; ++j) {
      const index = segmentation_data[j];
      hasLabel[index] = index;
    }
    const labels = hasLabel.filter((x) => x !== void 0);
    toReturn.push({ segmentation, labels });
  }
  return toReturn;
}
function remove_low_and_no_objects(class_logits, mask_logits, object_mask_threshold, num_labels) {
  const mask_probs_item = [];
  const pred_scores_item = [];
  const pred_labels_item = [];
  for (let j = 0; j < class_logits.dims[0]; ++j) {
    const cls = class_logits[j];
    const mask = mask_logits[j];
    const pred_label = max(cls.data)[1];
    if (pred_label === num_labels) {
      continue;
    }
    const scores = softmax(cls.data);
    const pred_score = scores[pred_label];
    if (pred_score > object_mask_threshold) {
      mask_probs_item.push(mask);
      pred_scores_item.push(pred_score);
      pred_labels_item.push(pred_label);
    }
  }
  return [mask_probs_item, pred_scores_item, pred_labels_item];
}
function check_segment_validity(mask_labels, mask_probs, k, mask_threshold = 0.5, overlap_mask_area_threshold = 0.8) {
  const mask_k = [];
  let mask_k_area = 0;
  let original_area = 0;
  const mask_probs_k_data = mask_probs[k].data;
  for (let i = 0; i < mask_labels.length; ++i) {
    if (mask_labels[i] === k) {
      mask_k.push(i);
      ++mask_k_area;
    }
    if (mask_probs_k_data[i] >= mask_threshold) {
      ++original_area;
    }
  }
  let mask_exists = mask_k_area > 0 && original_area > 0;
  if (mask_exists) {
    let area_ratio = mask_k_area / original_area;
    mask_exists = area_ratio > overlap_mask_area_threshold;
  }
  return [mask_exists, mask_k];
}
function compute_segments(mask_probs, pred_scores, pred_labels, mask_threshold, overlap_mask_area_threshold, label_ids_to_fuse = null, target_size = null) {
  const [height, width] = target_size ?? mask_probs[0].dims;
  const segmentation = new Tensor2(
    "int32",
    new Int32Array(height * width),
    [height, width]
  );
  const segments = [];
  if (target_size !== null) {
    for (let i = 0; i < mask_probs.length; ++i) {
      mask_probs[i] = interpolate(mask_probs[i], target_size, "bilinear", false);
    }
  }
  const mask_labels = new Int32Array(mask_probs[0].data.length);
  const bestScores = new Float32Array(mask_probs[0].data.length);
  for (let i = 0; i < mask_probs.length; ++i) {
    let score = pred_scores[i];
    const mask_probs_i_data = mask_probs[i].data;
    for (let j = 0; j < mask_probs_i_data.length; ++j) {
      mask_probs_i_data[j] *= score;
      if (mask_probs_i_data[j] > bestScores[j]) {
        mask_labels[j] = i;
        bestScores[j] = mask_probs_i_data[j];
      }
    }
  }
  let current_segment_id = 0;
  const segmentation_data = segmentation.data;
  for (let k = 0; k < pred_labels.length; ++k) {
    const pred_class = pred_labels[k];
    const [mask_exists, mask_k] = check_segment_validity(
      mask_labels,
      mask_probs,
      k,
      mask_threshold,
      overlap_mask_area_threshold
    );
    if (!mask_exists) {
      continue;
    }
    ++current_segment_id;
    for (const index of mask_k) {
      segmentation_data[index] = current_segment_id;
    }
    segments.push({
      id: current_segment_id,
      label_id: pred_class,
      // was_fused: should_fuse, TODO
      score: pred_scores[k]
    });
  }
  return [segmentation, segments];
}
function smart_resize(height, width, factor = 28, min_pixels = 56 * 56, max_pixels = 14 * 14 * 4 * 1280) {
  if (height < factor || width < factor) {
    throw new Error(`height:${height} or width:${width} must be larger than factor:${factor}`);
  } else if (Math.max(height, width) / Math.min(height, width) > 200) {
    throw new Error(
      `absolute aspect ratio must be smaller than 200, got ${Math.max(height, width) / Math.min(height, width)}`
    );
  }
  let h_bar = Math.round(height / factor) * factor;
  let w_bar = Math.round(width / factor) * factor;
  if (h_bar * w_bar > max_pixels) {
    const beta = Math.sqrt(height * width / max_pixels);
    h_bar = Math.floor(height / beta / factor) * factor;
    w_bar = Math.floor(width / beta / factor) * factor;
  } else if (h_bar * w_bar < min_pixels) {
    const beta = Math.sqrt(min_pixels / (height * width));
    h_bar = Math.ceil(height * beta / factor) * factor;
    w_bar = Math.ceil(width * beta / factor) * factor;
  }
  return [h_bar, w_bar];
}
function post_process_panoptic_segmentation(outputs, threshold = 0.5, mask_threshold = 0.5, overlap_mask_area_threshold = 0.8, label_ids_to_fuse = null, target_sizes = null) {
  if (label_ids_to_fuse === null) {
    console.warn("`label_ids_to_fuse` unset. No instance will be fused.");
    label_ids_to_fuse = /* @__PURE__ */ new Set();
  }
  const class_queries_logits = outputs.class_queries_logits ?? outputs.logits;
  const masks_queries_logits = outputs.masks_queries_logits ?? outputs.pred_masks;
  const mask_probs = masks_queries_logits.sigmoid();
  let [batch_size, num_queries, num_labels] = class_queries_logits.dims;
  num_labels -= 1;
  if (target_sizes !== null && target_sizes.length !== batch_size) {
    throw Error("Make sure that you pass in as many target sizes as the batch dimension of the logits");
  }
  let toReturn = [];
  for (let i = 0; i < batch_size; ++i) {
    let target_size = target_sizes !== null ? target_sizes[i] : null;
    let class_logits = class_queries_logits[i];
    let mask_logits = mask_probs[i];
    let [mask_probs_item, pred_scores_item, pred_labels_item] = remove_low_and_no_objects(class_logits, mask_logits, threshold, num_labels);
    if (pred_labels_item.length === 0) {
      let [height, width] = target_size ?? mask_logits.dims.slice(-2);
      let segmentation2 = new Tensor2(
        "int32",
        new Int32Array(height * width).fill(-1),
        [height, width]
      );
      toReturn.push({
        segmentation: segmentation2,
        segments_info: []
      });
      continue;
    }
    let [segmentation, segments] = compute_segments(
      mask_probs_item,
      pred_scores_item,
      pred_labels_item,
      mask_threshold,
      overlap_mask_area_threshold,
      label_ids_to_fuse,
      target_size
    );
    toReturn.push({
      segmentation,
      segments_info: segments
    });
  }
  return toReturn;
}
function post_process_instance_segmentation(outputs, threshold = 0.5, target_sizes = null) {
  throw new Error("`post_process_instance_segmentation` is not yet implemented.");
}
var ImageProcessor = class extends Callable {
  /**
   * Constructs a new `ImageProcessor`.
   * @param {ImageProcessorConfig} config The configuration object.
   */
  constructor(config) {
    super();
    this.image_mean = config.image_mean ?? config.mean;
    this.image_std = config.image_std ?? config.std;
    this.resample = config.resample ?? 2;
    this.do_rescale = config.do_rescale ?? true;
    this.rescale_factor = config.rescale_factor ?? 1 / 255;
    this.do_normalize = config.do_normalize;
    this.do_thumbnail = config.do_thumbnail;
    this.size = config.size ?? config.image_size;
    this.do_resize = config.do_resize ?? this.size !== void 0;
    this.size_divisibility = config.size_divisibility ?? config.size_divisor;
    this.do_center_crop = config.do_center_crop;
    this.crop_size = config.crop_size;
    this.do_convert_rgb = config.do_convert_rgb ?? true;
    this.do_crop_margin = config.do_crop_margin;
    this.pad_size = config.pad_size;
    this.do_pad = config.do_pad;
    this.min_pixels = config.min_pixels;
    this.max_pixels = config.max_pixels;
    if (this.do_pad && !this.pad_size && this.size && this.size.width !== void 0 && this.size.height !== void 0) {
      this.pad_size = this.size;
    }
    this.do_flip_channel_order = config.do_flip_channel_order ?? false;
    this.config = config;
  }
  /**
   * Resize the image to make a thumbnail. The image is resized so that no dimension is larger than any
   * corresponding dimension of the specified size.
   * @param {RawImage} image The image to be resized.
   * @param {{height:number, width:number}} size The size `{"height": h, "width": w}` to resize the image to.
   * @param {string | 0 | 1 | 2 | 3 | 4 | 5} [resample=2] The resampling filter to use.
   * @returns {Promise<RawImage>} The resized image.
   */
  async thumbnail(image, size, resample = 2) {
    const input_height = image.height;
    const input_width = image.width;
    const output_height = size.height;
    const output_width = size.width;
    let height = Math.min(input_height, output_height);
    let width = Math.min(input_width, output_width);
    if (height === input_height && width === input_width) {
      return image;
    }
    if (input_height > input_width) {
      width = Math.floor(input_width * height / input_height);
    } else if (input_width > input_height) {
      height = Math.floor(input_height * width / input_width);
    }
    return await image.resize(width, height, { resample });
  }
  /**
   * Crops the margin of the image. Gray pixels are considered margin (i.e., pixels with a value below the threshold).
   * @param {RawImage} image The image to be cropped.
   * @param {number} gray_threshold Value below which pixels are considered to be gray.
   * @returns {Promise<RawImage>} The cropped image.
   */
  async crop_margin(image, gray_threshold = 200) {
    const gray_image = image.clone().grayscale();
    const minValue = min(gray_image.data)[0];
    const maxValue = max(gray_image.data)[0];
    const diff = maxValue - minValue;
    if (diff === 0) {
      return image;
    }
    const threshold = gray_threshold / 255;
    let x_min = gray_image.width, y_min = gray_image.height, x_max = 0, y_max = 0;
    const gray_image_data = gray_image.data;
    for (let j = 0; j < gray_image.height; ++j) {
      const row = j * gray_image.width;
      for (let i = 0; i < gray_image.width; ++i) {
        if ((gray_image_data[row + i] - minValue) / diff < threshold) {
          x_min = Math.min(x_min, i);
          y_min = Math.min(y_min, j);
          x_max = Math.max(x_max, i);
          y_max = Math.max(y_max, j);
        }
      }
    }
    image = await image.crop([x_min, y_min, x_max, y_max]);
    return image;
  }
  /**
   * Pad the image by a certain amount.
   * @param {Float32Array} pixelData The pixel data to pad.
   * @param {number[]} imgDims The dimensions of the image (height, width, channels).
   * @param {{width:number; height:number}|number|'square'} padSize The dimensions of the padded image.
   * @param {Object} options The options for padding.
   * @param {'constant'|'symmetric'} [options.mode='constant'] The type of padding to add.
   * @param {boolean} [options.center=false] Whether to center the image.
   * @param {number|number[]} [options.constant_values=0] The constant value to use for padding.
   * @returns {[Float32Array, number[]]} The padded pixel data and image dimensions.
   */
  pad_image(pixelData, imgDims, padSize, {
    mode = "constant",
    center = false,
    constant_values = 0
  } = {}) {
    const [imageHeight, imageWidth, imageChannels] = imgDims;
    let paddedImageWidth, paddedImageHeight;
    if (typeof padSize === "number") {
      paddedImageWidth = padSize;
      paddedImageHeight = padSize;
    } else if (padSize === "square") {
      paddedImageWidth = paddedImageHeight = Math.max(imageHeight, imageWidth);
    } else {
      paddedImageWidth = padSize.width;
      paddedImageHeight = padSize.height;
    }
    if (paddedImageWidth !== imageWidth || paddedImageHeight !== imageHeight) {
      const paddedPixelData = new Float32Array(paddedImageWidth * paddedImageHeight * imageChannels);
      if (Array.isArray(constant_values)) {
        for (let i = 0; i < paddedPixelData.length; ++i) {
          paddedPixelData[i] = constant_values[i % imageChannels];
        }
      } else if (constant_values !== 0) {
        paddedPixelData.fill(constant_values);
      }
      const [left, top] = center ? [Math.floor((paddedImageWidth - imageWidth) / 2), Math.floor((paddedImageHeight - imageHeight) / 2)] : [0, 0];
      for (let i = 0; i < imageHeight; ++i) {
        const a = (i + top) * paddedImageWidth;
        const b = i * imageWidth;
        for (let j = 0; j < imageWidth; ++j) {
          const c = (a + j + left) * imageChannels;
          const d = (b + j) * imageChannels;
          for (let k = 0; k < imageChannels; ++k) {
            paddedPixelData[c + k] = pixelData[d + k];
          }
        }
      }
      if (mode === "symmetric") {
        if (center) {
          throw new Error("`center` padding is not supported when `mode` is set to `symmetric`.");
        }
        const h1 = imageHeight - 1;
        const w1 = imageWidth - 1;
        for (let i = 0; i < paddedImageHeight; ++i) {
          const a = i * paddedImageWidth;
          const b = calculateReflectOffset(i, h1) * imageWidth;
          for (let j = 0; j < paddedImageWidth; ++j) {
            if (i < imageHeight && j < imageWidth) continue;
            const c = (a + j) * imageChannels;
            const d = (b + calculateReflectOffset(j, w1)) * imageChannels;
            for (let k = 0; k < imageChannels; ++k) {
              paddedPixelData[c + k] = pixelData[d + k];
            }
          }
        }
      }
      pixelData = paddedPixelData;
      imgDims = [paddedImageHeight, paddedImageWidth, imageChannels];
    }
    return [pixelData, imgDims];
  }
  /**
   * Rescale the image' pixel values by `this.rescale_factor`.
   * @param {Float32Array} pixelData The pixel data to rescale.
   * @returns {void}
   */
  rescale(pixelData) {
    for (let i = 0; i < pixelData.length; ++i) {
      pixelData[i] = this.rescale_factor * pixelData[i];
    }
  }
  /**
   * Find the target (width, height) dimension of the output image after
   * resizing given the input image and the desired size.
   * @param {RawImage} image The image to resize.
   * @param {any} size The size to use for resizing the image. 
   * @returns {[number, number]} The target (width, height) dimension of the output image after resizing.
   */
  get_resize_output_image_size(image, size) {
    const [srcWidth, srcHeight] = image.size;
    let shortest_edge;
    let longest_edge;
    if (this.do_thumbnail) {
      const { height, width } = size;
      shortest_edge = Math.min(height, width);
    } else if (Number.isInteger(size)) {
      shortest_edge = size;
      longest_edge = this.config.max_size ?? shortest_edge;
    } else if (size !== void 0) {
      shortest_edge = size.shortest_edge;
      longest_edge = size.longest_edge;
    }
    if (shortest_edge !== void 0 || longest_edge !== void 0) {
      const shortResizeFactor = shortest_edge === void 0 ? 1 : Math.max(shortest_edge / srcWidth, shortest_edge / srcHeight);
      const newWidth = srcWidth * shortResizeFactor;
      const newHeight = srcHeight * shortResizeFactor;
      const longResizeFactor = longest_edge === void 0 ? 1 : Math.min(longest_edge / newWidth, longest_edge / newHeight);
      let finalWidth = Math.floor(Number((newWidth * longResizeFactor).toFixed(2)));
      let finalHeight = Math.floor(Number((newHeight * longResizeFactor).toFixed(2)));
      if (this.size_divisibility !== void 0) {
        [finalWidth, finalHeight] = enforce_size_divisibility([finalWidth, finalHeight], this.size_divisibility);
      }
      return [finalWidth, finalHeight];
    } else if (size !== void 0 && size.width !== void 0 && size.height !== void 0) {
      let newWidth = size.width;
      let newHeight = size.height;
      if (this.config.keep_aspect_ratio && this.config.ensure_multiple_of) {
        let scale_height = newHeight / srcHeight;
        let scale_width = newWidth / srcWidth;
        if (Math.abs(1 - scale_width) < Math.abs(1 - scale_height)) {
          scale_height = scale_width;
        } else {
          scale_width = scale_height;
        }
        newHeight = constraint_to_multiple_of(scale_height * srcHeight, this.config.ensure_multiple_of);
        newWidth = constraint_to_multiple_of(scale_width * srcWidth, this.config.ensure_multiple_of);
      }
      return [newWidth, newHeight];
    } else if (this.size_divisibility !== void 0) {
      return enforce_size_divisibility([srcWidth, srcHeight], this.size_divisibility);
    } else if (this.min_pixels !== void 0 && this.max_pixels !== void 0) {
      const factor = this.config.patch_size * this.config.merge_size;
      return smart_resize(srcHeight, srcWidth, factor, this.min_pixels, this.max_pixels);
    } else {
      throw new Error(`Could not resize image due to unsupported \`this.size\` option in config: ${JSON.stringify(size)}`);
    }
  }
  /**
   * Resizes the image.
   * @param {RawImage} image The image to resize.
   * @returns {Promise<RawImage>} The resized image.
   */
  async resize(image) {
    const [newWidth, newHeight] = this.get_resize_output_image_size(image, this.size);
    return await image.resize(newWidth, newHeight, {
      // @ts-expect-error TS2322
      resample: this.resample
    });
  }
  /**
   * @typedef {object} PreprocessedImage
   * @property {HeightWidth} original_size The original size of the image.
   * @property {HeightWidth} reshaped_input_size The reshaped input size of the image.
   * @property {Tensor} pixel_values The pixel values of the preprocessed image.
   */
  /**
   * Preprocesses the given image.
   *
   * @param {RawImage} image The image to preprocess.
   * @param {Object} overrides The overrides for the preprocessing options.
   * @returns {Promise<PreprocessedImage>} The preprocessed image.
   */
  async preprocess(image, {
    do_normalize = null,
    do_pad = null,
    do_convert_rgb = null,
    do_convert_grayscale = null,
    do_flip_channel_order = null
  } = {}) {
    if (this.do_crop_margin) {
      image = await this.crop_margin(image);
    }
    const [srcWidth, srcHeight] = image.size;
    if (do_convert_rgb ?? this.do_convert_rgb) {
      image = image.rgb();
    } else if (do_convert_grayscale) {
      image = image.grayscale();
    }
    if (this.do_resize) {
      image = await this.resize(image);
    }
    if (this.do_thumbnail) {
      image = await this.thumbnail(image, this.size, this.resample);
    }
    if (this.do_center_crop) {
      let crop_width;
      let crop_height;
      if (Number.isInteger(this.crop_size)) {
        crop_width = this.crop_size;
        crop_height = this.crop_size;
      } else {
        crop_width = this.crop_size.width;
        crop_height = this.crop_size.height;
      }
      image = await image.center_crop(crop_width, crop_height);
    }
    const reshaped_input_size = [image.height, image.width];
    let pixelData = Float32Array.from(image.data);
    let imgDims = [image.height, image.width, image.channels];
    if (this.do_rescale) {
      this.rescale(pixelData);
    }
    if (do_normalize ?? this.do_normalize) {
      let image_mean = this.image_mean;
      if (!Array.isArray(this.image_mean)) {
        image_mean = new Array(image.channels).fill(image_mean);
      }
      let image_std = this.image_std;
      if (!Array.isArray(this.image_std)) {
        image_std = new Array(image.channels).fill(image_std);
      }
      if (image_mean.length !== image.channels || image_std.length !== image.channels) {
        throw new Error(`When set to arrays, the length of \`image_mean\` (${image_mean.length}) and \`image_std\` (${image_std.length}) must match the number of channels in the image (${image.channels}).`);
      }
      for (let i = 0; i < pixelData.length; i += image.channels) {
        for (let j = 0; j < image.channels; ++j) {
          pixelData[i + j] = (pixelData[i + j] - image_mean[j]) / image_std[j];
        }
      }
    }
    if (do_pad ?? this.do_pad) {
      if (this.pad_size) {
        const padded = this.pad_image(pixelData, [image.height, image.width, image.channels], this.pad_size);
        [pixelData, imgDims] = padded;
      } else if (this.size_divisibility) {
        const [paddedWidth, paddedHeight] = enforce_size_divisibility([imgDims[1], imgDims[0]], this.size_divisibility);
        [pixelData, imgDims] = this.pad_image(pixelData, imgDims, { width: paddedWidth, height: paddedHeight });
      }
    }
    if (do_flip_channel_order ?? this.do_flip_channel_order) {
      if (imgDims[2] !== 3) {
        throw new Error("Flipping channel order is only supported for RGB images.");
      }
      for (let i = 0; i < pixelData.length; i += 3) {
        const temp = pixelData[i];
        pixelData[i] = pixelData[i + 2];
        pixelData[i + 2] = temp;
      }
    }
    const pixel_values = new Tensor2("float32", pixelData, imgDims).permute(2, 0, 1);
    return {
      original_size: [srcHeight, srcWidth],
      reshaped_input_size,
      pixel_values
    };
  }
  /**
   * Calls the feature extraction process on an array of images,
   * preprocesses each image, and concatenates the resulting
   * features into a single Tensor.
   * @param {RawImage[]} images The image(s) to extract features from.
   * @param {...any} args Additional arguments.
   * @returns {Promise<ImageProcessorResult>} An object containing the concatenated pixel values (and other metadata) of the preprocessed images.
   */
  async _call(images, ...args) {
    if (!Array.isArray(images)) {
      images = [images];
    }
    const imageData = await Promise.all(images.map((x) => this.preprocess(x)));
    const pixel_values = stack(imageData.map((x) => x.pixel_values), 0);
    return {
      pixel_values,
      // Original sizes of images
      original_sizes: imageData.map((x) => x.original_size),
      // Reshaped sizes of images, before padding or cropping
      reshaped_input_sizes: imageData.map((x) => x.reshaped_input_size)
    };
  }
  /**
   * Instantiate one of the processor classes of the library from a pretrained model.
   * 
   * The processor class to instantiate is selected based on the `image_processor_type` (or `feature_extractor_type`; legacy)
   * property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible)
   * 
   * @param {string} pretrained_model_name_or_path The name or path of the pretrained model. Can be either:
   * - A string, the *model id* of a pretrained processor hosted inside a model repo on huggingface.co.
   *   Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
   *   user or organization name, like `dbmdz/bert-base-german-cased`.
   * - A path to a *directory* containing processor files, e.g., `./my_model_directory/`.
   * @param {import('../utils/hub.js').PretrainedOptions} options Additional options for loading the processor.
   * 
   * @returns {Promise<ImageProcessor>} A new instance of the Processor class.
   */
  static async from_pretrained(pretrained_model_name_or_path, options = {}) {
    const preprocessorConfig = await getModelJSON(pretrained_model_name_or_path, IMAGE_PROCESSOR_NAME, true, options);
    return new this(preprocessorConfig);
  }
};

// src/models/image_processors.js
var image_processors_exports = {};
__export(image_processors_exports, {
  BeitFeatureExtractor: () => BeitFeatureExtractor,
  BitImageProcessor: () => BitImageProcessor,
  CLIPFeatureExtractor: () => CLIPFeatureExtractor,
  CLIPImageProcessor: () => CLIPImageProcessor,
  ChineseCLIPFeatureExtractor: () => ChineseCLIPFeatureExtractor,
  ConvNextFeatureExtractor: () => ConvNextFeatureExtractor,
  ConvNextImageProcessor: () => ConvNextImageProcessor,
  DINOv3ViTImageProcessor: () => DINOv3ViTImageProcessor,
  DPTFeatureExtractor: () => DPTFeatureExtractor,
  DPTImageProcessor: () => DPTImageProcessor,
  DeiTFeatureExtractor: () => DeiTFeatureExtractor,
  DeiTImageProcessor: () => DeiTImageProcessor,
  DetrFeatureExtractor: () => DetrFeatureExtractor,
  DetrImageProcessor: () => DetrImageProcessor,
  DonutFeatureExtractor: () => DonutFeatureExtractor,
  DonutImageProcessor: () => DonutImageProcessor,
  EfficientNetImageProcessor: () => EfficientNetImageProcessor,
  GLPNFeatureExtractor: () => GLPNFeatureExtractor,
  GroundingDinoImageProcessor: () => GroundingDinoImageProcessor,
  Idefics3ImageProcessor: () => Idefics3ImageProcessor,
  JinaCLIPImageProcessor: () => JinaCLIPImageProcessor,
  LlavaOnevisionImageProcessor: () => LlavaOnevisionImageProcessor,
  Mask2FormerImageProcessor: () => Mask2FormerImageProcessor,
  MaskFormerFeatureExtractor: () => MaskFormerFeatureExtractor,
  MaskFormerImageProcessor: () => MaskFormerImageProcessor,
  MobileNetV1FeatureExtractor: () => MobileNetV1FeatureExtractor,
  MobileNetV1ImageProcessor: () => MobileNetV1ImageProcessor,
  MobileNetV2FeatureExtractor: () => MobileNetV2FeatureExtractor,
  MobileNetV2ImageProcessor: () => MobileNetV2ImageProcessor,
  MobileNetV3FeatureExtractor: () => MobileNetV3FeatureExtractor,
  MobileNetV3ImageProcessor: () => MobileNetV3ImageProcessor,
  MobileNetV4FeatureExtractor: () => MobileNetV4FeatureExtractor,
  MobileNetV4ImageProcessor: () => MobileNetV4ImageProcessor,
  MobileViTFeatureExtractor: () => MobileViTFeatureExtractor,
  MobileViTImageProcessor: () => MobileViTImageProcessor,
  NougatImageProcessor: () => NougatImageProcessor,
  OwlViTFeatureExtractor: () => OwlViTFeatureExtractor,
  OwlViTImageProcessor: () => OwlViTImageProcessor,
  Owlv2ImageProcessor: () => Owlv2ImageProcessor,
  Phi3VImageProcessor: () => Phi3VImageProcessor,
  PvtImageProcessor: () => PvtImageProcessor,
  Qwen2VLImageProcessor: () => Qwen2VLImageProcessor,
  RTDetrImageProcessor: () => RTDetrImageProcessor,
  Sam2ImageProcessor: () => SamImageProcessor,
  SamImageProcessor: () => SamImageProcessor,
  SegformerFeatureExtractor: () => SegformerFeatureExtractor,
  SegformerImageProcessor: () => SegformerImageProcessor,
  SiglipImageProcessor: () => SiglipImageProcessor,
  SmolVLMImageProcessor: () => Idefics3ImageProcessor,
  Swin2SRImageProcessor: () => Swin2SRImageProcessor,
  VLMImageProcessor: () => VLMImageProcessor,
  ViTFeatureExtractor: () => ViTFeatureExtractor,
  ViTImageProcessor: () => ViTImageProcessor,
  VitMatteImageProcessor: () => VitMatteImageProcessor,
  VitPoseImageProcessor: () => VitPoseImageProcessor,
  YolosFeatureExtractor: () => YolosFeatureExtractor,
  YolosImageProcessor: () => YolosImageProcessor
});

// src/models/beit/image_processing_beit.js
var BeitFeatureExtractor = class extends ImageProcessor {
};

// src/models/bit/image_processing_bit.js
var BitImageProcessor = class extends ImageProcessor {
};

// src/models/chinese_clip/image_processing_chinese_clip.js
var ChineseCLIPFeatureExtractor = class extends ImageProcessor {
};

// src/models/clip/image_processing_clip.js
var CLIPImageProcessor = class extends ImageProcessor {
};
var CLIPFeatureExtractor = class extends CLIPImageProcessor {
};

// src/models/convnext/image_processing_convnext.js
var ConvNextImageProcessor = class extends ImageProcessor {
  constructor(config) {
    super(config);
    this.crop_pct = this.config.crop_pct ?? 224 / 256;
  }
  async resize(image) {
    const shortest_edge = this.size?.shortest_edge;
    if (shortest_edge === void 0) {
      throw new Error(`Size dictionary must contain 'shortest_edge' key.`);
    }
    if (shortest_edge < 384) {
      const resize_shortest_edge = Math.floor(shortest_edge / this.crop_pct);
      const [newWidth, newHeight] = this.get_resize_output_image_size(image, {
        shortest_edge: resize_shortest_edge
      });
      image = await image.resize(newWidth, newHeight, {
        resample: this.resample
      });
      image = await image.center_crop(shortest_edge, shortest_edge);
    } else {
      image = await image.resize(shortest_edge, shortest_edge, {
        resample: this.resample
      });
    }
    return image;
  }
};
var ConvNextFeatureExtractor = class extends ConvNextImageProcessor {
};

// src/models/deit/image_processing_deit.js
var DeiTImageProcessor = class extends ImageProcessor {
};
var DeiTFeatureExtractor = class extends DeiTImageProcessor {
};

// src/models/detr/image_processing_detr.js
var DetrImageProcessor = class extends ImageProcessor {
  /**
   * Calls the feature extraction process on an array of images, preprocesses
   * each image, and concatenates the resulting features into a single Tensor.
   * @param {import('../../utils/image.js').RawImage[]} images The image(s) to extract features from.
   * @returns {Promise<DetrFeatureExtractorResult>} An object containing the concatenated pixel values of the preprocessed images.
   */
  async _call(images) {
    const result = await super._call(images);
    const maskSize = [result.pixel_values.dims[0], 64, 64];
    const pixel_mask = full(maskSize, 1n);
    return { ...result, pixel_mask };
  }
  /** @type {typeof post_process_object_detection} */
  post_process_object_detection(...args) {
    return post_process_object_detection(...args);
  }
  /** @type {typeof post_process_panoptic_segmentation} */
  post_process_panoptic_segmentation(...args) {
    return post_process_panoptic_segmentation(...args);
  }
  /** @type {typeof post_process_instance_segmentation} */
  post_process_instance_segmentation(...args) {
    return post_process_instance_segmentation(...args);
  }
};
var DetrFeatureExtractor = class extends DetrImageProcessor {
};

// src/models/dinov3_vit/image_processing_dinov3_vit.js
var DINOv3ViTImageProcessor = class extends ImageProcessor {
};

// src/models/donut/image_processing_donut.js
var DonutImageProcessor = class extends ImageProcessor {
  pad_image(pixelData, imgDims, padSize, options = {}) {
    const [imageHeight, imageWidth, imageChannels] = imgDims;
    let image_mean = this.image_mean;
    if (!Array.isArray(this.image_mean)) {
      image_mean = new Array(imageChannels).fill(image_mean);
    }
    let image_std = this.image_std;
    if (!Array.isArray(image_std)) {
      image_std = new Array(imageChannels).fill(image_mean);
    }
    const constant_values = image_mean.map((x, i) => -x / image_std[i]);
    return super.pad_image(pixelData, imgDims, padSize, {
      center: true,
      // Since normalization is done after padding, we need to use certain constant values to ensure the same behaviour is observed.
      // For more information, see https://github.com/huggingface/transformers/blob/main/src/transformers/models/donut/image_processing_donut.py#L433-L451
      constant_values,
      ...options
    });
  }
};
var DonutFeatureExtractor = class extends DonutImageProcessor {
};

// src/models/dpt/image_processing_dpt.js
var DPTImageProcessor = class extends ImageProcessor {
};
var DPTFeatureExtractor = class extends DPTImageProcessor {
};

// src/models/efficientnet/image_processing_efficientnet.js
var EfficientNetImageProcessor = class extends ImageProcessor {
  constructor(config) {
    super(config);
    this.include_top = this.config.include_top ?? true;
    if (this.include_top) {
      this.image_std = this.image_std.map((x) => x * x);
    }
  }
};

// src/models/glpn/image_processing_glpn.js
var GLPNFeatureExtractor = class extends ImageProcessor {
};

// src/models/grounding_dino/image_processing_grounding_dino.js
var GroundingDinoImageProcessor = class extends ImageProcessor {
  /**
   * Calls the feature extraction process on an array of images, preprocesses
   * each image, and concatenates the resulting features into a single Tensor.
   * @param {import('../../utils/image.js').RawImage[]} images The image(s) to extract features from.
   * @returns {Promise<GroundingDinoFeatureExtractorResult>} An object containing the concatenated pixel values of the preprocessed images.
   */
  async _call(images) {
    const result = await super._call(images);
    const dims = result.pixel_values.dims;
    const pixel_mask = ones([dims[0], dims[2], dims[3]]);
    return { ...result, pixel_mask };
  }
};

// src/models/idefics3/image_processing_idefics3.js
var Idefics3ImageProcessor = class extends ImageProcessor {
  constructor(config) {
    super(config);
    this.do_image_splitting = config.do_image_splitting ?? true;
    this.max_image_size = config.max_image_size;
  }
  /**
   * @typedef {import('../../utils/image.js').RawImage} RawImage
   * @typedef {import('../../utils/tensor.js').Tensor} Tensor
   */
  /**
   * Calculate size to resize images to, to be multiples of `vision_encoder_max_size` while preserving the aspect ratio.
   * @param {Tensor} pixel_values Tensor of the image to resize.
   * @param {number} vision_encoder_max_size Maximum size of the output image. If the image is larger than this size,
   * it will be split into patches of this size, and the original image will be concatenated with the patches, resized to max_size.
   */
  get_resize_for_vision_encoder(pixel_values, vision_encoder_max_size) {
    let [height, width] = pixel_values.dims.slice(-2);
    const aspect_ratio = width / height;
    if (width >= height) {
      width = Math.ceil(width / vision_encoder_max_size) * vision_encoder_max_size;
      height = Math.floor(width / aspect_ratio);
      height = Math.ceil(height / vision_encoder_max_size) * vision_encoder_max_size;
    } else {
      height = Math.ceil(height / vision_encoder_max_size) * vision_encoder_max_size;
      width = Math.floor(height * aspect_ratio);
      width = Math.ceil(width / vision_encoder_max_size) * vision_encoder_max_size;
    }
    return { height, width };
  }
  /** @param {RawImage|RawImage[]|RawImage[][]} images */
  async _call(images, {
    do_image_splitting = null,
    return_row_col_info = false
  } = {}) {
    let batched_2d_images;
    if (!Array.isArray(images)) {
      batched_2d_images = [[images]];
    } else {
      if (images.length === 0 || !images[0]) {
        throw new Error("No images provided.");
      }
      if (!Array.isArray(images[0])) {
        batched_2d_images = [
          /** @type {RawImage[]} */
          images
        ];
      } else {
        batched_2d_images = /** @type {RawImage[][]} */
        images;
      }
    }
    let all_pixel_values = [];
    let images_list_rows = [];
    let images_list_cols = [];
    const original_sizes = [];
    const reshaped_input_sizes = [];
    for (const image_batch of batched_2d_images) {
      let images_list = await Promise.all(image_batch.map((x) => this.preprocess(x)));
      original_sizes.push(...images_list.map((x) => x.original_size));
      reshaped_input_sizes.push(...images_list.map((x) => x.reshaped_input_size));
      images_list.forEach((x) => x.pixel_values.unsqueeze_(0));
      const { longest_edge } = this.max_image_size;
      let images_tensor;
      if (do_image_splitting ?? this.do_image_splitting) {
        let image_rows = new Array(images_list.length);
        let image_cols = new Array(images_list.length);
        images_tensor = await Promise.all(images_list.map(async (x, i) => {
          const new_size = this.get_resize_for_vision_encoder(x.pixel_values, longest_edge);
          const resized = await interpolate_4d(x.pixel_values, {
            size: [new_size.height, new_size.width]
          });
          const { frames, num_splits_h, num_splits_w } = await this.split_image(resized, this.max_image_size);
          image_rows[i] = num_splits_h;
          image_cols[i] = num_splits_w;
          return cat(frames, 0);
        }));
        images_list_rows.push(image_rows);
        images_list_cols.push(image_cols);
      } else {
        const size = [longest_edge, longest_edge];
        images_tensor = await Promise.all(
          images_list.map((x) => interpolate_4d(x.pixel_values, { size }))
        );
        images_list_rows.push(new Array(images_list.length).fill(0));
        images_list_cols.push(new Array(images_list.length).fill(0));
      }
      all_pixel_values.push(cat(images_tensor, 0));
    }
    const batch_size = all_pixel_values.length;
    const [n, c, h, w] = all_pixel_values[0].dims;
    let pixel_values;
    let pixel_attention_mask;
    if (batch_size === 1) {
      pixel_values = all_pixel_values[0].unsqueeze_(0);
      pixel_attention_mask = full([batch_size, n, h, w], true);
    } else {
      const max_num_patches = Math.max(...all_pixel_values.map((x) => x.dims.at(0)));
      pixel_attention_mask = full([batch_size, max_num_patches, h, w], true);
      const pixel_attention_mask_data = pixel_attention_mask.data;
      const pixel_attention_mask_stride = max_num_patches * h * w;
      for (let i = 0; i < batch_size; ++i) {
        const num_patches = all_pixel_values[i].dims[0];
        if (num_patches < max_num_patches) {
          all_pixel_values[i] = cat([
            all_pixel_values[i],
            full([max_num_patches - num_patches, c, h, w], 0)
          ], 0);
          const start_offset = i * pixel_attention_mask_stride + num_patches * h * w;
          const end_offset = (i + 1) * pixel_attention_mask_stride;
          pixel_attention_mask_data.fill(false, start_offset, end_offset);
        }
      }
      pixel_values = stack(all_pixel_values, 0);
    }
    return {
      pixel_values,
      pixel_attention_mask,
      original_sizes,
      reshaped_input_sizes,
      ...return_row_col_info ? { rows: images_list_rows, cols: images_list_cols } : {}
    };
  }
  async split_image(pixel_values, { longest_edge }) {
    const max_height = longest_edge;
    const max_width = longest_edge;
    const frames = [];
    const [height, width] = pixel_values.dims.slice(-2);
    let num_splits_h = 0, num_splits_w = 0;
    if (height > max_height || width > max_width) {
      num_splits_h = Math.ceil(height / max_height);
      num_splits_w = Math.ceil(width / max_width);
      const optimal_height = Math.ceil(height / num_splits_h);
      const optimal_width = Math.ceil(width / num_splits_w);
      for (let r = 0; r < num_splits_h; ++r) {
        for (let c = 0; c < num_splits_w; ++c) {
          let start_x, start_y, end_x, end_y;
          if (r === num_splits_h - 1) {
            start_y = height - optimal_height;
            end_y = height;
          } else {
            start_y = r * optimal_height;
            end_y = (r + 1) * optimal_height;
          }
          if (c === num_splits_w - 1) {
            start_x = width - optimal_width;
            end_x = width;
          } else {
            start_x = c * optimal_width;
            end_x = (c + 1) * optimal_width;
          }
          const starts = [start_y, start_x];
          const ends = [end_y, end_x];
          const patch = await slice(pixel_values, starts, ends, [2, 3]);
          frames.push(patch);
        }
      }
      const global_image_height = max_height;
      const global_image_width = max_width;
      if (height !== global_image_height || width !== global_image_width) {
        pixel_values = await interpolate_4d(pixel_values, {
          size: [global_image_height, global_image_width]
        });
      }
    }
    frames.push(pixel_values);
    return { frames, num_splits_h, num_splits_w };
  }
};

// src/models/janus/image_processing_janus.js
var VLMImageProcessor = class extends ImageProcessor {
  constructor(config) {
    super({
      do_pad: true,
      pad_size: {
        width: config.image_size,
        height: config.image_size
      },
      ...config
    });
    this.constant_values = this.config.background_color.map((x) => x * this.rescale_factor);
  }
  pad_image(pixelData, imgDims, padSize, options) {
    return super.pad_image(pixelData, imgDims, padSize, {
      constant_values: this.constant_values,
      center: true,
      ...options
    });
  }
};

// src/models/jina_clip/image_processing_jina_clip.js
var JinaCLIPImageProcessor = class extends ImageProcessor {
  constructor(config) {
    const { resize_mode, fill_color, interpolation, size, ...other } = config;
    const new_size = resize_mode === "squash" ? { width: size, height: size } : resize_mode === "shortest" ? { shortest_edge: size } : { longest_edge: size };
    const resample = interpolation === "bicubic" ? 3 : 2;
    super({
      ...other,
      size: new_size,
      resample,
      do_center_crop: true,
      crop_size: size,
      do_normalize: true
    });
  }
};

// src/models/llava_onevision/image_processing_llava_onevision.js
var LlavaOnevisionImageProcessor = class extends ImageProcessor {
};

// src/models/maskformer/image_processing_maskformer.js
var MaskFormerImageProcessor = class extends ImageProcessor {
  /** @type {typeof post_process_panoptic_segmentation} */
  post_process_panoptic_segmentation(...args) {
    return post_process_panoptic_segmentation(...args);
  }
  /** @type {typeof post_process_instance_segmentation} */
  post_process_instance_segmentation(...args) {
    return post_process_instance_segmentation(...args);
  }
};
var MaskFormerFeatureExtractor = class extends MaskFormerImageProcessor {
};

// src/models/mask2former/image_processing_mask2former.js
var Mask2FormerImageProcessor = class extends MaskFormerImageProcessor {
};

// src/models/mobilenet_v1/image_processing_mobilenet_v1.js
var MobileNetV1ImageProcessor = class extends ImageProcessor {
};
var MobileNetV1FeatureExtractor = class extends MobileNetV1ImageProcessor {
};

// src/models/mobilenet_v2/image_processing_mobilenet_v2.js
var MobileNetV2ImageProcessor = class extends ImageProcessor {
};
var MobileNetV2FeatureExtractor = class extends MobileNetV2ImageProcessor {
};

// src/models/mobilenet_v3/image_processing_mobilenet_v3.js
var MobileNetV3ImageProcessor = class extends ImageProcessor {
};
var MobileNetV3FeatureExtractor = class extends MobileNetV3ImageProcessor {
};

// src/models/mobilenet_v4/image_processing_mobilenet_v4.js
var MobileNetV4ImageProcessor = class extends ImageProcessor {
};
var MobileNetV4FeatureExtractor = class extends MobileNetV4ImageProcessor {
};

// src/models/mobilevit/image_processing_mobilevit.js
var MobileViTImageProcessor = class extends ImageProcessor {
};
var MobileViTFeatureExtractor = class extends MobileViTImageProcessor {
};

// src/models/nougat/image_processing_nougat.js
var NougatImageProcessor = class extends DonutImageProcessor {
};

// src/models/owlvit/image_processing_owlvit.js
var OwlViTImageProcessor = class extends ImageProcessor {
  /** @type {typeof post_process_object_detection} */
  post_process_object_detection(...args) {
    return post_process_object_detection(...args);
  }
};
var OwlViTFeatureExtractor = class extends OwlViTImageProcessor {
};

// src/models/owlv2/image_processing_owlv2.js
var Owlv2ImageProcessor = class extends OwlViTImageProcessor {
};

// src/models/phi3_v/image_processing_phi3_v.js
var IMAGE_SIZE = 336;
var SLICE_AXES = [2, 3];
var { ceil, floor, sqrt } = Math;
var Phi3VImageProcessor = class extends ImageProcessor {
  constructor(config) {
    super({
      ...config,
      do_normalize: true,
      do_pad: true,
      pad_size: "custom",
      do_convert_rgb: true,
      do_resize: true
      // Smart resizing "hd_transform"
    });
    this._num_crops = config.num_crops;
  }
  calc_num_image_tokens_from_image_size(width, height) {
    const { num_img_tokens } = this.config;
    return floor((floor(height / IMAGE_SIZE) * floor(width / IMAGE_SIZE) + 1) * num_img_tokens + 1 + (floor(height / IMAGE_SIZE) + 1) * sqrt(num_img_tokens));
  }
  /** @type {ImageProcessor['get_resize_output_image_size']} */
  get_resize_output_image_size(image, size) {
    const hd_num = this._num_crops;
    const [width, height] = image.size;
    let ratio = width / height;
    let scale = 1;
    while (scale * Math.ceil(scale / ratio) <= hd_num) {
      scale += 1;
    }
    scale -= 1;
    const new_w = Math.floor(scale * 336);
    const new_h = Math.floor(new_w / ratio);
    return [new_w, new_h];
  }
  /** @type {ImageProcessor['pad_image']} */
  pad_image(pixelData, imgDims, padSize, options = {}) {
    const [imageHeight, imageWidth] = imgDims;
    const height = IMAGE_SIZE * ceil(imageHeight / IMAGE_SIZE);
    const width = IMAGE_SIZE * ceil(imageWidth / IMAGE_SIZE);
    const constant_values = [1, 1, 1].map((x, i) => (x - this.image_mean[i]) / this.image_std[i]);
    return super.pad_image(pixelData, imgDims, { width, height }, {
      center: true,
      constant_values,
      ...options
    });
  }
  async _call(images, {
    num_crops = null
  } = {}) {
    this._num_crops = num_crops ??= this.config.num_crops;
    if (num_crops < 4 || sqrt(num_crops) % 1 !== 0) {
      throw new Error("num_crops must be a square number >= 4");
    }
    if (!Array.isArray(images)) {
      images = [images];
    }
    const num_images = images.length;
    const imageData = await Promise.all(images.map((x) => this.preprocess(x)));
    const original_sizes = imageData.map((x) => x.original_size);
    const reshaped_input_sizes = imageData.map((x) => x.reshaped_input_size);
    const all_pixel_values = [];
    for (const { pixel_values: pixel_values2 } of imageData) {
      pixel_values2.unsqueeze_(0);
      const [height, width] = pixel_values2.dims.slice(-2);
      const batch_pixel_values = await interpolate_4d(pixel_values2, {
        size: [IMAGE_SIZE, IMAGE_SIZE],
        mode: "bicubic"
      });
      if (num_crops > 0) {
        const patches = [];
        const sqrt_patches = sqrt(num_crops);
        const patch_width = floor(width / sqrt_patches);
        const patch_height = floor(height / sqrt_patches);
        for (let y = 0; y < sqrt_patches; ++y) {
          for (let x = 0; x < sqrt_patches; ++x) {
            let start_x, start_y, end_x, end_y;
            if (y === sqrt_patches - 1) {
              start_y = height - patch_height;
              end_y = height;
            } else {
              start_y = y * patch_height;
              end_y = (y + 1) * patch_height;
            }
            if (x === sqrt_patches - 1) {
              start_x = width - patch_width;
              end_x = width;
            } else {
              start_x = x * patch_width;
              end_x = (x + 1) * patch_width;
            }
            const starts = [start_y, start_x];
            const ends = [end_y, end_x];
            const patch = await slice(pixel_values2, starts, ends, SLICE_AXES);
            patches.push(patch);
          }
        }
        const resized_tensors = await interpolate_4d(cat(patches, 0), {
          size: [IMAGE_SIZE, IMAGE_SIZE],
          mode: "bicubic"
        });
        all_pixel_values.push(cat([batch_pixel_values, resized_tensors], 0));
      } else {
        all_pixel_values.push(batch_pixel_values);
      }
    }
    const pixel_values = stack(all_pixel_values, 0);
    const sizes = reshaped_input_sizes.map((x) => x.map((y) => IMAGE_SIZE * ceil(y / IMAGE_SIZE)));
    const image_sizes = new Tensor2(
      "int64",
      sizes.flat(),
      [num_images, 2]
    );
    const num_img_tokens = sizes.map(
      ([height, width]) => this.calc_num_image_tokens_from_image_size(width, height)
    );
    return { pixel_values, original_sizes, reshaped_input_sizes, image_sizes, num_img_tokens };
  }
};

// src/models/pvt/image_processing_pvt.js
var PvtImageProcessor = class extends ImageProcessor {
};

// src/models/qwen2_vl/image_processing_qwen2_vl.js
var Qwen2VLImageProcessor = class extends ImageProcessor {
  async _call(images, ...args) {
    const { pixel_values, original_sizes, reshaped_input_sizes } = await super._call(images, ...args);
    let patches = pixel_values;
    const { temporal_patch_size, merge_size, patch_size } = this.config;
    if (patches.dims[0] === 1) {
      patches = cat(Array.from({ length: temporal_patch_size }, () => patches), 0);
    }
    const grid_t = patches.dims[0] / temporal_patch_size;
    const channel = patches.dims[1];
    const grid_h = Math.floor(patches.dims[2] / patch_size);
    const grid_w = Math.floor(patches.dims[3] / patch_size);
    const flatten_patches = patches.view(
      grid_t,
      temporal_patch_size,
      channel,
      Math.floor(grid_h / merge_size),
      merge_size,
      patch_size,
      Math.floor(grid_w / merge_size),
      merge_size,
      patch_size
    ).permute(0, 3, 6, 4, 7, 2, 1, 5, 8).view(
      grid_t * grid_h * grid_w,
      channel * temporal_patch_size * patch_size * patch_size
    );
    const image_grid_thw = new Tensor2("int64", [grid_t, grid_h, grid_w], [1, 3]);
    return {
      pixel_values: flatten_patches,
      image_grid_thw,
      original_sizes,
      reshaped_input_sizes
    };
  }
};

// src/models/rt_detr/image_processing_rt_detr.js
var RTDetrImageProcessor = class extends ImageProcessor {
  /** @type {typeof post_process_object_detection} */
  post_process_object_detection(...args) {
    return post_process_object_detection(...args);
  }
};

// src/models/sam/image_processing_sam.js
var SamImageProcessor = class extends ImageProcessor {
  /**
   * 
   * @param {any} input_points 
   * @param {import("../../base/image_processors_utils.js").HeightWidth[]} original_sizes 
   * @param {import("../../base/image_processors_utils.js").HeightWidth[]} reshaped_input_sizes 
   * @returns {Tensor}
   */
  reshape_input_points(input_points, original_sizes, reshaped_input_sizes, is_bounding_box = false) {
    input_points = structuredClone(input_points);
    let shape = calculateDimensions(input_points);
    if (shape.length === 3) {
      if (!is_bounding_box) {
        shape = [1, ...shape];
      }
      input_points = [input_points];
    } else if (shape.length !== 4) {
      throw Error("The input_points must be a 4D tensor of shape `batch_size`, `point_batch_size`, `nb_points_per_image`, `2`.");
    }
    for (let i = 0; i < input_points.length; ++i) {
      const [originalHeight, originalWidth] = original_sizes[i];
      const [reshapedHeight, reshapedWidth] = reshaped_input_sizes[i];
      const resizeFactors = [
        reshapedWidth / originalWidth,
        reshapedHeight / originalHeight
      ];
      for (let j = 0; j < input_points[i].length; ++j) {
        for (let k = 0; k < input_points[i][j].length; ++k) {
          for (let w = 0; w < input_points[i][j][k].length; ++w) {
            input_points[i][j][k][w] *= resizeFactors[w % 2];
          }
        }
      }
    }
    return new Tensor2(
      "float32",
      Float32Array.from(input_points.flat(Infinity)),
      shape
    );
  }
  /**
   * 
   * @param {any} input_labels 
   * @param {Tensor} input_points 
   * @returns {Tensor}
   */
  add_input_labels(input_labels, input_points) {
    let shape = calculateDimensions(input_labels);
    if (shape.length === 2) {
      shape = [1, ...shape];
      input_labels = [input_labels];
    } else if (shape.length !== 3) {
      throw Error("The input_points must be a 4D tensor of shape `batch_size`, `point_batch_size`, `nb_points_per_image`, `2`.");
    }
    if (shape.some((x, i) => x !== input_points.dims[i])) {
      throw Error(`The first ${shape.length} dimensions of 'input_points' and 'input_labels' must be the same.`);
    }
    return new Tensor2(
      "int64",
      input_labels.flat(Infinity).map(BigInt),
      shape
    );
  }
  /**
   * @param {any[]} images The URL(s) of the image(s) to extract features from.
   * @param {Object} [options] Additional options for the processor.
   * @param {any} [options.input_points=null] A 3D or 4D array, representing the input points provided by the user.
   * - 3D: `[point_batch_size, nb_points_per_image, 2]`. In this case, `batch_size` is assumed to be 1.
   * - 4D: `[batch_size, point_batch_size, nb_points_per_image, 2]`.
   * @param {any} [options.input_labels=null] A 2D or 3D array, representing the input labels for the points, used by the prompt encoder to encode the prompt.
   * - 2D: `[point_batch_size, nb_points_per_image]`. In this case, `batch_size` is assumed to be 1.
   * - 3D: `[batch_size, point_batch_size, nb_points_per_image]`.
   * @param {number[][][]} [options.input_boxes=null] A 3D array of shape `(batch_size, num_boxes, 4)`, representing the input boxes provided by the user.
   * This is used by the prompt encoder to encode the prompt. Generally yields to much better generated masks.
   * The processor will generate a tensor, with each dimension corresponding respectively to the image batch size,
   * the number of boxes per image and the coordinates of the top left and botton right point of the box.
   * In the order (`x1`, `y1`, `x2`, `y2`):
   * - `x1`: the x coordinate of the top left point of the input box
   * - `y1`: the y coordinate of the top left point of the input box
   * - `x2`: the x coordinate of the bottom right point of the input box
   * - `y2`: the y coordinate of the bottom right point of the input box
   * @returns {Promise<SamImageProcessorResult>}
   */
  async _call(images, {
    input_points = null,
    input_labels = null,
    input_boxes = null
  } = {}) {
    const processed = await super._call(images);
    if (input_points) {
      processed.input_points = this.reshape_input_points(
        input_points,
        processed.original_sizes,
        processed.reshaped_input_sizes
      );
    }
    if (input_labels) {
      if (!processed.input_points) {
        throw Error("`input_points` must be provided if `input_labels` are provided.");
      }
      processed.input_labels = this.add_input_labels(input_labels, processed.input_points);
    }
    if (input_boxes) {
      processed.input_boxes = this.reshape_input_points(
        input_boxes,
        processed.original_sizes,
        processed.reshaped_input_sizes,
        true
      );
    }
    return processed;
  }
  /**
   * Remove padding and upscale masks to the original image size.
   * @param {Tensor} masks Batched masks from the mask_decoder in (batch_size, num_channels, height, width) format.
   * @param {[number, number][]} original_sizes The original sizes of each image before it was resized to the model's expected input shape, in (height, width) format.
   * @param {[number, number][]} reshaped_input_sizes The size of each image as it is fed to the model, in (height, width) format. Used to remove padding.
   * @param {Object} options Optional parameters for post-processing.
   * @param {number} [options.mask_threshold] The threshold to use for binarizing the masks.
   * @param {boolean} [options.binarize] Whether to binarize the masks.
   * @param {Object} [options.pad_size] The target size the images were padded to before being passed to the model. If `null`, the target size is assumed to be the processor's `pad_size`.
   * @param {number} [options.pad_size.height] The height the images were padded to.
   * @param {number} [options.pad_size.width] The width the images were padded to.
   * @returns {Promise<Tensor[]>} Batched masks in batch_size, num_channels, height, width) format, where (height, width) is given by original_size.
   */
  async post_process_masks(masks, original_sizes, reshaped_input_sizes, {
    mask_threshold = 0,
    binarize = true,
    pad_size = null
  } = {}) {
    const output_masks = [];
    pad_size = pad_size ?? this.pad_size ?? this.size;
    const target_image_size = [pad_size.height, pad_size.width];
    for (let i = 0; i < original_sizes.length; ++i) {
      const original_size = original_sizes[i];
      const reshaped_input_size = reshaped_input_sizes[i];
      let interpolated_mask = await interpolate_4d(
        masks[i],
        { mode: "bilinear", size: target_image_size }
      );
      interpolated_mask = interpolated_mask.slice(null, null, [0, reshaped_input_size[0]], [0, reshaped_input_size[1]]);
      interpolated_mask = await interpolate_4d(
        interpolated_mask,
        { mode: "bilinear", size: original_size }
      );
      if (binarize) {
        const data = interpolated_mask.data;
        const binarizedMaskData = new Uint8Array(data.length);
        for (let i2 = 0; i2 < data.length; ++i2) {
          if (data[i2] > mask_threshold) {
            binarizedMaskData[i2] = 1;
          }
        }
        interpolated_mask = new Tensor2(
          "bool",
          binarizedMaskData,
          interpolated_mask.dims
        );
      }
      output_masks.push(interpolated_mask);
    }
    return output_masks;
  }
  /**
   * Generates a list of crop boxes of different sizes. Each layer has (2**i)**2 boxes for the ith layer.
   * @param {import("../../utils/image.js").RawImage} image Input original image
   * @param {number} target_size Target size of the resized image
   * @param {Object} options Options for generating crop boxes 
   * @param {number} [options.crop_n_layers] If >0, mask prediction will be run again on crops of the image.
   * Sets the number of layers to run, where each layer has 2**i_layer number of image crops.
   * @param {number} [options.overlap_ratio] Sets the degree to which crops overlap. In the first crop layer,
   * crops will overlap by this fraction of the image length. Later layers with more crops scale down this overlap.
   * @param {number} [options.points_per_crop] Number of points to sample from each crop.
   * @param {number} [options.crop_n_points_downscale_factor] The number of points-per-side sampled in layer n is
   * scaled down by crop_n_points_downscale_factor**n.
   * @returns {Object} An object containing the crop boxes, number of points per crop, cropped images, and input labels.
   */
  generate_crop_boxes(image, target_size, {
    crop_n_layers = 0,
    overlap_ratio = 512 / 1500,
    points_per_crop = 32,
    crop_n_points_downscale_factor = 1
  } = {}) {
  }
};

// src/models/segformer/image_processing_segformer.js
var SegformerImageProcessor = class extends ImageProcessor {
  /** @type {typeof post_process_semantic_segmentation} */
  post_process_semantic_segmentation(...args) {
    return post_process_semantic_segmentation(...args);
  }
};
var SegformerFeatureExtractor = class extends SegformerImageProcessor {
};

// src/models/siglip/image_processing_siglip.js
var SiglipImageProcessor = class extends ImageProcessor {
};

// src/models/swin2sr/image_processing_swin2sr.js
var Swin2SRImageProcessor = class extends ImageProcessor {
  pad_image(pixelData, imgDims, padSize, options = {}) {
    const [imageHeight, imageWidth, imageChannels] = imgDims;
    return super.pad_image(pixelData, imgDims, {
      // NOTE: For Swin2SR models, the original python implementation adds padding even when the image's width/height is already
      // a multiple of `pad_size`. However, this is most likely a bug (PR: https://github.com/mv-lab/swin2sr/pull/19).
      // For this reason, we only add padding when the image's width/height is not a multiple of `pad_size`.
      width: imageWidth + (padSize - imageWidth % padSize) % padSize,
      height: imageHeight + (padSize - imageHeight % padSize) % padSize
    }, {
      mode: "symmetric",
      center: false,
      constant_values: -1,
      ...options
    });
  }
};

// src/models/vit/image_processing_vit.js
var ViTImageProcessor = class extends ImageProcessor {
};
var ViTFeatureExtractor = class extends ViTImageProcessor {
};

// src/models/vitmatte/image_processing_vitmatte.js
var VitMatteImageProcessor = class extends ImageProcessor {
  /**
   * Calls the feature extraction process on an array of images, preprocesses
   * each image, and concatenates the resulting features into a single Tensor.
   * @param {import("../../utils/image.js").RawImage[]} images The image(s) to extract features from.
   * @param {import("../../utils/image.js").RawImage[]} trimaps The trimaps(s) to extract features from.
   * @returns {Promise<import("../../base/image_processors_utils.js").ImageProcessorResult>} An object containing the concatenated pixel values of the preprocessed images.
   */
  async _call(images, trimaps) {
    if (!Array.isArray(images)) {
      images = [images];
    }
    if (!Array.isArray(trimaps)) {
      trimaps = [trimaps];
    }
    const imageData = await Promise.all(images.map((x) => this.preprocess(x)));
    const trimapData = await Promise.all(trimaps.map((x) => this.preprocess(x, {
      do_normalize: false,
      do_convert_rgb: false,
      do_convert_grayscale: true
    })));
    const pixel_values = stack(imageData.map(
      // Concatenate images and trimaps
      (x, i) => cat([x.pixel_values, trimapData[i].pixel_values], 0)
    ), 0);
    return {
      pixel_values,
      // Original sizes of images
      original_sizes: imageData.map((x) => x.original_size),
      // Reshaped sizes of images, before padding or cropping
      reshaped_input_sizes: imageData.map((x) => x.reshaped_input_size)
    };
  }
};

// src/models/vitpose/image_processing_vitpose.js
var VitPoseImageProcessor = class extends ImageProcessor {
  /**
   * Transform the heatmaps into keypoint predictions and transform them back to the image.
   * NOTE: This is a naive implementation and does not include advanced post-processing techniques,
   * so the results may not be as accurate as the original implementation.
   * @param {import('../../utils/tensor.js').Tensor} outputs The model outputs.
   * @param {[number, number, number, number][][]} boxes List or array of bounding boxes for each image.
   * Each box should be a list of 4 floats representing the bounding box coordinates in COCO format (top_left_x, top_left_y, width, height).
   * @returns {{
   *   bbox: [number, number, number, number],
   *   scores: number[],
   *   labels: number[],
   *   keypoints: [number, number][]
   * }[][]} List of keypoints predictions for each image.
   */
  post_process_pose_estimation(outputs, boxes, {
    threshold = null
    // TODO:
    // kernel_size = 11,
    // target_sizes = null,
  } = {}) {
    const heatmaps = outputs.tolist();
    const [batch_size, num_classes, height, width] = outputs.dims;
    const results = [];
    for (let b = 0; b < batch_size; ++b) {
      const heatmap = heatmaps[b];
      const bboxes = boxes[b];
      const batch_results = [];
      for (let n = 0; n < bboxes.length; ++n) {
        const bbox = bboxes[n];
        const keypoints = [];
        const scores = [];
        const labels = [];
        const xScale = bbox.at(-2) / width;
        const yScale = bbox.at(-1) / height;
        for (let c = 0; c < heatmap.length; ++c) {
          let [xWeightedSum, yWeightedSum] = [0, 0];
          let sum = 0;
          let score = -Infinity;
          const row = heatmap[c];
          for (let y = 0; y < row.length; ++y) {
            const col = row[y];
            for (let x = 0; x < col.length; ++x) {
              const value = col[x];
              sum += value;
              score = Math.max(score, value);
              xWeightedSum += (x + 0.5) * value;
              yWeightedSum += y * value;
            }
          }
          if (threshold != null && score < threshold) continue;
          const keypoint = [
            xScale * xWeightedSum / sum,
            yScale * yWeightedSum / sum
          ];
          keypoints.push(keypoint);
          labels.push(c);
          scores.push(score);
        }
        batch_results.push({
          bbox,
          scores,
          labels,
          keypoints
        });
      }
      results.push(batch_results);
    }
    return results;
  }
};

// src/models/yolos/image_processing_yolos.js
var YolosImageProcessor = class extends ImageProcessor {
  /** @type {typeof post_process_object_detection} */
  post_process_object_detection(...args) {
    return post_process_object_detection(...args);
  }
};
var YolosFeatureExtractor = class extends YolosImageProcessor {
};

// src/models/auto/image_processing_auto.js
var AutoImageProcessor = class {
  /** @type {typeof ImageProcessor.from_pretrained} */
  static async from_pretrained(pretrained_model_name_or_path, options = {}) {
    const preprocessorConfig = await getModelJSON(pretrained_model_name_or_path, IMAGE_PROCESSOR_NAME, true, options);
    const key = preprocessorConfig.image_processor_type ?? preprocessorConfig.feature_extractor_type;
    let image_processor_class = image_processors_exports[key?.replace(/Fast$/, "")];
    if (!image_processor_class) {
      if (key !== void 0) {
        console.warn(`Image processor type '${key}' not found, assuming base ImageProcessor. Please report this at ${GITHUB_ISSUE_URL}.`);
      }
      image_processor_class = ImageProcessor;
    }
    return new image_processor_class(preprocessorConfig);
  }
};

// src/models/florence2/processing_florence2.js
var Florence2Processor = class extends Processor {
  static tokenizer_class = AutoTokenizer;
  static image_processor_class = AutoImageProcessor;
  constructor(config, components, chat_template) {
    super(config, components, chat_template);
    const {
      // @ts-expect-error TS2339
      tasks_answer_post_processing_type,
      // @ts-expect-error TS2339
      task_prompts_without_inputs,
      // @ts-expect-error TS2339
      task_prompts_with_input
    } = this.image_processor.config;
    this.tasks_answer_post_processing_type = new Map(Object.entries(tasks_answer_post_processing_type ?? {}));
    this.task_prompts_without_inputs = new Map(Object.entries(task_prompts_without_inputs ?? {}));
    this.task_prompts_with_input = new Map(Object.entries(task_prompts_with_input ?? {}));
    this.regexes = {
      quad_boxes: /(.+?)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>/gm,
      bboxes: /([^<]+)?<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>/gm
    };
    this.size_per_bin = 1e3;
  }
  /**
   * Helper function to construct prompts from input texts
   * @param {string|string[]} text
   * @returns {string[]}
   */
  construct_prompts(text) {
    if (typeof text === "string") {
      text = [text];
    }
    const prompts = [];
    for (const t of text) {
      if (this.task_prompts_without_inputs.has(t)) {
        prompts.push(this.task_prompts_without_inputs.get(t));
      } else {
        for (const [task, prompt] of this.task_prompts_with_input) {
          if (t.includes(task)) {
            prompts.push(prompt.replaceAll("{input}", t).replaceAll(task, ""));
            break;
          }
        }
        if (prompts.length !== text.length) {
          prompts.push(t);
        }
      }
    }
    return prompts;
  }
  /**
   * Post-process the output of the model to each of the task outputs.
   * @param {string} text The text to post-process.
   * @param {string} task The task to post-process the text for.
   * @param {[number, number]} image_size The size of the image. height x width.
   */
  post_process_generation(text, task, image_size) {
    const task_answer_post_processing_type = this.tasks_answer_post_processing_type.get(task) ?? "pure_text";
    text = text.replaceAll("<s>", "").replaceAll("</s>", "");
    let final_answer;
    switch (task_answer_post_processing_type) {
      case "pure_text":
        final_answer = text;
        break;
      case "description_with_bboxes":
      case "bboxes":
      case "phrase_grounding":
      case "ocr":
        const key = task_answer_post_processing_type === "ocr" ? "quad_boxes" : "bboxes";
        const matches = text.matchAll(this.regexes[key]);
        const labels = [];
        const items = [];
        for (const [_, label, ...locations] of matches) {
          labels.push(label ? label.trim() : labels.at(-1) ?? "");
          items.push(
            locations.map((x, i) => (
              // NOTE: Add 0.5 to use the center position of the bin as the coordinate.
              (Number(x) + 0.5) / this.size_per_bin * image_size[i % 2]
            ))
          );
        }
        final_answer = { labels, [key]: items };
        break;
      default:
        throw new Error(`Task "${task}" (of type "${task_answer_post_processing_type}") not yet implemented.`);
    }
    return { [task]: final_answer };
  }
  // NOTE: images and text are switched from the python version
  // `images` is required, `text` is optional
  async _call(images, text = null, kwargs = {}) {
    if (!images && !text) {
      throw new Error("Either text or images must be provided");
    }
    const image_inputs = await this.image_processor(images, kwargs);
    const text_inputs = text ? this.tokenizer(this.construct_prompts(text), kwargs) : {};
    return {
      ...image_inputs,
      ...text_inputs
    };
  }
};

// src/base/feature_extraction_utils.js
var FeatureExtractor = class extends Callable {
  /**
   * Constructs a new FeatureExtractor instance.
   *
   * @param {Object} config The configuration for the feature extractor.
   */
  constructor(config) {
    super();
    this.config = config;
  }
  /**
   * Instantiate one of the feature extractor classes of the library from a pretrained model.
   * 
   * The feature extractor class to instantiate is selected based on the `feature_extractor_type` property of
   * the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible)
   * 
   * @param {string} pretrained_model_name_or_path The name or path of the pretrained model. Can be either:
   * - A string, the *model id* of a pretrained feature_extractor hosted inside a model repo on huggingface.co.
   *   Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
   *   user or organization name, like `dbmdz/bert-base-german-cased`.
   * - A path to a *directory* containing feature_extractor files, e.g., `./my_model_directory/`.
   * @param {import('../utils/hub.js').PretrainedOptions} options Additional options for loading the feature_extractor.
   * 
   * @returns {Promise<FeatureExtractor>} A new instance of the Feature Extractor class.
   */
  static async from_pretrained(pretrained_model_name_or_path, options = {}) {
    const config = await getModelJSON(pretrained_model_name_or_path, FEATURE_EXTRACTOR_NAME, true, options);
    return new this(config);
  }
};
function validate_audio_inputs(audio, feature_extractor) {
  if (!(audio instanceof Float32Array || audio instanceof Float64Array)) {
    throw new Error(
      `${feature_extractor} expects input to be a Float32Array or a Float64Array, but got ${audio?.constructor?.name ?? typeof audio} instead. If using the feature extractor directly, remember to use \`read_audio(url, sampling_rate)\` to obtain the raw audio data of the file/url.`
    );
  }
}

// src/models/feature_extractors.js
var feature_extractors_exports = {};
__export(feature_extractors_exports, {
  ASTFeatureExtractor: () => ASTFeatureExtractor,
  ClapFeatureExtractor: () => ClapFeatureExtractor,
  DacFeatureExtractor: () => DacFeatureExtractor,
  EncodecFeatureExtractor: () => EncodecFeatureExtractor,
  Gemma3nAudioFeatureExtractor: () => Gemma3nAudioFeatureExtractor,
  ImageFeatureExtractor: () => ImageProcessor,
  MoonshineFeatureExtractor: () => MoonshineFeatureExtractor,
  ParakeetFeatureExtractor: () => ParakeetFeatureExtractor,
  PyAnnoteFeatureExtractor: () => PyAnnoteFeatureExtractor,
  SeamlessM4TFeatureExtractor: () => SeamlessM4TFeatureExtractor,
  SnacFeatureExtractor: () => SnacFeatureExtractor,
  SpeechT5FeatureExtractor: () => SpeechT5FeatureExtractor,
  Wav2Vec2FeatureExtractor: () => Wav2Vec2FeatureExtractor,
  WeSpeakerFeatureExtractor: () => WeSpeakerFeatureExtractor,
  WhisperFeatureExtractor: () => WhisperFeatureExtractor
});

// src/utils/audio.js
import fs3 from "fs";
async function read_audio(url2, sampling_rate) {
  if (typeof AudioContext === "undefined") {
    throw Error(
      "Unable to load audio from path/URL since `AudioContext` is not available in your environment. Instead, audio data should be passed directly to the pipeline/processor. For more information and some example code, see https://huggingface.co/docs/transformers.js/guides/node-audio-processing."
    );
  }
  const response = await (await getFile(url2)).arrayBuffer();
  const audioCTX = new AudioContext({ sampleRate: sampling_rate });
  if (typeof sampling_rate === "undefined") {
    console.warn(`No sampling rate provided, using default of ${audioCTX.sampleRate}Hz.`);
  }
  const decoded = await audioCTX.decodeAudioData(response);
  let audio;
  if (decoded.numberOfChannels === 2) {
    const SCALING_FACTOR = Math.sqrt(2);
    const left = decoded.getChannelData(0);
    const right = decoded.getChannelData(1);
    audio = new Float32Array(left.length);
    for (let i = 0; i < decoded.length; ++i) {
      audio[i] = SCALING_FACTOR * (left[i] + right[i]) / 2;
    }
  } else {
    audio = decoded.getChannelData(0);
  }
  return audio;
}
function generalized_cosine_window(M, a_0) {
  if (M < 1) {
    return new Float64Array();
  }
  if (M === 1) {
    return new Float64Array([1]);
  }
  const a_1 = 1 - a_0;
  const factor = 2 * Math.PI / (M - 1);
  const cos_vals = new Float64Array(M);
  for (let i = 0; i < M; ++i) {
    cos_vals[i] = a_0 - a_1 * Math.cos(i * factor);
  }
  return cos_vals;
}
function hanning(M) {
  return generalized_cosine_window(M, 0.5);
}
function hamming(M) {
  return generalized_cosine_window(M, 0.54);
}
var HERTZ_TO_MEL_MAPPING = {
  "htk": (freq) => 2595 * Math.log10(1 + freq / 700),
  "kaldi": (freq) => 1127 * Math.log(1 + freq / 700),
  "slaney": (freq, min_log_hertz = 1e3, min_log_mel = 15, logstep = 27 / Math.log(6.4)) => freq >= min_log_hertz ? min_log_mel + Math.log(freq / min_log_hertz) * logstep : 3 * freq / 200
};
function hertz_to_mel(freq, mel_scale = "htk") {
  const fn = HERTZ_TO_MEL_MAPPING[mel_scale];
  if (!fn) {
    throw new Error('mel_scale should be one of "htk", "slaney" or "kaldi".');
  }
  return typeof freq === "number" ? fn(freq) : freq.map((x) => fn(x));
}
var MEL_TO_HERTZ_MAPPING = {
  "htk": (mels) => 700 * (10 ** (mels / 2595) - 1),
  "kaldi": (mels) => 700 * (Math.exp(mels / 1127) - 1),
  "slaney": (mels, min_log_hertz = 1e3, min_log_mel = 15, logstep = Math.log(6.4) / 27) => mels >= min_log_mel ? min_log_hertz * Math.exp(logstep * (mels - min_log_mel)) : 200 * mels / 3
};
function mel_to_hertz(mels, mel_scale = "htk") {
  const fn = MEL_TO_HERTZ_MAPPING[mel_scale];
  if (!fn) {
    throw new Error('mel_scale should be one of "htk", "slaney" or "kaldi".');
  }
  return typeof mels === "number" ? fn(mels) : mels.map((x) => fn(x));
}
function _create_triangular_filter_bank(fft_freqs, filter_freqs) {
  const filter_diff = Float64Array.from(
    { length: filter_freqs.length - 1 },
    (_, i) => filter_freqs[i + 1] - filter_freqs[i]
  );
  const slopes = Array.from({
    length: fft_freqs.length
  }, () => new Array(filter_freqs.length));
  for (let j = 0; j < fft_freqs.length; ++j) {
    const slope = slopes[j];
    for (let i = 0; i < filter_freqs.length; ++i) {
      slope[i] = filter_freqs[i] - fft_freqs[j];
    }
  }
  const numFreqs = filter_freqs.length - 2;
  const ret = Array.from({ length: numFreqs }, () => new Array(fft_freqs.length));
  for (let j = 0; j < fft_freqs.length; ++j) {
    const slope = slopes[j];
    for (let i = 0; i < numFreqs; ++i) {
      const down = -slope[i] / filter_diff[i];
      const up = slope[i + 2] / filter_diff[i + 1];
      ret[i][j] = Math.max(0, Math.min(down, up));
    }
  }
  return ret;
}
function linspace(start, end, num) {
  const step = (end - start) / (num - 1);
  return Float64Array.from({ length: num }, (_, i) => start + step * i);
}
function mel_filter_bank(num_frequency_bins, num_mel_filters, min_frequency, max_frequency, sampling_rate, norm = null, mel_scale = "htk", triangularize_in_mel_space = false) {
  if (norm !== null && norm !== "slaney") {
    throw new Error('norm must be one of null or "slaney"');
  }
  if (num_frequency_bins < 2) {
    throw new Error(`Require num_frequency_bins: ${num_frequency_bins} >= 2`);
  }
  if (min_frequency > max_frequency) {
    throw new Error(`Require min_frequency: ${min_frequency} <= max_frequency: ${max_frequency}`);
  }
  const mel_min = hertz_to_mel(min_frequency, mel_scale);
  const mel_max = hertz_to_mel(max_frequency, mel_scale);
  const mel_freqs = linspace(mel_min, mel_max, num_mel_filters + 2);
  let filter_freqs = mel_to_hertz(mel_freqs, mel_scale);
  let fft_freqs;
  if (triangularize_in_mel_space) {
    const fft_bin_width = sampling_rate / ((num_frequency_bins - 1) * 2);
    fft_freqs = hertz_to_mel(Float64Array.from({ length: num_frequency_bins }, (_, i) => i * fft_bin_width), mel_scale);
    filter_freqs = mel_freqs;
  } else {
    fft_freqs = linspace(0, Math.floor(sampling_rate / 2), num_frequency_bins);
  }
  const mel_filters = _create_triangular_filter_bank(fft_freqs, filter_freqs);
  if (norm !== null && norm === "slaney") {
    for (let i = 0; i < num_mel_filters; ++i) {
      const filter = mel_filters[i];
      const enorm = 2 / (filter_freqs[i + 2] - filter_freqs[i]);
      for (let j = 0; j < num_frequency_bins; ++j) {
        filter[j] *= enorm;
      }
    }
  }
  return mel_filters;
}
function padReflect(array, left, right) {
  const padded = new array.constructor(array.length + left + right);
  const w = array.length - 1;
  for (let i = 0; i < array.length; ++i) {
    padded[left + i] = array[i];
  }
  for (let i = 1; i <= left; ++i) {
    padded[left - i] = array[calculateReflectOffset(i, w)];
  }
  for (let i = 1; i <= right; ++i) {
    padded[w + left + i] = array[calculateReflectOffset(w - i, w)];
  }
  return padded;
}
function _db_conversion_helper(spectrogram2, factor, reference, min_value, db_range) {
  if (reference <= 0) {
    throw new Error("reference must be greater than zero");
  }
  if (min_value <= 0) {
    throw new Error("min_value must be greater than zero");
  }
  reference = Math.max(min_value, reference);
  const logReference = Math.log10(reference);
  for (let i = 0; i < spectrogram2.length; ++i) {
    spectrogram2[i] = factor * Math.log10(Math.max(min_value, spectrogram2[i]) - logReference);
  }
  if (db_range !== null) {
    if (db_range <= 0) {
      throw new Error("db_range must be greater than zero");
    }
    const maxValue = max(spectrogram2)[0] - db_range;
    for (let i = 0; i < spectrogram2.length; ++i) {
      spectrogram2[i] = Math.max(spectrogram2[i], maxValue);
    }
  }
  return spectrogram2;
}
function amplitude_to_db(spectrogram2, reference = 1, min_value = 1e-5, db_range = null) {
  return _db_conversion_helper(spectrogram2, 20, reference, min_value, db_range);
}
function power_to_db(spectrogram2, reference = 1, min_value = 1e-10, db_range = null) {
  return _db_conversion_helper(spectrogram2, 10, reference, min_value, db_range);
}
async function spectrogram(waveform, window2, frame_length, hop_length, {
  fft_length = null,
  power = 1,
  center = true,
  pad_mode = "reflect",
  onesided = true,
  preemphasis = null,
  preemphasis_htk_flavor = true,
  mel_filters = null,
  mel_floor = 1e-10,
  log_mel = null,
  reference = 1,
  min_value = 1e-10,
  db_range = null,
  remove_dc_offset = null,
  // Custom parameters for efficiency reasons
  min_num_frames = null,
  max_num_frames = null,
  do_pad = true,
  transpose = false,
  mel_offset = 0
} = {}) {
  const window_length = window2.length;
  if (fft_length === null) {
    fft_length = frame_length;
  }
  if (frame_length > fft_length) {
    throw Error(`frame_length (${frame_length}) may not be larger than fft_length (${fft_length})`);
  }
  if (window_length !== frame_length) {
    throw new Error(`Length of the window (${window_length}) must equal frame_length (${frame_length})`);
  }
  if (hop_length <= 0) {
    throw new Error("hop_length must be greater than zero");
  }
  if (power === null && mel_filters !== null) {
    throw new Error(
      "You have provided `mel_filters` but `power` is `None`. Mel spectrogram computation is not yet supported for complex-valued spectrogram. Specify `power` to fix this issue."
    );
  }
  if (!preemphasis_htk_flavor) {
    throw new Error(
      "`preemphasis_htk_flavor=false` is not currently supported."
    );
  }
  if (center) {
    switch (pad_mode) {
      case "reflect": {
        const half_window = Math.floor((fft_length - 1) / 2) + 1;
        waveform = padReflect(waveform, half_window, half_window);
        break;
      }
      case "constant": {
        const padding = Math.floor(fft_length / 2);
        const padded = new waveform.constructor(waveform.length + 2 * padding);
        padded.set(waveform, padding);
        waveform = padded;
        break;
      }
      default:
        throw new Error(`pad_mode="${pad_mode}" not implemented yet.`);
    }
  }
  let num_frames = Math.floor(1 + Math.floor((waveform.length - frame_length) / hop_length));
  if (min_num_frames !== null && num_frames < min_num_frames) {
    num_frames = min_num_frames;
  }
  const num_frequency_bins = onesided ? Math.floor(fft_length / 2) + 1 : fft_length;
  let d1 = num_frames;
  let d1Max = num_frames;
  if (max_num_frames !== null) {
    if (max_num_frames > num_frames) {
      if (do_pad) {
        d1Max = max_num_frames;
      }
    } else {
      d1Max = d1 = max_num_frames;
    }
  }
  const fft = new FFT(fft_length);
  const inputBuffer = new Float64Array(fft_length);
  const outputBuffer = new Float64Array(fft.outputBufferSize);
  const transposedMagnitudeData = new Float32Array(num_frequency_bins * d1Max);
  for (let i = 0; i < d1; ++i) {
    const offset = i * hop_length;
    const buffer_size = Math.min(waveform.length - offset, frame_length);
    if (buffer_size !== frame_length) {
      inputBuffer.fill(0, 0, frame_length);
    }
    for (let j = 0; j < buffer_size; ++j) {
      inputBuffer[j] = waveform[offset + j];
    }
    if (remove_dc_offset) {
      let sum = 0;
      for (let j = 0; j < buffer_size; ++j) {
        sum += inputBuffer[j];
      }
      const mean2 = sum / buffer_size;
      for (let j = 0; j < buffer_size; ++j) {
        inputBuffer[j] -= mean2;
      }
    }
    if (preemphasis !== null) {
      for (let j = buffer_size - 1; j >= 1; --j) {
        inputBuffer[j] -= preemphasis * inputBuffer[j - 1];
      }
      inputBuffer[0] *= 1 - preemphasis;
    }
    for (let j = 0; j < window2.length; ++j) {
      inputBuffer[j] *= window2[j];
    }
    fft.realTransform(outputBuffer, inputBuffer);
    for (let j = 0; j < num_frequency_bins; ++j) {
      const j2 = j << 1;
      transposedMagnitudeData[j * d1Max + i] = outputBuffer[j2] ** 2 + outputBuffer[j2 + 1] ** 2;
    }
  }
  if (power !== null && power !== 2) {
    const pow = power / 2;
    for (let i = 0; i < transposedMagnitudeData.length; ++i) {
      transposedMagnitudeData[i] **= pow;
    }
  }
  const num_mel_filters = mel_filters.length;
  let mel_spec = await matmul(
    // TODO: Make `mel_filters` a Tensor during initialization
    new Tensor2("float32", mel_filters.flat(), [num_mel_filters, num_frequency_bins]),
    new Tensor2("float32", transposedMagnitudeData, [num_frequency_bins, d1Max])
  );
  if (transpose) {
    mel_spec = mel_spec.transpose(1, 0);
  }
  const mel_spec_data = (
    /** @type {Float32Array} */
    mel_spec.data
  );
  for (let i = 0; i < mel_spec_data.length; ++i) {
    mel_spec_data[i] = mel_offset + Math.max(mel_floor, mel_spec_data[i]);
  }
  if (power !== null && log_mel !== null) {
    const o = Math.min(mel_spec_data.length, d1 * num_mel_filters);
    switch (log_mel) {
      case "log":
        for (let i = 0; i < o; ++i) {
          mel_spec_data[i] = Math.log(mel_spec_data[i]);
        }
        break;
      case "log10":
        for (let i = 0; i < o; ++i) {
          mel_spec_data[i] = Math.log10(mel_spec_data[i]);
        }
        break;
      case "dB":
        if (power === 1) {
          amplitude_to_db(mel_spec_data, reference, min_value, db_range);
        } else if (power === 2) {
          power_to_db(mel_spec_data, reference, min_value, db_range);
        } else {
          throw new Error(`Cannot use log_mel option '${log_mel}' with power ${power}`);
        }
        break;
      default:
        throw new Error(`log_mel must be one of null, 'log', 'log10' or 'dB'. Got '${log_mel}'`);
    }
  }
  return mel_spec;
}
function window_function(window_length, name, {
  periodic = true,
  frame_length = null,
  center = true
} = {}) {
  const length = periodic ? window_length + 1 : window_length;
  let window2;
  switch (name) {
    case "boxcar":
      window2 = new Float64Array(length).fill(1);
      break;
    case "hann":
    case "hann_window":
      window2 = hanning(length);
      break;
    case "hamming":
      window2 = hamming(length);
      break;
    case "povey":
      window2 = hanning(length).map((x) => Math.pow(x, 0.85));
      break;
    default:
      throw new Error(`Unknown window type ${name}.`);
  }
  if (periodic) {
    window2 = window2.subarray(0, window_length);
  }
  if (frame_length === null) {
    return window2;
  }
  if (window_length > frame_length) {
    throw new Error(`Length of the window (${window_length}) may not be larger than frame_length (${frame_length})`);
  }
  return window2;
}
function encodeWAV(samples, rate) {
  let offset = 44;
  const buffer = new ArrayBuffer(offset + samples.length * 4);
  const view = new DataView(buffer);
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 4, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 3, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 4, true);
  view.setUint16(32, 4, true);
  view.setUint16(34, 32, true);
  writeString(view, 36, "data");
  view.setUint32(40, samples.length * 4, true);
  for (let i = 0; i < samples.length; ++i, offset += 4) {
    view.setFloat32(offset, samples[i], true);
  }
  return buffer;
}
function writeString(view, offset, string) {
  for (let i = 0; i < string.length; ++i) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}
var RawAudio = class {
  /**
   * Create a new `RawAudio` object.
   * @param {Float32Array} audio Audio data
   * @param {number} sampling_rate Sampling rate of the audio data
   */
  constructor(audio, sampling_rate) {
    this.audio = audio;
    this.sampling_rate = sampling_rate;
  }
  /**
   * Convert the audio to a wav file buffer.
   * @returns {ArrayBuffer} The WAV file.
   */
  toWav() {
    return encodeWAV(this.audio, this.sampling_rate);
  }
  /**
   * Convert the audio to a blob.
   * @returns {Blob}
   */
  toBlob() {
    const wav = this.toWav();
    const blob = new Blob([wav], { type: "audio/wav" });
    return blob;
  }
  /**
   * Save the audio to a wav file.
   * @param {string} path
   */
  async save(path3) {
    let fn;
    if (apis.IS_BROWSER_ENV) {
      if (apis.IS_WEBWORKER_ENV) {
        throw new Error("Unable to save a file from a Web Worker.");
      }
      fn = saveBlob;
    } else if (apis.IS_FS_AVAILABLE) {
      fn = async (path4, blob) => {
        let buffer = await blob.arrayBuffer();
        fs3.writeFileSync(path4, Buffer.from(buffer));
      };
    } else {
      throw new Error("Unable to save because filesystem is disabled in this environment.");
    }
    await fn(path3, this.toBlob());
  }
};

// src/models/audio_spectrogram_transformer/feature_extraction_audio_spectrogram_transformer.js
var ASTFeatureExtractor = class extends FeatureExtractor {
  constructor(config) {
    super(config);
    const sampling_rate = this.config.sampling_rate;
    const mel_filters = mel_filter_bank(
      257,
      // num_frequency_bins
      this.config.num_mel_bins,
      // num_mel_filters
      20,
      // min_frequency
      Math.floor(sampling_rate / 2),
      // max_frequency
      sampling_rate,
      // sampling_rate
      null,
      // norm
      "kaldi",
      // mel_scale
      true
      // triangularize_in_mel_space
    );
    this.mel_filters = mel_filters;
    this.window = window_function(400, "hann", {
      periodic: false
    });
    this.mean = this.config.mean;
    this.std = this.config.std;
  }
  /**
   * Computes the log-Mel spectrogram of the provided audio waveform.
   * @param {Float32Array|Float64Array} waveform The audio waveform to process.
   * @param {number} max_length The maximum number of frames to return.
   * @returns {Promise<Tensor>} An object containing the log-Mel spectrogram data as a Float32Array and its dimensions as an array of numbers.
   */
  async _extract_fbank_features(waveform, max_length) {
    return spectrogram(
      waveform,
      this.window,
      // window
      400,
      // frame_length
      160,
      // hop_length
      {
        fft_length: 512,
        power: 2,
        center: false,
        preemphasis: 0.97,
        mel_filters: this.mel_filters,
        log_mel: "log",
        mel_floor: 1192092955078125e-22,
        remove_dc_offset: true,
        // Custom
        max_num_frames: max_length,
        transpose: true
      }
    );
  }
  /**
   * Asynchronously extracts features from a given audio using the provided configuration.
   * @param {Float32Array|Float64Array} audio The audio data as a Float32Array/Float64Array.
   * @returns {Promise<{ input_values: Tensor }>} A Promise resolving to an object containing the extracted input features as a Tensor.
   */
  async _call(audio) {
    validate_audio_inputs(audio, "ASTFeatureExtractor");
    const features = await this._extract_fbank_features(audio, this.config.max_length);
    if (this.config.do_normalize) {
      const denom = this.std * 2;
      const features_data = features.data;
      for (let i = 0; i < features_data.length; ++i) {
        features_data[i] = (features_data[i] - this.mean) / denom;
      }
    }
    return {
      input_values: features.unsqueeze_(0)
    };
  }
};

// src/models/encodec/feature_extraction_encodec.js
var EncodecFeatureExtractor = class extends FeatureExtractor {
  /**
   * Asynchronously extracts input values from a given audio using the provided configuration.
   * @param {Float32Array|Float64Array} audio The audio data as a Float32Array/Float64Array.
   * @returns {Promise<{ input_values: Tensor; }>} The extracted input values.
   */
  async _call(audio) {
    validate_audio_inputs(audio, "EncodecFeatureExtractor");
    if (audio instanceof Float64Array) {
      audio = new Float32Array(audio);
    }
    const num_channels = this.config.feature_size;
    if (audio.length % num_channels !== 0) {
      throw new Error(`The length of the audio data must be a multiple of the number of channels (${num_channels}).`);
    }
    const shape = [
      1,
      /* batch_size */
      num_channels,
      /* num_channels */
      audio.length / num_channels
      /* num_samples */
    ];
    return {
      input_values: new Tensor2("float32", audio, shape)
    };
  }
};

// src/models/clap/feature_extraction_clap.js
var ClapFeatureExtractor = class extends FeatureExtractor {
  constructor(config) {
    super(config);
    this.mel_filters = mel_filter_bank(
      this.config.nb_frequency_bins,
      // num_frequency_bins
      this.config.feature_size,
      // num_mel_filters
      this.config.frequency_min,
      // min_frequency
      this.config.frequency_max,
      // max_frequency
      this.config.sampling_rate,
      // sampling_rate
      null,
      // norm
      "htk"
      // mel_scale
    );
    this.mel_filters_slaney = mel_filter_bank(
      this.config.nb_frequency_bins,
      // num_frequency_bins
      this.config.feature_size,
      // num_mel_filters
      this.config.frequency_min,
      // min_frequency
      this.config.frequency_max,
      // max_frequency
      this.config.sampling_rate,
      // sampling_rate
      "slaney",
      // norm
      "slaney"
      // mel_scale
    );
    this.window = window_function(this.config.fft_window_size, "hann");
  }
  /**
   * Extracts the mel spectrogram and prepares it for the mode based on the `truncation` and `padding` arguments.
   * 
   * Four different path are possible:
   *   - `truncation="fusion"` and the length of the waveform is greater than the max length: the mel spectrogram
   *     will be computed on the entire audio. 3 random crops and a dowsampled version of the full mel spectrogram
   *     are then stacked together. They will later be used for `feature_fusion`.
   *   - `truncation="rand_trunc"` and the length of the waveform is smaller than the max length: the audio is
   *     padded based on `padding`.
   *   - `truncation="fusion"` and the length of the waveform is smaller than the max length: the audio is padded
   *     based on `padding`, and is repeated `4` times.
   *   - `truncation="rand_trunc"` and the length of the waveform is greater than the max length: the mel
   *     spectrogram will be computed on a random crop of the waveform.
   * 
   * @param {Float32Array|Float64Array} waveform The input waveform.
   * @param {number} max_length The maximum length of the waveform.
   * @param {string} truncation The truncation strategy to use.
   * @param {string} padding The padding strategy to use.
   * @returns {Promise<Tensor>} An object containing the mel spectrogram data as a Float32Array, its dimensions as an array of numbers, and a boolean indicating whether the waveform was longer than the max length.
   * @private
   */
  async _get_input_mel(waveform, max_length, truncation, padding) {
    let input_mel;
    let longer = false;
    const diff = waveform.length - max_length;
    if (diff > 0) {
      if (truncation === "rand_trunc") {
        longer = true;
        const idx = Math.floor(Math.random() * (diff + 1));
        waveform = waveform.subarray(idx, idx + max_length);
        input_mel = await this._extract_fbank_features(waveform, this.mel_filters_slaney, this.config.nb_max_samples);
      } else {
        throw new Error(`Truncation strategy "${truncation}" not implemented`);
      }
    } else {
      if (diff < 0) {
        let padded = new Float64Array(max_length);
        padded.set(waveform);
        if (padding === "repeat") {
          for (let i = waveform.length; i < max_length; i += waveform.length) {
            padded.set(waveform.subarray(0, Math.min(waveform.length, max_length - i)), i);
          }
        } else if (padding === "repeatpad") {
          for (let i = waveform.length; i < -diff; i += waveform.length) {
            padded.set(waveform, i);
          }
        }
        waveform = padded;
      }
      if (truncation === "fusion") {
        throw new Error(`Truncation strategy "${truncation}" not implemented`);
      }
      input_mel = await this._extract_fbank_features(waveform, this.mel_filters_slaney, this.config.nb_max_samples);
    }
    return input_mel.unsqueeze_(0);
  }
  /**
   * Compute the log-mel spectrogram of the provided `waveform` using the Hann window.
   * In CLAP, two different filter banks are used depending on the truncation pattern:
   *  - `self.mel_filters`: they correspond to the default parameters of `torchaudio` which can be obtained from
   *    calling `torchaudio.transforms.MelSpectrogram().mel_scale.fb`. These filters are used when `truncation`
   *    is set to `"fusion"`.
   *  - `self.mel_filteres_slaney` : they correspond to the default parameters of `librosa` which used
   *    `librosa.filters.mel` when computing the mel spectrogram. These filters were only used in the original
   *    implementation when the truncation mode is not `"fusion"`.
   * 
   * @param {Float32Array|Float64Array} waveform The audio waveform to process.
   * @param {number[][]} mel_filters The mel filters to use.
   * @param {number} [max_length=null] The maximum number of frames to return.
   * @returns {Promise<Tensor>} An object containing the log-Mel spectrogram data as a Float32Array and its dimensions as an array of numbers.
   */
  async _extract_fbank_features(waveform, mel_filters, max_length = null) {
    return spectrogram(
      waveform,
      this.window,
      // window
      this.config.fft_window_size,
      // frame_length
      this.config.hop_length,
      // hop_length
      {
        power: 2,
        mel_filters,
        log_mel: "dB",
        // Custom
        max_num_frames: max_length,
        do_pad: false,
        transpose: true
      }
    );
  }
  /**
   * Asynchronously extracts features from a given audio using the provided configuration.
   * @param {Float32Array|Float64Array} audio The audio data as a Float32Array/Float64Array.
   * @returns {Promise<{ input_features: Tensor }>} A Promise resolving to an object containing the extracted input features as a Tensor.
   */
  async _call(audio, {
    max_length = null
  } = {}) {
    validate_audio_inputs(audio, "ClapFeatureExtractor");
    const padded_inputs = await this._get_input_mel(
      audio,
      max_length ?? this.config.nb_max_samples,
      this.config.truncation,
      this.config.padding
    );
    return {
      input_features: padded_inputs.unsqueeze_(0)
    };
  }
};

// src/models/dac/feature_extraction_dac.js
var DacFeatureExtractor = class extends EncodecFeatureExtractor {
};

// src/models/gemma3n/feature_extraction_gemma3n.js
var Gemma3nAudioFeatureExtractor = class extends FeatureExtractor {
  constructor(config) {
    super(config);
    const {
      fft_length,
      feature_size,
      min_frequency,
      max_frequency,
      sampling_rate,
      frame_length
    } = this.config;
    const mel_filters = mel_filter_bank(
      Math.floor(1 + fft_length / 2),
      // num_frequency_bins
      feature_size,
      // num_mel_filters
      min_frequency,
      // min_frequency
      max_frequency,
      // max_frequency
      sampling_rate,
      // sampling_rate
      null,
      // norm
      "htk",
      // mel_scale
      false
      // triangularize_in_mel_space
    );
    this.mel_filters = mel_filters;
    this.window = window_function(frame_length, "hann");
  }
  /**
   * Computes the log-Mel spectrogram of the provided audio waveform.
   * @param {Float32Array|Float64Array} waveform The audio waveform to process.
   * @param {number} max_length The maximum number of frames to return.
   * @returns {Promise<Tensor>} An object containing the log-Mel spectrogram data as a Float32Array and its dimensions as an array of numbers.
   */
  async _extract_fbank_features(waveform, max_length) {
    return spectrogram(
      waveform,
      this.window,
      // window
      this.config.frame_length,
      // frame_length
      this.config.hop_length,
      // hop_length
      {
        fft_length: this.config.fft_length,
        center: false,
        onesided: true,
        preemphasis: this.config.preemphasis,
        preemphasis_htk_flavor: this.config.preemphasis_htk_flavor,
        mel_filters: this.mel_filters,
        log_mel: "log",
        mel_floor: this.config.mel_floor,
        remove_dc_offset: false,
        // Custom
        transpose: true
      }
    );
  }
  /**
   * Asynchronously extracts features from a given audio using the provided configuration.
   * @param {Float32Array|Float64Array} audio The audio data as a Float32Array/Float64Array.
   * @param {Object} options Optional parameters for feature extraction.
   * @param {number} [options.max_length=480_000] If provided, defines the maximum length of the audio to allow.
   * Audio longer than this will be truncated if `truncation=True`.
   * @param {boolean} [options.truncation=true] Whether or not to truncate audio above `max_length`.
   * @param {boolean} [options.padding=true] Whether to pad the sequence to a multiple of `pad_to_multiple_of`.
   * @param {number} [options.pad_to_multiple_of=128] The number to pad the sequence to a multiple of.
   * @returns {Promise<{ input_features: Tensor, input_features_mask: Tensor }>} A Promise resolving to an object containing the extracted input features and attention masks as Tensors.
   */
  async _call(audio, {
    max_length = 48e4,
    truncation = true,
    padding = true,
    pad_to_multiple_of = 128
  } = {}) {
    validate_audio_inputs(audio, "Gemma3nAudioFeatureExtractor");
    if (truncation && audio.length > max_length) {
      audio = audio.slice(0, max_length);
    }
    if (padding && audio.length % pad_to_multiple_of !== 0) {
      const padding_length = pad_to_multiple_of - audio.length % pad_to_multiple_of;
      const padded_audio = new Float64Array(audio.length + padding_length);
      padded_audio.set(audio);
      if (this.config.padding_value !== 0) {
        padded_audio.fill(this.config.padding_value, audio.length);
      }
      audio = padded_audio;
    }
    const features = await this._extract_fbank_features(audio, this.config.max_length);
    const padded_attention_mask = full([1, features.dims[0]], true);
    return {
      input_features: features.unsqueeze_(0),
      input_features_mask: padded_attention_mask
    };
  }
};

// src/models/moonshine/feature_extraction_moonshine.js
var MoonshineFeatureExtractor = class extends FeatureExtractor {
  /**
   * Asynchronously extracts input values from a given audio using the provided configuration.
   * @param {Float32Array|Float64Array} audio The audio data as a Float32Array/Float64Array.
   * @returns {Promise<{ input_values: Tensor; }>} The extracted input values.
   */
  async _call(audio) {
    validate_audio_inputs(audio, "MoonshineFeatureExtractor");
    if (audio instanceof Float64Array) {
      audio = new Float32Array(audio);
    }
    const shape = [
      1,
      /* batch_size */
      audio.length
      /* num_samples */
    ];
    return {
      input_values: new Tensor2("float32", audio, shape)
    };
  }
};

// src/models/parakeet/feature_extraction_parakeet.js
var EPSILON = 1e-5;
var ParakeetFeatureExtractor = class extends FeatureExtractor {
  constructor(config) {
    super(config);
    this.config.mel_filters ??= mel_filter_bank(
      Math.floor(1 + this.config.n_fft / 2),
      // num_frequency_bins
      this.config.feature_size,
      // num_mel_filters
      0,
      // min_frequency
      this.config.sampling_rate / 2,
      // max_frequency
      this.config.sampling_rate,
      // sampling_rate
      "slaney",
      // norm
      "slaney"
      // mel_scale
    );
    const window2 = window_function(this.config.win_length, "hann", {
      periodic: false
    });
    this.window = new Float64Array(this.config.n_fft);
    const offset = Math.floor((this.config.n_fft - this.config.win_length) / 2);
    this.window.set(window2, offset);
  }
  /**
   * Computes the log-Mel spectrogram of the provided audio waveform.
   * @param {Float32Array|Float64Array} waveform The audio waveform to process.
   * @returns {Promise<Tensor>} An object containing the log-Mel spectrogram data as a Float32Array and its dimensions as an array of numbers.
   */
  async _extract_fbank_features(waveform) {
    const preemphasis = this.config.preemphasis;
    waveform = new Float64Array(waveform);
    for (let j = waveform.length - 1; j >= 1; --j) {
      waveform[j] -= preemphasis * waveform[j - 1];
    }
    const features = await spectrogram(
      waveform,
      this.window,
      // window
      this.window.length,
      // frame_length
      this.config.hop_length,
      // hop_length
      {
        fft_length: this.config.n_fft,
        power: 2,
        mel_filters: this.config.mel_filters,
        log_mel: "log",
        mel_floor: -Infinity,
        pad_mode: "constant",
        center: true,
        // Custom
        transpose: true,
        mel_offset: 2 ** -24
      }
    );
    return features;
  }
  /**
   * Asynchronously extracts features from a given audio using the provided configuration.
   * @param {Float32Array|Float64Array} audio The audio data as a Float32Array/Float64Array.
   * @returns {Promise<{ input_features: Tensor; attention_mask: Tensor; }>} A Promise resolving to an object containing the extracted input features as a Tensor.
   */
  async _call(audio) {
    validate_audio_inputs(audio, "ParakeetFeatureExtractor");
    const features = await this._extract_fbank_features(audio);
    const features_length = Math.floor(
      (audio.length + Math.floor(this.config.n_fft / 2) * 2 - this.config.n_fft) / this.config.hop_length
    );
    const features_data = (
      /** @type {Float32Array} */
      features.data
    );
    features_data.fill(0, features_length * features.dims[1]);
    const [num_frames, num_features] = features.dims;
    const sum = new Float64Array(num_features);
    const sum_sq = new Float64Array(num_features);
    for (let i = 0; i < features_length; ++i) {
      const offset = i * num_features;
      for (let j = 0; j < num_features; ++j) {
        const val = features_data[offset + j];
        sum[j] += val;
        sum_sq[j] += val * val;
      }
    }
    const divisor = features_length > 1 ? features_length - 1 : 1;
    for (let j = 0; j < num_features; ++j) {
      const mean2 = sum[j] / features_length;
      const variance = (sum_sq[j] - features_length * mean2 * mean2) / divisor;
      const std = Math.sqrt(variance) + EPSILON;
      const inv_std = 1 / std;
      for (let i = 0; i < features_length; ++i) {
        const index = i * num_features + j;
        features_data[index] = (features_data[index] - mean2) * inv_std;
      }
    }
    const mask_data = new BigInt64Array(num_frames);
    mask_data.fill(1n, 0, features_length);
    return {
      input_features: features.unsqueeze_(0),
      attention_mask: new Tensor2("int64", mask_data, [1, num_frames])
    };
  }
};

// src/models/pyannote/feature_extraction_pyannote.js
var PyAnnoteFeatureExtractor = class extends FeatureExtractor {
  /**
   * Asynchronously extracts features from a given audio using the provided configuration.
   * @param {Float32Array|Float64Array} audio The audio data as a Float32Array/Float64Array.
   * @returns {Promise<{ input_values: Tensor; }>} The extracted input features.
   */
  async _call(audio) {
    validate_audio_inputs(audio, "PyAnnoteFeatureExtractor");
    if (audio instanceof Float64Array) {
      audio = new Float32Array(audio);
    }
    const shape = [
      1,
      /* batch_size */
      1,
      /* num_channels */
      audio.length
      /* num_samples */
    ];
    return {
      input_values: new Tensor2("float32", audio, shape)
    };
  }
  /**
   * NOTE: Can return fractional values. `Math.ceil` will ensure correct value.
   * @param {number} samples The number of frames in the audio.
   * @returns {number} The number of frames in the audio.
   */
  samples_to_frames(samples) {
    return (samples - this.config.offset) / this.config.step;
  }
  /**
   * Post-processes the speaker diarization logits output by the model.
   * @param {import('../../utils/tensor.js').Tensor} logits The speaker diarization logits output by the model.
   * @param {number} num_samples Number of samples in the input audio.
   * @returns {Array<Array<{ id: number, start: number, end: number, confidence: number }>>} The post-processed speaker diarization results.
   */
  post_process_speaker_diarization(logits, num_samples) {
    const ratio = num_samples / this.samples_to_frames(num_samples) / this.config.sampling_rate;
    const results = [];
    for (const scores of logits.tolist()) {
      const accumulated_segments = [];
      let current_speaker = -1;
      for (let i = 0; i < scores.length; ++i) {
        const probabilities = softmax(scores[i]);
        const [score, id] = max(probabilities);
        const [start, end] = [i, i + 1];
        if (id !== current_speaker) {
          current_speaker = id;
          accumulated_segments.push({ id, start, end, score });
        } else {
          accumulated_segments.at(-1).end = end;
          accumulated_segments.at(-1).score += score;
        }
      }
      results.push(accumulated_segments.map(
        // Convert frame-space to time-space
        // and compute the confidence
        ({ id, start, end, score }) => ({
          id,
          start: start * ratio,
          end: end * ratio,
          confidence: score / (end - start)
        })
      ));
    }
    return results;
  }
};

// src/models/seamless_m4t/feature_extraction_seamless_m4t.js
var SeamlessM4TFeatureExtractor = class extends FeatureExtractor {
  constructor(config) {
    super(config);
    const sampling_rate = this.config.sampling_rate;
    const mel_filters = mel_filter_bank(
      257,
      // num_frequency_bins
      this.config.num_mel_bins,
      // num_mel_filters
      20,
      // min_frequency
      Math.floor(sampling_rate / 2),
      // max_frequency
      sampling_rate,
      // sampling_rate
      null,
      // norm
      "kaldi",
      // mel_scale
      true
      // triangularize_in_mel_space
    );
    this.mel_filters = mel_filters;
    this.window = window_function(400, "povey", {
      periodic: false
    });
  }
  /**
   * Computes the log-Mel spectrogram of the provided audio waveform.
   * @param {Float32Array|Float64Array} waveform The audio waveform to process.
   * @param {number} max_length The maximum number of frames to return.
   * @returns {Promise<Tensor>} An object containing the log-Mel spectrogram data as a Float32Array and its dimensions as an array of numbers.
   */
  async _extract_fbank_features(waveform, max_length) {
    waveform = waveform.map((x) => x * 32768);
    return spectrogram(
      waveform,
      this.window,
      // window
      400,
      // frame_length
      160,
      // hop_length
      {
        fft_length: 512,
        power: 2,
        center: false,
        preemphasis: 0.97,
        mel_filters: this.mel_filters,
        log_mel: "log",
        mel_floor: 1192092955078125e-22,
        remove_dc_offset: true,
        // Custom
        max_num_frames: max_length,
        transpose: true
      }
    );
  }
  /**
   * Asynchronously extracts features from a given audio using the provided configuration.
   * @param {Float32Array|Float64Array} audio The audio data as a Float32Array/Float64Array.
   * @param {Object} options Optional parameters for feature extraction.
   * @param {boolean} [options.padding=true] Whether to pad the sequence to a multiple of `pad_to_multiple_of`.
   * @param {number} [options.pad_to_multiple_of=2] The number to pad the sequence to a multiple of.
   * @param {boolean} [options.do_normalize_per_mel_bins=true] Whether or not to zero-mean unit-variance normalize the input per mel-channel.
   * @param {boolean} [options.return_attention_mask=true] Whether to return the attention mask.
   * @returns {Promise<{ input_features: Tensor, attention_mask?: Tensor }>} A Promise resolving to an object containing the extracted input features and attention masks as Tensors.
   */
  async _call(audio, {
    padding = true,
    pad_to_multiple_of = 2,
    do_normalize_per_mel_bins = true,
    return_attention_mask = true
  } = {}) {
    validate_audio_inputs(audio, "SeamlessM4TFeatureExtractor");
    let features = await this._extract_fbank_features(audio, this.config.max_length);
    if (do_normalize_per_mel_bins) {
      const [num_features, feature_size] = features.dims;
      const data = features.data;
      for (let i = 0; i < feature_size; ++i) {
        let sum = 0;
        for (let j = 0; j < num_features; ++j) {
          sum += data[j * feature_size + i];
        }
        const mean2 = sum / num_features;
        let variance = 0;
        for (let j = 0; j < num_features; ++j) {
          variance += (data[j * feature_size + i] - mean2) ** 2;
        }
        variance /= num_features - 1;
        const std = Math.sqrt(variance + 1e-7);
        for (let j = 0; j < num_features; ++j) {
          const index = j * feature_size + i;
          data[index] = (data[index] - mean2) / std;
        }
      }
    }
    let padded_attention_mask;
    if (padding) {
      const [num_frames2, num_channels2] = features.dims;
      const data = (
        /** @type {Float32Array} */
        features.data
      );
      const pad_size = num_frames2 % pad_to_multiple_of;
      if (pad_size > 0) {
        const padded_data = new Float32Array(num_channels2 * (num_frames2 + pad_size));
        padded_data.set(data);
        padded_data.fill(this.config.padding_value, data.length);
        const numPaddedFrames = num_frames2 + pad_size;
        features = new Tensor2(
          features.type,
          padded_data,
          [numPaddedFrames, num_channels2]
        );
        if (return_attention_mask) {
          padded_attention_mask = new Tensor2(
            "int64",
            new BigInt64Array(numPaddedFrames),
            [1, numPaddedFrames]
          );
          padded_attention_mask.data.fill(1n, 0, num_frames2);
        }
      }
    }
    const [num_frames, num_channels] = features.dims;
    const stride = this.config.stride;
    const remainder = num_frames % stride;
    if (remainder !== 0) {
      throw new Error(`The number of frames (${num_frames}) must be a multiple of the stride (${stride}).`);
    }
    const input_features = features.view(
      1,
      Math.floor(num_frames / stride),
      num_channels * stride
    );
    const result = { input_features };
    if (return_attention_mask) {
      const reshapedNumFrames = input_features.dims[1];
      const attention_mask_data = new BigInt64Array(reshapedNumFrames);
      if (padded_attention_mask) {
        const padded_attention_mask_data = padded_attention_mask.data;
        for (let i = 1, j = 0; i < num_frames; i += stride, ++j) {
          attention_mask_data[j] = padded_attention_mask_data[i];
        }
      } else {
        attention_mask_data.fill(1n);
      }
      result.attention_mask = new Tensor2(
        "int64",
        attention_mask_data,
        [1, reshapedNumFrames]
      );
    }
    return result;
  }
};

// src/models/snac/feature_extraction_snac.js
var SnacFeatureExtractor = class extends DacFeatureExtractor {
};

// src/models/speecht5/feature_extraction_speecht5.js
var SpeechT5FeatureExtractor = class extends FeatureExtractor {
};

// src/models/wav2vec2/feature_extraction_wav2vec2.js
var Wav2Vec2FeatureExtractor = class extends FeatureExtractor {
  /**
   * @param {Float32Array} input_values 
   * @returns {Float32Array} 
   */
  _zero_mean_unit_var_norm(input_values) {
    const sum = input_values.reduce((a, b) => a + b, 0);
    const mean2 = sum / input_values.length;
    const variance = input_values.reduce((a, b) => a + (b - mean2) ** 2, 0) / input_values.length;
    return input_values.map((x) => (x - mean2) / Math.sqrt(variance + 1e-7));
  }
  /**
   * Asynchronously extracts features from a given audio using the provided configuration.
   * @param {Float32Array|Float64Array} audio The audio data as a Float32Array/Float64Array.
   * @returns {Promise<{ input_values: Tensor; attention_mask: Tensor }>} A Promise resolving to an object containing the extracted input features and attention mask as Tensors.
   */
  async _call(audio) {
    validate_audio_inputs(audio, "Wav2Vec2FeatureExtractor");
    if (audio instanceof Float64Array) {
      audio = new Float32Array(audio);
    }
    let input_values = audio;
    if (this.config.do_normalize) {
      input_values = this._zero_mean_unit_var_norm(input_values);
    }
    const shape = [1, input_values.length];
    return {
      input_values: new Tensor2("float32", input_values, shape),
      attention_mask: new Tensor2("int64", new BigInt64Array(input_values.length).fill(1n), shape)
    };
  }
};

// src/models/wespeaker/feature_extraction_wespeaker.js
var WeSpeakerFeatureExtractor = class extends FeatureExtractor {
  constructor(config) {
    super(config);
    const sampling_rate = this.config.sampling_rate;
    const mel_filters = mel_filter_bank(
      257,
      // num_frequency_bins
      this.config.num_mel_bins,
      // num_mel_filters
      20,
      // min_frequency
      Math.floor(sampling_rate / 2),
      // max_frequency
      sampling_rate,
      // sampling_rate
      null,
      // norm
      "kaldi",
      // mel_scale
      true
      // triangularize_in_mel_space
    );
    this.mel_filters = mel_filters;
    this.window = window_function(400, "hamming", {
      periodic: false
    });
    this.min_num_frames = this.config.min_num_frames;
  }
  /**
   * Computes the log-Mel spectrogram of the provided audio waveform.
   * @param {Float32Array|Float64Array} waveform The audio waveform to process.
   * @returns {Promise<Tensor>} An object containing the log-Mel spectrogram data as a Float32Array and its dimensions as an array of numbers.
   */
  async _extract_fbank_features(waveform) {
    waveform = waveform.map((x) => x * 32768);
    return spectrogram(
      waveform,
      this.window,
      // window
      400,
      // frame_length
      160,
      // hop_length
      {
        fft_length: 512,
        power: 2,
        center: false,
        preemphasis: 0.97,
        mel_filters: this.mel_filters,
        log_mel: "log",
        mel_floor: 1192092955078125e-22,
        remove_dc_offset: true,
        // Custom
        transpose: true,
        min_num_frames: this.min_num_frames
      }
    );
  }
  /**
   * Asynchronously extracts features from a given audio using the provided configuration.
   * @param {Float32Array|Float64Array} audio The audio data as a Float32Array/Float64Array.
   * @returns {Promise<{ input_features: Tensor }>} A Promise resolving to an object containing the extracted input features as a Tensor.
   */
  async _call(audio) {
    validate_audio_inputs(audio, "WeSpeakerFeatureExtractor");
    const features = (await this._extract_fbank_features(audio)).unsqueeze_(0);
    if (this.config.fbank_centering_span === null) {
      const meanData = (
        /** @type {Float32Array} */
        features.mean(1).data
      );
      const featuresData = (
        /** @type {Float32Array} */
        features.data
      );
      const [batch_size, num_frames, feature_size] = features.dims;
      for (let i = 0; i < batch_size; ++i) {
        const offset1 = i * num_frames * feature_size;
        const offset2 = i * feature_size;
        for (let j = 0; j < num_frames; ++j) {
          const offset3 = offset1 + j * feature_size;
          for (let k = 0; k < feature_size; ++k) {
            featuresData[offset3 + k] -= meanData[offset2 + k];
          }
        }
      }
    }
    return {
      input_features: features
    };
  }
};

// src/models/whisper/feature_extraction_whisper.js
var WhisperFeatureExtractor = class extends FeatureExtractor {
  constructor(config) {
    super(config);
    this.config.mel_filters ??= mel_filter_bank(
      Math.floor(1 + this.config.n_fft / 2),
      // num_frequency_bins
      this.config.feature_size,
      // num_mel_filters
      0,
      // min_frequency
      8e3,
      // max_frequency
      this.config.sampling_rate,
      // sampling_rate
      "slaney",
      // norm
      "slaney"
      // mel_scale
    );
    this.window = window_function(this.config.n_fft, "hann");
  }
  /**
   * Computes the log-Mel spectrogram of the provided audio waveform.
   * @param {Float32Array|Float64Array} waveform The audio waveform to process.
   * @returns {Promise<Tensor>} An object containing the log-Mel spectrogram data as a Float32Array and its dimensions as an array of numbers.
   */
  async _extract_fbank_features(waveform) {
    const features = await spectrogram(
      waveform,
      this.window,
      // window
      this.config.n_fft,
      // frame_length
      this.config.hop_length,
      // hop_length
      {
        power: 2,
        mel_filters: this.config.mel_filters,
        log_mel: "log10",
        // Custom
        max_num_frames: Math.min(
          Math.floor(waveform.length / this.config.hop_length),
          this.config.nb_max_frames
          // 3000
        )
      }
    );
    const data = features.data;
    const maxValue = max(
      /** @type {Float32Array} */
      data
    )[0];
    for (let i = 0; i < data.length; ++i) {
      data[i] = (Math.max(data[i], maxValue - 8) + 4) / 4;
    }
    return features;
  }
  /**
   * Asynchronously extracts features from a given audio using the provided configuration.
   * @param {Float32Array|Float64Array} audio The audio data as a Float32Array/Float64Array.
   * @returns {Promise<{ input_features: Tensor }>} A Promise resolving to an object containing the extracted input features as a Tensor.
   */
  async _call(audio, {
    max_length = null
  } = {}) {
    validate_audio_inputs(audio, "WhisperFeatureExtractor");
    let waveform;
    const length = max_length ?? this.config.n_samples;
    if (audio.length > length) {
      if (audio.length > this.config.n_samples) {
        console.warn(
          "Attempting to extract features for audio longer than 30 seconds. If using a pipeline to extract transcript from a long audio clip, remember to specify `chunk_length_s` and/or `stride_length_s`."
        );
      }
      waveform = audio.slice(0, length);
    } else {
      waveform = new Float32Array(length);
      waveform.set(audio);
    }
    const features = await this._extract_fbank_features(waveform);
    return {
      input_features: features.unsqueeze_(0)
    };
  }
};

// src/models/auto/feature_extraction_auto.js
var AutoFeatureExtractor = class {
  /** @type {typeof FeatureExtractor.from_pretrained} */
  static async from_pretrained(pretrained_model_name_or_path, options = {}) {
    const preprocessorConfig = await getModelJSON(pretrained_model_name_or_path, FEATURE_EXTRACTOR_NAME, true, options);
    const key = preprocessorConfig.feature_extractor_type;
    const feature_extractor_class = feature_extractors_exports[key];
    if (!feature_extractor_class) {
      throw new Error(`Unknown feature_extractor_type: '${key}'. Please report this at ${GITHUB_ISSUE_URL}.`);
    }
    return new feature_extractor_class(preprocessorConfig);
  }
};

// src/models/gemma3n/processing_gemma3n.js
var Gemma3nProcessor = class extends Processor {
  static image_processor_class = AutoImageProcessor;
  static feature_extractor_class = AutoFeatureExtractor;
  static tokenizer_class = AutoTokenizer;
  static uses_processor_config = true;
  static uses_chat_template_file = true;
  constructor(config, components, chat_template) {
    super(config, components, chat_template);
    this.audio_seq_length = this.config.audio_seq_length;
    this.image_seq_length = this.config.image_seq_length;
    const {
      // Audio tokens
      audio_token_id,
      boa_token,
      audio_token,
      eoa_token,
      // Image tokens
      image_token_id,
      boi_token,
      image_token,
      eoi_token
    } = this.tokenizer.config;
    this.audio_token_id = audio_token_id;
    this.boa_token = boa_token;
    this.audio_token = audio_token;
    const audio_tokens_expanded = audio_token.repeat(this.audio_seq_length);
    this.full_audio_sequence = `

${boa_token}${audio_tokens_expanded}${eoa_token}

`;
    this.image_token_id = image_token_id;
    this.boi_token = boi_token;
    this.image_token = image_token;
    const image_tokens_expanded = image_token.repeat(this.image_seq_length);
    this.full_image_sequence = `

${boi_token}${image_tokens_expanded}${eoi_token}

`;
  }
  /**
   * 
   * @param {string|string[]} text 
   * @param {RawImage|RawImage[]|RawImage[][]} images
   * @param {RawAudio|RawAudio[]|RawAudio[][]} audio
   * @returns {Promise<any>}
   */
  async _call(text, images = null, audio = null, options = {}) {
    if (typeof text === "string") {
      text = [text];
    }
    let audio_inputs;
    if (audio) {
      audio_inputs = await this.feature_extractor(audio, options);
      text = text.map((prompt) => prompt.replaceAll(this.audio_token, this.full_audio_sequence));
    }
    let image_inputs;
    if (images) {
      image_inputs = await this.image_processor(images, options);
      text = text.map((prompt) => prompt.replaceAll(this.image_token, this.full_image_sequence));
    }
    let text_inputs = this.tokenizer(text, options);
    return {
      ...text_inputs,
      ...image_inputs,
      ...audio_inputs
    };
  }
};

// src/models/grounding_dino/processing_grounding_dino.js
function get_phrases_from_posmap(posmaps, input_ids) {
  const left_idx = 0;
  const right_idx = posmaps.dims.at(-1) - 1;
  const posmaps_list = posmaps.tolist();
  posmaps_list.fill(false, 0, left_idx + 1);
  posmaps_list.fill(false, right_idx);
  const input_ids_list = input_ids.tolist();
  return posmaps_list.map((val, idx) => val ? idx : null).filter((idx) => idx !== null).map((i) => input_ids_list[i]);
}
var GroundingDinoProcessor = class extends Processor {
  static tokenizer_class = AutoTokenizer;
  static image_processor_class = AutoImageProcessor;
  /**
   * @typedef {import('../../utils/image.js').RawImage} RawImage
   */
  /**
   * 
   * @param {RawImage|RawImage[]|RawImage[][]} images  
   * @param {string|string[]} text 
   * @returns {Promise<any>}
   */
  async _call(images, text, options = {}) {
    const image_inputs = images ? await this.image_processor(images, options) : {};
    const text_inputs = text ? this.tokenizer(text, options) : {};
    return {
      ...text_inputs,
      ...image_inputs
    };
  }
  post_process_grounded_object_detection(outputs, input_ids, {
    box_threshold = 0.25,
    text_threshold = 0.25,
    target_sizes = null
  } = {}) {
    const { logits, pred_boxes } = outputs;
    const batch_size = logits.dims[0];
    if (target_sizes !== null && target_sizes.length !== batch_size) {
      throw Error("Make sure that you pass in as many target sizes as the batch dimension of the logits");
    }
    const num_queries = logits.dims.at(1);
    const probs = logits.sigmoid();
    const scores = probs.max(-1).tolist();
    const boxes = pred_boxes.tolist().map((batch) => batch.map((box) => center_to_corners_format(box)));
    const results = [];
    for (let i = 0; i < batch_size; ++i) {
      const target_size = target_sizes !== null ? target_sizes[i] : null;
      if (target_size !== null) {
        boxes[i] = boxes[i].map((box) => box.map((x, j) => x * target_size[(j + 1) % 2]));
      }
      const batch_scores = scores[i];
      const final_scores = [];
      const final_phrases = [];
      const final_boxes = [];
      for (let j = 0; j < num_queries; ++j) {
        const score = batch_scores[j];
        if (score <= box_threshold) {
          continue;
        }
        const box = boxes[i][j];
        const prob = probs[i][j];
        final_scores.push(score);
        final_boxes.push(box);
        const phrases = get_phrases_from_posmap(prob.gt(text_threshold), input_ids[i]);
        final_phrases.push(phrases);
      }
      results.push({ scores: final_scores, boxes: final_boxes, labels: this.batch_decode(final_phrases) });
    }
    return results;
  }
};

// src/models/idefics3/processing_idefics3.js
function _prompt_split_image(image_seq_len, image_rows, image_cols, fake_token_around_image, image_token, global_img_token) {
  let text_split_images = "";
  for (let n_h = 0; n_h < image_rows; ++n_h) {
    for (let n_w = 0; n_w < image_cols; ++n_w) {
      text_split_images += fake_token_around_image + `<row_${n_h + 1}_col_${n_w + 1}>` + image_token.repeat(image_seq_len);
    }
    text_split_images += "\n";
  }
  text_split_images += `
${fake_token_around_image}${global_img_token}` + image_token.repeat(image_seq_len) + `${fake_token_around_image}`;
  return text_split_images;
}
function _prompt_single_image(image_seq_len, fake_token_around_image, image_token, global_img_token) {
  return `${fake_token_around_image}${global_img_token}` + image_token.repeat(image_seq_len) + `${fake_token_around_image}`;
}
function get_image_prompt_string(image_rows, image_cols, image_seq_len, fake_token_around_image, image_token, global_img_token) {
  if (image_rows === 0 && image_cols === 0) {
    return _prompt_single_image(
      image_seq_len,
      fake_token_around_image,
      image_token,
      global_img_token
    );
  }
  return _prompt_split_image(
    image_seq_len,
    image_rows,
    image_cols,
    fake_token_around_image,
    image_token,
    global_img_token
  );
}
var Idefics3Processor = class extends Processor {
  static image_processor_class = AutoImageProcessor;
  static tokenizer_class = AutoTokenizer;
  static uses_processor_config = true;
  fake_image_token = "<fake_token_around_image>";
  image_token = "<image>";
  global_img_token = "<global-img>";
  /**
   * 
   * @param {string|string[]} text 
   * @param {RawImage|RawImage[]|RawImage[][]} images  
   * @returns {Promise<any>}
   */
  async _call(text, images = null, options = {}) {
    options.return_row_col_info ??= true;
    let image_inputs;
    if (images) {
      image_inputs = await this.image_processor(images, options);
    }
    if (!Array.isArray(text)) {
      text = [text];
    }
    const image_rows = image_inputs.rows ?? [new Array(text.length).fill(0)];
    const image_cols = image_inputs.cols ?? [new Array(text.length).fill(0)];
    const image_seq_len = this.config.image_seq_len;
    const n_images_in_text = [];
    const prompt_strings = [];
    for (let i = 0; i < text.length; ++i) {
      const sample = text[i];
      const sample_rows = image_rows[i];
      const sample_cols = image_cols[i];
      n_images_in_text.push(count(sample, this.image_token));
      const image_prompt_strings = sample_rows.map(
        (n_rows, j) => get_image_prompt_string(
          n_rows,
          sample_cols[j],
          image_seq_len,
          this.fake_image_token,
          this.image_token,
          this.global_img_token
        )
      );
      const split_sample = sample.split(this.image_token);
      if (split_sample.length === 0) {
        throw new Error("The image token should be present in the text.");
      }
      let new_sample = split_sample[0];
      for (let j = 0; j < image_prompt_strings.length; ++j) {
        new_sample += image_prompt_strings[j] + split_sample[j + 1];
      }
      prompt_strings.push(new_sample);
    }
    const text_inputs = this.tokenizer(prompt_strings);
    return {
      ...text_inputs,
      ...image_inputs
    };
  }
};

// src/models/janus/processing_janus.js
var VLChatProcessor = class extends Processor {
  static image_processor_class = AutoImageProcessor;
  static tokenizer_class = AutoTokenizer;
  static uses_processor_config = true;
  constructor(config, components, chat_template) {
    super(config, components, chat_template);
    this.image_tag = this.config.image_tag;
    this.image_start_tag = this.config.image_start_tag;
    this.image_end_tag = this.config.image_end_tag;
    this.num_image_tokens = this.config.num_image_tokens;
  }
  /**
   * @typedef {Object} MultimodalMessageProperties Additional properties for multimodal messages.
   * @property {(RawImage | string | URL)[]} [images] The images in the message.
   * @typedef {(import('../../tokenizers.js').Message & MultimodalMessageProperties)[]} MultimodalConversation The conversation possibly containing multimodal inputs.
   */
  /**
   * @typedef {Object} VLCChatProcessorResult The processed input.
   * @property {Tensor} input_ids The input IDs.
   * @property {Tensor} attention_mask The attention mask.
   * @property {Tensor} images_seq_mask The image sequence mask.
   * @property {Tensor} images_emb_mask The image embedding mask.
   */
  /**
   * @param {MultimodalConversation} conversation The chat messages to process.
   * @param {Object} options Additional options for processing.
   * @param {RawImage|RawImage[]} [options.images] The images to process, if not set in the conversation.
   * @param {string} [options.chat_template="default"] The chat template to use.
   * @returns {Promise<VLCChatProcessorResult | VLCChatProcessorResult & import('../../base/image_processors_utils.js').ImageProcessorResult>} The processed input.
   */
  async _call(conversation, {
    images = null,
    chat_template = "default"
  } = {}) {
    if (!images) {
      images = await Promise.all(
        conversation.filter((msg) => msg.images).flatMap((msg) => msg.images).map((img) => RawImage.read(img))
      );
    } else if (!Array.isArray(images)) {
      images = [images];
    }
    const tokenizer = this.tokenizer;
    const result = tokenizer.apply_chat_template(conversation, {
      tokenize: false,
      add_generation_prompt: true,
      chat_template
    });
    const encode = (text) => tokenizer.encode(text, { add_special_tokens: false });
    const parts = (
      /** @type {string} */
      result.split(this.image_tag)
    );
    const num_images = parts.length - 1;
    if (images.length !== num_images) {
      throw new Error(`Number of images provided (${images.length}) does not match number of "${this.image_tag}" image tags (${num_images})`);
    }
    const [
      image_placeholder_tag_id,
      image_start_tag_id,
      image_end_tag_id
    ] = tokenizer.model.convert_tokens_to_ids([
      this.image_tag,
      this.image_start_tag,
      this.image_end_tag
    ]);
    let input_ids = encode(parts[0]);
    let images_seq_mask = new Array(input_ids.length).fill(false);
    for (let i = 1; i < parts.length; ++i) {
      const placeholder_image_tokens = new Array(this.num_image_tokens).fill(image_placeholder_tag_id);
      const tokens = encode(parts[i]);
      input_ids = mergeArrays(
        input_ids,
        [image_start_tag_id],
        placeholder_image_tokens,
        [image_end_tag_id],
        tokens
      );
      const image_mask = new Array(this.num_image_tokens).fill(true);
      images_seq_mask = mergeArrays(
        images_seq_mask,
        [false],
        image_mask,
        [false],
        new Array(tokens.length).fill(false)
      );
    }
    const dims = [1, input_ids.length];
    const final = {
      input_ids: new Tensor2("int64", input_ids, dims),
      attention_mask: new Tensor2("int64", new Array(input_ids.length).fill(1), dims),
      images_seq_mask: new Tensor2("bool", images_seq_mask, dims),
      images_emb_mask: new Tensor2(
        "bool",
        new Array(num_images * this.num_image_tokens).fill(true),
        [1, num_images, this.num_image_tokens]
      )
    };
    if (images && images.length > 0) {
      const image_inputs = await this.image_processor(images);
      image_inputs.pixel_values.unsqueeze_(0);
      return { ...final, ...image_inputs };
    }
    return final;
  }
};

// src/models/jina_clip/processing_jina_clip.js
var JinaCLIPProcessor = class extends Processor {
  static tokenizer_class = AutoTokenizer;
  static image_processor_class = AutoImageProcessor;
  async _call(text = null, images = null, kwargs = {}) {
    if (!text && !images) {
      throw new Error("Either text or images must be provided");
    }
    const text_inputs = text ? this.tokenizer(text, kwargs) : {};
    const image_inputs = images ? await this.image_processor(images, kwargs) : {};
    return {
      ...text_inputs,
      ...image_inputs
    };
  }
};

// src/models/llava/processing_llava.js
var LlavaProcessor = class extends Processor {
  static tokenizer_class = AutoTokenizer;
  static image_processor_class = AutoImageProcessor;
  static uses_processor_config = true;
  /**
   * @typedef {import('../../utils/image.js').RawImage} RawImage
   */
  // `images` is required, `text` is optional
  async _call(images, text = null, kwargs = {}) {
    const image_inputs = await this.image_processor(images, kwargs);
    if (text) {
      const [height, width] = image_inputs.pixel_values.dims.slice(-2);
      const { image_token, patch_size, num_additional_image_tokens } = this.config;
      const num_image_tokens = Math.floor(
        height / patch_size
      ) * Math.floor(width / patch_size) + num_additional_image_tokens;
      text = structuredClone(text);
      if (!Array.isArray(text)) {
        text = [text];
      }
      for (let i = 0; i < text.length; ++i) {
        text[i] = text[i].replace(image_token, image_token.repeat(num_image_tokens));
      }
    }
    const text_inputs = text ? this.tokenizer(text, kwargs) : {};
    return {
      ...image_inputs,
      ...text_inputs
    };
  }
};

// src/models/mgp_str/processing_mgp_str.js
var DECODE_TYPE_MAPPING = {
  "char": ["char_decode", 1],
  "bpe": ["bpe_decode", 2],
  "wp": ["wp_decode", 102]
};
var MgpstrProcessor = class extends Processor {
  static tokenizer_class = AutoTokenizer;
  static image_processor_class = AutoImageProcessor;
  /**
   * @returns {import('../../tokenizers.js').MgpstrTokenizer} The character tokenizer.
   */
  get char_tokenizer() {
    return this.components.char_tokenizer;
  }
  /**
   * @returns {import('../../tokenizers.js').GPT2Tokenizer} The BPE tokenizer.
   */
  get bpe_tokenizer() {
    return this.components.bpe_tokenizer;
  }
  /**
   * @returns {import('../../tokenizers.js').BertTokenizer} The WordPiece tokenizer.
   */
  get wp_tokenizer() {
    return this.components.wp_tokenizer;
  }
  /**
   * Helper function to decode the model prediction logits.
   * @param {import('../../utils/tensor.js').Tensor} pred_logits Model prediction logits.
   * @param {string} format Type of model prediction. Must be one of ['char', 'bpe', 'wp'].
   * @returns {[string[], number[]]} The decoded sentences and their confidence scores.
   */
  _decode_helper(pred_logits, format2) {
    if (!DECODE_TYPE_MAPPING.hasOwnProperty(format2)) {
      throw new Error(`Format ${format2} is not supported.`);
    }
    const [decoder_name, eos_token] = DECODE_TYPE_MAPPING[format2];
    const decoder = this[decoder_name].bind(this);
    const [batch_size, batch_max_length] = pred_logits.dims;
    const conf_scores = [];
    const all_ids = [];
    const pred_logits_list = pred_logits.tolist();
    for (let i = 0; i < batch_size; ++i) {
      const logits = pred_logits_list[i];
      const ids = [];
      const scores = [];
      for (let j = 1; j < batch_max_length; ++j) {
        const [max_prob, max_prob_index] = max(softmax(logits[j]));
        scores.push(max_prob);
        if (max_prob_index == eos_token) {
          break;
        }
        ids.push(max_prob_index);
      }
      const confidence_score = scores.length > 0 ? scores.reduce((a, b) => a * b, 1) : 0;
      all_ids.push(ids);
      conf_scores.push(confidence_score);
    }
    const decoded = decoder(all_ids);
    return [decoded, conf_scores];
  }
  /**
   * Convert a list of lists of char token ids into a list of strings by calling char tokenizer.
   * @param {number[][]} sequences List of tokenized input ids.
   * @returns {string[]} The list of char decoded sentences.
   */
  char_decode(sequences) {
    return this.char_tokenizer.batch_decode(sequences).map((str) => str.replaceAll(" ", ""));
  }
  /**
   * Convert a list of lists of BPE token ids into a list of strings by calling BPE tokenizer.
   * @param {number[][]} sequences List of tokenized input ids.
   * @returns {string[]} The list of BPE decoded sentences.
   */
  bpe_decode(sequences) {
    return this.bpe_tokenizer.batch_decode(sequences);
  }
  /**
   * Convert a list of lists of word piece token ids into a list of strings by calling word piece tokenizer.
   * @param {number[][]} sequences List of tokenized input ids.
   * @returns {string[]} The list of wp decoded sentences.
   */
  wp_decode(sequences) {
    return this.wp_tokenizer.batch_decode(sequences).map((str) => str.replaceAll(" ", ""));
  }
  /**
   * Convert a list of lists of token ids into a list of strings by calling decode.
   * @param {[import('../../utils/tensor.js').Tensor, import('../../utils/tensor.js').Tensor, import('../../utils/tensor.js').Tensor]} sequences List of tokenized input ids.
   * @returns {{generated_text: string[], scores: number[], char_preds: string[], bpe_preds: string[], wp_preds: string[]}}
   * Dictionary of all the outputs of the decoded results.
   * - generated_text: The final results after fusion of char, bpe, and wp.
   * - scores: The final scores after fusion of char, bpe, and wp.
   * - char_preds: The list of character decoded sentences.
   * - bpe_preds: The list of BPE decoded sentences.
   * - wp_preds: The list of wp decoded sentences.
   */
  // @ts-expect-error The type of this method is not compatible with the one in the base class.
  batch_decode([char_logits, bpe_logits, wp_logits]) {
    const [char_preds, char_scores] = this._decode_helper(char_logits, "char");
    const [bpe_preds, bpe_scores] = this._decode_helper(bpe_logits, "bpe");
    const [wp_preds, wp_scores] = this._decode_helper(wp_logits, "wp");
    const generated_text = [];
    const scores = [];
    for (let i = 0; i < char_preds.length; ++i) {
      const [max_score, max_score_index] = max([char_scores[i], bpe_scores[i], wp_scores[i]]);
      generated_text.push([char_preds[i], bpe_preds[i], wp_preds[i]][max_score_index]);
      scores.push(max_score);
    }
    return {
      generated_text,
      scores,
      char_preds,
      bpe_preds,
      wp_preds
    };
  }
  /** @type {typeof Processor.from_pretrained} */
  static async from_pretrained(...args) {
    const base = await super.from_pretrained(...args);
    const bpe_tokenizer = await AutoTokenizer.from_pretrained("Xenova/gpt2");
    const wp_tokenizer = await AutoTokenizer.from_pretrained("Xenova/bert-base-uncased");
    base.components = {
      image_processor: base.image_processor,
      char_tokenizer: base.tokenizer,
      bpe_tokenizer,
      wp_tokenizer
    };
    return base;
  }
  async _call(images, text = null) {
    const result = await this.image_processor(images);
    if (text) {
      result.labels = this.tokenizer(text).input_ids;
    }
    return result;
  }
};

// src/models/moonshine/processing_moonshine.js
var MoonshineProcessor = class extends Processor {
  static tokenizer_class = AutoTokenizer;
  static feature_extractor_class = AutoFeatureExtractor;
  /**
   * Calls the feature_extractor function with the given audio input.
   * @param {any} audio The audio input to extract features from.
   * @returns {Promise<any>} A Promise that resolves with the extracted features.
   */
  async _call(audio) {
    return await this.feature_extractor(audio);
  }
};

// src/models/owlvit/processing_owlvit.js
var OwlViTProcessor = class extends Processor {
  static tokenizer_class = AutoTokenizer;
  static image_processor_class = AutoImageProcessor;
};

// src/models/phi3_v/processing_phi3_v.js
var IMAGE_TOKEN = "<|image|>";
var IMAGE_TOKEN_PATTERN = /<\|image_\d+\|>/g;
var Phi3VProcessor = class extends Processor {
  static image_processor_class = AutoImageProcessor;
  static tokenizer_class = AutoTokenizer;
  /**
   * 
   * @param {string|string[]} text 
   * @param {RawImage|RawImage[]} images 
   * @param  { { padding?: boolean, truncation?: boolean, num_crops?: number } | undefined } options
   * @returns {Promise<any>}
   */
  async _call(text, images = null, {
    padding = true,
    truncation = true,
    num_crops = null
  } = {}) {
    if (!Array.isArray(text)) {
      text = [text];
    }
    let text_inputs, image_inputs;
    if (images) {
      image_inputs = await this.image_processor(images, { num_crops });
      const { num_img_tokens } = image_inputs;
      const prompt_chunks = text.map((t, i) => t.split(IMAGE_TOKEN_PATTERN).join(IMAGE_TOKEN.repeat(num_img_tokens[i])));
      text_inputs = this.tokenizer(prompt_chunks, { padding, truncation });
      const image_token_id = this.tokenizer.model.convert_tokens_to_ids([IMAGE_TOKEN])[0];
      text_inputs.input_ids.map_((id) => id == image_token_id ? -id : id);
    } else {
      text_inputs = this.tokenizer(text);
    }
    return {
      ...text_inputs,
      ...image_inputs
    };
  }
};

// src/models/paligemma/processing_paligemma.js
var IMAGE_TOKEN2 = "<image>";
function build_string_from_input(prompt, bos_token, image_seq_len, image_token, num_images) {
  return `${image_token.repeat(image_seq_len * num_images)}${bos_token}${prompt}
`;
}
var PaliGemmaProcessor = class extends Processor {
  static tokenizer_class = AutoTokenizer;
  static image_processor_class = AutoImageProcessor;
  static uses_processor_config = false;
  /**
   * @typedef {import('../../utils/image.js').RawImage} RawImage
   */
  // `images` is required, `text` is optional
  async _call(images, text = null, kwargs = {}) {
    if (!text) {
      console.warn(
        "You are using PaliGemma without a text prefix. It will perform as a picture-captioning model."
      );
      text = "";
    }
    if (!Array.isArray(images)) {
      images = [images];
    }
    if (!Array.isArray(text)) {
      text = [text];
    }
    const bos_token = this.tokenizer.bos_token;
    const image_seq_length = this.image_processor.config.image_seq_length;
    let input_strings;
    if (text.some((t) => t.includes(IMAGE_TOKEN2))) {
      input_strings = text.map(
        (sample) => {
          const expanded_sample = sample.replaceAll(IMAGE_TOKEN2, IMAGE_TOKEN2.repeat(image_seq_length));
          const bos_rfind_index = expanded_sample.lastIndexOf(IMAGE_TOKEN2);
          const bos_index = bos_rfind_index === -1 ? 0 : bos_rfind_index + IMAGE_TOKEN2.length;
          return expanded_sample.slice(0, bos_index) + bos_token + expanded_sample.slice(bos_index) + "\n";
        }
      );
    } else {
      console.warn(
        "You are passing both `text` and `images` to `PaliGemmaProcessor`. The processor expects special image tokens in the text, as many tokens as there are images per each text. It is recommended to add `<image>` tokens in the very beginning of your text. For this call, we will infer how many images each text has and add special tokens."
      );
      input_strings = text.map(
        (sample) => build_string_from_input(
          sample,
          bos_token,
          image_seq_length,
          IMAGE_TOKEN2,
          images.length
        )
      );
    }
    const text_inputs = this.tokenizer(input_strings, kwargs);
    const image_inputs = await this.image_processor(images, kwargs);
    return {
      ...image_inputs,
      ...text_inputs
    };
  }
};

// src/models/pyannote/processing_pyannote.js
var PyAnnoteProcessor = class extends Processor {
  static feature_extractor_class = PyAnnoteFeatureExtractor;
  /**
   * Calls the feature_extractor function with the given audio input.
   * @param {any} audio The audio input to extract features from.
   * @returns {Promise<any>} A Promise that resolves with the extracted features.
   */
  async _call(audio) {
    return await this.feature_extractor(audio);
  }
  /** @type {PyAnnoteFeatureExtractor['post_process_speaker_diarization']} */
  post_process_speaker_diarization(...args) {
    return (
      /** @type {PyAnnoteFeatureExtractor} */
      this.feature_extractor.post_process_speaker_diarization(...args)
    );
  }
  get sampling_rate() {
    return this.feature_extractor.config.sampling_rate;
  }
};

// src/models/qwen2_vl/processing_qwen2_vl.js
var Qwen2VLProcessor = class extends Processor {
  static image_processor_class = AutoImageProcessor;
  static tokenizer_class = AutoTokenizer;
  /**
   * 
   * @param {string|string[]} text 
   * @param {RawImage|RawImage[]} images 
   * @param  {...any} args 
   * @returns {Promise<any>}
   */
  async _call(text, images = null, ...args) {
    if (!Array.isArray(text)) {
      text = [text];
    }
    let image_inputs, image_grid_thw;
    if (images) {
      image_inputs = await this.image_processor(images);
      image_grid_thw = image_inputs.image_grid_thw;
    }
    if (image_grid_thw) {
      let merge_length = this.image_processor.config.merge_size ** 2;
      let index = 0;
      const image_grid_thw_list = image_grid_thw.tolist();
      text = text.map((t) => {
        while (t.includes("<|image_pad|>")) {
          const prod = Number(image_grid_thw_list[index++].reduce((a, b) => a * b, 1n));
          t = t.replace("<|image_pad|>", "<|placeholder|>".repeat(Math.floor(prod / merge_length)));
        }
        return t.replaceAll("<|placeholder|>", "<|image_pad|>");
      });
    }
    const text_inputs = this.tokenizer(text);
    return {
      ...text_inputs,
      ...image_inputs
      // TODO: ...videos_inputs,
    };
  }
};

// src/models/sam/processing_sam.js
var SamProcessor = class extends Processor {
  static image_processor_class = AutoImageProcessor;
  async _call(...args) {
    return await this.image_processor(...args);
  }
  post_process_masks(...args) {
    return this.image_processor.post_process_masks(...args);
  }
  reshape_input_points(...args) {
    return this.image_processor.reshape_input_points(...args);
  }
};

// src/models/sam2/processing_sam2.js
var Sam2VideoProcessor = class extends SamProcessor {
};

// src/models/speecht5/processing_speecht5.js
var SpeechT5Processor = class extends Processor {
  static tokenizer_class = AutoTokenizer;
  static feature_extractor_class = AutoFeatureExtractor;
  /**
   * Calls the feature_extractor function with the given input.
   * @param {any} input The input to extract features from.
   * @returns {Promise<any>} A Promise that resolves with the extracted features.
   */
  async _call(input) {
    return await this.feature_extractor(input);
  }
};

// src/models/ultravox/processing_ultravox.js
var UltravoxProcessor = class extends Processor {
  static tokenizer_class = AutoTokenizer;
  static feature_extractor_class = AutoFeatureExtractor;
  static uses_processor_config = true;
  /**
   * @param {string} text The text input to process.
   * @param {Float32Array} audio The audio input to process.
   */
  async _call(text, audio = null, kwargs = {}) {
    if (Array.isArray(text)) {
      throw new Error("Batched inputs are not supported yet.");
    }
    let audio_inputs = {};
    if (audio) {
      const audio_len = audio.length;
      const { input_features } = await this.feature_extractor(audio, {
        ...kwargs,
        max_length: audio_len
      });
      const nb_encoder_frames = Math.round(audio_len / this.config.encoder_ds_factor + 1e-4);
      const audio_embed_frames = 1 + Math.ceil(nb_encoder_frames / this.config.stack_factor);
      audio_inputs["audio_token_len"] = [audio_embed_frames];
      audio_inputs["audio_values"] = input_features;
      const image_token = this.config.audio_placeholder;
      if (!text.includes(image_token)) {
        throw new Error(`The input text does not contain the image token ${image_token}.`);
      }
      text = text.replaceAll(image_token, image_token.repeat(audio_embed_frames));
    }
    const text_inputs = this.tokenizer(text, {
      add_special_tokens: false,
      ...kwargs
    });
    return {
      ...text_inputs,
      ...audio_inputs
    };
  }
};

// src/models/voxtral/processing_voxtral.js
var AUDIO_TOKEN = "[AUDIO]";
var BEGIN_AUDIO_TOKEN = "[BEGIN_AUDIO]";
var NUM_AUDIO_TOKENS = 375;
function chunk(audio, n_samples) {
  const chunks = [];
  for (let i = 0; i < audio.length; i += n_samples) {
    chunks.push(audio.subarray(i, Math.min(i + n_samples, audio.length)));
  }
  return chunks;
}
var VoxtralProcessor = class extends Processor {
  static tokenizer_class = AutoTokenizer;
  static feature_extractor_class = AutoFeatureExtractor;
  static uses_processor_config = false;
  /**
   * @param {string} text The text input to process.
   * @param {Float32Array|Float32Array[]} audio The audio input(s) to process.
   */
  async _call(text, audio = null, kwargs = {}) {
    if (Array.isArray(text)) {
      throw new Error("Batched inputs are not supported yet.");
    }
    const audio_inputs = {};
    if (audio) {
      if (!text.includes(AUDIO_TOKEN)) {
        throw new Error(`The input text does not contain the audio token ${AUDIO_TOKEN}.`);
      }
      if (!Array.isArray(audio)) {
        audio = [audio];
      }
      const text_parts = text.split(AUDIO_TOKEN);
      const num_audio_tokens = text_parts.length - 1;
      if (num_audio_tokens !== audio.length) {
        throw new Error(`The number of audio inputs (${audio.length}) does not match the number of audio tokens in the text (${num_audio_tokens}).`);
      }
      const n_samples = this.feature_extractor.config.n_samples;
      const audio_chunks = audio.map((a) => chunk(a, n_samples));
      const chunk_counts = audio_chunks.map((chunks) => chunks.length);
      const all_chunks = audio_chunks.flat();
      const features = (await Promise.all(
        all_chunks.map((audio_input) => this.feature_extractor(audio_input, kwargs))
      )).map((x) => x.input_features);
      audio_inputs["audio_values"] = features.length > 1 ? cat(features, 0) : features[0];
      let new_text = text_parts[0];
      for (let i = 0; i < chunk_counts.length; ++i) {
        new_text += BEGIN_AUDIO_TOKEN;
        for (let j = 0; j < chunk_counts[i]; ++j) {
          new_text += AUDIO_TOKEN.repeat(NUM_AUDIO_TOKENS);
        }
        new_text += text_parts[i + 1];
      }
      text = new_text;
    }
    const text_inputs = this.tokenizer(text, {
      add_special_tokens: false,
      ...kwargs
    });
    return {
      ...text_inputs,
      ...audio_inputs
    };
  }
};

// src/models/wav2vec2/processing_wav2vec2.js
var Wav2Vec2Processor = class extends Processor {
  static tokenizer_class = AutoTokenizer;
  static feature_extractor_class = AutoFeatureExtractor;
  /**
   * Calls the feature_extractor function with the given audio input.
   * @param {any} audio The audio input to extract features from.
   * @returns {Promise<any>} A Promise that resolves with the extracted features.
   */
  async _call(audio) {
    return await this.feature_extractor(audio);
  }
};

// src/models/wav2vec2_with_lm/processing_wav2vec2_with_lm.js
var Wav2Vec2ProcessorWithLM = class extends Processor {
  static tokenizer_class = AutoTokenizer;
  static feature_extractor_class = AutoFeatureExtractor;
  /**
   * Calls the feature_extractor function with the given audio input.
   * @param {any} audio The audio input to extract features from.
   * @returns {Promise<any>} A Promise that resolves with the extracted features.
   */
  async _call(audio) {
    return await this.feature_extractor(audio);
  }
};

// src/models/whisper/processing_whisper.js
var WhisperProcessor = class extends Processor {
  static tokenizer_class = AutoTokenizer;
  static feature_extractor_class = AutoFeatureExtractor;
  /**
   * Calls the feature_extractor function with the given audio input.
   * @param {any} audio The audio input to extract features from.
   * @returns {Promise<any>} A Promise that resolves with the extracted features.
   */
  async _call(audio) {
    return await this.feature_extractor(audio);
  }
};

// src/models/auto/processing_auto.js
var AutoProcessor = class {
  /** @type {typeof Processor.from_pretrained} */
  static async from_pretrained(pretrained_model_name_or_path, options = {}) {
    const preprocessorConfig = await getModelJSON(pretrained_model_name_or_path, IMAGE_PROCESSOR_NAME, true, options);
    const { image_processor_type, feature_extractor_type, processor_class } = preprocessorConfig;
    if (processor_class && processors_exports[processor_class]) {
      return processors_exports[processor_class].from_pretrained(pretrained_model_name_or_path, options);
    }
    if (!image_processor_type && !feature_extractor_type) {
      throw new Error("No `image_processor_type` or `feature_extractor_type` found in the config.");
    }
    const components = {};
    if (image_processor_type) {
      const image_processor_class = image_processors_exports[image_processor_type.replace(/Fast$/, "")];
      if (!image_processor_class) {
        throw new Error(`Unknown image_processor_type: '${image_processor_type}'.`);
      }
      components.image_processor = new image_processor_class(preprocessorConfig);
    }
    if (feature_extractor_type) {
      const image_processor_class = image_processors_exports[feature_extractor_type];
      if (image_processor_class) {
        components.image_processor = new image_processor_class(preprocessorConfig);
      } else {
        const feature_extractor_class = feature_extractors_exports[feature_extractor_type];
        if (!feature_extractor_class) {
          throw new Error(`Unknown feature_extractor_type: '${feature_extractor_type}'.`);
        }
        components.feature_extractor = new feature_extractor_class(preprocessorConfig);
      }
    }
    const config = {};
    return new Processor(config, components, null);
  }
};

// src/pipelines.js
async function prepareImages(images) {
  if (!Array.isArray(images)) {
    images = [images];
  }
  return await Promise.all(images.map((x) => RawImage.read(x)));
}
async function prepareAudios(audios, sampling_rate) {
  if (!Array.isArray(audios)) {
    audios = [audios];
  }
  return await Promise.all(audios.map((x) => {
    if (typeof x === "string" || x instanceof URL) {
      return read_audio(x, sampling_rate);
    } else if (x instanceof Float64Array) {
      return new Float32Array(x);
    }
    return x;
  }));
}
function get_bounding_box(box, asInteger) {
  if (asInteger) {
    box = box.map((x) => x | 0);
  }
  const [xmin, ymin, xmax, ymax] = box;
  return { xmin, ymin, xmax, ymax };
}
var Pipeline = class extends Callable {
  /**
   * Create a new Pipeline.
   * @param {Object} options An object containing the following properties:
   * @param {string} [options.task] The task of the pipeline. Useful for specifying subtasks.
   * @param {PreTrainedModel} [options.model] The model used by the pipeline.
   * @param {PreTrainedTokenizer} [options.tokenizer=null] The tokenizer used by the pipeline (if any).
   * @param {Processor} [options.processor=null] The processor used by the pipeline (if any).
   */
  constructor({ task, model, tokenizer = null, processor = null }) {
    super();
    this.task = task;
    this.model = model;
    this.tokenizer = tokenizer;
    this.processor = processor;
  }
  /** @type {DisposeType} */
  async dispose() {
    await this.model.dispose();
  }
};
var TextClassificationPipeline = class extends /** @type {new (options: TextPipelineConstructorArgs) => TextClassificationPipelineType} */
Pipeline {
  /**
   * Create a new TextClassificationPipeline.
   * @param {TextPipelineConstructorArgs} options An object used to instantiate the pipeline.
   */
  constructor(options) {
    super(options);
  }
  /** @type {TextClassificationPipelineCallback} */
  async _call(texts, {
    top_k = 1
  } = {}) {
    const model_inputs = this.tokenizer(texts, {
      padding: true,
      truncation: true
    });
    const outputs = await this.model(model_inputs);
    const function_to_apply = (
      // @ts-expect-error TS2339
      this.model.config.problem_type === "multi_label_classification" ? (batch) => batch.sigmoid() : (batch) => new Tensor2(
        "float32",
        softmax(batch.data),
        batch.dims
      )
    );
    const id2label = this.model.config.id2label;
    const toReturn = [];
    for (const batch of outputs.logits) {
      const output = function_to_apply(batch);
      const scores = await topk(output, top_k);
      const values = scores[0].tolist();
      const indices = scores[1].tolist();
      const vals = indices.map((x, i) => ({
        label: id2label ? id2label[x] : `LABEL_${x}`,
        score: values[i]
      }));
      if (top_k === 1) {
        toReturn.push(...vals);
      } else {
        toReturn.push(vals);
      }
    }
    return Array.isArray(texts) || top_k === 1 ? (
      /** @type {TextClassificationOutput} */
      toReturn
    ) : (
      /** @type {TextClassificationOutput[]} */
      toReturn[0]
    );
  }
};
var TokenClassificationPipeline = class extends /** @type {new (options: TextPipelineConstructorArgs) => TokenClassificationPipelineType} */
Pipeline {
  /**
   * Create a new TokenClassificationPipeline.
   * @param {TextPipelineConstructorArgs} options An object used to instantiate the pipeline.
   */
  constructor(options) {
    super(options);
  }
  /** @type {TokenClassificationPipelineCallback} */
  async _call(texts, {
    ignore_labels = ["O"]
  } = {}) {
    const isBatched = Array.isArray(texts);
    const model_inputs = this.tokenizer(isBatched ? texts : [texts], {
      padding: true,
      truncation: true
    });
    const outputs = await this.model(model_inputs);
    const logits = outputs.logits;
    const id2label = this.model.config.id2label;
    const toReturn = [];
    for (let i = 0; i < logits.dims[0]; ++i) {
      const ids = model_inputs.input_ids[i];
      const batch = logits[i];
      const tokens = [];
      for (let j = 0; j < batch.dims[0]; ++j) {
        const tokenData = batch[j];
        const topScoreIndex = max(tokenData.data)[1];
        const entity = id2label ? id2label[topScoreIndex] : `LABEL_${topScoreIndex}`;
        if (ignore_labels.includes(entity)) {
          continue;
        }
        const word = this.tokenizer.decode([ids[j].item()], { skip_special_tokens: true });
        if (word === "") {
          continue;
        }
        const scores = softmax(tokenData.data);
        tokens.push({
          entity,
          score: scores[topScoreIndex],
          index: j,
          word
          // TODO: Add support for start and end
          // start: null,
          // end: null,
        });
      }
      toReturn.push(tokens);
    }
    return isBatched ? toReturn : toReturn[0];
  }
};
var QuestionAnsweringPipeline = class extends /** @type {new (options: TextPipelineConstructorArgs) => QuestionAnsweringPipelineType} */
Pipeline {
  /**
   * Create a new QuestionAnsweringPipeline.
   * @param {TextPipelineConstructorArgs} options An object used to instantiate the pipeline.
   */
  constructor(options) {
    super(options);
  }
  /** @type {QuestionAnsweringPipelineCallback} */
  async _call(question, context, {
    top_k = 1
  } = {}) {
    const inputs = this.tokenizer(question, {
      text_pair: context,
      padding: true,
      truncation: true
    });
    const { start_logits, end_logits } = await this.model(inputs);
    const input_ids = inputs.input_ids.tolist();
    const attention_mask = inputs.attention_mask.tolist();
    const special_tokens = this.tokenizer.all_special_ids;
    const toReturn = [];
    for (let j = 0; j < start_logits.dims[0]; ++j) {
      const ids = input_ids[j];
      const sepIndex = ids.findIndex(
        (x) => (
          // We use == to match bigint with number
          // @ts-ignore
          x == this.tokenizer.sep_token_id
        )
      );
      const valid_mask = attention_mask[j].map((y, ix) => y == 1 && (ix === 0 || ix > sepIndex && special_tokens.findIndex((x) => x == ids[ix]) === -1));
      const start = start_logits[j].tolist();
      const end = end_logits[j].tolist();
      for (let i = 1; i < start.length; ++i) {
        if (attention_mask[j] == 0 || i <= sepIndex || special_tokens.findIndex((x) => x == ids[i]) !== -1) {
          start[i] = -Infinity;
          end[i] = -Infinity;
        }
      }
      const start_scores = softmax(start).map((x, i) => [x, i]);
      const end_scores = softmax(end).map((x, i) => [x, i]);
      start_scores[0][0] = 0;
      end_scores[0][0] = 0;
      const options = product(start_scores, end_scores).filter((x) => x[0][1] <= x[1][1]).map((x) => [x[0][1], x[1][1], x[0][0] * x[1][0]]).sort((a, b) => b[2] - a[2]);
      for (let k = 0; k < Math.min(options.length, top_k); ++k) {
        const [start2, end2, score] = options[k];
        const answer_tokens = ids.slice(start2, end2 + 1);
        const answer = this.tokenizer.decode(answer_tokens, {
          skip_special_tokens: true
        });
        toReturn.push({
          answer,
          score
        });
      }
    }
    return top_k === 1 ? toReturn[0] : toReturn;
  }
};
var FillMaskPipeline = class extends /** @type {new (options: TextPipelineConstructorArgs) => FillMaskPipelineType} */
Pipeline {
  /**
   * Create a new FillMaskPipeline.
   * @param {TextPipelineConstructorArgs} options An object used to instantiate the pipeline.
   */
  constructor(options) {
    super(options);
  }
  /** @type {FillMaskPipelineCallback} */
  async _call(texts, {
    top_k = 5
  } = {}) {
    const model_inputs = this.tokenizer(texts, {
      padding: true,
      truncation: true
    });
    const { logits } = await this.model(model_inputs);
    const toReturn = [];
    const input_ids = model_inputs.input_ids.tolist();
    for (let i = 0; i < input_ids.length; ++i) {
      const ids = input_ids[i];
      const mask_token_index = ids.findIndex(
        (x) => (
          // We use == to match bigint with number
          // @ts-ignore
          x == this.tokenizer.mask_token_id
        )
      );
      if (mask_token_index === -1) {
        throw Error(`Mask token (${this.tokenizer.mask_token}) not found in text.`);
      }
      const itemLogits = logits[i][mask_token_index];
      const scores = await topk(new Tensor2(
        "float32",
        softmax(itemLogits.data),
        itemLogits.dims
      ), top_k);
      const values = scores[0].tolist();
      const indices = scores[1].tolist();
      toReturn.push(indices.map((x, i2) => {
        const sequence = ids.slice();
        sequence[mask_token_index] = x;
        return {
          score: values[i2],
          token: Number(x),
          token_str: this.tokenizer.decode([x]),
          sequence: this.tokenizer.decode(sequence, { skip_special_tokens: true })
        };
      }));
    }
    return Array.isArray(texts) ? toReturn : toReturn[0];
  }
};
var Text2TextGenerationPipeline = class extends /** @type {new (options: TextPipelineConstructorArgs) => Text2TextGenerationPipelineType} */
Pipeline {
  /** @type {'generated_text'} */
  _key = "generated_text";
  /**
   * Create a new Text2TextGenerationPipeline.
   * @param {TextPipelineConstructorArgs} options An object used to instantiate the pipeline.
   */
  constructor(options) {
    super(options);
  }
  /** @type {Text2TextGenerationPipelineCallback} */
  async _call(texts, generate_kwargs = {}) {
    if (!Array.isArray(texts)) {
      texts = [texts];
    }
    if (this.model.config.prefix) {
      texts = texts.map((x) => this.model.config.prefix + x);
    }
    const task_specific_params = this.model.config.task_specific_params;
    if (task_specific_params && task_specific_params[this.task]) {
      if (task_specific_params[this.task].prefix) {
        texts = texts.map((x) => task_specific_params[this.task].prefix + x);
      }
    }
    const tokenizer = this.tokenizer;
    const tokenizer_options = {
      padding: true,
      truncation: true
    };
    let inputs;
    if (this instanceof TranslationPipeline && "_build_translation_inputs" in tokenizer) {
      inputs = tokenizer._build_translation_inputs(texts, tokenizer_options, generate_kwargs);
    } else {
      inputs = tokenizer(texts, tokenizer_options);
    }
    const outputTokenIds = await this.model.generate({ ...inputs, ...generate_kwargs });
    return tokenizer.batch_decode(
      /** @type {Tensor} */
      outputTokenIds,
      {
        skip_special_tokens: true
      }
    ).map((text) => ({ [this._key]: text }));
  }
};
var SummarizationPipeline = class extends /** @type {new (options: TextPipelineConstructorArgs) => SummarizationPipelineType} */
/** @type {any} */
Text2TextGenerationPipeline {
  /** @type {'summary_text'} */
  _key = "summary_text";
  /**
   * Create a new SummarizationPipeline.
   * @param {TextPipelineConstructorArgs} options An object used to instantiate the pipeline.
   */
  constructor(options) {
    super(options);
  }
};
var TranslationPipeline = class extends /** @type {new (options: TextPipelineConstructorArgs) => TranslationPipelineType} */
/** @type {any} */
Text2TextGenerationPipeline {
  /** @type {'translation_text'} */
  _key = "translation_text";
  /**
   * Create a new TranslationPipeline.
   * @param {TextPipelineConstructorArgs} options An object used to instantiate the pipeline.
   */
  constructor(options) {
    super(options);
  }
};
function isChat(x) {
  return Array.isArray(x) && x.every((x2) => "role" in x2 && "content" in x2);
}
var TextGenerationPipeline = class extends /** @type {new (options: TextPipelineConstructorArgs) => TextGenerationPipelineType} */
Pipeline {
  /**
   * Create a new TextGenerationPipeline.
   * @param {TextPipelineConstructorArgs} options An object used to instantiate the pipeline.
   */
  constructor(options) {
    super(options);
  }
  /** @type {TextGenerationPipelineCallback} */
  async _call(texts, generate_kwargs = {}) {
    let isBatched = false;
    let isChatInput = false;
    let add_special_tokens = generate_kwargs.add_special_tokens ?? (this.tokenizer.add_bos_token || this.tokenizer.add_eos_token) ?? false;
    let inputs;
    if (typeof texts === "string") {
      inputs = texts = [texts];
    } else if (Array.isArray(texts) && texts.every((x) => typeof x === "string")) {
      isBatched = true;
      inputs = /** @type {string[]} */
      texts;
    } else {
      if (isChat(texts)) {
        texts = [
          /** @type {Chat} */
          texts
        ];
      } else if (Array.isArray(texts) && texts.every(isChat)) {
        isBatched = true;
      } else {
        throw new Error("Input must be a string, an array of strings, a Chat, or an array of Chats");
      }
      isChatInput = true;
      inputs = /** @type {string[]} */
      /** @type {Chat[]} */
      texts.map(
        (x) => this.tokenizer.apply_chat_template(x, {
          tokenize: false,
          add_generation_prompt: true
        })
      );
      add_special_tokens = false;
    }
    const return_full_text = isChatInput ? false : generate_kwargs.return_full_text ?? true;
    this.tokenizer.padding_side = "left";
    const text_inputs = this.tokenizer(inputs, {
      add_special_tokens,
      padding: true,
      truncation: true
    });
    const outputTokenIds = (
      /** @type {Tensor} */
      await this.model.generate({
        ...text_inputs,
        ...generate_kwargs
      })
    );
    const decoded = this.tokenizer.batch_decode(outputTokenIds, {
      skip_special_tokens: true
    });
    let promptLengths;
    if (!return_full_text && text_inputs.input_ids.dims.at(-1) > 0) {
      promptLengths = this.tokenizer.batch_decode(text_inputs.input_ids, {
        skip_special_tokens: true
      }).map((x) => x.length);
    }
    const toReturn = Array.from({ length: texts.length }, (_) => []);
    for (let i = 0; i < decoded.length; ++i) {
      const textIndex = Math.floor(i / outputTokenIds.dims[0] * texts.length);
      if (promptLengths) {
        decoded[i] = decoded[i].slice(promptLengths[textIndex]);
      }
      toReturn[textIndex].push({
        generated_text: isChatInput ? [
          .../** @type {Chat[]} */
          texts[textIndex],
          { role: "assistant", content: decoded[i] }
        ] : decoded[i]
      });
    }
    return !isBatched && toReturn.length === 1 ? toReturn[0] : toReturn;
  }
};
var ZeroShotClassificationPipeline = class extends /** @type {new (options: TextPipelineConstructorArgs) => ZeroShotClassificationPipelineType} */
Pipeline {
  /**
   * Create a new ZeroShotClassificationPipeline.
   * @param {TextPipelineConstructorArgs} options An object used to instantiate the pipeline.
   */
  constructor(options) {
    super(options);
    this.label2id = Object.fromEntries(
      Object.entries(
        /** @type {any} */
        this.model.config.label2id
      ).map(
        ([k, v]) => [k.toLowerCase(), v]
      )
    );
    this.entailment_id = this.label2id["entailment"];
    if (this.entailment_id === void 0) {
      console.warn("Could not find 'entailment' in label2id mapping. Using 2 as entailment_id.");
      this.entailment_id = 2;
    }
    this.contradiction_id = this.label2id["contradiction"] ?? this.label2id["not_entailment"];
    if (this.contradiction_id === void 0) {
      console.warn("Could not find 'contradiction' in label2id mapping. Using 0 as contradiction_id.");
      this.contradiction_id = 0;
    }
  }
  /** @type {ZeroShotClassificationPipelineCallback} */
  async _call(texts, candidate_labels, {
    hypothesis_template = "This example is {}.",
    multi_label = false
  } = {}) {
    const isBatched = Array.isArray(texts);
    if (!isBatched) {
      texts = [
        /** @type {string} */
        texts
      ];
    }
    if (!Array.isArray(candidate_labels)) {
      candidate_labels = [candidate_labels];
    }
    const hypotheses = candidate_labels.map(
      (x) => hypothesis_template.replace("{}", x)
    );
    const softmaxEach = multi_label || candidate_labels.length === 1;
    const toReturn = [];
    for (const premise of texts) {
      const entails_logits = [];
      for (const hypothesis of hypotheses) {
        const inputs = this.tokenizer(premise, {
          text_pair: hypothesis,
          padding: true,
          truncation: true
        });
        const outputs = await this.model(inputs);
        if (softmaxEach) {
          entails_logits.push([
            outputs.logits.data[this.contradiction_id],
            outputs.logits.data[this.entailment_id]
          ]);
        } else {
          entails_logits.push(outputs.logits.data[this.entailment_id]);
        }
      }
      const scores = softmaxEach ? entails_logits.map((x) => softmax(x)[1]) : softmax(entails_logits);
      const scores_sorted = scores.map((x, i) => [x, i]).sort((a, b) => b[0] - a[0]);
      toReturn.push({
        sequence: premise,
        labels: scores_sorted.map((x) => candidate_labels[x[1]]),
        scores: scores_sorted.map((x) => x[0])
      });
    }
    return isBatched ? toReturn : toReturn[0];
  }
};
var FeatureExtractionPipeline = class extends /** @type {new (options: TextPipelineConstructorArgs) => FeatureExtractionPipelineType} */
Pipeline {
  /**
   * Create a new FeatureExtractionPipeline.
   * @param {TextPipelineConstructorArgs} options An object used to instantiate the pipeline.
   */
  constructor(options) {
    super(options);
  }
  /** @type {FeatureExtractionPipelineCallback} */
  async _call(texts, {
    pooling = (
      /** @type {'none'} */
      "none"
    ),
    normalize = false,
    quantize = false,
    precision = (
      /** @type {'binary'} */
      "binary"
    )
  } = {}) {
    const model_inputs = this.tokenizer(texts, {
      padding: true,
      truncation: true
    });
    const outputs = await this.model(model_inputs);
    let result = outputs.last_hidden_state ?? outputs.logits ?? outputs.token_embeddings;
    switch (pooling) {
      case "none":
        break;
      case "mean":
        result = mean_pooling(result, model_inputs.attention_mask);
        break;
      case "first_token":
      case "cls":
        result = result.slice(null, 0);
        break;
      case "last_token":
      case "eos":
        result = result.slice(null, -1);
        break;
      default:
        throw Error(`Pooling method '${pooling}' not supported.`);
    }
    if (normalize) {
      result = result.normalize(2, -1);
    }
    if (quantize) {
      result = quantize_embeddings(result, precision);
    }
    return result;
  }
};
var ImageFeatureExtractionPipeline = class extends /** @type {new (options: ImagePipelineConstructorArgs) => ImageFeatureExtractionPipelineType} */
Pipeline {
  /**
   * Create a new ImageFeatureExtractionPipeline.
   * @param {ImagePipelineConstructorArgs} options An object used to instantiate the pipeline.
   */
  constructor(options) {
    super(options);
  }
  /** @type {ImageFeatureExtractionPipelineCallback} */
  async _call(images, {
    pool = null
  } = {}) {
    const preparedImages = await prepareImages(images);
    const { pixel_values } = await this.processor(preparedImages);
    const outputs = await this.model({ pixel_values });
    let result;
    if (pool) {
      if (!("pooler_output" in outputs)) {
        throw Error(`No pooled output was returned. Make sure the model has a 'pooler' layer when using the 'pool' option.`);
      }
      result = outputs.pooler_output;
    } else {
      result = outputs.last_hidden_state ?? outputs.logits ?? outputs.image_embeds;
    }
    return result;
  }
};
var AudioClassificationPipeline = class extends /** @type {new (options: AudioPipelineConstructorArgs) => AudioClassificationPipelineType} */
Pipeline {
  /**
   * Create a new AudioClassificationPipeline.
   * @param {AudioPipelineConstructorArgs} options An object used to instantiate the pipeline.
   */
  constructor(options) {
    super(options);
  }
  /** @type {AudioClassificationPipelineCallback} */
  async _call(audio, {
    top_k = 5
  } = {}) {
    const sampling_rate = this.processor.feature_extractor.config.sampling_rate;
    const preparedAudios = await prepareAudios(audio, sampling_rate);
    const id2label = this.model.config.id2label;
    const toReturn = [];
    for (const aud of preparedAudios) {
      const inputs = await this.processor(aud);
      const output = await this.model(inputs);
      const logits = output.logits[0];
      const scores = await topk(new Tensor2(
        "float32",
        softmax(logits.data),
        logits.dims
      ), top_k);
      const values = scores[0].tolist();
      const indices = scores[1].tolist();
      const vals = indices.map((x, i) => ({
        label: (
          /** @type {string} */
          id2label ? id2label[x] : `LABEL_${x}`
        ),
        score: (
          /** @type {number} */
          values[i]
        )
      }));
      toReturn.push(vals);
    }
    ;
    return Array.isArray(audio) ? toReturn : toReturn[0];
  }
};
var ZeroShotAudioClassificationPipeline = class extends /** @type {new (options: TextAudioPipelineConstructorArgs) => ZeroShotAudioClassificationPipelineType} */
Pipeline {
  /**
   * Create a new ZeroShotAudioClassificationPipeline.
   * @param {TextAudioPipelineConstructorArgs} options An object used to instantiate the pipeline.
   */
  constructor(options) {
    super(options);
  }
  /** @type {ZeroShotAudioClassificationPipelineCallback} */
  async _call(audio, candidate_labels, {
    hypothesis_template = "This is a sound of {}."
  } = {}) {
    const single = !Array.isArray(audio);
    if (single) {
      audio = [
        /** @type {AudioInput} */
        audio
      ];
    }
    const texts = candidate_labels.map(
      (x) => hypothesis_template.replace("{}", x)
    );
    const text_inputs = this.tokenizer(texts, {
      padding: true,
      truncation: true
    });
    const sampling_rate = this.processor.feature_extractor.config.sampling_rate;
    const preparedAudios = await prepareAudios(audio, sampling_rate);
    const toReturn = [];
    for (const aud of preparedAudios) {
      const audio_inputs = await this.processor(aud);
      const output = await this.model({ ...text_inputs, ...audio_inputs });
      const probs = softmax(output.logits_per_audio.data);
      toReturn.push([...probs].map((x, i) => ({
        score: x,
        label: candidate_labels[i]
      })));
    }
    return single ? toReturn[0] : toReturn;
  }
};
var AutomaticSpeechRecognitionPipeline = class extends /** @type {new (options: TextAudioPipelineConstructorArgs) => AutomaticSpeechRecognitionPipelineType} */
Pipeline {
  /**
   * Create a new AutomaticSpeechRecognitionPipeline.
   * @param {TextAudioPipelineConstructorArgs} options An object used to instantiate the pipeline.
   */
  constructor(options) {
    super(options);
  }
  /** @type {AutomaticSpeechRecognitionPipelineCallback} */
  async _call(audio, kwargs = {}) {
    switch (this.model.config.model_type) {
      case "whisper":
      case "lite-whisper":
        return this._call_whisper(audio, kwargs);
      case "wav2vec2":
      case "wav2vec2-bert":
      case "unispeech":
      case "unispeech-sat":
      case "hubert":
      case "parakeet_ctc":
        return this._call_wav2vec2(audio, kwargs);
      case "moonshine":
        return this._call_moonshine(audio, kwargs);
      default:
        throw new Error(`AutomaticSpeechRecognitionPipeline does not support model type '${this.model.config.model_type}'.`);
    }
  }
  /**
   * @type {AutomaticSpeechRecognitionPipelineCallback}
   * @private
   */
  async _call_wav2vec2(audio, kwargs) {
    if (kwargs.language) {
      console.warn('`language` parameter is not yet supported for `wav2vec2` models, defaulting to "English".');
    }
    if (kwargs.task) {
      console.warn('`task` parameter is not yet supported for `wav2vec2` models, defaulting to "transcribe".');
    }
    const single = !Array.isArray(audio);
    if (single) {
      audio = [
        /** @type {AudioInput} */
        audio
      ];
    }
    const sampling_rate = this.processor.feature_extractor.config.sampling_rate;
    const preparedAudios = await prepareAudios(audio, sampling_rate);
    const toReturn = [];
    for (const aud of preparedAudios) {
      const inputs = await this.processor(aud);
      const output = await this.model(inputs);
      const logits = output.logits[0];
      const predicted_ids = [];
      for (const item of logits) {
        predicted_ids.push(max(item.data)[1]);
      }
      const predicted_sentences = this.tokenizer.decode(predicted_ids, { skip_special_tokens: true }).trim();
      toReturn.push({ text: predicted_sentences });
    }
    return single ? toReturn[0] : toReturn;
  }
  /**
   * @type {AutomaticSpeechRecognitionPipelineCallback}
   * @private
   */
  async _call_whisper(audio, kwargs) {
    const return_timestamps = kwargs.return_timestamps ?? false;
    const chunk_length_s = kwargs.chunk_length_s ?? 0;
    const force_full_sequences = kwargs.force_full_sequences ?? false;
    let stride_length_s = kwargs.stride_length_s ?? null;
    const generation_config = { ...kwargs };
    if (return_timestamps === "word") {
      generation_config["return_token_timestamps"] = true;
      generation_config["return_timestamps"] = false;
    }
    const single = !Array.isArray(audio);
    if (single) {
      audio = [
        /** @type {AudioInput} */
        audio
      ];
    }
    const time_precision = this.processor.feature_extractor.config.chunk_length / this.model.config.max_source_positions;
    const hop_length = this.processor.feature_extractor.config.hop_length;
    const sampling_rate = this.processor.feature_extractor.config.sampling_rate;
    const preparedAudios = await prepareAudios(audio, sampling_rate);
    const toReturn = [];
    for (const aud of preparedAudios) {
      let chunks = [];
      if (chunk_length_s > 0) {
        if (stride_length_s === null) {
          stride_length_s = chunk_length_s / 6;
        } else if (chunk_length_s <= stride_length_s) {
          throw Error("`chunk_length_s` must be larger than `stride_length_s`.");
        }
        const window2 = sampling_rate * chunk_length_s;
        const stride = sampling_rate * stride_length_s;
        const jump = window2 - 2 * stride;
        let offset = 0;
        while (true) {
          const offset_end = offset + window2;
          const subarr = aud.subarray(offset, offset_end);
          const feature = await this.processor(subarr);
          const is_first = offset === 0;
          const is_last = offset_end >= aud.length;
          chunks.push({
            stride: [
              subarr.length,
              is_first ? 0 : stride,
              is_last ? 0 : stride
            ],
            input_features: feature.input_features,
            is_last
          });
          if (is_last) break;
          offset += jump;
        }
      } else {
        chunks = [{
          stride: [aud.length, 0, 0],
          input_features: (await this.processor(aud)).input_features,
          is_last: true
        }];
      }
      for (const chunk2 of chunks) {
        generation_config.num_frames = Math.floor(chunk2.stride[0] / hop_length);
        const data = await this.model.generate({
          inputs: chunk2.input_features,
          ...generation_config
        });
        if (return_timestamps === "word") {
          chunk2.tokens = data.sequences.tolist()[0];
          chunk2.token_timestamps = data.token_timestamps.tolist()[0].map(
            (x) => round(x, 2)
          );
        } else {
          chunk2.tokens = /** @type {Tensor} */
          data[0].tolist();
        }
        chunk2.stride = chunk2.stride.map((x) => x / sampling_rate);
      }
      const [full_text, optional] = this.tokenizer._decode_asr(chunks, {
        time_precision,
        return_timestamps,
        force_full_sequences
      });
      toReturn.push({ text: full_text, ...optional });
    }
    return single ? toReturn[0] : toReturn;
  }
  /**
   * @type {AutomaticSpeechRecognitionPipelineCallback}
   * @private
   */
  async _call_moonshine(audio, kwargs) {
    const single = !Array.isArray(audio);
    if (single) {
      audio = [
        /** @type {AudioInput} */
        audio
      ];
    }
    const sampling_rate = this.processor.feature_extractor.config.sampling_rate;
    const preparedAudios = await prepareAudios(audio, sampling_rate);
    const toReturn = [];
    for (const aud of preparedAudios) {
      const inputs = await this.processor(aud);
      const max_new_tokens = Math.floor(aud.length / sampling_rate) * 6;
      const outputs = await this.model.generate({ max_new_tokens, ...kwargs, ...inputs });
      const text = this.processor.batch_decode(
        /** @type {Tensor} */
        outputs,
        { skip_special_tokens: true }
      )[0];
      toReturn.push({ text });
    }
    return single ? toReturn[0] : toReturn;
  }
};
var ImageToTextPipeline = class extends /** @type {new (options: TextImagePipelineConstructorArgs) => ImageToTextPipelineType} */
Pipeline {
  /**
   * Create a new ImageToTextPipeline.
   * @param {TextImagePipelineConstructorArgs} options An object used to instantiate the pipeline.
   */
  constructor(options) {
    super(options);
  }
  /** @type {ImageToTextPipelineCallback} */
  async _call(images, generate_kwargs = {}) {
    const isBatched = Array.isArray(images);
    const preparedImages = await prepareImages(images);
    const { pixel_values } = await this.processor(preparedImages);
    const toReturn = [];
    for (const batch of pixel_values) {
      batch.dims = [1, ...batch.dims];
      const output = await this.model.generate({ inputs: batch, ...generate_kwargs });
      const decoded = this.tokenizer.batch_decode(
        /** @type {Tensor} */
        output,
        {
          skip_special_tokens: true
        }
      ).map((x) => ({ generated_text: x.trim() }));
      toReturn.push(decoded);
    }
    return isBatched ? toReturn : toReturn[0];
  }
};
var ImageClassificationPipeline = class extends /** @type {new (options: ImagePipelineConstructorArgs) => ImageClassificationPipelineType} */
Pipeline {
  /**
   * Create a new ImageClassificationPipeline.
   * @param {ImagePipelineConstructorArgs} options An object used to instantiate the pipeline.
   */
  constructor(options) {
    super(options);
  }
  /** @type {ImageClassificationPipelineCallback} */
  async _call(images, {
    top_k = 5
  } = {}) {
    const preparedImages = await prepareImages(images);
    const { pixel_values } = await this.processor(preparedImages);
    const output = await this.model({ pixel_values });
    const id2label = this.model.config.id2label;
    const toReturn = [];
    for (const batch of output.logits) {
      const scores = await topk(new Tensor2(
        "float32",
        softmax(batch.data),
        batch.dims
      ), top_k);
      const values = scores[0].tolist();
      const indices = scores[1].tolist();
      const vals = indices.map((x, i) => ({
        label: (
          /** @type {string} */
          id2label ? id2label[x] : `LABEL_${x}`
        ),
        score: (
          /** @type {number} */
          values[i]
        )
      }));
      toReturn.push(vals);
    }
    return Array.isArray(images) ? toReturn : toReturn[0];
  }
};
var ImageSegmentationPipeline = class extends /** @type {new (options: ImagePipelineConstructorArgs) => ImageSegmentationPipelineType} */
Pipeline {
  /**
   * Create a new ImageSegmentationPipeline.
   * @param {ImagePipelineConstructorArgs} options An object used to instantiate the pipeline.
   */
  constructor(options) {
    super(options);
    this.subtasks_mapping = {
      // Mapping of subtasks to their corresponding post-processing function names.
      panoptic: "post_process_panoptic_segmentation",
      instance: "post_process_instance_segmentation",
      semantic: "post_process_semantic_segmentation"
    };
  }
  /** @type {ImageSegmentationPipelineCallback} */
  async _call(images, {
    threshold = 0.5,
    mask_threshold = 0.5,
    overlap_mask_area_threshold = 0.8,
    label_ids_to_fuse = null,
    target_sizes = null,
    subtask = null
  } = {}) {
    const isBatched = Array.isArray(images);
    if (isBatched && images.length !== 1) {
      throw Error("Image segmentation pipeline currently only supports a batch size of 1.");
    }
    const preparedImages = await prepareImages(images);
    const imageSizes = preparedImages.map((x) => [x.height, x.width]);
    const inputs = await this.processor(preparedImages);
    const { inputNames, outputNames } = this.model.sessions["model"];
    if (!inputNames.includes("pixel_values")) {
      if (inputNames.length !== 1) {
        throw Error(`Expected a single input name, but got ${inputNames.length} inputs: ${inputNames}.`);
      }
      const newName = inputNames[0];
      if (newName in inputs) {
        throw Error(`Input name ${newName} already exists in the inputs.`);
      }
      inputs[newName] = inputs.pixel_values;
    }
    const output = await this.model(inputs);
    let fn = null;
    if (subtask !== null) {
      fn = this.subtasks_mapping[subtask];
    } else if (this.processor.image_processor) {
      for (const [task, func] of Object.entries(this.subtasks_mapping)) {
        if (func in this.processor.image_processor) {
          fn = this.processor.image_processor[func].bind(this.processor.image_processor);
          subtask = task;
          break;
        }
      }
    }
    const id2label = this.model.config.id2label;
    const annotation = [];
    if (!subtask) {
      const epsilon = 1e-5;
      const result = output[outputNames[0]];
      for (let i = 0; i < imageSizes.length; ++i) {
        const size = imageSizes[i];
        const item = result[i];
        if (item.data.some((x) => x < -epsilon || x > 1 + epsilon)) {
          item.sigmoid_();
        }
        const mask = await RawImage.fromTensor(item.mul_(255).to("uint8")).resize(size[1], size[0]);
        annotation.push({
          label: null,
          score: null,
          mask
        });
      }
    } else if (subtask === "panoptic" || subtask === "instance") {
      const processed = fn(
        output,
        threshold,
        mask_threshold,
        overlap_mask_area_threshold,
        label_ids_to_fuse,
        target_sizes ?? imageSizes
        // TODO FIX?
      )[0];
      const segmentation = processed.segmentation;
      for (const segment of processed.segments_info) {
        const maskData = new Uint8ClampedArray(segmentation.data.length);
        for (let i = 0; i < segmentation.data.length; ++i) {
          if (segmentation.data[i] === segment.id) {
            maskData[i] = 255;
          }
        }
        const mask = new RawImage(maskData, segmentation.dims[1], segmentation.dims[0], 1);
        annotation.push({
          score: segment.score,
          label: id2label[segment.label_id],
          mask
        });
      }
    } else if (subtask === "semantic") {
      const { segmentation, labels } = fn(output, target_sizes ?? imageSizes)[0];
      for (const label of labels) {
        const maskData = new Uint8ClampedArray(segmentation.data.length);
        for (let i = 0; i < segmentation.data.length; ++i) {
          if (segmentation.data[i] === label) {
            maskData[i] = 255;
          }
        }
        const mask = new RawImage(maskData, segmentation.dims[1], segmentation.dims[0], 1);
        annotation.push({
          score: null,
          label: id2label[label],
          mask
        });
      }
    } else {
      throw Error(`Subtask ${subtask} not supported.`);
    }
    return annotation;
  }
};
var BackgroundRemovalPipeline = class extends /** @type {new (options: ImagePipelineConstructorArgs) => BackgroundRemovalPipelineType} */
/** @type {any} */
ImageSegmentationPipeline {
  /**
   * Create a new BackgroundRemovalPipeline.
   * @param {ImagePipelineConstructorArgs} options An object used to instantiate the pipeline.
   */
  constructor(options) {
    super(options);
  }
  /** @type {BackgroundRemovalPipelineCallback} */
  async _call(images, options = {}) {
    const isBatched = Array.isArray(images);
    if (isBatched && images.length !== 1) {
      throw Error("Background removal pipeline currently only supports a batch size of 1.");
    }
    const preparedImages = await prepareImages(images);
    const masks = await super._call(images, options);
    const result = preparedImages.map((img, i) => {
      const cloned = img.clone();
      cloned.putAlpha(masks[i].mask);
      return cloned;
    });
    return result;
  }
};
var ZeroShotImageClassificationPipeline = class extends /** @type {new (options: TextImagePipelineConstructorArgs) => ZeroShotImageClassificationPipelineType} */
Pipeline {
  /**
   * Create a new ZeroShotImageClassificationPipeline.
   * @param {TextImagePipelineConstructorArgs} options An object used to instantiate the pipeline.
   */
  constructor(options) {
    super(options);
  }
  /** @type {ZeroShotImageClassificationPipelineCallback} */
  async _call(images, candidate_labels, {
    hypothesis_template = "This is a photo of {}"
  } = {}) {
    const isBatched = Array.isArray(images);
    const preparedImages = await prepareImages(images);
    const texts = candidate_labels.map(
      (x) => hypothesis_template.replace("{}", x)
    );
    const text_inputs = this.tokenizer(texts, {
      padding: this.model.config.model_type === "siglip" ? "max_length" : true,
      truncation: true
    });
    const { pixel_values } = await this.processor(preparedImages);
    const output = await this.model({ ...text_inputs, pixel_values });
    const function_to_apply = this.model.config.model_type === "siglip" ? (batch) => batch.sigmoid().data : (batch) => softmax(batch.data);
    const toReturn = [];
    for (const batch of output.logits_per_image) {
      const probs = function_to_apply(batch);
      const result = [...probs].map((x, i) => ({
        score: x,
        label: candidate_labels[i]
      }));
      result.sort((a, b) => b.score - a.score);
      toReturn.push(result);
    }
    return isBatched ? toReturn : toReturn[0];
  }
};
var ObjectDetectionPipeline = class extends /** @type {new (options: ImagePipelineConstructorArgs) => ObjectDetectionPipelineType} */
Pipeline {
  /**
   * Create a new ObjectDetectionPipeline.
   * @param {ImagePipelineConstructorArgs} options An object used to instantiate the pipeline.
   */
  constructor(options) {
    super(options);
  }
  /** @type {ObjectDetectionPipelineCallback} */
  async _call(images, {
    threshold = 0.9,
    percentage = false
  } = {}) {
    const isBatched = Array.isArray(images);
    if (isBatched && images.length !== 1) {
      throw Error("Object detection pipeline currently only supports a batch size of 1.");
    }
    const preparedImages = await prepareImages(images);
    const imageSizes = percentage ? null : preparedImages.map((x) => [x.height, x.width]);
    const { pixel_values, pixel_mask } = await this.processor(preparedImages);
    const output = await this.model({ pixel_values, pixel_mask });
    const processed = this.processor.image_processor.post_process_object_detection(output, threshold, imageSizes);
    const id2label = this.model.config.id2label;
    const result = processed.map((batch) => batch.boxes.map((box, i) => ({
      score: batch.scores[i],
      label: id2label[batch.classes[i]],
      box: get_bounding_box(box, !percentage)
    })));
    return isBatched ? result : result[0];
  }
};
var ZeroShotObjectDetectionPipeline = class extends /** @type {new (options: TextImagePipelineConstructorArgs) => ZeroShotObjectDetectionPipelineType} */
Pipeline {
  /**
   * Create a new ZeroShotObjectDetectionPipeline.
   * @param {TextImagePipelineConstructorArgs} options An object used to instantiate the pipeline.
   */
  constructor(options) {
    super(options);
  }
  /** @type {ZeroShotObjectDetectionPipelineCallback} */
  async _call(images, candidate_labels, {
    threshold = 0.1,
    top_k = null,
    percentage = false
  } = {}) {
    const isBatched = Array.isArray(images);
    const preparedImages = await prepareImages(images);
    const text_inputs = this.tokenizer(candidate_labels, {
      padding: true,
      truncation: true
    });
    const model_inputs = await this.processor(preparedImages);
    const toReturn = [];
    for (let i = 0; i < preparedImages.length; ++i) {
      const image = preparedImages[i];
      const imageSize = percentage ? null : [[image.height, image.width]];
      const pixel_values = model_inputs.pixel_values[i].unsqueeze_(0);
      const output = await this.model({ ...text_inputs, pixel_values });
      let result;
      if ("post_process_grounded_object_detection" in this.processor) {
        const processed = this.processor.post_process_grounded_object_detection(
          output,
          text_inputs.input_ids,
          {
            // TODO: support separate threshold values
            box_threshold: threshold,
            text_threshold: threshold,
            target_sizes: imageSize
          }
        )[0];
        result = processed.boxes.map((box, i2) => ({
          score: processed.scores[i2],
          label: processed.labels[i2],
          box: get_bounding_box(box, !percentage)
        }));
      } else {
        const processed = this.processor.image_processor.post_process_object_detection(output, threshold, imageSize, true)[0];
        result = processed.boxes.map((box, i2) => ({
          score: processed.scores[i2],
          label: candidate_labels[processed.classes[i2]],
          box: get_bounding_box(box, !percentage)
        }));
      }
      result.sort((a, b) => b.score - a.score);
      if (top_k !== null) {
        result = result.slice(0, top_k);
      }
      toReturn.push(result);
    }
    return isBatched ? toReturn : toReturn[0];
  }
};
var DocumentQuestionAnsweringPipeline = class extends /** @type {new (options: TextImagePipelineConstructorArgs) => DocumentQuestionAnsweringPipelineType} */
Pipeline {
  /**
   * Create a new DocumentQuestionAnsweringPipeline.
   * @param {TextImagePipelineConstructorArgs} options An object used to instantiate the pipeline.
   */
  constructor(options) {
    super(options);
  }
  /** @type {DocumentQuestionAnsweringPipelineCallback} */
  async _call(image, question, generate_kwargs = {}) {
    const preparedImage = (await prepareImages(image))[0];
    const { pixel_values } = await this.processor(preparedImage);
    const task_prompt = `<s_docvqa><s_question>${question}</s_question><s_answer>`;
    const decoder_input_ids = this.tokenizer(task_prompt, {
      add_special_tokens: false,
      padding: true,
      truncation: true
    }).input_ids;
    const output = await this.model.generate({
      inputs: pixel_values,
      // @ts-expect-error TS2339
      max_length: this.model.config.decoder.max_position_embeddings,
      decoder_input_ids,
      ...generate_kwargs
    });
    const decoded = this.tokenizer.batch_decode(
      /** @type {Tensor} */
      output
    )[0];
    const match = decoded.match(/<s_answer>(.*?)<\/s_answer>/);
    let answer = null;
    if (match && match.length >= 2) {
      answer = match[1].trim();
    }
    return [{ answer }];
  }
};
var TextToAudioPipeline = class extends /** @type {new (options: TextToAudioPipelineConstructorArgs) => TextToAudioPipelineType} */
Pipeline {
  DEFAULT_VOCODER_ID = "Xenova/speecht5_hifigan";
  /**
   * Create a new TextToAudioPipeline.
   * @param {TextToAudioPipelineConstructorArgs} options An object used to instantiate the pipeline.
   */
  constructor(options) {
    super(options);
    this.vocoder = options.vocoder ?? null;
  }
  /** @type {TextToAudioPipelineCallback} */
  async _call(text_inputs, {
    speaker_embeddings = null
  } = {}) {
    if (this.processor) {
      return this._call_text_to_spectrogram(text_inputs, { speaker_embeddings });
    } else {
      return this._call_text_to_waveform(text_inputs);
    }
  }
  async _call_text_to_waveform(text_inputs) {
    const inputs = this.tokenizer(text_inputs, {
      padding: true,
      truncation: true
    });
    const { waveform } = await this.model(inputs);
    const sampling_rate = this.model.config.sampling_rate;
    return new RawAudio(
      waveform.data,
      sampling_rate
    );
  }
  async _call_text_to_spectrogram(text_inputs, { speaker_embeddings }) {
    if (!this.vocoder) {
      console.log("No vocoder specified, using default HifiGan vocoder.");
      this.vocoder = await AutoModel.from_pretrained(this.DEFAULT_VOCODER_ID, { dtype: "fp32" });
    }
    if (typeof speaker_embeddings === "string" || speaker_embeddings instanceof URL) {
      speaker_embeddings = new Float32Array(
        await (await fetch(speaker_embeddings)).arrayBuffer()
      );
    }
    if (speaker_embeddings instanceof Float32Array) {
      speaker_embeddings = new Tensor2(
        "float32",
        speaker_embeddings,
        [1, speaker_embeddings.length]
      );
    } else if (!(speaker_embeddings instanceof Tensor2)) {
      throw new Error("Speaker embeddings must be a `Tensor`, `Float32Array`, `string`, or `URL`.");
    }
    const { input_ids } = this.tokenizer(text_inputs, {
      padding: true,
      truncation: true
    });
    const { waveform } = await this.model.generate_speech(input_ids, speaker_embeddings, { vocoder: this.vocoder });
    const sampling_rate = this.processor.feature_extractor.config.sampling_rate;
    return new RawAudio(
      waveform.data,
      sampling_rate
    );
  }
};
var ImageToImagePipeline = class extends /** @type {new (options: ImagePipelineConstructorArgs) => ImageToImagePipelineType} */
Pipeline {
  /**
   * Create a new ImageToImagePipeline.
   * @param {ImagePipelineConstructorArgs} options An object used to instantiate the pipeline.
   */
  constructor(options) {
    super(options);
  }
  /** @type {ImageToImagePipelineCallback} */
  async _call(images) {
    const preparedImages = await prepareImages(images);
    const inputs = await this.processor(preparedImages);
    const outputs = await this.model(inputs);
    const toReturn = [];
    for (const batch of outputs.reconstruction) {
      const output = batch.squeeze().clamp_(0, 1).mul_(255).round_().to("uint8");
      toReturn.push(RawImage.fromTensor(output));
    }
    return toReturn.length > 1 ? toReturn : toReturn[0];
  }
};
var DepthEstimationPipeline = class extends /** @type {new (options: ImagePipelineConstructorArgs) => DepthEstimationPipelineType} */
Pipeline {
  /**
   * Create a new DepthEstimationPipeline.
   * @param {ImagePipelineConstructorArgs} options An object used to instantiate the pipeline.
   */
  constructor(options) {
    super(options);
  }
  /** @type {DepthEstimationPipelineCallback} */
  async _call(images) {
    const preparedImages = await prepareImages(images);
    const inputs = await this.processor(preparedImages);
    const { predicted_depth } = await this.model(inputs);
    const toReturn = [];
    for (let i = 0; i < preparedImages.length; ++i) {
      const batch = predicted_depth[i];
      const [height, width] = batch.dims.slice(-2);
      const [new_width, new_height] = preparedImages[i].size;
      const prediction = (await interpolate_4d(batch.view(1, 1, height, width), {
        size: [new_height, new_width],
        mode: "bilinear"
      })).view(new_height, new_width);
      const minval = (
        /** @type {number} */
        prediction.min().item()
      );
      const maxval = (
        /** @type {number} */
        prediction.max().item()
      );
      const formatted = prediction.sub(minval).div_(maxval - minval).mul_(255).to("uint8").unsqueeze(0);
      const depth = RawImage.fromTensor(formatted);
      toReturn.push({
        predicted_depth: prediction,
        depth
      });
    }
    return toReturn.length > 1 ? toReturn : toReturn[0];
  }
};
var SUPPORTED_TASKS = Object.freeze({
  "text-classification": {
    "tokenizer": AutoTokenizer,
    "pipeline": TextClassificationPipeline,
    "model": AutoModelForSequenceClassification,
    "default": {
      // TODO: replace with original
      // "model": "distilbert-base-uncased-finetuned-sst-2-english",
      "model": "Xenova/distilbert-base-uncased-finetuned-sst-2-english"
    },
    "type": "text"
  },
  "token-classification": {
    "tokenizer": AutoTokenizer,
    "pipeline": TokenClassificationPipeline,
    "model": AutoModelForTokenClassification,
    "default": {
      // TODO: replace with original
      // "model": "Davlan/bert-base-multilingual-cased-ner-hrl",
      "model": "Xenova/bert-base-multilingual-cased-ner-hrl"
    },
    "type": "text"
  },
  "question-answering": {
    "tokenizer": AutoTokenizer,
    "pipeline": QuestionAnsweringPipeline,
    "model": AutoModelForQuestionAnswering,
    "default": {
      // TODO: replace with original
      // "model": "distilbert-base-cased-distilled-squad",
      "model": "Xenova/distilbert-base-cased-distilled-squad"
    },
    "type": "text"
  },
  "fill-mask": {
    "tokenizer": AutoTokenizer,
    "pipeline": FillMaskPipeline,
    "model": AutoModelForMaskedLM,
    "default": {
      // TODO: replace with original
      // "model": "bert-base-uncased",
      "model": "Xenova/bert-base-uncased"
    },
    "type": "text"
  },
  "summarization": {
    "tokenizer": AutoTokenizer,
    "pipeline": SummarizationPipeline,
    "model": AutoModelForSeq2SeqLM,
    "default": {
      // TODO: replace with original
      // "model": "sshleifer/distilbart-cnn-6-6",
      "model": "Xenova/distilbart-cnn-6-6"
    },
    "type": "text"
  },
  "translation": {
    "tokenizer": AutoTokenizer,
    "pipeline": TranslationPipeline,
    "model": AutoModelForSeq2SeqLM,
    "default": {
      // TODO: replace with original
      // "model": "t5-small",
      "model": "Xenova/t5-small"
    },
    "type": "text"
  },
  "text2text-generation": {
    "tokenizer": AutoTokenizer,
    "pipeline": Text2TextGenerationPipeline,
    "model": AutoModelForSeq2SeqLM,
    "default": {
      // TODO: replace with original
      // "model": "google/flan-t5-small",
      "model": "Xenova/flan-t5-small"
    },
    "type": "text"
  },
  "text-generation": {
    "tokenizer": AutoTokenizer,
    "pipeline": TextGenerationPipeline,
    "model": AutoModelForCausalLM,
    "default": {
      // TODO: replace with original
      // "model": "gpt2",
      "model": "Xenova/gpt2"
    },
    "type": "text"
  },
  "zero-shot-classification": {
    "tokenizer": AutoTokenizer,
    "pipeline": ZeroShotClassificationPipeline,
    "model": AutoModelForSequenceClassification,
    "default": {
      // TODO: replace with original
      // "model": "typeform/distilbert-base-uncased-mnli",
      "model": "Xenova/distilbert-base-uncased-mnli"
    },
    "type": "text"
  },
  "audio-classification": {
    "pipeline": AudioClassificationPipeline,
    "model": AutoModelForAudioClassification,
    "processor": AutoProcessor,
    "default": {
      // TODO: replace with original
      // "model": "superb/wav2vec2-base-superb-ks",
      "model": "Xenova/wav2vec2-base-superb-ks"
    },
    "type": "audio"
  },
  "zero-shot-audio-classification": {
    "tokenizer": AutoTokenizer,
    "pipeline": ZeroShotAudioClassificationPipeline,
    "model": AutoModel,
    "processor": AutoProcessor,
    "default": {
      // TODO: replace with original
      // "model": "laion/clap-htsat-fused",
      "model": "Xenova/clap-htsat-unfused"
    },
    "type": "multimodal"
  },
  "automatic-speech-recognition": {
    "tokenizer": AutoTokenizer,
    "pipeline": AutomaticSpeechRecognitionPipeline,
    "model": [AutoModelForSpeechSeq2Seq, AutoModelForCTC],
    "processor": AutoProcessor,
    "default": {
      // TODO: replace with original
      // "model": "openai/whisper-tiny.en",
      "model": "Xenova/whisper-tiny.en"
    },
    "type": "multimodal"
  },
  "text-to-audio": {
    "tokenizer": AutoTokenizer,
    "pipeline": TextToAudioPipeline,
    "model": [AutoModelForTextToWaveform, AutoModelForTextToSpectrogram],
    "processor": [
      AutoProcessor,
      /* Some don't use a processor */
      null
    ],
    "default": {
      // TODO: replace with original
      // "model": "microsoft/speecht5_tts",
      "model": "Xenova/speecht5_tts"
    },
    "type": "text"
  },
  "image-to-text": {
    "tokenizer": AutoTokenizer,
    "pipeline": ImageToTextPipeline,
    "model": AutoModelForVision2Seq,
    "processor": AutoProcessor,
    "default": {
      // TODO: replace with original
      // "model": "nlpconnect/vit-gpt2-image-captioning",
      "model": "Xenova/vit-gpt2-image-captioning"
    },
    "type": "multimodal"
  },
  "image-classification": {
    // no tokenizer
    "pipeline": ImageClassificationPipeline,
    "model": AutoModelForImageClassification,
    "processor": AutoProcessor,
    "default": {
      // TODO: replace with original
      // "model": "google/vit-base-patch16-224",
      "model": "Xenova/vit-base-patch16-224"
    },
    "type": "multimodal"
  },
  "image-segmentation": {
    // no tokenizer
    "pipeline": ImageSegmentationPipeline,
    "model": [AutoModelForImageSegmentation, AutoModelForSemanticSegmentation, AutoModelForUniversalSegmentation],
    "processor": AutoProcessor,
    "default": {
      // TODO: replace with original
      // "model": "facebook/detr-resnet-50-panoptic",
      "model": "Xenova/detr-resnet-50-panoptic"
    },
    "type": "multimodal"
  },
  "background-removal": {
    // no tokenizer
    "pipeline": BackgroundRemovalPipeline,
    "model": [AutoModelForImageSegmentation, AutoModelForSemanticSegmentation, AutoModelForUniversalSegmentation],
    "processor": AutoProcessor,
    "default": {
      "model": "Xenova/modnet"
    },
    "type": "image"
  },
  "zero-shot-image-classification": {
    "tokenizer": AutoTokenizer,
    "pipeline": ZeroShotImageClassificationPipeline,
    "model": AutoModel,
    "processor": AutoProcessor,
    "default": {
      // TODO: replace with original
      // "model": "openai/clip-vit-base-patch32",
      "model": "Xenova/clip-vit-base-patch32"
    },
    "type": "multimodal"
  },
  "object-detection": {
    // no tokenizer
    "pipeline": ObjectDetectionPipeline,
    "model": AutoModelForObjectDetection,
    "processor": AutoProcessor,
    "default": {
      // TODO: replace with original
      // "model": "facebook/detr-resnet-50",
      "model": "Xenova/detr-resnet-50"
    },
    "type": "multimodal"
  },
  "zero-shot-object-detection": {
    "tokenizer": AutoTokenizer,
    "pipeline": ZeroShotObjectDetectionPipeline,
    "model": AutoModelForZeroShotObjectDetection,
    "processor": AutoProcessor,
    "default": {
      // TODO: replace with original
      // "model": "google/owlvit-base-patch32",
      "model": "Xenova/owlvit-base-patch32"
    },
    "type": "multimodal"
  },
  "document-question-answering": {
    "tokenizer": AutoTokenizer,
    "pipeline": DocumentQuestionAnsweringPipeline,
    "model": AutoModelForDocumentQuestionAnswering,
    "processor": AutoProcessor,
    "default": {
      // TODO: replace with original
      // "model": "naver-clova-ix/donut-base-finetuned-docvqa",
      "model": "Xenova/donut-base-finetuned-docvqa"
    },
    "type": "multimodal"
  },
  "image-to-image": {
    // no tokenizer
    "pipeline": ImageToImagePipeline,
    "model": AutoModelForImageToImage,
    "processor": AutoProcessor,
    "default": {
      // TODO: replace with original
      // "model": "caidas/swin2SR-classical-sr-x2-64",
      "model": "Xenova/swin2SR-classical-sr-x2-64"
    },
    "type": "image"
  },
  "depth-estimation": {
    // no tokenizer
    "pipeline": DepthEstimationPipeline,
    "model": AutoModelForDepthEstimation,
    "processor": AutoProcessor,
    "default": {
      // TODO: replace with original
      // "model": "Intel/dpt-large",
      "model": "Xenova/dpt-large"
    },
    "type": "image"
  },
  // This task serves as a useful interface for dealing with sentence-transformers (https://huggingface.co/sentence-transformers).
  "feature-extraction": {
    "tokenizer": AutoTokenizer,
    "pipeline": FeatureExtractionPipeline,
    "model": AutoModel,
    "default": {
      // TODO: replace with original
      // "model": "sentence-transformers/all-MiniLM-L6-v2",
      "model": "Xenova/all-MiniLM-L6-v2"
    },
    "type": "text"
  },
  "image-feature-extraction": {
    "processor": AutoProcessor,
    "pipeline": ImageFeatureExtractionPipeline,
    "model": [AutoModelForImageFeatureExtraction, AutoModel],
    "default": {
      // TODO: replace with original
      // "model": "google/vit-base-patch16-224",
      "model": "Xenova/vit-base-patch16-224-in21k"
    },
    "type": "image"
  }
});
var TASK_ALIASES = Object.freeze({
  "sentiment-analysis": "text-classification",
  "ner": "token-classification",
  // "vqa": "visual-question-answering", // TODO: Add
  "asr": "automatic-speech-recognition",
  "text-to-speech": "text-to-audio",
  // Add for backwards compatibility
  "embeddings": "feature-extraction"
});
async function pipeline(task, model = null, {
  progress_callback = null,
  config = null,
  cache_dir = null,
  local_files_only = false,
  revision = "main",
  device = null,
  dtype = null,
  subfolder = "onnx",
  use_external_data_format = null,
  model_file_name = null,
  session_options = {}
} = {}) {
  task = TASK_ALIASES[task] ?? task;
  const pipelineInfo = SUPPORTED_TASKS[task.split("_", 1)[0]];
  if (!pipelineInfo) {
    throw Error(`Unsupported pipeline: ${task}. Must be one of [${Object.keys(SUPPORTED_TASKS)}]`);
  }
  if (!model) {
    model = pipelineInfo.default.model;
    console.log(`No model specified. Using default model: "${model}".`);
  }
  const pretrainedOptions = {
    progress_callback,
    config,
    cache_dir,
    local_files_only,
    revision,
    device,
    dtype,
    subfolder,
    use_external_data_format,
    model_file_name,
    session_options
  };
  const classes = /* @__PURE__ */ new Map([
    ["tokenizer", pipelineInfo.tokenizer],
    ["model", pipelineInfo.model],
    ["processor", pipelineInfo.processor]
  ]);
  const results = await loadItems(classes, model, pretrainedOptions);
  results.task = task;
  dispatchCallback(progress_callback, {
    "status": "ready",
    "task": task,
    "model": model
  });
  const pipelineClass = pipelineInfo.pipeline;
  return new pipelineClass(results);
}
async function loadItems(mapping, model, pretrainedOptions) {
  const result = /* @__PURE__ */ Object.create(null);
  const promises = [];
  for (const [name, cls] of mapping.entries()) {
    if (!cls) continue;
    let promise;
    if (Array.isArray(cls)) {
      promise = new Promise(async (resolve, reject) => {
        let e;
        for (const c of cls) {
          if (c === null) {
            resolve(null);
            return;
          }
          try {
            resolve(await c.from_pretrained(model, pretrainedOptions));
            return;
          } catch (err) {
            if (err.message?.includes("Unsupported model type")) {
              e = err;
            } else if (err.message?.includes("Could not locate file")) {
              e = err;
            } else {
              reject(err);
              return;
            }
          }
        }
        reject(e);
      });
    } else {
      promise = cls.from_pretrained(model, pretrainedOptions);
    }
    result[name] = promise;
    promises.push(promise);
  }
  await Promise.all(promises);
  for (const [name, promise] of Object.entries(result)) {
    result[name] = await promise;
  }
  return result;
}

// src/utils/video.js
var RawVideoFrame = class {
  /**
   * @param {RawImage} image
   * @param {number} timestamp
   */
  constructor(image, timestamp) {
    this.image = image;
    this.timestamp = timestamp;
  }
};
var RawVideo = class {
  /**
   * @param {RawVideoFrame[]|RawImage[]} frames
   * @param {number} duration
   */
  constructor(frames, duration) {
    if (frames.length > 0 && frames[0] instanceof RawImage) {
      frames = frames.map((image, i) => new RawVideoFrame(image, (i + 1) / (frames.length + 1) * duration));
    }
    this.frames = /** @type {RawVideoFrame[]} */
    frames;
    this.duration = duration;
  }
  get width() {
    return this.frames[0].image.width;
  }
  get height() {
    return this.frames[0].image.height;
  }
  get fps() {
    return this.frames.length / this.duration;
  }
};
async function load_video(src, { num_frames = null, fps = null } = {}) {
  if (!apis.IS_BROWSER_ENV) {
    throw new Error("`load_video` is currently only supported in browser environments.");
  }
  if (num_frames == null && fps == null) {
    throw new Error("Either num_frames or fps must be provided.");
  }
  const frames = [];
  const video = document.createElement("video");
  video.crossOrigin = "anonymous";
  video.muted = true;
  if (typeof src === "string") {
    video.src = src;
  } else if (src instanceof Blob) {
    video.src = URL.createObjectURL(src);
  } else if (src instanceof HTMLVideoElement) {
    video.src = src.src;
  } else {
    throw new Error("Invalid URL or video element provided.");
  }
  await new Promise((resolve) => video.onloadedmetadata = resolve);
  if (video.seekable.start(0) === video.seekable.end(0)) {
    const response = await fetch(video.src);
    const blob = await response.blob();
    video.src = URL.createObjectURL(blob);
    await new Promise((resolve) => video.onloadedmetadata = resolve);
  }
  const duration = video.duration;
  let count2, step;
  if (num_frames != null) {
    count2 = num_frames;
    step = num_frames === 1 ? 0 : duration / (num_frames - 1);
  } else {
    step = 1 / fps;
    count2 = Math.floor(duration / step);
  }
  let sampleTimes = [];
  for (let i = 0; i < count2; ++i) {
    sampleTimes.push(num_frames === 1 ? duration / 2 : i * step);
  }
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  for (const t of sampleTimes) {
    video.currentTime = t;
    await new Promise((resolve) => {
      video.onseeked = resolve;
    });
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const frameData = new RawImage(imageData.data, canvas.width, canvas.height, 4);
    const frame = new RawVideoFrame(frameData, t);
    frames.push(frame);
  }
  video.remove();
  return new RawVideo(frames, duration);
}

// src/generation/streamers.js
var BaseStreamer = class {
  /**
   * Function that is called by `.generate()` to push new tokens
   * @param {bigint[][]} value 
   */
  put(value) {
    throw Error("Not implemented");
  }
  /**
   * Function that is called by `.generate()` to signal the end of generation
   */
  end() {
    throw Error("Not implemented");
  }
};
var stdout_write = apis.IS_PROCESS_AVAILABLE ? (x) => process.stdout.write(x) : (x) => console.log(x);
var TextStreamer = class extends BaseStreamer {
  /**
   * 
   * @param {import('../tokenizers.js').PreTrainedTokenizer} tokenizer
   * @param {Object} options
   * @param {boolean} [options.skip_prompt=false] Whether to skip the prompt tokens
   * @param {boolean} [options.skip_special_tokens=true] Whether to skip special tokens when decoding
   * @param {function(string): void} [options.callback_function=null] Function to call when a piece of text is ready to display
   * @param {function(bigint[]): void} [options.token_callback_function=null] Function to call when a new token is generated
   * @param {Object} [options.decode_kwargs={}] Additional keyword arguments to pass to the tokenizer's decode method
   */
  constructor(tokenizer, {
    skip_prompt = false,
    callback_function = null,
    token_callback_function = null,
    skip_special_tokens = true,
    decode_kwargs = {},
    ...kwargs
  } = {}) {
    super();
    this.tokenizer = tokenizer;
    this.skip_prompt = skip_prompt;
    this.callback_function = callback_function ?? stdout_write;
    this.token_callback_function = token_callback_function;
    this.decode_kwargs = { skip_special_tokens, ...decode_kwargs, ...kwargs };
    this.token_cache = [];
    this.print_len = 0;
    this.next_tokens_are_prompt = true;
  }
  /**
   * Receives tokens, decodes them, and prints them to stdout as soon as they form entire words.
   * @param {bigint[][]} value 
   */
  put(value) {
    if (value.length > 1) {
      throw Error("TextStreamer only supports batch size of 1");
    }
    const is_prompt = this.next_tokens_are_prompt;
    if (is_prompt) {
      this.next_tokens_are_prompt = false;
      if (this.skip_prompt) return;
    }
    const tokens = value[0];
    this.token_callback_function?.(tokens);
    this.token_cache = mergeArrays(this.token_cache, tokens);
    const text = this.tokenizer.decode(this.token_cache, this.decode_kwargs);
    let printable_text;
    if (is_prompt || text.endsWith("\n")) {
      printable_text = text.slice(this.print_len);
      this.token_cache = [];
      this.print_len = 0;
    } else if (text.length > 0 && is_chinese_char(text.charCodeAt(text.length - 1))) {
      printable_text = text.slice(this.print_len);
      this.print_len += printable_text.length;
    } else {
      printable_text = text.slice(this.print_len, text.lastIndexOf(" ") + 1);
      this.print_len += printable_text.length;
    }
    this.on_finalized_text(printable_text, false);
  }
  /**
   * Flushes any remaining cache and prints a newline to stdout.
   */
  end() {
    let printable_text;
    if (this.token_cache.length > 0) {
      const text = this.tokenizer.decode(this.token_cache, this.decode_kwargs);
      printable_text = text.slice(this.print_len);
      this.token_cache = [];
      this.print_len = 0;
    } else {
      printable_text = "";
    }
    this.next_tokens_are_prompt = true;
    this.on_finalized_text(printable_text, true);
  }
  /**
   * Prints the new text to stdout. If the stream is ending, also prints a newline.
   * @param {string} text 
   * @param {boolean} stream_end 
   */
  on_finalized_text(text, stream_end) {
    if (text.length > 0) {
      this.callback_function?.(text);
    }
    if (stream_end && this.callback_function === stdout_write && apis.IS_PROCESS_AVAILABLE) {
      this.callback_function?.("\n");
    }
  }
};
var WhisperTextStreamer = class extends TextStreamer {
  /**
   * @param {import('../tokenizers.js').WhisperTokenizer} tokenizer
   * @param {Object} options
   * @param {boolean} [options.skip_prompt=false] Whether to skip the prompt tokens
   * @param {function(string): void} [options.callback_function=null] Function to call when a piece of text is ready to display
   * @param {function(bigint[]): void} [options.token_callback_function=null] Function to call when a new token is generated
   * @param {function(number): void} [options.on_chunk_start=null] Function to call when a new chunk starts
   * @param {function(number): void} [options.on_chunk_end=null] Function to call when a chunk ends
   * @param {function(): void} [options.on_finalize=null] Function to call when the stream is finalized
   * @param {number} [options.time_precision=0.02] Precision of the timestamps
   * @param {boolean} [options.skip_special_tokens=true] Whether to skip special tokens when decoding
   * @param {Object} [options.decode_kwargs={}] Additional keyword arguments to pass to the tokenizer's decode method
   */
  constructor(tokenizer, {
    skip_prompt = false,
    callback_function = null,
    token_callback_function = null,
    on_chunk_start = null,
    on_chunk_end = null,
    on_finalize = null,
    time_precision = 0.02,
    skip_special_tokens = true,
    decode_kwargs = {}
  } = {}) {
    super(tokenizer, {
      skip_prompt,
      skip_special_tokens,
      callback_function,
      token_callback_function,
      decode_kwargs
    });
    this.timestamp_begin = tokenizer.timestamp_begin;
    this.on_chunk_start = on_chunk_start;
    this.on_chunk_end = on_chunk_end;
    this.on_finalize = on_finalize;
    this.time_precision = time_precision;
    this.waiting_for_timestamp = false;
  }
  /**
   * @param {bigint[][]} value 
   */
  put(value) {
    if (value.length > 1) {
      throw Error("WhisperTextStreamer only supports batch size of 1");
    }
    const tokens = value[0];
    if (tokens.length === 1) {
      const offset = Number(tokens[0]) - this.timestamp_begin;
      if (offset >= 0) {
        const time = offset * this.time_precision;
        if (this.waiting_for_timestamp) {
          this.on_chunk_end?.(time);
        } else {
          this.on_chunk_start?.(time);
        }
        this.waiting_for_timestamp = !this.waiting_for_timestamp;
        this.token_callback_function?.(tokens);
        return;
      }
    }
    return super.put(value);
  }
  end() {
    super.end();
    this.on_finalize?.();
  }
};
export {
  ASTFeatureExtractor,
  ASTForAudioClassification,
  ASTModel,
  ASTPreTrainedModel,
  AlbertForMaskedLM,
  AlbertForQuestionAnswering,
  AlbertForSequenceClassification,
  AlbertModel,
  AlbertPreTrainedModel,
  AlbertTokenizer,
  ArceeForCausalLM,
  ArceeModel,
  ArceePreTrainedModel,
  AudioClassificationPipeline,
  AutoConfig,
  AutoFeatureExtractor,
  AutoImageProcessor,
  AutoModel,
  AutoModelForAudioClassification,
  AutoModelForAudioFrameClassification,
  AutoModelForAudioTextToText,
  AutoModelForCTC,
  AutoModelForCausalLM,
  AutoModelForDepthEstimation,
  AutoModelForDocumentQuestionAnswering,
  AutoModelForImageClassification,
  AutoModelForImageFeatureExtraction,
  AutoModelForImageMatting,
  AutoModelForImageSegmentation,
  AutoModelForImageTextToText,
  AutoModelForImageToImage,
  AutoModelForMaskGeneration,
  AutoModelForMaskedLM,
  AutoModelForNormalEstimation,
  AutoModelForObjectDetection,
  AutoModelForPoseEstimation,
  AutoModelForQuestionAnswering,
  AutoModelForSemanticSegmentation,
  AutoModelForSeq2SeqLM,
  AutoModelForSequenceClassification,
  AutoModelForSpeechSeq2Seq,
  AutoModelForTextToSpectrogram,
  AutoModelForTextToWaveform,
  AutoModelForTokenClassification,
  AutoModelForUniversalSegmentation,
  AutoModelForVision2Seq,
  AutoModelForXVector,
  AutoModelForZeroShotObjectDetection,
  AutoProcessor,
  AutoTokenizer,
  AutomaticSpeechRecognitionPipeline,
  BackgroundRemovalPipeline,
  BartForConditionalGeneration,
  BartForSequenceClassification,
  BartModel,
  BartPretrainedModel,
  BartTokenizer,
  BaseModelOutput,
  BaseStreamer,
  BeitFeatureExtractor,
  BeitForImageClassification,
  BeitModel,
  BeitPreTrainedModel,
  BertForMaskedLM,
  BertForQuestionAnswering,
  BertForSequenceClassification,
  BertForTokenClassification,
  BertModel,
  BertPreTrainedModel,
  BertTokenizer,
  BitImageProcessor,
  BlenderbotForConditionalGeneration,
  BlenderbotModel,
  BlenderbotPreTrainedModel,
  BlenderbotSmallForConditionalGeneration,
  BlenderbotSmallModel,
  BlenderbotSmallPreTrainedModel,
  BlenderbotSmallTokenizer,
  BlenderbotTokenizer,
  BloomForCausalLM,
  BloomModel,
  BloomPreTrainedModel,
  BloomTokenizer,
  CLIPFeatureExtractor,
  CLIPImageProcessor,
  CLIPModel,
  CLIPPreTrainedModel,
  CLIPSegForImageSegmentation,
  CLIPSegModel,
  CLIPSegPreTrainedModel,
  CLIPTextModel,
  CLIPTextModelWithProjection,
  CLIPTokenizer,
  CLIPVisionModel,
  CLIPVisionModelWithProjection,
  CamembertForMaskedLM,
  CamembertForQuestionAnswering,
  CamembertForSequenceClassification,
  CamembertForTokenClassification,
  CamembertModel,
  CamembertPreTrainedModel,
  CamembertTokenizer,
  CausalLMOutput,
  CausalLMOutputWithPast,
  ChineseCLIPFeatureExtractor,
  ChineseCLIPModel,
  ChineseCLIPPreTrainedModel,
  ClapAudioModelWithProjection,
  ClapFeatureExtractor,
  ClapModel,
  ClapPreTrainedModel,
  ClapTextModelWithProjection,
  ClassifierFreeGuidanceLogitsProcessor,
  CodeGenForCausalLM,
  CodeGenModel,
  CodeGenPreTrainedModel,
  CodeGenTokenizer,
  CodeLlamaTokenizer,
  CohereForCausalLM,
  CohereModel,
  CoherePreTrainedModel,
  CohereTokenizer,
  ConvBertForMaskedLM,
  ConvBertForQuestionAnswering,
  ConvBertForSequenceClassification,
  ConvBertForTokenClassification,
  ConvBertModel,
  ConvBertPreTrainedModel,
  ConvBertTokenizer,
  ConvNextFeatureExtractor,
  ConvNextForImageClassification,
  ConvNextImageProcessor,
  ConvNextModel,
  ConvNextPreTrainedModel,
  ConvNextV2ForImageClassification,
  ConvNextV2Model,
  ConvNextV2PreTrainedModel,
  DFineForObjectDetection,
  DFineModel,
  DFinePreTrainedModel,
  DINOv3ConvNextModel,
  DINOv3ConvNextPreTrainedModel,
  DINOv3ViTImageProcessor,
  DINOv3ViTModel,
  DINOv3ViTPreTrainedModel,
  DPTFeatureExtractor,
  DPTForDepthEstimation,
  DPTImageProcessor,
  DPTModel,
  DPTPreTrainedModel,
  DacDecoderModel,
  DacDecoderOutput,
  DacEncoderModel,
  DacEncoderOutput,
  DacFeatureExtractor,
  DacModel,
  DacPreTrainedModel,
  DataTypeMap,
  DebertaForMaskedLM,
  DebertaForQuestionAnswering,
  DebertaForSequenceClassification,
  DebertaForTokenClassification,
  DebertaModel,
  DebertaPreTrainedModel,
  DebertaTokenizer,
  DebertaV2ForMaskedLM,
  DebertaV2ForQuestionAnswering,
  DebertaV2ForSequenceClassification,
  DebertaV2ForTokenClassification,
  DebertaV2Model,
  DebertaV2PreTrainedModel,
  DebertaV2Tokenizer,
  DecisionTransformerModel,
  DecisionTransformerPreTrainedModel,
  DeiTFeatureExtractor,
  DeiTForImageClassification,
  DeiTImageProcessor,
  DeiTModel,
  DeiTPreTrainedModel,
  DepthAnythingForDepthEstimation,
  DepthAnythingPreTrainedModel,
  DepthEstimationPipeline,
  DepthProForDepthEstimation,
  DepthProPreTrainedModel,
  DetrFeatureExtractor,
  DetrForObjectDetection,
  DetrForSegmentation,
  DetrImageProcessor,
  DetrModel,
  DetrObjectDetectionOutput,
  DetrPreTrainedModel,
  DetrSegmentationOutput,
  Dinov2ForImageClassification,
  Dinov2Model,
  Dinov2PreTrainedModel,
  Dinov2WithRegistersForImageClassification,
  Dinov2WithRegistersModel,
  Dinov2WithRegistersPreTrainedModel,
  DistilBertForMaskedLM,
  DistilBertForQuestionAnswering,
  DistilBertForSequenceClassification,
  DistilBertForTokenClassification,
  DistilBertModel,
  DistilBertPreTrainedModel,
  DistilBertTokenizer,
  DocumentQuestionAnsweringPipeline,
  DonutFeatureExtractor,
  DonutImageProcessor,
  DonutSwinModel,
  DonutSwinPreTrainedModel,
  EdgeTamModel,
  EdgeTamPreTrainedModel,
  EfficientNetForImageClassification,
  EfficientNetImageProcessor,
  EfficientNetModel,
  EfficientNetPreTrainedModel,
  ElectraForMaskedLM,
  ElectraForQuestionAnswering,
  ElectraForSequenceClassification,
  ElectraForTokenClassification,
  ElectraModel,
  ElectraPreTrainedModel,
  ElectraTokenizer,
  EncodecFeatureExtractor,
  EosTokenCriteria,
  Ernie4_5_ForCausalLM,
  Ernie4_5_Model,
  Ernie4_5_PretrainedModel,
  Ernie4_5_Tokenizer,
  EsmForMaskedLM,
  EsmForSequenceClassification,
  EsmForTokenClassification,
  EsmModel,
  EsmPreTrainedModel,
  EsmTokenizer,
  ExaoneForCausalLM,
  ExaoneModel,
  ExaonePreTrainedModel,
  FFT,
  FalconForCausalLM,
  FalconModel,
  FalconPreTrainedModel,
  FalconTokenizer,
  FastViTForImageClassification,
  FastViTModel,
  FastViTPreTrainedModel,
  FeatureExtractionPipeline,
  FeatureExtractor,
  FillMaskPipeline,
  Florence2ForConditionalGeneration,
  Florence2PreTrainedModel,
  Florence2Processor,
  ForcedBOSTokenLogitsProcessor,
  ForcedEOSTokenLogitsProcessor,
  GLPNFeatureExtractor,
  GLPNForDepthEstimation,
  GLPNModel,
  GLPNPreTrainedModel,
  GPT2LMHeadModel,
  GPT2Model,
  GPT2PreTrainedModel,
  GPT2Tokenizer,
  GPTBigCodeForCausalLM,
  GPTBigCodeModel,
  GPTBigCodePreTrainedModel,
  GPTJForCausalLM,
  GPTJModel,
  GPTJPreTrainedModel,
  GPTNeoForCausalLM,
  GPTNeoModel,
  GPTNeoPreTrainedModel,
  GPTNeoXForCausalLM,
  GPTNeoXModel,
  GPTNeoXPreTrainedModel,
  GPTNeoXTokenizer,
  Gemma2ForCausalLM,
  Gemma2Model,
  Gemma2PreTrainedModel,
  Gemma3ForCausalLM,
  Gemma3Model,
  Gemma3PreTrainedModel,
  Gemma3nAudioFeatureExtractor,
  Gemma3nForConditionalGeneration,
  Gemma3nPreTrainedModel,
  Gemma3nProcessor,
  GemmaForCausalLM,
  GemmaModel,
  GemmaPreTrainedModel,
  GemmaTokenizer,
  GlmForCausalLM,
  GlmModel,
  GlmPreTrainedModel,
  GraniteForCausalLM,
  GraniteModel,
  GraniteMoeHybridForCausalLM,
  GraniteMoeHybridModel,
  GraniteMoeHybridPreTrainedModel,
  GranitePreTrainedModel,
  Grok1Tokenizer,
  GroundingDinoForObjectDetection,
  GroundingDinoImageProcessor,
  GroundingDinoPreTrainedModel,
  GroundingDinoProcessor,
  GroupViTModel,
  GroupViTPreTrainedModel,
  HeliumForCausalLM,
  HeliumModel,
  HeliumPreTrainedModel,
  HerbertTokenizer,
  HieraForImageClassification,
  HieraModel,
  HieraPreTrainedModel,
  HubertForCTC,
  HubertForSequenceClassification,
  HubertModel,
  HubertPreTrainedModel,
  IJepaForImageClassification,
  IJepaModel,
  IJepaPreTrainedModel,
  Idefics3ForConditionalGeneration,
  Idefics3ImageProcessor,
  Idefics3PreTrainedModel,
  Idefics3Processor,
  ImageClassificationPipeline,
  ImageFeatureExtractionPipeline,
  ImageProcessor as ImageFeatureExtractor,
  ImageMattingOutput,
  ImageProcessor,
  ImageSegmentationPipeline,
  ImageToImagePipeline,
  ImageToTextPipeline,
  InterruptableStoppingCriteria,
  JAISLMHeadModel,
  JAISModel,
  JAISPreTrainedModel,
  JinaCLIPImageProcessor,
  JinaCLIPModel,
  JinaCLIPPreTrainedModel,
  JinaCLIPProcessor,
  JinaCLIPTextModel,
  JinaCLIPVisionModel,
  Lfm2ForCausalLM,
  Lfm2Model,
  Lfm2PreTrainedModel,
  LiteWhisperForConditionalGeneration,
  Llama4ForCausalLM,
  Llama4PreTrainedModel,
  LlamaForCausalLM,
  LlamaModel,
  LlamaPreTrainedModel,
  LlamaTokenizer,
  LlavaForConditionalGeneration,
  LlavaOnevisionForConditionalGeneration,
  LlavaOnevisionImageProcessor,
  LlavaPreTrainedModel,
  LlavaProcessor,
  LlavaQwen2ForCausalLM,
  LogitsProcessor,
  LogitsProcessorList,
  LogitsWarper,
  LongT5ForConditionalGeneration,
  LongT5Model,
  LongT5PreTrainedModel,
  M2M100ForConditionalGeneration,
  M2M100Model,
  M2M100PreTrainedModel,
  M2M100Tokenizer,
  MBart50Tokenizer,
  MBartForCausalLM,
  MBartForConditionalGeneration,
  MBartForSequenceClassification,
  MBartModel,
  MBartPreTrainedModel,
  MBartTokenizer,
  MPNetForMaskedLM,
  MPNetForQuestionAnswering,
  MPNetForSequenceClassification,
  MPNetForTokenClassification,
  MPNetModel,
  MPNetPreTrainedModel,
  MPNetTokenizer,
  MT5ForConditionalGeneration,
  MT5Model,
  MT5PreTrainedModel,
  MarianMTModel,
  MarianModel,
  MarianPreTrainedModel,
  MarianTokenizer,
  Mask2FormerImageProcessor,
  MaskFormerFeatureExtractor,
  MaskFormerForInstanceSegmentation,
  MaskFormerImageProcessor,
  MaskFormerModel,
  MaskFormerPreTrainedModel,
  MaskedLMOutput,
  MaxLengthCriteria,
  Metric3DForDepthEstimation,
  Metric3DPreTrainedModel,
  Metric3Dv2ForDepthEstimation,
  Metric3Dv2PreTrainedModel,
  MgpstrForSceneTextRecognition,
  MgpstrModelOutput,
  MgpstrPreTrainedModel,
  MgpstrProcessor,
  MgpstrTokenizer,
  MimiDecoderModel,
  MimiDecoderOutput,
  MimiEncoderModel,
  MimiEncoderOutput,
  MimiModel,
  MimiPreTrainedModel,
  MinLengthLogitsProcessor,
  MinNewTokensLengthLogitsProcessor,
  MistralForCausalLM,
  MistralModel,
  MistralPreTrainedModel,
  MobileBertForMaskedLM,
  MobileBertForQuestionAnswering,
  MobileBertForSequenceClassification,
  MobileBertModel,
  MobileBertPreTrainedModel,
  MobileBertTokenizer,
  MobileLLMForCausalLM,
  MobileLLMModel,
  MobileLLMPreTrainedModel,
  MobileNetV1FeatureExtractor,
  MobileNetV1ForImageClassification,
  MobileNetV1ForSemanticSegmentation,
  MobileNetV1ImageProcessor,
  MobileNetV1Model,
  MobileNetV1PreTrainedModel,
  MobileNetV2FeatureExtractor,
  MobileNetV2ForImageClassification,
  MobileNetV2ForSemanticSegmentation,
  MobileNetV2ImageProcessor,
  MobileNetV2Model,
  MobileNetV2PreTrainedModel,
  MobileNetV3FeatureExtractor,
  MobileNetV3ForImageClassification,
  MobileNetV3ForSemanticSegmentation,
  MobileNetV3ImageProcessor,
  MobileNetV3Model,
  MobileNetV3PreTrainedModel,
  MobileNetV4FeatureExtractor,
  MobileNetV4ForImageClassification,
  MobileNetV4ForSemanticSegmentation,
  MobileNetV4ImageProcessor,
  MobileNetV4Model,
  MobileNetV4PreTrainedModel,
  MobileViTFeatureExtractor,
  MobileViTForImageClassification,
  MobileViTImageProcessor,
  MobileViTModel,
  MobileViTPreTrainedModel,
  MobileViTV2ForImageClassification,
  MobileViTV2Model,
  MobileViTV2PreTrainedModel,
  ModelOutput,
  ModernBertDecoderForCausalLM,
  ModernBertDecoderModel,
  ModernBertDecoderPreTrainedModel,
  ModernBertForMaskedLM,
  ModernBertForSequenceClassification,
  ModernBertForTokenClassification,
  ModernBertModel,
  ModernBertPreTrainedModel,
  Moondream1ForConditionalGeneration,
  MoonshineFeatureExtractor,
  MoonshineForConditionalGeneration,
  MoonshineModel,
  MoonshinePreTrainedModel,
  MoonshineProcessor,
  MptForCausalLM,
  MptModel,
  MptPreTrainedModel,
  MultiModalityCausalLM,
  MultiModalityPreTrainedModel,
  MusicgenForCausalLM,
  MusicgenForConditionalGeneration,
  MusicgenModel,
  MusicgenPreTrainedModel,
  NanoChatForCausalLM,
  NanoChatModel,
  NanoChatPreTrainedModel,
  NeoBertForMaskedLM,
  NeoBertForQuestionAnswering,
  NeoBertForSequenceClassification,
  NeoBertForTokenClassification,
  NeoBertModel,
  NeoBertPreTrainedModel,
  NllbTokenizer,
  NoBadWordsLogitsProcessor,
  NoRepeatNGramLogitsProcessor,
  NomicBertModel,
  NomicBertPreTrainedModel,
  NougatImageProcessor,
  NougatTokenizer,
  OPTForCausalLM,
  OPTModel,
  OPTPreTrainedModel,
  ObjectDetectionPipeline,
  Olmo2ForCausalLM,
  Olmo2Model,
  Olmo2PreTrainedModel,
  OlmoForCausalLM,
  OlmoModel,
  OlmoPreTrainedModel,
  OpenELMForCausalLM,
  OpenELMModel,
  OpenELMPreTrainedModel,
  OwlViTFeatureExtractor,
  OwlViTForObjectDetection,
  OwlViTImageProcessor,
  OwlViTModel,
  OwlViTPreTrainedModel,
  OwlViTProcessor,
  Owlv2ForObjectDetection,
  Owlv2ImageProcessor,
  Owlv2Model,
  Owlv2PreTrainedModel,
  PaliGemmaForConditionalGeneration,
  PaliGemmaPreTrainedModel,
  PaliGemmaProcessor,
  ParakeetFeatureExtractor,
  ParakeetForCTC,
  ParakeetPreTrainedModel,
  PatchTSMixerForPrediction,
  PatchTSMixerModel,
  PatchTSMixerPreTrainedModel,
  PatchTSTForPrediction,
  PatchTSTModel,
  PatchTSTPreTrainedModel,
  Phi3ForCausalLM,
  Phi3Model,
  Phi3PreTrainedModel,
  Phi3VForCausalLM,
  Phi3VImageProcessor,
  Phi3VPreTrainedModel,
  Phi3VProcessor,
  PhiForCausalLM,
  PhiModel,
  PhiPreTrainedModel,
  Pipeline,
  PreTrainedModel,
  PreTrainedTokenizer,
  PretrainedConfig,
  PretrainedMixin,
  Processor,
  PvtForImageClassification,
  PvtImageProcessor,
  PvtModel,
  PvtPreTrainedModel,
  PyAnnoteFeatureExtractor,
  PyAnnoteForAudioFrameClassification,
  PyAnnoteModel,
  PyAnnotePreTrainedModel,
  PyAnnoteProcessor,
  QuestionAnsweringModelOutput,
  QuestionAnsweringPipeline,
  Qwen2ForCausalLM,
  Qwen2Model,
  Qwen2PreTrainedModel,
  Qwen2Tokenizer,
  Qwen2VLForConditionalGeneration,
  Qwen2VLImageProcessor,
  Qwen2VLPreTrainedModel,
  Qwen2VLProcessor,
  Qwen3ForCausalLM,
  Qwen3Model,
  Qwen3PreTrainedModel,
  RFDetrForObjectDetection,
  RFDetrModel,
  RFDetrObjectDetectionOutput,
  RFDetrPreTrainedModel,
  RTDetrForObjectDetection,
  RTDetrImageProcessor,
  RTDetrModel,
  RTDetrObjectDetectionOutput,
  RTDetrPreTrainedModel,
  RTDetrV2ForObjectDetection,
  RTDetrV2Model,
  RTDetrV2ObjectDetectionOutput,
  RTDetrV2PreTrainedModel,
  RawAudio,
  RawImage,
  RawVideo,
  RawVideoFrame,
  RepetitionPenaltyLogitsProcessor,
  ResNetForImageClassification,
  ResNetModel,
  ResNetPreTrainedModel,
  RoFormerForMaskedLM,
  RoFormerForQuestionAnswering,
  RoFormerForSequenceClassification,
  RoFormerForTokenClassification,
  RoFormerModel,
  RoFormerPreTrainedModel,
  RoFormerTokenizer,
  RobertaForMaskedLM,
  RobertaForQuestionAnswering,
  RobertaForSequenceClassification,
  RobertaForTokenClassification,
  RobertaModel,
  RobertaPreTrainedModel,
  RobertaTokenizer,
  SamImageProcessor as Sam2ImageProcessor,
  Sam2ImageSegmentationOutput,
  Sam2VideoProcessor,
  SamImageProcessor,
  SamImageSegmentationOutput,
  SamModel,
  SamPreTrainedModel,
  SamProcessor,
  SapiensForDepthEstimation,
  SapiensForNormalEstimation,
  SapiensForSemanticSegmentation,
  SapiensPreTrainedModel,
  SeamlessM4TFeatureExtractor,
  SegformerFeatureExtractor,
  SegformerForImageClassification,
  SegformerForSemanticSegmentation,
  SegformerImageProcessor,
  SegformerModel,
  SegformerPreTrainedModel,
  Seq2SeqLMOutput,
  SequenceClassifierOutput,
  SiglipImageProcessor,
  SiglipModel,
  SiglipPreTrainedModel,
  SiglipTextModel,
  SiglipTokenizer,
  SiglipVisionModel,
  SmolLM3ForCausalLM,
  SmolLM3Model,
  SmolLM3PreTrainedModel,
  SmolVLMForConditionalGeneration,
  Idefics3ImageProcessor as SmolVLMImageProcessor,
  Idefics3Processor as SmolVLMProcessor,
  SnacDecoderModel,
  SnacEncoderModel,
  SnacFeatureExtractor,
  SnacModel,
  SnacPreTrainedModel,
  SpeechT5FeatureExtractor,
  SpeechT5ForSpeechToText,
  SpeechT5ForTextToSpeech,
  SpeechT5HifiGan,
  SpeechT5Model,
  SpeechT5PreTrainedModel,
  SpeechT5Processor,
  SpeechT5Tokenizer,
  SqueezeBertForMaskedLM,
  SqueezeBertForQuestionAnswering,
  SqueezeBertForSequenceClassification,
  SqueezeBertModel,
  SqueezeBertPreTrainedModel,
  SqueezeBertTokenizer,
  StableLmForCausalLM,
  StableLmModel,
  StableLmPreTrainedModel,
  Starcoder2ForCausalLM,
  Starcoder2Model,
  Starcoder2PreTrainedModel,
  StoppingCriteria,
  StoppingCriteriaList,
  StyleTextToSpeech2Model,
  StyleTextToSpeech2PreTrainedModel,
  SummarizationPipeline,
  SuppressTokensAtBeginLogitsProcessor,
  Swin2SRForImageSuperResolution,
  Swin2SRImageProcessor,
  Swin2SRModel,
  Swin2SRPreTrainedModel,
  SwinForImageClassification,
  SwinForSemanticSegmentation,
  SwinModel,
  SwinPreTrainedModel,
  T5ForConditionalGeneration,
  T5Model,
  T5PreTrainedModel,
  T5Tokenizer,
  TableTransformerForObjectDetection,
  TableTransformerModel,
  TableTransformerObjectDetectionOutput,
  TableTransformerPreTrainedModel,
  TemperatureLogitsWarper,
  Tensor2 as Tensor,
  Text2TextGenerationPipeline,
  TextClassificationPipeline,
  TextGenerationPipeline,
  TextStreamer,
  TextToAudioPipeline,
  TokenClassificationPipeline,
  TokenClassifierOutput,
  TokenizerModel,
  TopKLogitsWarper,
  TopPLogitsWarper,
  TrOCRForCausalLM,
  TrOCRPreTrainedModel,
  TranslationPipeline,
  UltravoxModel,
  UltravoxPreTrainedModel,
  UltravoxProcessor,
  UniSpeechForCTC,
  UniSpeechForSequenceClassification,
  UniSpeechModel,
  UniSpeechPreTrainedModel,
  UniSpeechSatForAudioFrameClassification,
  UniSpeechSatForCTC,
  UniSpeechSatForSequenceClassification,
  UniSpeechSatModel,
  UniSpeechSatPreTrainedModel,
  VLChatProcessor,
  VLMImageProcessor,
  VaultGemmaForCausalLM,
  VaultGemmaModel,
  VaultGemmaPreTrainedModel,
  ViTFeatureExtractor,
  ViTForImageClassification,
  ViTImageProcessor,
  ViTMAEModel,
  ViTMAEPreTrainedModel,
  ViTMSNForImageClassification,
  ViTMSNModel,
  ViTMSNPreTrainedModel,
  ViTModel,
  ViTPreTrainedModel,
  VisionEncoderDecoderModel,
  VitMatteForImageMatting,
  VitMatteImageProcessor,
  VitMattePreTrainedModel,
  VitPoseForPoseEstimation,
  VitPoseImageProcessor,
  VitPosePreTrainedModel,
  VitsModel,
  VitsModelOutput,
  VitsPreTrainedModel,
  VitsTokenizer,
  VoxtralForConditionalGeneration,
  VoxtralProcessor,
  Wav2Vec2BertForCTC,
  Wav2Vec2BertForSequenceClassification,
  Wav2Vec2BertModel,
  Wav2Vec2BertPreTrainedModel,
  Wav2Vec2CTCTokenizer,
  Wav2Vec2FeatureExtractor,
  Wav2Vec2ForAudioFrameClassification,
  Wav2Vec2ForCTC,
  Wav2Vec2ForSequenceClassification,
  Wav2Vec2Model,
  Wav2Vec2PreTrainedModel,
  Wav2Vec2Processor,
  Wav2Vec2ProcessorWithLM,
  WavLMForAudioFrameClassification,
  WavLMForCTC,
  WavLMForSequenceClassification,
  WavLMForXVector,
  WavLMModel,
  WavLMPreTrainedModel,
  WeSpeakerFeatureExtractor,
  WeSpeakerResNetModel,
  WeSpeakerResNetPreTrainedModel,
  WhisperFeatureExtractor,
  WhisperForConditionalGeneration,
  WhisperModel,
  WhisperPreTrainedModel,
  WhisperProcessor,
  WhisperTextStreamer,
  WhisperTimeStampLogitsProcessor,
  WhisperTokenizer,
  XLMForQuestionAnswering,
  XLMForSequenceClassification,
  XLMForTokenClassification,
  XLMModel,
  XLMPreTrainedModel,
  XLMRobertaForMaskedLM,
  XLMRobertaForQuestionAnswering,
  XLMRobertaForSequenceClassification,
  XLMRobertaForTokenClassification,
  XLMRobertaModel,
  XLMRobertaPreTrainedModel,
  XLMRobertaTokenizer,
  XLMTokenizer,
  XLMWithLMHeadModel,
  XVectorOutput,
  YolosFeatureExtractor,
  YolosForObjectDetection,
  YolosImageProcessor,
  YolosModel,
  YolosObjectDetectionOutput,
  YolosPreTrainedModel,
  ZeroShotAudioClassificationPipeline,
  ZeroShotClassificationPipeline,
  ZeroShotImageClassificationPipeline,
  ZeroShotObjectDetectionPipeline,
  bankers_round,
  cat,
  cos_sim,
  dot,
  dynamic_time_warping,
  env,
  full,
  full_like,
  getCacheShapes,
  hamming,
  hanning,
  interpolate,
  interpolate_4d,
  interpolate_data,
  is_chinese_char,
  layer_norm,
  load_image,
  load_video,
  log_softmax,
  magnitude,
  matmul,
  max,
  mean,
  mean_pooling,
  medianFilter,
  mel_filter_bank,
  min,
  ones,
  ones_like,
  permute,
  permute_data,
  pipeline,
  quantize_embeddings,
  rand,
  read_audio,
  rfft,
  round,
  slice,
  softmax,
  spectrogram,
  stack,
  std_mean,
  topk,
  window_function,
  zeros,
  zeros_like
};
//# sourceMappingURL=transformers.web.js.map
