# Configuration

All global configuration lives on the `env` object exported from the package.
Mutating fields on `env` changes library-wide behavior at runtime.

```javascript
import { env, LogLevel } from "@huggingface/transformers";
```

## Quick examples

<!-- @generated:start id=examples:env -->
**Example:** Load models from your own server and disable remote downloads.

```javascript
import { env } from '@huggingface/transformers';
env.allowRemoteModels = false;
env.localModelPath = '/path/to/local/models/';
```

**Example:** Point the filesystem cache at a custom directory (Node.js).

```javascript
import { env } from '@huggingface/transformers';
env.cacheDir = '/path/to/cache/directory/';
```
<!-- @generated:end id=examples:env -->

## All options

<!-- @generated:start id=typedef:TransformersEnvironment -->
| Option | Type | Description |
|--------|------|-------------|
| `version` | `string` | This version of Transformers.js. |
| `backends` | `object` | Expose environment variables of different backends, allowing users to set these variables if they want to. |
| `logLevel` | `number` | The logging level. Use LogLevel enum values. Defaults to LogLevel.ERROR. |
| `allowRemoteModels` | `boolean` | Whether to allow loading of remote files, defaults to `true`. If set to `false`, it will have the same effect as setting `local_files_only=true` when loading pipelines, models, tokenizers, processors, etc. |
| `remoteHost` | `string` | Host URL to load models from. Defaults to the Hugging Face Hub. |
| `remotePathTemplate` | `string` | Path template to fill in and append to `remoteHost` when loading models. |
| `allowLocalModels` | `boolean` | Whether to allow loading of local files, defaults to `false` if running in-browser, and `true` otherwise. If set to `false`, it will skip the local file check and try to load the model from the remote host. |
| `localModelPath` | `string` | Path to load local models from. Defaults to `/models/`. |
| `useFS` | `boolean` | Whether to use the file system to load files. By default, it is `true` if available. |
| `useBrowserCache` | `boolean` | Whether to use Cache API to cache models. By default, it is `true` if available. |
| `useFSCache` | `boolean` | Whether to use the file system to cache files. By default, it is `true` if available. |
| `cacheDir` | `string\|null` | The directory to use for caching files with the file system. By default, it is `./.cache`. |
| `useCustomCache` | `boolean` | Whether to use a custom cache system (defined by `customCache`), defaults to `false`. |
| `customCache` | `CacheInterface\|null` | The custom cache to use. Defaults to `null`. Note: this must be an object which implements the `match` and `put` functions of the Web Cache API. For more information, see https://developer.mozilla.org/en-US/docs/Web/API/Cache. |
| `useWasmCache` | `boolean` | Whether to pre-load and cache WASM binaries and the WASM factory (.mjs) for ONNX Runtime. Defaults to `true` when cache is available. This can improve performance and enables offline usage by avoiding repeated downloads. |
| `cacheKey` | `string` | The cache key to use for storing models and WASM binaries. Defaults to 'transformers-cache'. |
| `experimental_useCrossOriginStorage` | `boolean` | Whether to use the Cross-Origin Storage API to cache model files across origins, allowing different sites to share the same cached model weights. Defaults to `false`. Requires the Cross-Origin Storage Chrome extension: https://chromewebstore.google.com/detail/cross-origin-storage/denpnpcgjgikjpoglpjefakmdcbmlgih. The `experimental_` prefix indicates that the underlying browser API is not yet standardised and may change or be removed without a major version bump. For more information, see https://github.com/WICG/cross-origin-storage. |
| `fetch` | `(input: string \| URL, init?: any) => Promise<any>` | The fetch function to use. Defaults to `fetch`. |
<!-- @generated:end id=typedef:TransformersEnvironment -->

## Log levels

Pass one of these values to `env.logLevel`:

| Value          | Numeric |
|----------------|---------|
| `LogLevel.DEBUG`   | `10` |
| `LogLevel.INFO`    | `20` |
| `LogLevel.WARNING` | `30` (default) |
| `LogLevel.ERROR`   | `40` |
| `LogLevel.NONE`    | `50` |

Higher numbers suppress more output. Setting `env.logLevel` also propagates to
ONNX Runtime, so you get matching verbosity from the inference backend.

## Common patterns

**Development (fast iteration with remote models):**

```javascript
env.allowRemoteModels = true;
env.useFSCache = true;
env.logLevel = LogLevel.INFO;
```

**Production Node.js server (air-gapped, local only):**

```javascript
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = "/opt/models";
```

**Browser app with a CDN mirror:**

```javascript
env.remoteHost = "https://cdn.example.com";
env.useBrowserCache = true;
```

## Custom fetch (private / gated models, retries, etc.)

Override `env.fetch` to inject auth headers, retry logic, or abort signals:

```javascript
env.fetch = (url, init) =>
  fetch(url, {
    ...init,
    headers: { ...init?.headers, Authorization: `Bearer ${process.env.HF_TOKEN}` },
  });
```

## Custom cache backends

```javascript
env.useCustomCache = true;
env.customCache = {
  async match(key) { /* return Response or undefined */ },
  async put(key, response) { /* persist */ },
};
```

The cache must implement the Web Cache API's `match` and `put` methods.

## Inspecting models before loading

`ModelRegistry` reports which files a model needs, whether they're cached
locally, which dtypes the model ships with, and can clear caches selectively.
Useful for pre-flight UI and disk management.

```javascript
import { ModelRegistry } from "@huggingface/transformers";

const task = "feature-extraction";
const modelId = "onnx-community/all-MiniLM-L6-v2-ONNX";

const cached = await ModelRegistry.is_pipeline_cached(task, modelId);
if (!cached) {
  const files = await ModelRegistry.get_pipeline_files(task, modelId);
  // ask the user to confirm before `pipeline(...)` downloads these.
}

const dtypes = await ModelRegistry.get_available_dtypes(modelId);
const dtype = ["q4", "q8", "fp16", "fp32"].find((d) => dtypes.includes(d));
const pipe = await pipeline(task, modelId, { dtype });
```

<!-- @generated:start id=class:ModelRegistry -->
Static class for cache and file management operations.

**Methods**

- `get_files(modelId, [options])` → `Promise<string[]>` — Get all files (model, tokenizer, processor) needed for a model.
- `get_pipeline_files(task, modelId, [options])` → `Promise<string[]>` — Get all files needed for a specific pipeline task.
- `get_model_files(modelId, [options])` → `Promise<string[]>` — Get model files needed for a specific model.
- `get_tokenizer_files(modelId)` → `Promise<string[]>` — Get tokenizer files needed for a specific model.
- `get_processor_files(modelId)` → `Promise<string[]>` — Get processor files needed for a specific model.
- `get_available_dtypes(modelId, [options])` → `Promise<string[]>` — Detects which quantization levels (dtypes) are available for a model by checking which ONNX files exist on the hub or locally.
- `is_cached(modelId, [options])` → `Promise<boolean>` — Quickly checks if a model is fully cached by verifying `config.json` is present, then confirming all required files are cached.
- `is_cached_files(modelId, [options])` → `Promise<CacheCheckResult>` — Checks if all files for a given model are already cached, with per-file detail.
- `is_pipeline_cached(task, modelId, [options])` → `Promise<boolean>` — Quickly checks if all files for a specific pipeline task are cached by verifying `config.json` is present, then confirming all required files are cached.
- `is_pipeline_cached_files(task, modelId, [options])` → `Promise<CacheCheckResult>` — Checks if all files for a specific pipeline task are already cached, with per-file detail.
- `get_file_metadata(path_or_repo_id, filename, [options])` → `Promise<{exists: boolean, size?: number, contentType?: string, fromCache?: boolean}>` — Get metadata for a specific file without downloading it.
- `clear_cache(modelId, [options])` → `Promise<CacheClearResult>` — Clears all cached files for a given model.
- `clear_pipeline_cache(task, modelId, [options])` → `Promise<CacheClearResult>` — Clears all cached files for a specific pipeline task.
<!-- @generated:end id=class:ModelRegistry -->

To reclaim disk space for a specific model or task:

```javascript
await ModelRegistry.clear_cache(modelId);
await ModelRegistry.clear_pipeline_cache(task, modelId);
```
