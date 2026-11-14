# Use custom models

<include>
{
    "path": "../snippets/3_custom-usage.snippet"
}
</include>

## Resource-loading precedence

Global `env` values are defaults. A loader's `options.env` object can override only the session-scopable resource-loading fields: `allowRemoteModels`, `remoteHost`, `remotePathTemplate`, `allowLocalModels`, `localModelPath`, `fetch`, `hfToken`, and `cacheDir`. Cache backend selection, filesystem capability, logging, and backend settings remain global.

```javascript
import { pipeline } from "@huggingface/transformers";

// Two pipelines in the same process, each loading from its own host and cache directory.
const publicPipe = await pipeline(
  "feature-extraction",
  "Xenova/all-MiniLM-L6-v2",
);
const privatePipe = await pipeline(
  "feature-extraction",
  "acme/internal-embeddings",
  {
    env: {
      remoteHost: "https://hub.internal.example/",
      hfToken: process.env.ACME_HUB_TOKEN,
      cacheDir: "/var/cache/acme-models/",
    },
  },
);
```

Fields that a loader's `options.env` does not override are read from the global `env` when they are needed, so later changes to the global `env` still apply. Pipelines also use their `options.env` when they download image or audio inputs given as URLs.

In Node.js, `env.remoteHost` defaults to `HF_ENDPOINT` when set, otherwise `https://huggingface.co/`. `env.hfToken` defaults to `HF_TOKEN`, falling back to `HF_ACCESS_TOKEN`. Both variables are read when they are needed, not when the library is imported, so they can be loaded afterwards (for example, with `dotenv`). Explicit global assignments replace those defaults, and an `options.env` value takes precedence for that loader. Tokens read only from environment variables are sent exclusively to `https://huggingface.co` and `https://hf.co`; tokens explicitly assigned to `env.hfToken` or `options.env.hfToken` may also be sent to the resolved `remoteHost` origin. Browser environments do not send token authorization headers.

The `cache_dir` and `local_files_only` options keep working as before, and are applied after environment resolution:

- `local_files_only: true` prevents remote requests even if `allowRemoteModels` is `true`.
- `cache_dir` overrides both `options.env.cacheDir` and global `env.cacheDir` for that call.

## Filesystem cache namespaces

Session-scoped hosts and tokens do **not** partition the filesystem cache. Persisted files keep the existing model ID/revision/file layout, without automatic host or token namespaces. If two hosts serve different content under the same model ID and revision, use separate `options.env.cacheDir` values (or explicit `cache_dir` values):

```javascript
import { AutoTokenizer } from "@huggingface/transformers";

const modelId = "acme/shared-model";
const tokenizerA = await AutoTokenizer.from_pretrained(modelId, {
  revision: "main",
  env: {
    remoteHost: "https://hub-a.example/",
    cacheDir: "/var/cache/models/host-a/",
  },
});
const tokenizerB = await AutoTokenizer.from_pretrained(modelId, {
  revision: "main",
  env: {
    remoteHost: "https://hub-b.example/",
    cacheDir: "/var/cache/models/host-b/",
  },
});
```

Cache directories separate content namespaces, not credential authorization boundaries: loading a cached file does not reauthorize it with the current token. Rotating a token within the same content namespace should keep using the same directory. An explicit `cache_dir` still takes precedence over session and global `cacheDir` settings.

Filesystem cache writers stage downloads before acquiring a per-file mutation lock. Publication holds
the lock while renaming the payload and then atomically replacing its version marker; deletion holds
the same lock while checking the version and removing the payload and marker. This orders stale
deletion before a replacement, or makes it observe the replacement's new version and refuse deletion.
The guarantee requires all writers/deleters sharing the directory to use this protocol; older library
versions and external tools do not participate. Reads remain lock-free, including read-only/offline use.

Lock files are adjacent to version markers with an added `.lock` suffix. Contention waits at most one
second before failing; optional cleanup remains best-effort. Locks are not automatically stolen after
a process crash, since a paused process can still be alive. If a crash leaves a lock behind, remove it
only after confirming no process is mutating that entry. No payload/marker migration or custom-cache
locking interface is required.
