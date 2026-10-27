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

In Node.js, `env.remoteHost` defaults to `HF_ENDPOINT` when set, otherwise `https://huggingface.co/`. `env.hfToken` defaults to `HF_TOKEN`, falling back to `HF_ACCESS_TOKEN`. Both variables are read when they are needed, not when the library is imported, so they can be loaded afterwards (for example, with `dotenv`). Explicit global assignments replace those defaults, and an `options.env` value takes precedence for that loader. Tokens are sent only to the configured Hub origin (and the official Hugging Face Hub origins), never to arbitrary image or audio URLs. Browser environments do not send token authorization headers.

The `cache_dir` and `local_files_only` options keep working as before, and are applied after environment resolution:

- `local_files_only: true` prevents remote requests even if `allowRemoteModels` is `true`.
- `cache_dir` overrides both `options.env.cacheDir` and global `env.cacheDir` for that call.
