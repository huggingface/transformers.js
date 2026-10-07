# Use custom models

<include>
{
    "path": "../snippets/3_custom-usage.snippet"
}
</include>

## Resource-loading precedence

Global `env` values are defaults. A loader's `options.env` object can override only the session-scopable resource-loading fields: `allowRemoteModels`, `remoteHost`, `remotePathTemplate`, `allowLocalModels`, `localModelPath`, `fetch`, `hfToken`, and `cacheDir`. Cache backend selection, filesystem capability, logging, and backend settings remain global.

```javascript
import { pipeline } from '@huggingface/transformers';

// Two pipelines in the same process, each loading from its own host and cache directory.
const publicPipe = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
const privatePipe = await pipeline('feature-extraction', 'acme/internal-embeddings', {
    env: {
        remoteHost: 'https://hub.internal.example/',
        hfToken: process.env.ACME_HUB_TOKEN,
        cacheDir: '/var/cache/acme-models/',
    },
});
```

A pipeline exposes the overrides it was created with as `pipe.sessionEnv` and the effective, resolved configuration as `pipe.env`. `pipe.env` is resolved on access, so later changes to the global `env` are reflected unless the session overrides that field.

In Node.js, the initial `env.remoteHost` is `HF_ENDPOINT` when set, otherwise `https://huggingface.co/`. The initial `env.hfToken` uses `HF_TOKEN`, falling back to `HF_ACCESS_TOKEN`. Explicit global assignments replace those initial values, and an `options.env` value takes precedence for that loader. Tokens are sent only to the configured Hub origin (and the official Hugging Face Hub origins), never to arbitrary image or audio URLs. Browser environments do not send token authorization headers.

The deprecated per-call options are applied after environment resolution:

- `local_files_only: true` prevents remote requests even if `allowRemoteModels` is `true`. Prefer `options.env.allowRemoteModels: false` for session-scoped loading.
- `cache_dir` overrides both `options.env.cacheDir` and global `env.cacheDir` for that call. Prefer `options.env.cacheDir` for session-scoped caching, or global `env.cacheDir` for one application-wide location.
