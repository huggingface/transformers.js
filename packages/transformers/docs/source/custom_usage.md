# Use custom models

<include>
{
    "path": "../snippets/3_custom-usage.snippet"
}
</include>

## Resource-loading precedence

Global `env` values are defaults. A loader's `options.env` object can override only the session-scopable resource-loading fields: `allowRemoteModels`, `remoteHost`, `remotePathTemplate`, `allowLocalModels`, `localModelPath`, `fetch`, and `hfToken`. Cache infrastructure, filesystem capability, logging, and backend settings remain global.

In Node.js, the initial `env.remoteHost` is `HF_ENDPOINT` when set, otherwise `https://huggingface.co/`. The initial `env.hfToken` uses `HF_TOKEN`, falling back to `HF_ACCESS_TOKEN`. Explicit global assignments replace those initial values, and an `options.env` value takes precedence for that loader. Tokens are sent only to the configured Hub origin (and the official Hugging Face Hub origins), never to arbitrary image or audio URLs. Browser environments do not send token authorization headers.

The deprecated per-call options are applied after environment resolution:

- `local_files_only: true` prevents remote requests even if `allowRemoteModels` is `true`. Prefer `options.env.allowRemoteModels: false` for session-scoped loading.
- `cache_dir` overrides global `env.cacheDir` for that call. Prefer global `env.cacheDir` when one application-wide cache location is sufficient.
