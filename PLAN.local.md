# Plan: `response_format` (JSON Schema constrained decoding) in Transformers.js

## Goal

Add OpenAI-compatible structured output support to Transformers.js, backed by
[llguidance](https://github.com/guidance-ai/llguidance) (Rust, compiled to WASM) for
grammar-constrained decoding. Guarantees schema-valid JSON output instead of
prompt-and-hope + post-hoc validation.

## Target API

```js
const pipe = await pipeline(
  "text-generation",
  "onnx-community/gemma-4-E2B-it-ONNX",
  {
    device: "webgpu",
    dtype: "q4f16",
  }
);

const schema = {
  type: "array",
  items: { type: "string" },
};

await pipe(messages, {
  max_new_tokens: 1024,
  response_format: { type: "json_schema", json_schema: schema },
});
```

- `response_format` is a new option accepted by the `pipeline()` call function (text-generation
  pipeline first, generalizable later).
- Mirrors OpenAI's `response_format` shape so it's a drop-in mental model for JS devs.
- Internally resolves to attaching a constrained `LogitsProcessor` to the `generate()` call.
- `type: "json_object"` (schema-less, valid-JSON-only) should be supported too as a cheap
  first milestone (no schema compilation needed, just JSON grammar).

## Why llguidance

Already covered in prior research — key points to keep in mind while implementing:

- No pre-computation/startup cost (unlike Outlines), fast enough to compute masks on-the-fly
  per step (~50μs on native CPU for a 128k vocab).
- Proven to run in a browser (merged into Chromium for `window.ai`'s `responseConstraint`),
  though Chromium's actual wiring code is closed-source — we can't copy it, only use the
  open-source llguidance engine itself.
- Tokenizer integration is a **one-time setup cost**, not a per-step cost: llguidance builds a
  `TokTrie` from the full vocab (as raw bytes) once; per-step calls are just
  `compute_mask()` → apply bitmask to logits → `commit_token()`. No decoding in the hot loop.

## Architecture overview

```
response_format (user input)
        │
        ▼
schema → llguidance grammar compiler (WASM)  ─┐
        │                                     │ (one-time per generate() call)
        ▼                                     │
tokenizer vocab → TokTrie (WASM)  ─────────────┘ (one-time per model load, cached)
        │
        ▼
SchemaConstrainedLogitsProcessor (new LogitsProcessor subclass)
        │  each decode step:
        │    mask = interpreter.compute_mask()
        │    logits[i] = -Infinity where mask bit is 0
        │    (after sampling) interpreter.commit_token(sampled_id)
        ▼
generate() loop (existing, unchanged) → LogitsProcessorList → sampler
```

## Phases

### Phase 0 — Confirm/extend the plug-in point

- Verify `generate()`'s public `logits_processor` option (appended after built-ins) is
  sufficient, or needs a small extension to support processors that need a `commit_token`-style
  callback *after* sampling (llguidance requires this — most existing `LogitsProcessor`s only
  hook pre-sampling).
- Likely need a small addition to the generation loop: an optional post-sample hook
  (e.g. `processor.onTokenSampled?.(tokenId)`), since `LogitsProcessorList` today assumes
  stateless-after-`_call` processors.
- File: `src/generation/logits_process.js`, `src/models/modeling_utils.js` (generate loop).

### Phase 1 — Spike: compile llguidance to WASM, measure feasibility

- Compile llguidance's `parser` crate to `wasm32-unknown-unknown` (or `wasm32-wasi` if easier,
  but prefer unknown-unknown for browser bundle compatibility). Check if a maintained
  wasm-bindgen target already exists upstream before building our own.
- Measure and report:
  - Bundle size (gzip) of the compiled `.wasm` — this is the #1 go/no-go metric for a
    browser-shipped library.
  - Per-step `compute_mask()` latency through the WASM boundary + JS marshaling (not just
    native Rust numbers) for a realistic vocab size (Gemma/Llama ~256k, ~128k).
  - One-time `TokTrie` build latency for a full model vocab.
- Decision gate: if bundle size or per-step latency is unacceptable, fall back to a
  JSON-Schema-only hand-rolled JS grammar (Jsonformer-style: deterministic structural tokens,
  masking only on leaf values) as a lighter-weight v1. Document the fallback trigger criteria
  before starting Phase 2 so this isn't an open-ended detour later.

### Phase 2 — Tokenizer → llguidance bridge (one-time per model)

- Build an adapter that extracts, once per model/tokenizer load:
  - `tokens: bytes[]` — every vocab entry as **raw UTF-8 bytes**, not the human-readable
    decoded form. For byte-level BPE (GPT-2/Llama-style, e.g. `Ġgazed`), this means resolving
    the byte-level encoding table to actual bytes, not just using `tokenizer.decode()`.
  - `eos_token_id`, `bos_token_id`, `special_token_ids` — already available on Transformers.js
    tokenizer objects, just need to be surfaced in the shape llguidance expects.
- Pass this to the WASM module once to construct the `LLTokenizer` / `TokTrie`.
- Cache the resulting handle per model instance (same caching pattern as `ModelRegistry`) —
  rebuilding the trie per `generate()` call is wasted work if the model doesn't change.
- File: new `src/generation/grammar/tokenizer_bridge.js` (or similar).

### Phase 3 — `SchemaConstrainedLogitsProcessor`

```js
class SchemaConstrainedLogitsProcessor extends LogitsProcessor {
  constructor(schema, llguidanceTokenizer) {
    super();
    // compile schema -> llguidance grammar (LLInterpreter), once per generate() call
  }

  _call(input_ids, logits) {
    const { mask } = this.interpreter.compute_mask();
    // apply mask (bitset, vocab_size/32 elements) to logits typed array
    return logits;
  }

  onTokenSampled(tokenId) {
    this.interpreter.commit_token(tokenId);
  }
}
```

- Needs efficient bitmask application against Transformers.js's logits representation
  (likely a `Tensor`/`Float32Array`) — avoid per-element JS loops if possible, use typed-array
  ops.
- Handle the "grammar reached a stop state" signal from llguidance to force/allow EOS.
- File: `src/generation/logits_process.js`.

### Phase 4 — Wire up `response_format` in the pipeline

- Add `response_format` to the text-generation pipeline's call options
  (`src/pipelines.js`, `TextGenerationPipeline`).
- Validation:
  - `type: "json_object"` → JSON-only grammar, no schema compilation.
  - `type: "json_schema"` → compile `json_schema` field via llguidance.
  - Unsupported schema features → throw a clear error before generation starts (mirror Chrome's
    `NotSupportedError` behavior — fail fast, not mid-generation).
- Translate into a `SchemaConstrainedLogitsProcessor` instance, append to the
  `logits_processor` list passed into `generate()`.
- Decide default behavior on schema-includes-context-window cost: unlike Chrome, we probably
  should **not** auto-inject the schema into the prompt by default (grammar constraint alone
  should be sufficient and cheaper on context budget) — but expose an opt-in flag if empirically
  output quality benefits from also showing the schema in-prompt (worth A/B testing once
  working).

### Phase 5 — Batch generation handling

- llguidance's `Constraint`/`LLInterpreter` is single-sequence/stateful.
- For Transformers.js's batched `generate()`, need one interpreter instance **per batch item**,
  each tracking its own grammar state and getting masked independently.
- Scope: v1 may explicitly restrict `response_format` to `batch_size === 1` and throw otherwise,
  documented as a known limitation, with batched support as a fast-follow.

### Phase 6 — Tests & docs

- Unit tests: schema compilation, mask correctness on known small vocabs (reuse llguidance's
  own JSON Schema Test Suite harness pattern if feasible).
- Integration test: run a small ONNX model end-to-end with a nested schema (object + array +
  enum + required fields), assert `JSON.parse()` succeeds and matches schema on every run
  (structural guarantee — should never flake).
- Perf test: measure generation-loop overhead with/without constrained decoding on WebGPU vs
  WASM backend.
- Docs: new guide page + `response_format` API reference, modeled on OpenAI's docs since the
  shape is intentionally familiar.

## Open risks (carry into implementation, don't resolve in the plan)

1. **WASM bundle size** — could be the single blocking issue; resolve in Phase 1 before
   committing further engineering time.
2. **Post-sample hook** — `LogitsProcessorList`/`generate()` today may not have a clean seam for
   `commit_token()`-style post-sampling callbacks; needs a small core API addition.
3. **Byte-level tokenizer edge cases** — getting the raw-bytes extraction wrong will silently
   produce incorrect masks (accepting/rejecting wrong tokens) rather than an obvious crash —
   needs solid test coverage, not just "it compiled."
4. **Batching** — v1 scoping to single-sequence generation is a real product limitation worth
   flagging in the RFC/issue up front, not discovering mid-implementation.
5. **WebGPU/WASM boundary latency** — mask computation happening off the main compute path
   (CPU, via WASM) needs to not stall the GPU forward pass; may need to run concurrently
   (per llguidance's own recommendation: run `compute_mask()` while logits are being computed).

## Suggested order of work for a first PR

1. Phase 1 spike (throwaway branch, just prove bundle size + latency are acceptable).
2. Phase 0 + Phase 2 (core plug-in point + tokenizer bridge) — no user-facing API yet.
3. Phase 3 (`SchemaConstrainedLogitsProcessor`) with a manual/internal test using `generate()`
   directly (no pipeline API yet).
4. Phase 4 (`response_format` on the pipeline) — first user-facing milestone.
5. Phase 5 (batching) and Phase 6 (tests/docs) as follow-ups, potentially separate PRs.