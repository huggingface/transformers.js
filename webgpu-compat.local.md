# WebGPU runtime integration contract

## Resolved Architecture

Transformers.js and the WebGPU runtime have agreed on:

- pull and plan session modes;
- terminal fast-path iteration with awaited iterator `return()`;
- leased CPU logits with synchronous idempotent release;
- prefill and decode token semantics;
- separate CPU and plan generation capabilities;
- bounded token-pipeline depth capabilities;
- batch-one, all-ones-mask V1 inputs;
- uint32 runtime token batches and Transformers.js-owned int64 sequences;
- Transformers.js-owned generation policy, callbacks, stopping, streaming, and finalization;
- runtime-owned KV cache and native inference state;
- optional generation-only models without `forward()`;
- load/session cancellation and defensive model-side session cleanup;
- native QAT dtype behavior: omitted/`auto` is supported and `q4f16` throws.

## Default Artifact Provider

Option A is accepted for the first adapter.

`Gemma4E2B.load()` may use the existing runtime `ResourceRootIo` when Transformers.js does not supply an `artifactProvider`.

Initial behavior:

- `revision`, injected fetch/auth, abort, weight progress, and runtime browser caching are mapped where supported;
- `local_files_only` is rejected before metadata or weight loading rather than ignored;
- an explicitly supplied `artifactProvider` takes precedence over runtime IO;
- unsupported cache-policy fields are rejected or documented, never silently treated as equivalent;
- exact Transformers.js cache identity, offline policy, and shared Hub caching are deferred to a separate ranged-provider project.

This does not block a later move to a Transformers.js-provided ranged implementation. The adapter's model-loading code consumes a small internal random-access source abstraction so runtime IO and a future Transformers.js provider share the same safetensors path.

## Provider Contract

```ts
interface InferenceArtifactProvider {
  readJson<T>(file: string, options?: { signal?: AbortSignal }): Promise<T>;

  openByteSource(
    file: string,
    options?: {
      signal?: AbortSignal;
      onProgress?: (event: ArtifactProgressEvent) => void;
    },
  ): Promise<RandomAccessByteSource>;
}

interface RandomAccessByteSource {
  /** May become defined after transport metadata arrives. */
  readonly size?: number;

  /** Read bytes in the half-open interval [begin, end). */
  read(
    begin: number,
    end: number,
    options?: { signal?: AbortSignal },
  ): Promise<Uint8Array>;

  /** Idempotently wait for pending reads, then release the source. */
  close(): Promise<void>;
}
```

### Range Convention

`read(begin, end)` is half-open: `[begin, end)`.

Both values must be non-negative safe integers and `end >= begin`. An empty range returns an owned zero-length `Uint8Array`.

This matches the runtime `ByteSource.readRange()` contract and safetensors offsets.

### Concurrency

Reads may execute concurrently and complete out of order.

The provider must not rely on one global mutable seek cursor. HTTP range requests naturally satisfy this. Filesystem implementations must use positional reads.

A provider may serialize internally when required by its backing store, but it must preserve each read's independent range and result.

### Returned Bytes

Every successful `read()` returns an owned `Uint8Array` whose contents remain valid and unchanged after:

- later reads;
- out-of-order completion of other reads;
- cache writes;
- `close()`.

The runtime may retain returned bytes until the consuming upload, decode, or transcode operation completes. Providers must not return a view into a reused scratch buffer.

### Closing with Pending Reads

`close()` waits for reads that were already started, then releases the source. It does not implicitly abort pending reads.

Once closing starts:

- new reads reject;
- existing reads settle normally or through their supplied abort signal;
- `close()` observes all settlements before resolving;
- read failures remain owned by the corresponding read promises and do not become unhandled rejections.

A provider may internally abort transport work during process-wide teardown, but `close()` still waits for that work to settle.

`close()` is idempotent. Concurrent calls share the same close operation. Calls after closure resolve without repeating resource release.

### Unknown Size

An unknown `size` is acceptable, including at open time.

Safetensors header parsing may read an initial probe and derive tensor ranges without total file length. When size is unknown:

- complete-file length validation is unavailable;
- progress totals may be unknown;
- size-dependent cache metadata remains unavailable until size is discovered.

If transport metadata later reveals the size, `size` may be a getter that changes once from `undefined` to a stable non-negative safe integer. Once defined, it must not change.

For ranges wholly inside the file, `read()` returns exactly `end - begin` bytes. When size is unknown and an initial probe extends past EOF, returning the available shorter prefix is acceptable. Arbitrary short reads for known-valid tensor ranges are errors.

### Validation and Limits

The runtime imposes no alignment requirement on `begin`, `end`, or range length.

Provider validation requirements:

- `begin` and `end` are safe integers;
- `begin >= 0`;
- `end >= begin`;
- when `size` is known, `end <= size`;
- a successful known-valid range contains exactly `end - begin` bytes.

There is no protocol-level maximum range size. The runtime weight planner chunks and coalesces reads under its own memory and concurrency policy. A provider with implementation limits rejects oversized requests clearly instead of truncating them.

HTTP-specific alignment, multipart behavior, and minimum request sizes do not leak into this interface.

### Node Filesystem Sources

The same interface supports Node filesystem-backed sources.

Node implementations use positional reads and normally expose a known file size. They follow the same requirements:

- half-open ranges;
- concurrent independent positional reads;
- owned returned arrays;
- per-read abort checks;
- idempotent close that drains pending reads.

## Adapter Precedence and Errors

The first `Gemma4E2B` adapter resolves artifacts in this order:

1. Use `options.artifactProvider` when supplied.
2. Otherwise reject `local_files_only: true` because runtime HTTP IO cannot guarantee Transformers.js local-only semantics.
3. Otherwise resolve the fixed `Gemma4E2B.modelId` and `revision` through runtime IO.
4. Map auth/fetch, abort, progress, and supported cache options explicitly.
5. Throw on every supplied option that would otherwise imply unsupported offline or cache behavior.

An artifact-provider failure is propagated as-is after source cleanup. The adapter does not retry through runtime IO after an explicit provider fails because that could violate local-only, auth, revision, or cache policy.

## Generation Capabilities

The loaded model exposes final device-dependent capabilities. The exported backend class may also expose advisory static capabilities.

```ts
interface GenerationCapabilitiesV1 {
  readonly sessionVersion: 1;
  readonly maxBatchSize: 1;

  readonly cpuModes: readonly ["greedy", "multinomial"];
  readonly planModes: readonly ["greedy"];
  readonly declarativePlans: readonly ["argmax"];

  readonly cpuLogits: true;
  readonly tokenPipeline: {
    readonly defaultDepth: 4;
    readonly maxDepth: 4;
  };

  readonly customJavaScriptLogitsProcessors: "cpu-fallback";
  readonly customJavaScriptStoppingCriteria: true;
  readonly streamers: true;

  readonly cacheReorder: false;
  readonly cacheExpand: false;
  readonly returnScores: "cpu-fallback";
  readonly returnLogits: "cpu-fallback";
  readonly returnAttentions: false;
  readonly returnHiddenStates: false;
}
```

## Generation Session Contract

```ts
interface TokenBatch {
  readonly data: Uint32Array;
  readonly shape: readonly [batch: 1, sequenceLength: number];
}

interface PrefillInputs {
  readonly inputIds: TokenBatch;
  readonly attentionMask?: {
    readonly data: Uint8Array;
    readonly shape: readonly [batch: 1, sequenceLength: number];
  };
  readonly signal?: AbortSignal;
}

interface DecodeInputs {
  readonly tokenIds: {
    readonly data: Uint32Array;
    readonly shape: readonly [batch: 1, one: 1];
  };
  readonly signal?: AbortSignal;
}

interface LogitsLeaseV1 {
  readonly version: 1;
  readonly dtype: "float32";
  readonly shape: readonly [batch: 1, vocabularySize: number];

  /** Exactly one call; caller owns the returned array. */
  read(): Promise<Float32Array>;

  /** Synchronous, idempotent borrow release. */
  release(): void;
}

interface RuntimeGenerationPlanV1 {
  readonly version: 1;
  readonly processors: readonly [];
  readonly sampler: { readonly op: "argmax" };
  readonly maxNewTokens: number;
  readonly pipelineDepth?: number;
}

interface RuntimeTokenDecision {
  /** Owned, retainable host copy. */
  readonly tokenIds: Uint32Array;
}

interface AutoregressiveSessionV1 {
  readonly version: 1;
  readonly batchSize: 1;
  readonly maxSequenceLength: number;
  readonly consumedTokens: number;

  prefill(inputs: PrefillInputs): Promise<LogitsLeaseV1>;
  decode(inputs: DecodeInputs): Promise<LogitsLeaseV1>;

  /**
   * Alternative terminal execution mode. Performs prefill internally.
   * Iterator completion or return makes the session terminal.
   */
  generateWithPlan(
    inputs: PrefillInputs,
    plan: RuntimeGenerationPlanV1,
  ): AsyncIterable<RuntimeTokenDecision>;

  dispose(): Promise<void>;
}
```

Runtime validation rejects concurrent operations, pull/plan mode mixing, decode before prefill, decode with an active lease, repeated lease reads, and operations after terminal state.

## Fast-Path Lifecycle

`generateWithPlan()` performs prefill internally and is an alternative to pull mode.

Each yield represents exactly one selected token in logical generation order. Transformers.js may stop after any yielded token. Breaking iteration invokes and awaits iterator `return()`.

After `return()` resolves:

- no new GPU work is submitted;
- every submitted result has an attached rejection observer;
- all submitted result promises have settled;
- mapped and staging buffers are released or destroyed;
- the session is terminal;
- `session.dispose()` completes deterministically and idempotently.

Speculative KV writes need no rollback because the terminal session is disposed rather than reused. The runtime owns decisions and staging resources that were never yielded. Yielded `tokenIds` are owned host copies and need no release.

## Pull-Mode Lifecycle

`prefill()` consumes the complete prompt and returns last-position logits for the first token decision. It does not sample.

`decode()` consumes exactly the selected token from the preceding logits and returns logits for the following decision. EOS is committed and streamed by Transformers.js but is not decoded after it terminates generation.

The runtime owns cache position, KV writes, RoPE state, causal and sliding masks, and architecture-specific positions. Its consumed-token count is authoritative for native inference state.

Logits remain unchanged until lease release. No overwriting session operation begins while a lease is active. `read()` may be called exactly once and returns an owned row-major `Float32Array`. Transformers.js waits for `read()` to settle and then calls `release()` in `finally` before the next decode operation.

## Cancellation and Disposal

Session creation, prefill, and decode accept `AbortSignal`.

Before submission, abort rejects immediately with `signal.reason`. After submission, the session becomes terminal, stops subsequent submissions, drains submitted GPU and readback work, and rejects at a cleanup-safe boundary.

Device loss rejects active operations and makes the loaded model unusable. Ordinary abort, controller callback errors, stopping criteria, and session input errors terminate only that session.

`session.dispose()` is safe and idempotent after partial initialization, operation failure, cancellation, and device loss. It waits for tracked work and mapped staging buffers.

Normal disposal is session-first. Model disposal defensively marks remaining sessions terminal, awaits their disposal, and then releases model weights and adapter-owned runtime resources.

## Gemma4E2B Adapter

```ts
interface Gemma4E2BLoadOptions extends InferenceBackendLoadOptions {
  readonly device?: "webgpu";
  readonly dtype?: "auto";
  readonly signal?: AbortSignal;
  readonly artifactProvider?: InferenceArtifactProvider;
  readonly progress_callback?: (event: ArtifactProgressEvent) => void;
}

interface Gemma4E2BLoadedModel {
  readonly config: PretrainedConfig;
  readonly generationCapabilities: GenerationCapabilitiesV1;

  createAutoregressiveSession(options: {
    readonly batchSize: 1;
    readonly maxSequenceLength: number;
    readonly signal?: AbortSignal;
  }): Promise<AutoregressiveSessionV1>;

  dispose(): Promise<void>;
}

export class Gemma4E2B {
  static readonly modelId = "google/gemma-4-E2B-it-qat-mobile-transformers";
  static readonly generationCapabilities: StaticGenerationCapabilitiesV1;

  static load(options: Gemma4E2BLoadOptions): Promise<Gemma4E2BLoadedModel>;
}
```

The adapter accepts `device: 'webgpu'` and omitted/`dtype: 'auto'`. Unsupported devices and dtypes, including `q4f16`, throw without fallback.

The adapter omits demo-owned tokenization, chat templates, text decoding, history, streamers, stopping, sampling policy, and final output formatting. Transformers.js owns those concerns.

The Transformers.js shared config and runtime config are checked for identity-critical agreement, including vocabulary size and maximum positions.

## Implementation Status

No runtime contract decision is outstanding.

Transformers.js has implemented the controller, pull-session driver, CPU logits fallback, fast argmax plan commit, iterator cleanup, capability validation, option passthrough, and public provider types.

Implementation remains in the WebGPU package for:

- the `Gemma4E2B` adapter;
- autoregressive session lifecycle;
- CPU logits leases;
- fast argmax iteration;
- cancellation and device-loss propagation;
- model/session cleanup tracking;
- runtime IO option mapping;
- artifact-provider adaptation.
