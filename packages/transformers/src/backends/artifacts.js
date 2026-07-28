/**
 * @file Runtime-neutral random-access artifact provider contracts.
 * @module backends/artifacts
 */

/**
 * @typedef {Object} ArtifactProgressEvent
 * @property {string} file
 * @property {number} loaded
 * @property {number} [total]
 */

/**
 * @typedef {Object} RandomAccessByteSource
 * @property {number} [size] Stable byte length when known. It may initially be undefined and become defined after transport metadata arrives.
 * @property {(begin: number, end: number, options?: {signal?: AbortSignal}) => Promise<Uint8Array>} read Read an independent half-open byte range `[begin, end)`. Bounds are non-negative safe integers with `end >= begin`. Reads may complete out of order, and every result is an owned array that remains valid after later reads and close.
 * @property {() => Promise<void>} close Idempotently reject new reads, wait for reads already in progress to settle, and release the source without implicitly aborting those reads.
 */

/**
 * @typedef {Object} InferenceArtifactProvider
 * @property {<T>(file: string, options?: {signal?: AbortSignal}) => Promise<T>} readJson
 * @property {(file: string, options?: {signal?: AbortSignal, onProgress?: (event: ArtifactProgressEvent) => void}) => Promise<RandomAccessByteSource>} openByteSource Open a source supporting concurrent, independently positioned reads.
 */

/**
 * Validate the provider shape without opening an artifact. Range and ownership conformance is
 * validated by the consuming runtime, which owns the source lifecycle.
 *
 * @param {unknown} provider
 * @returns {asserts provider is InferenceArtifactProvider|undefined}
 */
export function validateInferenceArtifactProvider(provider) {
    if (provider === undefined) return;
    const candidate = /** @type {any} */ (provider);
    if (
        provider === null ||
        typeof provider !== 'object' ||
        typeof candidate.readJson !== 'function' ||
        typeof candidate.openByteSource !== 'function'
    ) {
        throw new TypeError('`artifactProvider` must implement `readJson()` and `openByteSource()`.');
    }
}
