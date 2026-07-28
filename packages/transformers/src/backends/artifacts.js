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
 * @property {(begin: number, end: number, options?: {signal?: AbortSignal}) => Promise<Uint8Array>} read Read an independent half-open byte range `[begin, end)`. The returned array is owned by the caller and remains valid after later reads and close.
 * @property {() => Promise<void>} close Idempotently reject new reads, drain reads already in progress, and release the source.
 */

/**
 * @typedef {Object} InferenceArtifactProvider
 * @property {<T>(file: string, options?: {signal?: AbortSignal}) => Promise<T>} readJson
 * @property {(file: string, options?: {signal?: AbortSignal, onProgress?: (event: ArtifactProgressEvent) => void}) => Promise<RandomAccessByteSource>} openByteSource Open a source supporting concurrent, independently positioned reads.
 */

export {};
