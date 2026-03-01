/**
 * @file Utility for memoizing promises by key.
 *
 * Ensures that a given async operation is only initiated once per key.
 * Subsequent calls with the same key return the same pending or resolved promise.
 * Rejected promises are evicted from the cache so callers can retry.
 */

/** @type {Map<string, Promise<any>>} */
const cache = new Map();

/**
 * Returns the cached promise for `key`, or calls `factory` to create one and caches it.
 * Subsequent calls with the same key return the same promise whether it is still
 * pending or already resolved, so the factory is never invoked more than once per key.
 * If the promise rejects, the entry is removed from the cache so the operation can be retried.
 *
 * @template T
 * @param {string} key A unique identifier for this async operation.
 * @param {() => Promise<T>} factory A function that returns the promise to memoize.
 *   Only called when no entry exists for `key`.
 * @returns {Promise<T>}
 */
export function memoizePromise(key, factory) {
    if (cache.has(key)) {
        return cache.get(key);
    }
    const promise = factory().then(
        (value) => value,
        (err) => {
            cache.delete(key);
            return Promise.reject(err);
        },
    );
    cache.set(key, promise);
    return promise;
}
