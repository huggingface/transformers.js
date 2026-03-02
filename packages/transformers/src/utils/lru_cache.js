/**
 * A simple Least Recently Used (LRU) cache implementation in JavaScript.
 * This cache stores key-value pairs and evicts the least recently used item
 * when the capacity is exceeded.
 */
export class LRUCache {
    /** @type {number} */
    #capacity;

    /** @type {Map<any, any>} */
    #cache;

    /**
     * Creates an LRUCache instance.
     * @param {number} capacity The maximum number of items the cache can hold.
     */
    constructor(capacity) {
        this.#capacity = capacity;
        this.#cache = new Map();
    }

    /**
     * Retrieves the value associated with the given key and marks the key as recently used.
     * @param {any} key The key to retrieve.
     * @returns {any} The value associated with the key, or undefined if the key does not exist.
     */
    get(key) {
        if (!this.#cache.has(key)) return undefined;
        const value = this.#cache.get(key);
        this.#cache.delete(key);
        this.#cache.set(key, value);
        return value;
    }

    /**
     * Inserts or updates the key-value pair in the cache.
     * If the key already exists, it is updated and marked as recently used.
     * If the cache exceeds its capacity, the least recently used item is evicted.
     * @param {any} key The key to add or update.
     * @param {any} value The value to associate with the key.
     */
    put(key, value) {
        if (this.#cache.has(key)) {
            this.#cache.delete(key);
        }
        this.#cache.set(key, value);
        if (this.#cache.size > this.#capacity) {
            this.#cache.delete(this.#cache.keys().next().value);
        }
    }

    /**
     * Removes the entry for the given key from the cache.
     * @param {any} key The key to delete.
     * @returns {boolean} `true` if the entry existed and was removed, `false` otherwise.
     */
    delete(key) {
        return this.#cache.delete(key);
    }

    /**
     * Clears the cache.
     */
    clear() {
        this.#cache.clear();
    }
}
