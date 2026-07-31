/**
 * Let there be order amidst the chaos.
 *
 * This file implements Mersenne Twister 19937, matching Python's `random` module exactly for reproducibility.
 *
 * ```javascript
 * import { random } from '@huggingface/transformers';
 *
 * random.seed(42);
 * random.random();           // 0.6394267984578837  (matches Python)
 * random.gauss(0, 1);        // normal-distributed value
 * random.choices(['a','b'], [3, 1]);  // weighted pick
 *
 * const arr = [1, 2, 3, 4, 5];
 * random.shuffle(arr);       // in-place Fisher-Yates shuffle
 *
 * // Use a separate instance to avoid affecting the global state:
 * const rng = new random.Random(42);
 * rng.random();              // 0.6394267984578837  (same seed, independent state)
 * ```
 *
 * **Note on Reproducibility:**
 * Similarly to the [Python random](https://docs.python.org/3/library/random.html#notes-on-reproducibility)
 * module, it is useful to be able to reproduce the sequences given by a pseudo-random number generator.
 * By reusing a seed value, the same sequence should be reproducible from run to run as long as multiple
 * threads or asynchronous operations are not running concurrently.
 *
 * @module utils/random
 */

import { apis } from '../env.js';

/**
 * Mersenne Twister 19937 PRNG, matching Python's `random.Random` class exactly.
 *
 * Each instance has its own independent state, so seeding one instance does not
 * affect any other instance or the global helper functions.
 *
 * @example
 * const rng1 = new Random(42);
 * const rng2 = new Random(42);
 * rng1.random() === rng2.random(); // true (same seed, independent state)
 */
export class Random {
    constructor(seed) {
        this._mt = new Uint32Array(624);
        this._idx = 625;
        this._gauss_next = null;
        this._random_fn = this.random.bind(this);
        this.seed(seed);
    }

    /**
     * Seeds this instance's PRNG.
     *
     * When called with a number, initializes the state deterministically from that value.
     * When called with no arguments (or `undefined`/`null`), seeds from OS entropy
     * via `crypto.getRandomValues`, matching Python's `random.seed()` behaviour.
     *
     * @param {number} [n] The seed value. Omit to seed from OS entropy.
     */
    seed(n) {
        if (n === undefined || n === null) {
            if (apis.IS_CRYPTO_AVAILABLE) {
                const buf = new Uint32Array(1);
                crypto.getRandomValues(buf);
                n = buf[0];
            } else {
                n = Date.now() >>> 0;
            }
        }
        const mt = this._mt;
        const u = (a, b) => Math.imul(a, b) >>> 0,
            key = [];
        for (let v = n || 0; v > 0; v = Math.floor(v / 0x100000000)) key.push(v & 0xffffffff);
        if (!key.length) key.push(0);
        mt[0] = 19650218;
        for (let k = 1; k < 624; ++k) mt[k] = (u(1812433253, mt[k - 1] ^ (mt[k - 1] >>> 30)) + k) >>> 0;
        let i = 1,
            j = 0;
        for (let k = Math.max(624, key.length); k > 0; --k, ++i, ++j) {
            if (i >= 624) {
                mt[0] = mt[623];
                i = 1;
            }
            if (j >= key.length) j = 0;
            mt[i] = ((mt[i] ^ u(mt[i - 1] ^ (mt[i - 1] >>> 30), 1664525)) + key[j] + j) >>> 0;
        }
        for (let k = 623; k > 0; --k, ++i) {
            if (i >= 624) {
                mt[0] = mt[623];
                i = 1;
            }
            mt[i] = ((mt[i] ^ u(mt[i - 1] ^ (mt[i - 1] >>> 30), 1566083941)) - i) >>> 0;
        }
        mt[0] = 0x80000000;
        this._idx = 624;
        this._gauss_next = null;
    }

    /**
     * Generates a random unsigned 32-bit integer.
     *
     * Performs the "twist" step when the state buffer is exhausted,
     * then applies the standard MT19937 tempering transform.
     *
     * @returns {number} A random integer in the range [0, 2^32 - 1].
     */
    _int32() {
        const mt = this._mt;
        if (this._idx >= 624) {
            for (let k = 0; k < 624; ++k) {
                // twist
                const y = (mt[k] & 0x80000000) | (mt[(k + 1) % 624] & 0x7fffffff);
                mt[k] = (mt[(k + 397) % 624] ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0)) >>> 0;
            }
            this._idx = 0;
        }
        let y = mt[this._idx++];
        y ^= y >>> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >>> 18;
        return y >>> 0;
    }

    /**
     * Generates a random floating-point number in the half-open interval [0, 1).
     *
     * Combines two 32-bit integers (using 53 bits of precision) to produce
     * a uniformly distributed double, matching Python's `random.random()`.
     *
     * @returns {number} A random float in [0, 1).
     */
    random() {
        return ((this._int32() >>> 5) * 67108864.0 + (this._int32() >>> 6)) / 9007199254740992.0;
    }

    /**
     * Generates a random number from a Gaussian (normal) distribution.
     *
     * Uses the Box-Muller transform with a cached spare value,
     * matching Python's `random.gauss()` output for the same seed.
     *
     * @param {number} [mu=0] The mean of the distribution.
     * @param {number} [sigma=1] The standard deviation of the distribution.
     * @returns {number} A normally distributed random value.
     */
    gauss(mu = 0, sigma = 1) {
        let z = this._gauss_next;
        this._gauss_next = null;
        if (z === null) {
            const x2pi = this.random() * 2 * Math.PI,
                g2rad = Math.sqrt(-2 * Math.log(1 - this.random()));
            z = Math.cos(x2pi) * g2rad;
            this._gauss_next = Math.sin(x2pi) * g2rad;
        }
        return mu + z * sigma;
    }

    /**
     * Shuffles an array in-place using the Fisher-Yates algorithm.
     *
     * Uses rejection sampling via `getrandbits`-style bit masking to ensure
     * a uniform distribution, matching Python's `random.shuffle()`.
     *
     * @param {any[]} arr The array to shuffle in-place.
     */
    shuffle(arr) {
        for (let i = arr.length - 1; i > 0; --i) {
            const k = 32 - Math.clz32(i + 1);
            let r = this._int32() >>> (32 - k);
            while (r > i) r = this._int32() >>> (32 - k);
            const t = arr[i];
            arr[i] = arr[r];
            arr[r] = t;
        }
    }

    /**
     * Selects a single element from a weighted population.
     *
     * Matches Python's `random.choices(population, weights=weights, k=1)[0]`
     *
     * @param {any[]} population The array of items to choose from.
     * @param {number[]} weights An array of non-negative weights, one per population element.
     * @returns {*} A single randomly selected element from the population.
     */
    choices(population, weights) {
        return population[_weightedIndexWith(this._random_fn, weights)];
    }
}

/**
 * Returns a random index into `weights`, where each index's probability
 * is proportional to its weight. Uses a linear scan: O(n) time, O(1) memory.
 *
 * @param {() => number} randomFn A function returning a uniform random float in [0, 1).
 * @param {ArrayLike<number>} weights Non-negative weights.
 * @returns {number} A randomly selected index in `[0, weights.length)`.
 */
function _weightedIndexWith(randomFn, weights) {
    let sum = 0;
    for (let i = 0; i < weights.length; ++i) sum += weights[i];
    let x = randomFn() * sum;
    for (let i = 0; i < weights.length; ++i) {
        x -= weights[i];
        if (x < 0) return i;
    }
    return weights.length - 1; // floating-point guard
}

// Global default instance: mirrors the module-level functions in Python's `random` module.
const _default = new Random();
export const random = Object.freeze({
    Random,
    seed: _default.seed.bind(_default),
    random: _default.random.bind(_default),
    gauss: _default.gauss.bind(_default),
    shuffle: _default.shuffle.bind(_default),
    choices: _default.choices.bind(_default),
});

// Private helper function, used by LogitsSampler, but not exported as part of the public API.
export const _weightedIndex = (weights) => _weightedIndexWith(random.random, weights);
