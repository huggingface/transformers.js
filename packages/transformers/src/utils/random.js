/**
 * Let there be order amidst the chaos.
 *
 * This file implements Mersenne Twister 19937, matching Python's `random` module exactly for reproducibility.
 *
 * @example
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
 * @module utils/random
 */

import { apis } from '../env.js';

const mt = new Uint32Array(624);
let idx = 625;
let _gauss_next = null;

/**
 * Seeds the Mersenne Twister PRNG.
 *
 * When called with a number, initializes the state deterministically from that value.
 * When called with no arguments (or `undefined`/`null`), seeds from OS entropy
 * via `crypto.getRandomValues`, matching Python's `random.seed()` behaviour.
 *
 * @param {number} [n] The seed value. Omit to seed from OS entropy.
 */
export function seed(n) {
    if (n === undefined || n === null) {
        if (apis.IS_CRYPTO_AVAILABLE) {
            const buf = new Uint32Array(1);
            crypto.getRandomValues(buf);
            n = buf[0];
        } else {
            n = Date.now() >>> 0;
        }
    }
    const u = (a, b) => Math.imul(a, b) >>> 0,
        key = [];
    for (let v = n || 0; v > 0; v = Math.floor(v / 0x100000000)) key.push(v & 0xffffffff);
    if (!key.length) key.push(0);
    mt[0] = 19650218;
    for (idx = 1; idx < 624; ++idx) mt[idx] = (u(1812433253, mt[idx - 1] ^ (mt[idx - 1] >>> 30)) + idx) >>> 0;
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
    idx = 624;
    _gauss_next = null;
}

/**
 * Generates a random unsigned 32-bit integer.
 *
 * Performs the "twist" step when the state buffer is exhausted,
 * then applies the standard MT19937 tempering transform.
 *
 * @returns {number} A random integer in the range [0, 2^32 - 1].
 */
function int32() {
    if (idx >= 624) {
        for (let k = 0; k < 624; ++k) {
            // twist
            const y = (mt[k] & 0x80000000) | (mt[(k + 1) % 624] & 0x7fffffff);
            mt[k] = (mt[(k + 397) % 624] ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0)) >>> 0;
        }
        idx = 0;
    }
    let y = mt[idx++];
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
export function random() {
    return ((int32() >>> 5) * 67108864.0 + (int32() >>> 6)) / 9007199254740992.0;
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
export function gauss(mu = 0, sigma = 1) {
    let z = _gauss_next;
    _gauss_next = null;
    if (z === null) {
        const x2pi = random() * 2 * Math.PI,
            g2rad = Math.sqrt(-2 * Math.log(1 - random()));
        z = Math.cos(x2pi) * g2rad;
        _gauss_next = Math.sin(x2pi) * g2rad;
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
export function shuffle(arr) {
    for (let i = arr.length - 1; i > 0; --i) {
        const k = 32 - Math.clz32(i + 1);
        let r = int32() >>> (32 - k);
        while (r > i) r = int32() >>> (32 - k);
        const t = arr[i];
        arr[i] = arr[r];
        arr[r] = t;
    }
}

/**
 * Returns a random index into `weights`, where each index's probability
 * is proportional to its weight. Uses a linear scan: O(n) time, O(1) memory.
 *
 * @param {ArrayLike<number>} weights Non-negative weights.
 * @returns {number} A randomly selected index in `[0, weights.length)`.
 * @private
 */
export function _weightedIndex(weights) {
    let sum = 0;
    for (let i = 0; i < weights.length; ++i) sum += weights[i];
    let x = random() * sum;
    for (let i = 0; i < weights.length; ++i) {
        x -= weights[i];
        if (x < 0) return i;
    }
    return weights.length - 1; // floating-point guard
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
export function choices(population, weights) {
    return population[_weightedIndex(weights)];
}

// Auto-seed from OS entropy on module load.
seed();
