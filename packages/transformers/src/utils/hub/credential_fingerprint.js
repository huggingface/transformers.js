// SHA-256 keeps memoization keys free of credentials without retaining a token-to-id map.
// This synchronous implementation works in browser bundles as well as Node.
const ROUND_CONSTANTS = new Uint32Array([
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
    0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
    0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
    0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2,
]);
const rotate = (value, bits) => (value >>> bits) | (value << (32 - bits));

/**
 * A non-secret credential fingerprint, not a persisted cache namespace or authorization mechanism.
 * @param {string|null|undefined} token
 * @returns {string|null}
 * @private
 */
export function credentialFingerprint(token) {
    if (token == null) return null;
    // JSON escaping also distinguishes ill-formed UTF-16 strings before UTF-8 encoding.
    const input = new TextEncoder().encode(JSON.stringify(token));
    const bytes = new Uint8Array(Math.ceil((input.length + 9) / 64) * 64);
    bytes.set(input);
    bytes[input.length] = 0x80;
    const data = new DataView(bytes.buffer);
    data.setUint32(bytes.length - 8, Math.floor(input.length / 0x20000000));
    data.setUint32(bytes.length - 4, input.length * 8);
    const hash = new Uint32Array([
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ]);
    const words = new Uint32Array(64);
    for (let offset = 0; offset < bytes.length; offset += 64) {
        for (let i = 0; i < 16; ++i) words[i] = data.getUint32(offset + i * 4);
        for (let i = 16; i < 64; ++i) {
            const a = words[i - 15];
            const b = words[i - 2];
            const s0 = rotate(a, 7) ^ rotate(a, 18) ^ (a >>> 3);
            const s1 = rotate(b, 17) ^ rotate(b, 19) ^ (b >>> 10);
            words[i] = (words[i - 16] + s0 + words[i - 7] + s1) >>> 0;
        }
        let [a, b, c, d, e, f, g, h] = hash;
        for (let i = 0; i < 64; ++i) {
            const sum1 = rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25);
            const choose = (e & f) ^ (~e & g);
            const t1 = (h + sum1 + choose + ROUND_CONSTANTS[i] + words[i]) >>> 0;
            const sum0 = rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22);
            const majority = (a & b) ^ (a & c) ^ (b & c);
            [a, b, c, d, e, f, g, h] = [(t1 + sum0 + majority) >>> 0, a, b, c, (d + t1) >>> 0, e, f, g];
        }
        const result = [a, b, c, d, e, f, g, h];
        for (let i = 0; i < 8; ++i) hash[i] = (hash[i] + result[i]) >>> 0;
    }
    return Array.from(hash, (word) => word.toString(16).padStart(8, '0')).join('');
}
