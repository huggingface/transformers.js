// Explicit tradeoff: retain a token-to-ID map for this module's lifetime instead of maintaining
// a hand-written hash. Tokens already live in request settings; this map additionally retains each
// distinct token so process-local memoization can separate credentials without putting raw tokens
// in key strings. This is hygiene against accidental log/debug exposure, not a security boundary
// or a persisted cache namespace. IDs are local to this module instance, not stable across restarts.
/** @type {Map<string, number>} */
const credentialIds = new Map();
let nextCredentialId = 1;

/**
 * A non-secret credential identity, not a persisted cache namespace or authorization mechanism.
 * @param {string|null|undefined} token
 * @returns {number|null}
 * @private
 */
export function credentialFingerprint(token) {
    if (token == null) return null;
    let id = credentialIds.get(token);
    if (id === undefined) {
        id = nextCredentialId++;
        credentialIds.set(token, id);
    }
    return id;
}
