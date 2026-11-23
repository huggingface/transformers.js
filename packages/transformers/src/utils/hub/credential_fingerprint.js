// Maps each distinct token to a process-local ID, so that memoization keys can separate credentials
// without containing raw tokens. This guards against accidental log/debug exposure; it is not a
// security boundary or a persisted cache namespace. The map retains every token it has seen, and
// IDs are not stable across restarts.
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
