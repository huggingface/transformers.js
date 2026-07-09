/**
 * @module generation/grammar/tokenizer_bridge
 */

const encoder = new TextEncoder();
const tokenizerCache = new WeakMap();

function decodeToken(tokenizer, token_id) {
    try {
        return tokenizer.decode([token_id], { skip_special_tokens: false, clean_up_tokenization_spaces: false });
    } catch {
        return '';
    }
}

/**
 * Creates or returns the cached llguidance tokenizer handle for a Transformers.js tokenizer.
 *
 * @param {import('../../tokenization_utils.js').PreTrainedTokenizer} tokenizer The tokenizer to bridge.
 * @param {Object} runtime The normalized llguidance runtime.
 * @returns {Promise<Object>}
 */
export async function getLLGuidanceTokenizer(tokenizer, runtime) {
    let runtimeCache = tokenizerCache.get(tokenizer);
    if (!runtimeCache) {
        runtimeCache = new WeakMap();
        tokenizerCache.set(tokenizer, runtimeCache);
    }

    let cached = runtimeCache.get(runtime);
    if (cached) {
        return cached;
    }

    cached = Promise.resolve().then(() => {
        const vocab = tokenizer.get_vocab();
        const tokens = [];
        for (const [token, id] of Object.entries(vocab)) {
            tokens[id] = encoder.encode(decodeToken(tokenizer, id) || token);
        }

        return runtime.createTokenizer({
            tokens,
            eos_token_id: tokenizer.eos_token_id,
            bos_token_id: tokenizer.bos_token_id,
            special_token_ids: tokenizer.all_special_ids ?? [],
        });
    });
    runtimeCache.set(runtime, cached);
    return cached;
}
