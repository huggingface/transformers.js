/**
 * Cache tokenizer id->token maps for stable and fast boundary detection.
 * @type {WeakMap<any, Map<number, string>>}
 */
const TOKEN_ID_TO_TEXT_CACHE = new WeakMap();

/**
 * @param {any} tokenizer
 * @returns {Map<number, string>}
 */
function getIdToTokenMap(tokenizer) {
    let cached = TOKEN_ID_TO_TEXT_CACHE.get(tokenizer);
    if (cached) return cached;

    cached = new Map();
    if (tokenizer?.get_vocab) {
        const vocab = tokenizer.get_vocab();
        // get_vocab() may return a Map or a plain Object depending on the tokenizer backend.
        const entries = vocab instanceof Map ? vocab.entries() : Object.entries(vocab);
        for (const [token, id] of entries) {
            if (Number.isInteger(id)) {
                cached.set(id, token);
            }
        }
    }
    TOKEN_ID_TO_TEXT_CACHE.set(tokenizer, cached);
    return cached;
}

/**
 * Resolve per-token text and word boundary metadata in a tokenizer-agnostic way.
 * Uses raw vocab token (if available) for boundary markers, and decoded token text for display.
 * @param {any} tokenizer
 * @param {number} id
 * @returns {{ raw: string, clean: string, startsNewWord: boolean }}
 */
function resolveTokenPiece(tokenizer, id) {
    const rawToken = getIdToTokenMap(tokenizer).get(id) ?? '';
    const decoded = tokenizer.decode([id], {
        skip_special_tokens: true,
        clean_up_tokenization_spaces: false,
    });

    // SentencePiece/BPE boundary markers used by common tokenizers.
    const startsWithBoundaryMarker = /^(?:▁|Ġ)+/.test(rawToken);
    const startsWithWhitespace = /^\s+/.test(decoded);
    const startsNewWord = startsWithBoundaryMarker || startsWithWhitespace;

    // Human readable token text.
    let clean = decoded.replace(/^\s+/, '');
    if (!clean) {
        clean = rawToken.replace(/^(?:▁|Ġ|Ċ)+/, '').replace(/^ +/, '');
    }

    return { raw: rawToken || decoded, clean, startsNewWord };
}

/**
 * Decode token ids into final transcription text.
 * @param {any} tokenizer
 * @param {number[]} token_ids
 * @returns {string}
 */
export function decodeTransducerText(tokenizer, token_ids) {
    if (!Array.isArray(token_ids) || token_ids.length === 0) return '';
    if (!tokenizer) return token_ids.join(' ');
    return tokenizer.decode(token_ids, { skip_special_tokens: true }).trim();
}

/**
 * Build detailed word/token outputs with optional confidence aggregation.
 * @param {any} tokenizer
 * @param {number[]} token_ids
 * @param {[number, number][]} token_timestamps
 * @param {number[] | null} token_confidences
 * @returns {{
 *  words: Array<{ text: string, start_time: number, end_time: number, confidence?: number }>,
 *  tokens: Array<{ token: string, raw_token: string, is_word_start: boolean, start_time: number, end_time: number, confidence?: number }>,
 *  word_confidences: number[] | null,
 *  word_avg: number | null,
 * }}
 */
export function buildTransducerDetailedOutputs(tokenizer, token_ids, token_timestamps, token_confidences = null) {
    if (!tokenizer || token_ids.length === 0 || token_timestamps.length === 0) {
        return { words: [], tokens: [], word_confidences: null, word_avg: null };
    }

    /** @type {Array<{ id: number, token: string, raw_token: string, is_word_start: boolean, start_time: number, end_time: number, confidence?: number }>} */
    const tokens = [];
    /** @type {Array<{ text: string, start_time: number, end_time: number, confidence?: number }>} */
    const words = [];

    /** @type {{ text: string, start: number, end: number, confs: number[] } | null} */
    let current = null;

    for (let i = 0; i < token_ids.length; ++i) {
        const id = token_ids[i];
        const ts = token_timestamps[i];
        const piece = resolveTokenPiece(tokenizer, id);
        const raw = piece.raw;
        const startsNewWord = piece.startsNewWord;
        const clean = piece.clean;
        if (!clean) continue;

        const tok = {
            id,
            token: clean,
            raw_token: raw,
            is_word_start: startsNewWord,
            start_time: ts[0],
            end_time: ts[1],
        };
        const conf = token_confidences?.[i];
        if (conf != null && Number.isFinite(conf)) {
            tok.confidence = Math.round(conf * 1e6) / 1e6;
        }
        tokens.push(tok);

        if (!current || startsNewWord) {
            if (current) {
                const text = current.text.trim();
                if (text) {
                    /** @type {{ text: string, start_time: number, end_time: number, confidence?: number }} */
                    const word = {
                        text,
                        start_time: current.start,
                        end_time: current.end,
                    };
                    if (current.confs.length > 0) {
                        word.confidence =
                            Math.round((current.confs.reduce((a, b) => a + b, 0) / current.confs.length) * 1e6) / 1e6;
                    }
                    words.push(word);
                }
            }
            current = {
                text: clean,
                start: ts[0],
                end: ts[1],
                confs: conf != null && Number.isFinite(conf) ? [conf] : [],
            };
        } else {
            current.text += clean;
            current.end = ts[1];
            if (conf != null && Number.isFinite(conf)) {
                current.confs.push(conf);
            }
        }
    }

    if (current) {
        const text = current.text.trim();
        if (text) {
            /** @type {{ text: string, start_time: number, end_time: number, confidence?: number }} */
            const word = {
                text,
                start_time: current.start,
                end_time: current.end,
            };
            if (current.confs.length > 0) {
                word.confidence =
                    Math.round((current.confs.reduce((a, b) => a + b, 0) / current.confs.length) * 1e6) / 1e6;
            }
            words.push(word);
        }
    }

    const word_confidences = words.some((x) => x.confidence != null) ? words.map((x) => x.confidence ?? 0) : null;
    const word_avg =
        word_confidences && word_confidences.length > 0
            ? Math.round((word_confidences.reduce((a, b) => a + b, 0) / word_confidences.length) * 1e6) / 1e6
            : null;

    return { words, tokens, word_confidences, word_avg };
}
