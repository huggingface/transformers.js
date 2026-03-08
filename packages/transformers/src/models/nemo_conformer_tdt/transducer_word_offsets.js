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
        if (Array.isArray(vocab)) {
            for (let id = 0; id < vocab.length; ++id) {
                if (typeof vocab[id] === 'string') {
                    cached.set(id, vocab[id]);
                }
            }
        } else if (vocab instanceof Map) {
            for (const [token, id] of vocab.entries()) {
                if (Number.isInteger(id)) {
                    cached.set(id, token);
                }
            }
        } else if (vocab && typeof vocab === 'object') {
            for (const [token, id] of Object.entries(vocab)) {
                if (Number.isInteger(id)) {
                    cached.set(id, token);
                }
            }
        }
    }
    TOKEN_ID_TO_TEXT_CACHE.set(tokenizer, cached);
    return cached;
}

/**
 * Resolve per-token text and word boundary metadata in a tokenizer-agnostic way.
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

    const startsWithBoundaryMarker = /^(?:▁|Ġ)+/.test(rawToken);
    const startsWithWhitespace = /^\s+/.test(decoded);
    const startsNewWord = startsWithBoundaryMarker || startsWithWhitespace;

    let clean = decoded.replace(/^\s+/, '');
    if (!clean) {
        clean = rawToken.replace(/^(?:▁|Ġ|Ċ)+/, '').replace(/^ +/, '');
    }

    return { raw: rawToken || decoded, clean, startsNewWord };
}

/**
 * @param {string} fullText
 * @param {number} cursor
 * @param {string} tokenText
 * @returns {{ cursor: number, text: string, skippedWhitespace: boolean }}
 */
function consumeAlignedTokenText(fullText, cursor, tokenText) {
    let skippedWhitespace = false;
    while (cursor < fullText.length && /\s/.test(fullText[cursor])) {
        skippedWhitespace = true;
        cursor += 1;
    }

    if (!tokenText) {
        return { cursor, text: '', skippedWhitespace };
    }

    if (fullText.startsWith(tokenText, cursor)) {
        return {
            cursor: cursor + tokenText.length,
            text: fullText.slice(cursor, cursor + tokenText.length),
            skippedWhitespace,
        };
    }

    const next = fullText.indexOf(tokenText, cursor);
    if (next !== -1 && /^\s*$/.test(fullText.slice(cursor, next))) {
        return {
            cursor: next + tokenText.length,
            text: fullText.slice(next, next + tokenText.length),
            skippedWhitespace: skippedWhitespace || next > cursor,
        };
    }

    return {
        cursor: cursor + tokenText.length,
        text: tokenText,
        skippedWhitespace,
    };
}

/**
 * @param {Array<{ text: string, startTime: number, endTime: number, confidence?: number }>} words
 * @param {{ text: string, start: number, end: number, confs: number[] } | null} current
 */
function finalizeAndPushWord(words, current) {
    if (!current) return;

    const text = current.text.trim();
    if (!text) return;

    /** @type {{ text: string, startTime: number, endTime: number, confidence?: number }} */
    const word = {
        text,
        startTime: current.start,
        endTime: current.end,
    };
    if (current.confs.length > 0) {
        word.confidence = Math.round((current.confs.reduce((a, b) => a + b, 0) / current.confs.length) * 1e6) / 1e6;
    }
    words.push(word);
}

/**
 * @param {any} tokenizer
 * @param {number[]} token_ids
 * @param {[number, number][]} token_timestamps
 * @param {number[] | null} token_confidences
 * @param {string} fullText
 * @returns {{
 *  words: Array<{ text: string, startTime: number, endTime: number, confidence?: number }>,
 *  tokens: Array<{ id: number, token: string, rawToken: string, isWordStart: boolean, startTime: number, endTime: number, confidence?: number }>,
 *  wordAverage: number | null,
 * }}
 */
export function buildTransducerWordOffsets(
    tokenizer,
    token_ids,
    token_timestamps,
    token_confidences = null,
    fullText = '',
) {
    if (token_ids.length !== token_timestamps.length) {
        throw new Error(
            `buildTransducerWordOffsets expects equal lengths for token_ids (${token_ids.length}) and token_timestamps (${token_timestamps.length}).`,
        );
    }
    if (token_confidences && token_confidences.length !== token_ids.length) {
        throw new Error(
            `buildTransducerWordOffsets expects token_confidences length (${token_confidences.length}) to match token_ids length (${token_ids.length}).`,
        );
    }
    if (token_ids.length === 0) {
        return { words: [], tokens: [], wordAverage: null };
    }
    if (!tokenizer) {
        throw new Error('buildTransducerWordOffsets requires a tokenizer for non-empty token_ids.');
    }

    /** @type {Array<{ id: number, token: string, rawToken: string, isWordStart: boolean, startTime: number, endTime: number, confidence?: number }>} */
    const tokens = [];
    /** @type {Array<{ text: string, startTime: number, endTime: number, confidence?: number }>} */
    const words = [];
    let textCursor = 0;

    /** @type {{ text: string, start: number, end: number, confs: number[] } | null} */
    let current = null;

    for (let i = 0; i < token_ids.length; ++i) {
        const id = token_ids[i];
        const ts = token_timestamps[i];
        const piece = resolveTokenPiece(tokenizer, id);
        const raw = piece.raw;
        const clean = piece.clean;
        if (!clean) continue;

        const aligned = consumeAlignedTokenText(fullText, textCursor, clean);
        textCursor = aligned.cursor;
        const tokenText = aligned.text || clean;
        const startsNewWord = !current || aligned.skippedWhitespace || piece.startsNewWord;

        const tok = {
            id,
            token: tokenText,
            rawToken: raw,
            isWordStart: startsNewWord,
            startTime: ts[0],
            endTime: ts[1],
        };
        const conf = token_confidences?.[i];
        if (conf != null && Number.isFinite(conf)) {
            tok.confidence = Math.round(conf * 1e6) / 1e6;
        }
        tokens.push(tok);

        if (!current || startsNewWord) {
            finalizeAndPushWord(words, current);
            current = {
                text: tokenText,
                start: ts[0],
                end: ts[1],
                confs: conf != null && Number.isFinite(conf) ? [conf] : [],
            };
        } else {
            current.text += tokenText;
            current.end = ts[1];
            if (conf != null && Number.isFinite(conf)) {
                current.confs.push(conf);
            }
        }
    }

    finalizeAndPushWord(words, current);

    let wordAverage = null;
    if (words.some((x) => x.confidence != null)) {
        const validConfidences = words.map((x) => x.confidence).filter((x) => x != null);
        if (validConfidences.length > 0) {
            wordAverage =
                Math.round((validConfidences.reduce((a, b) => a + b, 0) / validConfidences.length) * 1e6) / 1e6;
        }
    }

    return { words, tokens, wordAverage };
}
