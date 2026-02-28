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
 * Build word-level timestamps from token ids and token-level timestamps.
 * @param {any} tokenizer
 * @param {number[]} token_ids
 * @param {[number, number][]} token_timestamps
 * @returns {{ text: string, timestamp: [number, number] }[]}
 */
export function buildTransducerWordTimestamps(tokenizer, token_ids, token_timestamps) {
    if (!tokenizer || token_ids.length === 0 || token_timestamps.length === 0) {
        return [];
    }

    const words = [];
    let current = null;

    for (let i = 0; i < token_ids.length; ++i) {
        const id = token_ids[i];
        const ts = token_timestamps[i];
        const piece = tokenizer.decode([id], {
            skip_special_tokens: true,
            clean_up_tokenization_spaces: false,
        });

        if (!piece) continue;

        const startsNewWord = /^\s+/.test(piece) || piece.startsWith('▁');
        const normalizedPiece = piece.replace(/^\s+/, '').replace(/^▁+/, '');
        if (!normalizedPiece) continue;

        if (!current || startsNewWord) {
            if (current) {
                const text = current.text.trim();
                if (text) {
                    words.push({
                        text,
                        timestamp: [current.start, current.end],
                    });
                }
            }
            current = {
                text: normalizedPiece,
                start: ts[0],
                end: ts[1],
            };
        } else {
            current.text += normalizedPiece;
            current.end = ts[1];
        }
    }

    if (current) {
        const text = current.text.trim();
        if (text) {
            words.push({
                text,
                timestamp: [current.start, current.end],
            });
        }
    }

    return words;
}
