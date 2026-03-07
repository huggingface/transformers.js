const NEMO_SEGMENT_BREAK_REGEX = /[.!?;:]["')\]]*$/;
const NEMO_MAX_WORD_GAP_S = 0.8;

/**
 * @param {Array<{ text: string, start_time: number, end_time: number }>} words
 * @returns {string}
 */
export function joinTimedWords(words) {
    let text = '';
    for (const word of words) {
        const part = word.text ?? '';
        if (!part) continue;
        if (!text) {
            text = part;
        } else if (/^[,.;:!?)}\]]+$/.test(part)) {
            text += part;
        } else {
            text += ` ${part}`;
        }
    }
    return text;
}

/**
 * @param {Array<{ text: string, start_time: number, end_time: number }>} words
 * @returns {Array<{ text: string, timestamp: [number, number] }>}
 */
export function buildWordChunks(words) {
    return words.map((word) => ({
        text: word.text,
        timestamp: [word.start_time, word.end_time],
    }));
}

/**
 * @param {Array<{ text: string, start_time: number, end_time: number }>} words
 * @returns {string}
 */
export function buildSegmentText(words) {
    return joinTimedWords(words);
}

/**
 * @param {Array<{ text: string, start_time: number, end_time: number }>} words
 * @param {[number, number] | null} utterance_timestamp
 * @param {string} text
 * @returns {Array<{ text: string, timestamp: [number, number] }>}
 */
export function buildNemoSegmentChunks(words, utterance_timestamp = null, text = '') {
    if (!Array.isArray(words) || words.length === 0) {
        if (utterance_timestamp) {
            return [{ text, timestamp: utterance_timestamp }];
        }
        return [];
    }

    /** @type {Array<{ text: string, timestamp: [number, number] }>} */
    const chunks = [];
    /** @type {typeof words} */
    let current = [];
    for (const word of words) {
        const prev = current.at(-1);
        if (prev) {
            const gap_s = Math.max(0, word.start_time - prev.end_time);
            const shouldBreak =
                NEMO_SEGMENT_BREAK_REGEX.test(prev.text) ||
                gap_s > NEMO_MAX_WORD_GAP_S;
            if (shouldBreak) {
                chunks.push({
                    text: buildSegmentText(current),
                    timestamp: [current[0].start_time, current[current.length - 1].end_time],
                });
                current = [];
            }
        }
        current.push(word);
    }

    if (current.length > 0) {
        chunks.push({
            text: buildSegmentText(current),
            timestamp: [current[0].start_time, current[current.length - 1].end_time],
        });
    }

    return chunks;
}
