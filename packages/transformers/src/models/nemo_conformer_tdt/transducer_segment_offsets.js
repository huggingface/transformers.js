const NEMO_STRONG_SENTENCE_END_REGEX = /[!?…](?:["')\]]+)?$/u;
const NEMO_PERIOD_SENTENCE_END_REGEX = /\.(?:["')\]]+)?$/u;
const NEMO_TRAILING_CLOSERS_REGEX = /["')\]]+$/gu;
const NEMO_LEADING_OPENERS_REGEX = /^[("'“‘\[{]+/u;
const NEMO_DOTTED_ACRONYM_REGEX = /^(?:[A-Z]\.){2,}$/;
const NEMO_SINGLE_LETTER_ENUM_REGEX = /^[A-Z]\.$/;
const NEMO_ROMAN_ENUM_REGEX = /^(?:[IVXLCDM]+)\.$/i;
const NEMO_NUMERIC_ENUM_REGEX = /^\d+\.$/;
const NEMO_FALLBACK_SEGMENT_GAP_S = 3.0;
const NEMO_NON_BREAKING_PERIOD_WORDS = new Set([
    'mr.',
    'mrs.',
    'ms.',
    'dr.',
    'prof.',
    'sr.',
    'jr.',
    'vs.',
    'etc.',
    'e.g.',
    'i.e.',
]);

/**
 * @param {Array<{ text: string, startTime: number, endTime: number }>} words
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
 * @param {Array<{ text: string, startTime: number, endTime: number }>} words
 * @returns {Array<{ text: string, timestamp: [number, number] }>}
 */
export function buildWordChunks(words) {
    return words.map((word) => ({
        text: word.text,
        timestamp: [word.startTime, word.endTime],
    }));
}

/**
 * @param {Array<{ text: string, startTime: number, endTime: number }>} words
 * @returns {string}
 */
export function buildSegmentText(words) {
    return joinTimedWords(words);
}

function stripTrailingClosers(text) {
    return String(text ?? '').replace(NEMO_TRAILING_CLOSERS_REGEX, '');
}

function looksLikeSentenceStart(text) {
    const cleaned = String(text ?? '').replace(NEMO_LEADING_OPENERS_REGEX, '');
    return /^[A-Z]/.test(cleaned);
}

/**
 * Conservative sentence-boundary heuristic for ASR word timestamps.
 * Favors under-segmentation over mid-sentence false positives.
 *
 * @param {{ text: string }} currentWord
 * @param {{ text: string } | null} nextWord
 * @param {number} gap_s
 * @returns {boolean}
 */
export function shouldEndSentenceAfterWord(currentWord, nextWord, gap_s = 0) {
    if (!nextWord) {
        return false;
    }

    if (gap_s >= NEMO_FALLBACK_SEGMENT_GAP_S) {
        return true;
    }

    const currentText = String(currentWord?.text ?? '');
    if (!currentText) {
        return false;
    }

    if (NEMO_STRONG_SENTENCE_END_REGEX.test(currentText)) {
        return true;
    }

    if (!NEMO_PERIOD_SENTENCE_END_REGEX.test(currentText)) {
        return false;
    }

    const stripped = stripTrailingClosers(currentText);
    const lowered = stripped.toLowerCase();
    if (
        NEMO_NON_BREAKING_PERIOD_WORDS.has(lowered) ||
        NEMO_DOTTED_ACRONYM_REGEX.test(stripped) ||
        NEMO_SINGLE_LETTER_ENUM_REGEX.test(stripped) ||
        NEMO_ROMAN_ENUM_REGEX.test(stripped) ||
        NEMO_NUMERIC_ENUM_REGEX.test(stripped)
    ) {
        return false;
    }

    return looksLikeSentenceStart(nextWord.text);
}

/**
 * Partition timed words into conservative sentence-like segments.
 *
 * @param {Array<{ text: string, startTime: number, endTime: number }>} words
 * @returns {Array<{ words: Array<{ text: string, startTime: number, endTime: number }>, text: string, timestamp: [number, number] }>}
 */
export function partitionNemoWordsIntoSegments(words) {
    if (!Array.isArray(words) || words.length === 0) {
        return [];
    }

    /** @type {Array<{ words: Array<{ text: string, startTime: number, endTime: number }>, text: string, timestamp: [number, number] }>} */
    const segments = [];
    /** @type {typeof words} */
    let current = [];
    for (let i = 0; i < words.length; ++i) {
        const word = words[i];
        current.push(word);

        const nextWord = words[i + 1] ?? null;
        const gap_s = nextWord ? Math.max(0, nextWord.startTime - word.endTime) : 0;
        if (shouldEndSentenceAfterWord(word, nextWord, gap_s)) {
            segments.push({
                words: current,
                text: buildSegmentText(current),
                timestamp: [current[0].startTime, current[current.length - 1].endTime],
            });
            current = [];
        }
    }

    if (current.length > 0) {
        segments.push({
            words: current,
            text: buildSegmentText(current),
            timestamp: [current[0].startTime, current[current.length - 1].endTime],
        });
    }

    return segments;
}

/**
 * @param {Array<{ text: string, startTime: number, endTime: number }>} words
 * @param {[number, number] | null} utteranceTimestamp
 * @param {string} text
 * @returns {Array<{ text: string, timestamp: [number, number] }>}
 */
export function buildNemoSegmentChunks(words, utteranceTimestamp = null, text = '') {
    if (!Array.isArray(words) || words.length === 0) {
        if (utteranceTimestamp) {
            return [{ text, timestamp: utteranceTimestamp }];
        }
        return [];
    }

    return partitionNemoWordsIntoSegments(words).map((segment) => ({
        text: segment.text,
        timestamp: segment.timestamp,
    }));
}
