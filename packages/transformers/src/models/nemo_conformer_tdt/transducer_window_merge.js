import { decodeTransducerText } from './transducer_text.js';
import { joinTimedWords } from './transducer_segment_offsets.js';

/**
 * @param {Float32Array|Float64Array} audio
 * @param {number} sampling_rate
 * @param {number} chunk_length_s
 * @param {number | null} stride_length_s
 * @returns {Array<{audio: Float32Array|Float64Array, start_s: number, end_s: number, left_stride_s: number, right_stride_s: number}>}
 */
export function buildNemoWindowSpecs(audio, sampling_rate, chunk_length_s, stride_length_s) {
    if (!(chunk_length_s > 0)) {
        return [
            {
                audio,
                start_s: 0,
                end_s: audio.length / sampling_rate,
                left_stride_s: 0,
                right_stride_s: 0,
            },
        ];
    }

    if (stride_length_s === null) {
        stride_length_s = chunk_length_s / 6;
    } else if (!(stride_length_s >= 0)) {
        throw Error('`stride_length_s` must be non-negative.');
    }
    if (chunk_length_s <= 2 * stride_length_s) {
        throw Error('`chunk_length_s` must be larger than `2 * stride_length_s` for Nemo windowed decoding.');
    }

    const window = Math.floor(sampling_rate * chunk_length_s);
    const stride = Math.floor(sampling_rate * stride_length_s);
    const jump = window - 2 * stride;

    /** @type {Array<{audio: Float32Array|Float64Array, start_s: number, end_s: number, left_stride_s: number, right_stride_s: number}>} */
    const windows = [];
    let offset = 0;
    while (true) {
        const offset_end = offset + window;
        const subarr = audio.subarray(offset, offset_end);
        const is_first = offset === 0;
        const is_last = offset_end >= audio.length;
        windows.push({
            audio: subarr,
            start_s: offset / sampling_rate,
            end_s: (offset + subarr.length) / sampling_rate,
            left_stride_s: is_first ? 0 : stride / sampling_rate,
            right_stride_s: is_last ? 0 : stride / sampling_rate,
        });
        if (is_last) break;
        offset += jump;
    }

    return windows;
}

function shouldKeepTimedItem(start_time, end_time, keep_start_s, keep_end_s, is_first_window, is_last_window) {
    const midpoint = (start_time + end_time) / 2;
    if (!is_first_window && midpoint < keep_start_s) {
        return false;
    }
    if (!is_last_window && midpoint >= keep_end_s) {
        return false;
    }
    return true;
}

function dedupeMergedWords(words) {
    /** @type {typeof words} */
    const merged = [];
    for (const word of words) {
        const prev = merged.at(-1);
        if (
            prev &&
            prev.text === word.text &&
            word.start_time < prev.end_time
        ) {
            const prevDuration = prev.end_time - prev.start_time;
            const nextDuration = word.end_time - word.start_time;
            if (nextDuration > prevDuration) {
                merged[merged.length - 1] = word;
            }
            continue;
        }
        merged.push(word);
    }
    return merged;
}

function dedupeMergedTokens(tokens) {
    /** @type {typeof tokens} */
    const merged = [];
    for (const token of tokens) {
        const prev = merged.at(-1);
        if (
            prev &&
            prev.id === token.id &&
            prev.raw_token === token.raw_token &&
            token.start_time < prev.end_time
        ) {
            const prevDuration = prev.end_time - prev.start_time;
            const nextDuration = token.end_time - token.start_time;
            if (nextDuration > prevDuration) {
                merged[merged.length - 1] = token;
            }
            continue;
        }
        merged.push(token);
    }
    return merged;
}

/**
 * @param {any} tokenizer
 * @param {Array<{ window: { start_s: number, end_s: number, left_stride_s: number, right_stride_s: number }, output: { words?: any[], tokens?: any[] } }>} windowResults
 * @returns {{ text: string, tokens: any[], words: any[], utterance_timestamp: [number, number] | null }}
 */
export function mergeNemoWindowResults(tokenizer, windowResults) {
    /** @type {Array<{ id: number, token: string, raw_token: string, is_word_start: boolean, start_time: number, end_time: number, confidence?: number }>} */
    const mergedTokens = [];
    /** @type {Array<{ text: string, start_time: number, end_time: number, confidence?: number }>} */
    const mergedWords = [];

    for (const { window, output } of windowResults) {
        const keep_start_s = window.start_s + window.left_stride_s;
        const keep_end_s = window.end_s - window.right_stride_s;
        const is_first_window = window.left_stride_s === 0;
        const is_last_window = window.right_stride_s === 0;

        for (const token of output.tokens ?? []) {
            if (
                shouldKeepTimedItem(
                    token.start_time,
                    token.end_time,
                    keep_start_s,
                    keep_end_s,
                    is_first_window,
                    is_last_window,
                )
            ) {
                mergedTokens.push(token);
            }
        }

        for (const word of output.words ?? []) {
            if (
                shouldKeepTimedItem(
                    word.start_time,
                    word.end_time,
                    keep_start_s,
                    keep_end_s,
                    is_first_window,
                    is_last_window,
                )
            ) {
                mergedWords.push(word);
            }
        }
    }

    const tokens = dedupeMergedTokens(mergedTokens);
    const words = dedupeMergedWords(mergedWords);
    const text =
        words.length > 0
            ? joinTimedWords(words)
            : tokens.length > 0 && typeof tokenizer?.decode === 'function'
              ? decodeTransducerText(tokenizer, tokens.map((token) => token.id))
              : '';
    const utterance_timestamp =
        words.length > 0
            ? /** @type {[number, number]} */ ([words[0].start_time, words[words.length - 1].end_time])
            : tokens.length > 0
              ? /** @type {[number, number]} */ ([tokens[0].start_time, tokens[tokens.length - 1].end_time])
              : null;

    return { text, tokens, words, utterance_timestamp };
}
