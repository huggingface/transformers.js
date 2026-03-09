import { buildTransducerWordOffsets } from './transducer_word_offsets.js';

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
 *  words: Array<{ text: string, startTime: number, endTime: number, confidence?: number }>,
 *  tokens: Array<{ id: number, token: string, rawToken: string, isWordStart: boolean, startTime: number, endTime: number, confidence?: number }>,
 *  wordAverage: number | null,
 * }}
 */
export function buildTransducerDetailedOutputs(tokenizer, token_ids, token_timestamps, token_confidences = null) {
    const fullText = decodeTransducerText(tokenizer, token_ids);
    return buildTransducerWordOffsets(tokenizer, token_ids, token_timestamps, token_confidences, fullText);
}
