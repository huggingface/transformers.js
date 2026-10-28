import { Processor } from '../../processing_utils.js';
import { AutoTokenizer } from '../auto/tokenization_auto.js';

// Until tokenizers.js implements Unigram byte fallback, expand unknown pieces
// before they are converted to IDs. Limit the workaround to this processor.
// TODO: remove this adapter once https://github.com/huggingface/tokenizers.js/issues/26 is fixed.
function encode(tokenizer, text) {
    return tokenizer.tokenize(text).flatMap((piece) => {
        const id = tokenizer.convert_tokens_to_ids(piece);
        if (id != null) return [id];
        const bytes = tokenizer.convert_tokens_to_ids(
            Array.from(new TextEncoder().encode(piece), (b) => `<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`),
        );
        return bytes.every((id) => id != null) ? bytes : [tokenizer.unk_token_id];
    });
}

/**
 * Text processor for PocketTTS-style models such as Gradium Phonon. Port of Python's `PocketTTSProcessor`
 * text preparation: `split_text()` and the per-input `frames_after_eos` of its `__call__`.
 */
export class PocketTTSProcessor extends Processor {
    static tokenizer_class = AutoTokenizer;
    static uses_processor_config = true;

    /**
     * Split text into sentence-aligned chunks of at most `max_tokens` tokens, as Python's `PocketTTSProcessor.split_text`.
     * @param {string} text The text to split.
     * @param {number} [max_tokens=50] Maximum number of tokens per chunk (a soft limit).
     * @returns {string[]} The text chunks, in order.
     */
    split_text(text, max_tokens = 50) {
        if (typeof text !== 'string' || !text.trim()) {
            throw new Error('PocketTTSProcessor expects a non-empty text string.');
        }
        // Split the whole text after sentence ends (not decimal points), split long sentences after
        // clause ends, then group greedily up to `max_tokens` (a soft limit).
        const tokens = encode(this.tokenizer, text.trim());
        const boundary_ids = (pieces) =>
            new Set(
                /** @type {number[]} */ (this.tokenizer.convert_tokens_to_ids(pieces)).filter(
                    (id) => Number.isInteger(id) && id !== this.tokenizer.unk_token_id,
                ),
            );
        const sentence_ends = boundary_ids(['.', '!', '...', '?']);
        const clause_ends = boundary_ids([',', ';', ':']);
        // Five tokens cover a four-byte Unicode digit plus punctuation/space.
        const is_decimal_point = (ids, i) =>
            /\p{Nd}\.$/u.test(this.tokenizer.decode(ids.slice(Math.max(0, i - 5), i))) &&
            /^\p{Nd}/u.test(this.tokenizer.decode(ids.slice(i, i + 5)));
        const split_after = (ids, boundaries) => {
            const segments = [];
            let start = 0;
            for (let i = 1; i < ids.length; ++i) {
                if (!boundaries.has(ids[i - 1]) || boundaries.has(ids[i])) continue;
                if (boundaries === sentence_ends && is_decimal_point(ids, i)) continue;
                segments.push(ids.slice(start, i));
                start = i;
            }
            segments.push(ids.slice(start));
            return segments;
        };
        const segments = split_after(tokens, sentence_ends).flatMap((sentence) =>
            sentence.length > max_tokens ? split_after(sentence, clause_ends) : [sentence],
        );
        const chunks = [];
        let chunk = [];
        let chunk_length = 0;
        const push = () => {
            const decoded = chunk.join(' ').trim();
            if (decoded) chunks.push(decoded);
        };
        for (const segment of segments) {
            if (chunk_length && chunk_length + segment.length > max_tokens) {
                push();
                chunk = [];
                chunk_length = 0;
            }
            // Python joins decoded pieces with spaces before re-encoding a chunk,
            // even when the original text had no space after punctuation.
            chunk.push(this.tokenizer.decode(segment));
            chunk_length += segment.length;
        }
        if (chunk_length) push();
        return chunks;
    }

    /**
     * Prepare text as generation inputs, one per `split_text()` chunk. Unlike Python's `__call__`, which takes
     * chat conversations, this takes plain text; each chunk is encoded as Python's chat template encodes it.
     * @param {string} text The text to speak.
     * @param {Object} [options]
     * @param {number} [options.max_tokens=50] Maximum number of tokens per chunk.
     * @returns {Promise<{ input_ids: number[], frames_after_eos: number }[]>} Inputs for `generate_speech()`.
     */
    async _call(text, { max_tokens = 50 } = {}) {
        const { short_text_max_words = 4, frames_after_eos_short_text = 4, frames_after_eos = 2 } = this.config;
        return this.split_text(text, max_tokens).map((chunk) => ({
            input_ids: encode(this.tokenizer, chunk),
            frames_after_eos:
                chunk.split(/\s+/u).length <= short_text_max_words ? frames_after_eos_short_text : frames_after_eos,
        }));
    }
}
