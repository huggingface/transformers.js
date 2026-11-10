/**
 * @module processors
 */
import { Processor } from '../../processing_utils.js';
import { AutoTokenizer } from '../auto/tokenization_auto.js';
import { IMAGE_PROCESSOR_NAME } from '../../utils/constants.js';
import { getModelJSON } from '../../utils/hub.js';
import { Tensor, full } from '../../utils/tensor.js';

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
 * Prepares Pocket TTS prompts and precomputed voices, and splits long text.
 */
export class PocketTTSProcessor extends Processor {
    static classes = ['tokenizer_class'];
    static tokenizer_class = AutoTokenizer;

    /** @type {typeof Processor.from_pretrained} */
    static async from_pretrained(pretrained_model_name_or_path, options = {}) {
        const [config, tokenizer] = await Promise.all([
            getModelJSON(pretrained_model_name_or_path, IMAGE_PROCESSOR_NAME, true, options),
            this.tokenizer_class.from_pretrained(pretrained_model_name_or_path, options),
        ]);
        return new this(config, { tokenizer }, null);
    }

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
        let chunk = '',
            chunk_length = 0;
        for (const segment of segments) {
            if (chunk_length && chunk_length + segment.length > max_tokens) {
                chunks.push(chunk.trim());
                chunk = '';
                chunk_length = 0;
            }
            // Python joins decoded pieces with spaces before re-encoding a chunk,
            // even when the original text had no space after punctuation.
            chunk += (chunk ? ' ' : '') + this.tokenizer.decode(segment);
            chunk_length += segment.length;
        }
        if (chunk) chunks.push(chunk.trim());
        return chunks;
    }

    /**
     * Prepare a rendered chat prompt and a precomputed voice for generation.
     * The current ONNX export accepts one prompt and one voice per call.
     * @param {string} text Prompt from `apply_chat_template(..., { add_generation_prompt: true })`.
     * @param {Object} [options]
     * @param {Tensor[]} [options.audio_prompt_latents] One voice tensor of shape `[frames, audio_prompt_dim]`.
     * @returns {Promise<Object>} `input_ids`, `attention_mask`, voice latents and masks, and `frames_after_eos`.
     */
    async _call(text, { audio_prompt_latents = [] } = {}) {
        if (typeof text !== 'string' || !text.endsWith('<|audio_bos|>')) {
            throw new Error('Prepare one prompt with apply_chat_template and add_generation_prompt=true.');
        }
        const placeholders = text.match(/<\|audio_prompt\|>/g) ?? [];
        if (placeholders.length !== audio_prompt_latents.length || placeholders.length > 1) {
            throw new Error('Pass one audio_prompt_latents tensor per voice placeholder (at most one).');
        }
        const voice = audio_prompt_latents[0];
        if (voice && (voice.type !== 'float32' || voice.dims.length !== 2 || !voice.size)) {
            throw new Error('audio_prompt_latents must contain a float32 tensor of shape [frames, audio_prompt_dim].');
        }
        const prompt = text.replace('<|audio_prompt|>', '<|audio_prompt|>'.repeat(voice?.dims[0] ?? 0));
        const ids = encode(this.tokenizer, prompt);
        const num_words = this.tokenizer.decode(ids, { skip_special_tokens: true }).trim().split(/\s+/u).length;
        const { short_text_max_words = 4, frames_after_eos_short_text = 4, frames_after_eos = 2 } = this.config;
        return {
            input_ids: new Tensor('int64', ids, [1, ids.length]),
            attention_mask: full([1, ids.length], 1n),
            ...(voice && {
                audio_prompt_latents: voice.view(1, ...voice.dims),
                audio_prompt_latents_mask: full([1, voice.dims[0]], 1n),
            }),
            frames_after_eos: new Tensor(
                'int64',
                [num_words <= short_text_max_words ? frames_after_eos_short_text : frames_after_eos],
                [1],
            ),
        };
    }
}
