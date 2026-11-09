import { PreTrainedTokenizer } from '../../tokenization_utils.js';
import { Tensor } from '../../utils/tensor.js';

/**
 * Empty padding value for word labels.
 * @type {number}
 */
const DEFAULT_PAD_TOKEN_LABEL = -100;

/**
 * Tokenizer for LayoutLMv3. Unlike most other tokenizers, LayoutLMv3 expects
 * pre-tokenized words together with normalized bounding boxes, since OCR is
 * performed before tokenization.
 *
 * The tokenizer follows the RoBERTa byte-level BPE scheme: each word is
 * tokenized independently with a leading space, and special tokens (`<s>` and
 * `</s>`) are added around the sequence.
 */
export class LayoutLMv3Tokenizer extends PreTrainedTokenizer {
    /**
     * Create a new LayoutLMv3Tokenizer instance.
     * @param {Object} tokenizerJSON The JSON of the tokenizer.
     * @param {Object} tokenizerConfig The config of the tokenizer.
     */
    constructor(tokenizerJSON, tokenizerConfig) {
        super(tokenizerJSON, tokenizerConfig);

        const cls_token = tokenizerConfig.cls_token ?? tokenizerConfig.bos_token;
        this.cls_token = typeof cls_token === 'object' && cls_token !== null ? cls_token.content : cls_token;
        this.cls_token_id = this._tokenizer.token_to_id(this.cls_token);

        this.cls_token_box = tokenizerConfig.cls_token_box ?? [0, 0, 0, 0];
        this.sep_token_box = tokenizerConfig.sep_token_box ?? [0, 0, 0, 0];
        this.pad_token_box = tokenizerConfig.pad_token_box ?? [0, 0, 0, 0];
        this.pad_token_label = tokenizerConfig.pad_token_label ?? DEFAULT_PAD_TOKEN_LABEL;
        this.only_label_first_subword = tokenizerConfig.only_label_first_subword ?? true;
    }

    /**
     * Tokenize a list of words and attach a bounding box and optional label to
     * each generated subword token.
     * @param {string[]} words The words to tokenize.
     * @param {number[][]|null} boxes The bounding boxes corresponding to the words.
     * @param {number[]|null} word_labels The labels corresponding to the words.
     * @returns {{ ids: number[], boxes: number[][], labels: number[] }} The tokenized word sequence.
     * @private
     */
    _tokenize_words(words, boxes, word_labels) {
        /** @type {number[]} */
        const ids = [];
        /** @type {number[][]} */
        const token_boxes = [];
        /** @type {number[]} */
        const labels = [];

        for (let i = 0; i < words.length; ++i) {
            const word = words[i];
            const box = boxes?.[i] ?? this.pad_token_box;
            const { ids: word_ids } = this._tokenizer.encode(word, { add_special_tokens: false });

            for (let j = 0; j < word_ids.length; ++j) {
                ids.push(word_ids[j]);
                token_boxes.push(box);
                if (word_labels !== null && word_labels !== undefined) {
                    const label = word_labels[i] ?? this.pad_token_label;
                    labels.push(j === 0 || !this.only_label_first_subword ? label : this.pad_token_label);
                }
            }
        }

        return { ids, boxes: token_boxes, labels };
    }

    /**
     * Build a single encoded example from pre-tokenized words.
     * @param {Object} example The example to encode.
     * @param {string[]|null} example.text_words The words of the first sequence (only used for pairs).
     * @param {string[]} example.words The words of the document sequence.
     * @param {number[][]|null} example.boxes The bounding boxes of the document words.
     * @param {number[]|null} example.word_labels The labels of the document words.
     * @param {boolean} add_special_tokens Whether to add the special tokens.
     * @returns {{ input_ids: number[], attention_mask: number[], bbox: number[][], labels?: number[], first_token_count: number, second_token_count: number }} The encoded example.
     * @private
     */
    _encode_example({ text_words, words, boxes, word_labels }, add_special_tokens) {
        const first = text_words === null ? null : this._tokenize_words(text_words, null, null);
        const second = this._tokenize_words(words, boxes, word_labels);

        /** @type {number[]} */
        const input_ids = [];
        /** @type {number[][]} */
        const bbox = [];
        /** @type {number[]} */
        const labels = [];
        const has_labels = word_labels !== null && word_labels !== undefined;

        if (add_special_tokens) {
            input_ids.push(this.cls_token_id);
            bbox.push(this.cls_token_box);
            if (has_labels) labels.push(this.pad_token_label);
        }

        if (first !== null) {
            input_ids.push(...first.ids);
            bbox.push(...first.boxes);
            if (has_labels) labels.push(...new Array(first.ids.length).fill(this.pad_token_label));

            if (add_special_tokens) {
                input_ids.push(this.sep_token_id, this.sep_token_id);
                bbox.push(this.sep_token_box, this.sep_token_box);
                if (has_labels) labels.push(this.pad_token_label, this.pad_token_label);
            }
        }

        input_ids.push(...second.ids);
        bbox.push(...second.boxes);
        if (has_labels) labels.push(...second.labels);

        if (add_special_tokens) {
            input_ids.push(this.sep_token_id);
            bbox.push(this.sep_token_box);
            if (has_labels) labels.push(this.pad_token_label);
        }

        const encoded = {
            input_ids,
            attention_mask: new Array(input_ids.length).fill(1),
            bbox,
            first_token_count: first === null ? 0 : first.ids.length,
            second_token_count: second.ids.length,
        };
        if (has_labels) {
            encoded.labels = labels;
        }
        return encoded;
    }

    /**
     * Tokenize one or more pre-tokenized inputs with word-level bounding boxes.
     *
     * @param {string[]|string[][]|string} text The words of one example, a batch of words, or (when `text_pair` is given) the first sequence.
     * @param {Object} [options] Additional tokenization options.
     * @param {string[]|string[][]|null} [options.text_pair=null] The words of the document sequence when `text` is the first sequence.
     * @param {number[][]|number[][][]} [options.boxes] The bounding boxes corresponding to the words.
     * @param {number[]|number[][]|null} [options.word_labels=null] The labels corresponding to the words.
     * @param {boolean} [options.add_special_tokens=true] Whether to add the special tokens.
     * @param {boolean|'max_length'} [options.padding=false] Whether to pad the input sequences.
     * @param {boolean|null} [options.truncation=null] Whether to truncate the input sequences.
     * @param {number|null} [options.max_length=null] The maximum length of the returned sequences.
     * @param {boolean} [options.return_tensor=true] Whether to return the results as Tensors or arrays.
     * @returns {Object} The encoded inputs.
     */
    // @ts-expect-error The LayoutLMv3 tokenizer requires word-level inputs and bounding boxes.
    _call(
        text,
        {
            text_pair = null,
            boxes = null,
            word_labels = null,
            add_special_tokens = true,
            padding = false,
            truncation = null,
            max_length = null,
            return_tensor = true,
        } = {},
    ) {
        if (boxes === null || boxes === undefined) {
            throw new Error('You must provide corresponding bounding boxes.');
        }

        /** @type {{ text_words: string[]|null, words: string[], boxes: number[][], word_labels: number[]|null }[]} */
        let examples;
        let is_batched;

        if (text_pair !== null && text_pair !== undefined) {
            if (typeof text === 'string') {
                is_batched = false;
                examples = [
                    {
                        text_words: text.trim() === '' ? [] : text.trim().split(/\s+/),
                        words: /** @type {string[]} */ (text_pair),
                        boxes: /** @type {number[][]} */ (boxes),
                        word_labels: /** @type {number[]|null} */ (word_labels),
                    },
                ];
            } else if (Array.isArray(text)) {
                if (!Array.isArray(text_pair) || text.length !== text_pair.length) {
                    throw new Error('text and text_pair must have the same length.');
                }
                if (!Array.isArray(boxes) || text.length !== boxes.length) {
                    throw new Error('You must provide words and boxes for an equal amount of examples.');
                }
                is_batched = true;
                examples = text.map((first_sequence, i) => {
                    const words = text_pair[i];
                    if (!Array.isArray(words)) {
                        throw new Error(
                            'Words must be of type `List[str]` (single pretokenized example), or `List[List[str]]` (batch of pretokenized examples).',
                        );
                    }
                    return {
                        text_words: Array.isArray(first_sequence)
                            ? first_sequence
                            : String(first_sequence).trim() === ''
                              ? []
                              : String(first_sequence).trim().split(/\s+/),
                        words,
                        boxes: /** @type {number[][][]} */ (boxes)[i],
                        word_labels:
                            word_labels === null || word_labels === undefined
                                ? null
                                : /** @type {number[][]} */ (word_labels)[i],
                    };
                });
            } else {
                throw new Error(
                    'text input must be of type `str` (single example) or `List[str]` (batch of examples).',
                );
            }
        } else {
            if (!Array.isArray(text)) {
                throw new Error(
                    'Words must be of type `List[str]` (single pretokenized example), or `List[List[str]]` (batch of pretokenized examples).',
                );
            }

            is_batched = text.length > 0 && Array.isArray(text[0]);
            if (is_batched) {
                if (!Array.isArray(boxes) || text.length !== boxes.length) {
                    throw new Error('You must provide words and boxes for an equal amount of examples.');
                }
                examples = text.map((words, i) => ({
                    text_words: null,
                    words: /** @type {string[]} */ (words),
                    boxes: /** @type {number[][][]} */ (boxes)[i],
                    word_labels:
                        word_labels === null || word_labels === undefined
                            ? null
                            : /** @type {number[][]} */ (word_labels)[i],
                }));
            } else {
                examples = [
                    {
                        text_words: null,
                        words: /** @type {string[]} */ (text),
                        boxes: /** @type {number[][]} */ (boxes),
                        word_labels: /** @type {number[]|null} */ (word_labels),
                    },
                ];
            }
        }

        // Validate that every example has as many boxes as words.
        for (const example of examples) {
            if (!Array.isArray(example.words) || !Array.isArray(example.boxes)) {
                throw new Error('Words and boxes must be arrays.');
            }
            if (example.words.length !== example.boxes.length) {
                throw new Error('You must provide as many words as there are bounding boxes.');
            }
            if (
                example.word_labels !== null &&
                (!Array.isArray(example.word_labels) || example.word_labels.length !== example.words.length)
            ) {
                throw new Error('You must provide as many labels as there are words.');
            }
        }

        const special_token_count = add_special_tokens ? (text_pair !== null && text_pair !== undefined ? 4 : 2) : 0;
        const max_sequence_length = max_length ?? this.model_max_length;

        if (truncation && max_sequence_length < special_token_count) {
            throw new Error('`max_length` must be greater than or equal to the number of special tokens.');
        }

        /** @type {ReturnType<LayoutLMv3Tokenizer['_encode_example']>[]} */
        const encoded_examples = examples.map((example) => {
            const encoded = this._encode_example(example, add_special_tokens);

            if (truncation && encoded.input_ids.length > max_sequence_length) {
                // Special tokens are always kept, so we only truncate word tokens.
                let excess = encoded.input_ids.length - max_sequence_length;
                let first_token_count = encoded.first_token_count;
                let second_token_count = encoded.second_token_count;

                const remove_token = (index) => {
                    encoded.input_ids.splice(index, 1);
                    encoded.attention_mask.splice(index, 1);
                    encoded.bbox.splice(index, 1);
                    if (encoded.labels) {
                        encoded.labels.splice(index, 1);
                    }
                };

                while (excess > 0) {
                    if (second_token_count >= first_token_count && second_token_count > 0) {
                        // Remove the last token of the document sequence.
                        const index =
                            (add_special_tokens ? 1 : 0) +
                            first_token_count +
                            (first_token_count > 0 ? 2 : 0) +
                            second_token_count -
                            1;
                        remove_token(index);
                        --second_token_count;
                    } else if (first_token_count > 0) {
                        // Remove the last token of the first sequence.
                        const index = (add_special_tokens ? 1 : 0) + first_token_count - 1;
                        remove_token(index);
                        --first_token_count;
                    } else {
                        throw new Error('Unable to truncate the input sequence.');
                    }
                    --excess;
                }
            }

            return encoded;
        });

        for (const example of encoded_examples) {
            delete example.first_token_count;
            delete example.second_token_count;
        }

        // Determine the padding length, if applicable.
        let target_length = null;
        if (padding) {
            target_length =
                padding === 'max_length'
                    ? max_sequence_length
                    : Math.max(...encoded_examples.map((example) => example.input_ids.length));
        }

        if (return_tensor && !padding && !truncation) {
            const first_length = encoded_examples[0]?.input_ids.length;
            if (encoded_examples.some((example) => example.input_ids.length !== first_length)) {
                throw new Error(
                    'Unable to create tensor, you should probably activate truncation and/or padding ' +
                        "with 'padding=true' and 'truncation=true' to have batched tensors with the same length.",
                );
            }
        }

        if (padding) {
            for (const example of encoded_examples) {
                const difference = target_length - example.input_ids.length;
                if (difference <= 0) continue;

                const pad_data = {
                    input_ids: new Array(difference).fill(this.pad_token_id),
                    attention_mask: new Array(difference).fill(0),
                    bbox: new Array(difference).fill(this.pad_token_box),
                    labels: example.labels ? new Array(difference).fill(this.pad_token_label) : undefined,
                };

                for (const key of Object.keys(example)) {
                    if (this.padding_side === 'left') {
                        example[key] = [...pad_data[key], ...example[key]];
                    } else {
                        example[key] = [...example[key], ...pad_data[key]];
                    }
                }
            }
        }

        if (return_tensor) {
            const batch_size = encoded_examples.length;
            const sequence_length = encoded_examples[0].input_ids.length;
            const dims = [batch_size, sequence_length];

            /** @type {Record<string, Tensor>} */
            const result = {
                input_ids: new Tensor(
                    'int64',
                    BigInt64Array.from(encoded_examples.flatMap((example) => example.input_ids).map(BigInt)),
                    dims,
                ),
                attention_mask: new Tensor(
                    'int64',
                    BigInt64Array.from(encoded_examples.flatMap((example) => example.attention_mask).map(BigInt)),
                    dims,
                ),
                bbox: new Tensor(
                    'int64',
                    BigInt64Array.from(encoded_examples.flatMap((example) => example.bbox.flat()).map(BigInt)),
                    [...dims, 4],
                ),
            };

            if (encoded_examples[0].labels) {
                result.labels = new Tensor(
                    'int64',
                    BigInt64Array.from(encoded_examples.flatMap((example) => example.labels).map(BigInt)),
                    dims,
                );
            }

            return result;
        }

        /** @type {Record<string, any>} */
        const result = {};
        for (const key of Object.keys(encoded_examples[0])) {
            result[key] = encoded_examples.map((example) => example[key]);
        }

        if (!is_batched) {
            for (const key of Object.keys(result)) {
                result[key] = result[key][0];
            }
        }

        return result;
    }
}
