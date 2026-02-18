/**
 * @file Tokenization utilities
 *
 * @module tokenizers
 */

import { Tokenizer } from '@huggingface/tokenizers';
import { Template } from '@huggingface/jinja';
import { Callable } from './utils/generic.js';

import { isIntegralNumber, mergeArrays } from './utils/core.js';
import { getModelJSON } from './utils/hub.js';
import { max } from './utils/maths.js';
import { Tensor } from './utils/tensor.js';

/**
 * @typedef {import('./utils/hub.js').PretrainedOptions} PretrainedTokenizerOptions
 */

/**
 * Loads a tokenizer from the specified path.
 * @param {string} pretrained_model_name_or_path The path to the tokenizer directory.
 * @param {PretrainedTokenizerOptions} options Additional options for loading the tokenizer.
 * @returns {Promise<any[]>} A promise that resolves with information about the loaded tokenizer.
 */
export async function loadTokenizer(pretrained_model_name_or_path, options) {
    const info = await Promise.all([
        getModelJSON(pretrained_model_name_or_path, 'tokenizer.json', true, options),
        getModelJSON(pretrained_model_name_or_path, 'tokenizer_config.json', true, options),
    ]);
    return info;
}

/**
 * Helper function to convert a tensor to a list before decoding.
 * @param {Tensor} tensor The tensor to convert.
 * @returns {number[]} The tensor as a list.
 */
export function prepareTensorForDecode(tensor) {
    const dims = tensor.dims;
    switch (dims.length) {
        case 1:
            return tensor.tolist();
        case 2:
            if (dims[0] !== 1) {
                throw new Error(
                    'Unable to decode tensor with `batch size !== 1`. Use `tokenizer.batch_decode(...)` for batched inputs.',
                );
            }
            return tensor.tolist()[0];
        default:
            throw new Error(`Expected tensor to have 1-2 dimensions, got ${dims.length}.`);
    }
}

const SPECIAL_TOKEN_ATTRIBUTES = [
    'bos_token',
    'eos_token',
    'unk_token',
    'sep_token',
    'pad_token',
    'cls_token',
    'mask_token',
    // additional_special_tokens (TODO)
];

/**
 * @typedef {Object} Message
 * @property {string} role The role of the message (e.g., "user" or "assistant" or "system").
 * @property {string} content The content of the message.
 */

/**
 *
 * Helper function for padding values of an object, which are each arrays.
 * NOTE: No additional checks are made here for validity of arguments.
 * @param {Record<string, any[]>} item The input object.
 * @param {number} length The length to pad to.
 * @param {(key: string) => any} value_fn Determine the value to fill the array, based on its key.
 * @param {string} side Which side to pad the array.
 * @private
 */
function padHelper(item, length, value_fn, side) {
    for (const key of Object.keys(item)) {
        const diff = length - item[key].length;
        const value = value_fn(key);

        const padData = new Array(diff).fill(value);
        item[key] = side === 'right' ? mergeArrays(item[key], padData) : mergeArrays(padData, item[key]);
    }
}

/**
 * Helper function for truncating values of an object, which are each arrays.
 * NOTE: No additional checks are made here for validity of arguments.
 * @param {Record<string, any[]>} item The input object.
 * @param {number} length The length to truncate to.
 * @private
 */
function truncateHelper(item, length) {
    // Setting .length to a lower value truncates the array in-place:
    // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/length
    for (const key of Object.keys(item)) {
        item[key].length = length;
    }
}

/**
 * Returns the value of the first matching key in the tokenizer config object.
 * @param {Object} config The tokenizer config object.
 * @param {...string} keys One or more keys to search for in the tokenizer config object.
 * @returns {string|null} The value associated with the first matching key, or null if no match is found.
 * @throws {Error} If an object is found for a matching key and its __type property is not "AddedToken".
 * @private
 */
function getTokenFromConfig(config, ...keys) {
    for (const key of keys) {
        if (!Object.hasOwn(config, key)) continue;
        const item = config[key];
        if (!item) continue;

        if (typeof item === 'object') {
            if (item.__type === 'AddedToken') {
                return item.content;
            } else {
                throw Error(`Unknown token: ${item}`);
            }
        } else {
            return item;
        }
    }
    return null;
}

/**
 *
 * @param {import('@huggingface/tokenizers').Tokenizer} tokenizer
 * @returns {import('@huggingface/tokenizers').AddedToken[]}
 * @private
 */
function getSpecialTokens(tokenizer) {
    const special = [];
    for (const value of tokenizer.get_added_tokens_decoder().values()) {
        if (value.special) special.push(value);
    }
    return special;
}

export class PreTrainedTokenizer extends Callable {
    return_token_type_ids = false;

    padding_side = 'right';
    /**
     * Create a new PreTrainedTokenizer instance.
     * @param {Object} tokenizerJSON The JSON of the tokenizer.
     * @param {Object} tokenizerConfig The config of the tokenizer.
     */
    constructor(tokenizerJSON, tokenizerConfig) {
        super();

        this._tokenizerJSON = tokenizerJSON;
        this._tokenizerConfig = tokenizerConfig;
        this._tokenizer = new Tokenizer(tokenizerJSON, tokenizerConfig);

        this.config = tokenizerConfig;

        this.padding_side = tokenizerConfig.padding_side ?? this.padding_side;

        // Set mask token if present (otherwise will be undefined, which is fine)
        this.mask_token = getTokenFromConfig(tokenizerConfig, 'mask_token');
        this.mask_token_id = this._tokenizer.token_to_id(this.mask_token);

        this.pad_token = getTokenFromConfig(tokenizerConfig, 'pad_token', 'eos_token');
        this.pad_token_id = this._tokenizer.token_to_id(this.pad_token);

        this.sep_token = getTokenFromConfig(tokenizerConfig, 'sep_token');
        this.sep_token_id = this._tokenizer.token_to_id(this.sep_token);

        this.unk_token = getTokenFromConfig(tokenizerConfig, 'unk_token');
        this.unk_token_id = this._tokenizer.token_to_id(this.unk_token);

        this.bos_token = getTokenFromConfig(tokenizerConfig, 'bos_token');
        this.bos_token_id = this._tokenizer.token_to_id(this.bos_token);

        this.eos_token = getTokenFromConfig(tokenizerConfig, 'eos_token');
        this.eos_token_id = this._tokenizer.token_to_id(this.eos_token);

        this.chat_template = tokenizerConfig.chat_template ?? null;
        if (Array.isArray(this.chat_template)) {
            // Chat templates are stored as lists of dicts with fixed key names,
            // we reconstruct that into a single dict while loading them.
            const chat_template = Object.create(null);
            for (const { name, template } of this.chat_template) {
                if (typeof name !== 'string' || typeof template !== 'string') {
                    throw new Error('Chat template must be a list of objects with "name" and "template" properties');
                }
                chat_template[name] = template;
            }
            this.chat_template = chat_template;
        }
        this._compiled_template_cache = new Map();

        const special_tokens = getSpecialTokens(this._tokenizer);
        this.all_special_ids = special_tokens.map((t) => t.id);
        this.all_special_tokens = special_tokens.map((t) => t.content);
    }

    /**
     * Loads a pre-trained tokenizer from the given `pretrained_model_name_or_path`.
     *
     * @param {string} pretrained_model_name_or_path The path to the pre-trained tokenizer.
     * @param {PretrainedTokenizerOptions} options Additional options for loading the tokenizer.
     *
     * @throws {Error} Throws an error if the tokenizer.json or tokenizer_config.json files are not found in the `pretrained_model_name_or_path`.
     * @returns {Promise<PreTrainedTokenizer>} A new instance of the `PreTrainedTokenizer` class.
     */
    static async from_pretrained(
        pretrained_model_name_or_path,
        { progress_callback = null, config = null, cache_dir = null, local_files_only = false, revision = 'main' } = {},
    ) {
        const info = await loadTokenizer(pretrained_model_name_or_path, {
            progress_callback,
            config,
            cache_dir,
            local_files_only,
            revision,
        });

        // @ts-ignore
        return new this(...info);
    }

    get_vocab() {
        return this._tokenizer.get_vocab();
    }

    get model_max_length() {
        return this._tokenizerConfig.model_max_length ?? Infinity;
    }

    get add_eos_token() {
        return this._tokenizerConfig.add_eos_token;
    }
    get add_bos_token() {
        return this._tokenizerConfig.add_bos_token;
    }

    /**
     * Converts a token string (or a sequence of tokens) into a single integer id (or a sequence of ids), using the vocabulary.
     *
     * @template {string|string[]} T
     * @param {T} tokens One or several token(s) to convert to token id(s).
     * @returns {T extends string ? number : number[]} The token id or list of token ids.
     */
    convert_tokens_to_ids(tokens) {
        if (typeof tokens === 'string') {
            return /** @type {any} */ (this._tokenizer.token_to_id(tokens));
        } else {
            return /** @type {any} */ (tokens.map((token) => this._tokenizer.token_to_id(token)));
        }
    }

    /**
     * @typedef {number[]|number[][]|Tensor} BatchEncodingItem
     *
     * @typedef {Object} BatchEncoding Holds the output of the tokenizer's call function.
     * @property {BatchEncodingItem} input_ids List of token ids to be fed to a model.
     * @property {BatchEncodingItem} attention_mask List of indices specifying which tokens should be attended to by the model.
     * @property {BatchEncodingItem} [token_type_ids] List of token type ids to be fed to a model.
     */

    /**
     * Encode/tokenize the given text(s).
     * @param {string|string[]} text The text to tokenize.
     * @param {Object} options An optional object containing the following properties:
     * @param {string|string[]} [options.text_pair=null] Optional second sequence to be encoded. If set, must be the same type as text.
     * @param {boolean|'max_length'} [options.padding=false] Whether to pad the input sequences.
     * @param {boolean} [options.add_special_tokens=true] Whether or not to add the special tokens associated with the corresponding model.
     * @param {boolean} [options.truncation=null] Whether to truncate the input sequences.
     * @param {number} [options.max_length=null] Maximum length of the returned list and optionally padding length.
     * @param {boolean} [options.return_tensor=true] Whether to return the results as Tensors or arrays.
     * @param {boolean} [options.return_token_type_ids=null] Whether to return the token type ids.
     * @returns {BatchEncoding} Object to be passed to the model.
     */
    _call(
        // Required positional arguments
        text,

        // Optional keyword arguments
        {
            text_pair = null,
            add_special_tokens = true,
            padding = false,
            truncation = null,
            max_length = null,
            return_tensor = true, // Different to HF
            return_token_type_ids = null,
        } = {},
    ) {
        const isBatched = Array.isArray(text);

        let encodedTokens;

        if (isBatched) {
            if (text.length === 0) {
                throw Error('text array must be non-empty');
            }

            if (text_pair !== null) {
                if (!Array.isArray(text_pair)) {
                    throw Error('text_pair must also be an array');
                } else if (text.length !== text_pair.length) {
                    throw Error('text and text_pair must have the same length');
                }

                encodedTokens = text.map((t, i) =>
                    this._encode_plus(t, { text_pair: text_pair[i], add_special_tokens, return_token_type_ids }),
                );
            } else {
                encodedTokens = text.map((x) => this._encode_plus(x, { add_special_tokens, return_token_type_ids }));
            }
        } else {
            if (text === null || text === undefined) {
                throw Error('text may not be null or undefined');
            }

            if (Array.isArray(text_pair)) {
                throw Error(
                    'When specifying `text_pair`, since `text` is a string, `text_pair` must also be a string (i.e., not an array).',
                );
            }

            // For single input, we just wrap in an array, and then unwrap later.
            encodedTokens = [this._encode_plus(text, { text_pair, add_special_tokens, return_token_type_ids })];
        }
        // At this point, `encodedTokens` is batched, of shape [batch_size, tokens].
        // However, array may be jagged. So, we may need pad to max_length.
        if (max_length === null) {
            max_length = this.model_max_length;
        } else if (truncation === null) {
            if (padding === true) {
                console.warn(
                    '`max_length` is ignored when `padding: true` and there is no truncation strategy. ' +
                        "To pad to max length, use `padding: 'max_length'`.",
                );
                max_length = this.model_max_length;
            } else if (padding === false) {
                console.warn(
                    'Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation: true` to explicitly truncate examples to max length.',
                );
                truncation = true;
            }
        }

        // padding: 'max_length' doesn't require any additional calculation
        // but padding: true has to calculate max_length from the sequences
        if (padding === true) {
            max_length = Math.min(max(encodedTokens.map((x) => x.input_ids.length))[0], max_length ?? Infinity);
        }

        // Ensure it is less than model max length
        max_length = Math.min(max_length, this.model_max_length ?? Infinity);

        if (padding || truncation) {
            // Perform padding and/or truncation
            for (let i = 0; i < encodedTokens.length; ++i) {
                if (encodedTokens[i].input_ids.length === max_length) {
                    continue;
                } else if (encodedTokens[i].input_ids.length > max_length) {
                    // possibly truncate
                    if (truncation) {
                        truncateHelper(encodedTokens[i], max_length);
                    }
                } else {
                    // t.length < max_length
                    // possibly pad
                    if (padding) {
                        padHelper(
                            encodedTokens[i],
                            max_length,
                            (key) => (key === 'input_ids' ? this.pad_token_id : 0),
                            this.padding_side,
                        );
                    }
                }
            }
        }

        const result = {};

        if (return_tensor) {
            if (!(padding && truncation)) {
                // Not, guaranteed that all items have same length, so
                // we perform additional check

                if (
                    encodedTokens.some((x) => {
                        for (const key of Object.keys(x)) {
                            if (x[key].length !== encodedTokens[0][key]?.length) {
                                return true;
                            }
                        }
                        return false;
                    })
                ) {
                    throw Error(
                        'Unable to create tensor, you should probably activate truncation and/or padding ' +
                            "with 'padding=true' and 'truncation=true' to have batched tensors with the same length.",
                    );
                }
            }

            // Now we actually convert to tensor
            // NOTE: In the same way as the python library, we return a batched tensor, regardless of
            // whether we have a single input or multiple inputs.
            const dims = [encodedTokens.length, encodedTokens[0].input_ids.length];

            for (const key of Object.keys(encodedTokens[0])) {
                result[key] = new Tensor(
                    'int64',
                    BigInt64Array.from(encodedTokens.flatMap((x) => x[key]).map(BigInt)),
                    dims,
                );
            }
        } else {
            for (const key of Object.keys(encodedTokens[0])) {
                result[key] = encodedTokens.map((x) => x[key]);
            }

            // If not returning a tensor, we match the input type
            if (!isBatched) {
                // Input was not batched, so we unwrap
                for (const key of Object.keys(result)) {
                    result[key] = result[key][0];
                }
            }
        }

        return /** @type {BatchEncoding} */ (result);
    }

    /**
     * Encodes a single text using the preprocessor pipeline of the tokenizer.
     *
     * @param {string|null} text The text to encode.
     * @returns {string[]|null} The encoded tokens.
     */
    _encode_text(text) {
        if (text === null) return null;
        return this._tokenizer.encode(text).tokens;
    }

    /**
     * Encodes a single text or a pair of texts using the model's tokenizer.
     *
     * @param {string} text The text to encode.
     * @param {Object} options An optional object containing the following properties:
     * @param {string} [options.text_pair=null] The optional second text to encode.
     * @param {boolean} [options.add_special_tokens=true] Whether or not to add the special tokens associated with the corresponding model.
     * @param {boolean} [options.return_token_type_ids=null] Whether to return token_type_ids.
     * @returns {{input_ids: number[], attention_mask: number[], token_type_ids?: number[]}} An object containing the encoded text.
     * @private
     */
    _encode_plus(text, { text_pair = null, add_special_tokens = true, return_token_type_ids = null } = {}) {
        const { ids, attention_mask, token_type_ids } = this._tokenizer.encode(text, {
            text_pair,
            add_special_tokens,
            return_token_type_ids: return_token_type_ids ?? this.return_token_type_ids,
        });
        return {
            input_ids: ids,
            attention_mask,
            ...(token_type_ids ? { token_type_ids } : {}),
        };
    }

    /**
     * Converts a string into a sequence of tokens.
     * @param {string} text The sequence to be encoded.
     * @param {Object} options An optional object containing the following properties:
     * @param {string} [options.pair] A second sequence to be encoded with the first.
     * @param {boolean} [options.add_special_tokens=false] Whether or not to add the special tokens associated with the corresponding model.
     * @returns {string[]} The list of tokens.
     */
    tokenize(text, { pair = null, add_special_tokens = false } = {}) {
        return this._tokenizer.tokenize(text, { text_pair: pair, add_special_tokens });
    }

    /**
     * Encodes a single text or a pair of texts using the model's tokenizer.
     *
     * @param {string} text The text to encode.
     * @param {Object} options An optional object containing the following properties:
     * @param {string} [options.text_pair=null] The optional second text to encode.
     * @param {boolean} [options.add_special_tokens=true] Whether or not to add the special tokens associated with the corresponding model.
     * @param {boolean} [options.return_token_type_ids=null] Whether to return token_type_ids.
     * @returns {number[]} An array of token IDs representing the encoded text(s).
     */
    encode(text, { text_pair = null, add_special_tokens = true, return_token_type_ids = null } = {}) {
        return this._tokenizer.encode(text, {
            text_pair,
            add_special_tokens,
            return_token_type_ids,
        }).ids;
    }

    /**
     * Decode a batch of tokenized sequences.
     * @param {number[][]|Tensor} batch List/Tensor of tokenized input sequences.
     * @param {Object} decode_args (Optional) Object with decoding arguments.
     * @returns {string[]} List of decoded sequences.
     */
    batch_decode(batch, decode_args = {}) {
        if (batch instanceof Tensor) {
            batch = batch.tolist();
        }
        return batch.map((x) => this.decode(x, decode_args));
    }

    /**
     * Decodes a sequence of token IDs back to a string.
     *
     * @param {number[]|bigint[]|Tensor} token_ids List/Tensor of token IDs to decode.
     * @param {Object} [decode_args={}]
     * @param {boolean} [decode_args.skip_special_tokens=false] If true, special tokens are removed from the output string.
     * @param {boolean} [decode_args.clean_up_tokenization_spaces=true] If true, spaces before punctuations and abbreviated forms are removed.
     *
     * @returns {string} The decoded string.
     * @throws {Error} If `token_ids` is not a non-empty array of integers.
     */
    decode(token_ids, decode_args = {}) {
        if (token_ids instanceof Tensor) {
            token_ids = prepareTensorForDecode(token_ids);
        }

        if (!Array.isArray(token_ids) || token_ids.length === 0 || !isIntegralNumber(token_ids[0])) {
            throw Error('token_ids must be a non-empty array of integers.');
        }

        return this.decode_single(token_ids, decode_args);
    }

    /**
     * Decode a single list of token ids to a string.
     * @param {number[]|bigint[]} token_ids List of token ids to decode
     * @param {Object} decode_args Optional arguments for decoding
     * @param {boolean} [decode_args.skip_special_tokens=false] Whether to skip special tokens during decoding
     * @param {boolean} [decode_args.clean_up_tokenization_spaces=null] Whether to clean up tokenization spaces during decoding.
     * If null, the value is set to `this.decoder.cleanup` if it exists, falling back to `this.clean_up_tokenization_spaces` if it exists, falling back to `true`.
     * @returns {string} The decoded string
     */
    decode_single(token_ids, { skip_special_tokens = false, clean_up_tokenization_spaces = null }) {
        return this._tokenizer.decode(token_ids, {
            skip_special_tokens,
            clean_up_tokenization_spaces,
        });
    }

    /**
     * Retrieve the chat template string used for tokenizing chat messages. This template is used
     * internally by the `apply_chat_template` method and can also be used externally to retrieve the model's chat
     * template for better generation tracking.
     *
     * @param {Object} options An optional object containing the following properties:
     * @param {string} [options.chat_template=null]
     * A Jinja template or the name of a template to use for this conversion.
     * It is usually not necessary to pass anything to this argument,
     * as the model's template will be used by default.
     * @param {Object[]} [options.tools=null]
     * A list of tools (callable functions) that will be accessible to the model. If the template does not
     * support function calling, this argument will have no effect. Each tool should be passed as a JSON Schema,
     * giving the name, description and argument types for the tool. See our
     * [chat templating guide](https://huggingface.co/docs/transformers/main/en/chat_templating#automated-function-conversion-for-tool-use)
     * for more information.
     * @returns {string} The chat template string.
     */
    get_chat_template({ chat_template = null, tools = null } = {}) {
        // First, handle the cases when the model has a dict of multiple templates
        if (this.chat_template && typeof this.chat_template === 'object') {
            const template_dict = this.chat_template;

            if (chat_template !== null && Object.hasOwn(template_dict, chat_template)) {
                // The user can pass the name of a template to the chat template argument instead of an entire template
                chat_template = template_dict[chat_template];
            } else if (chat_template === null) {
                if (tools !== null && 'tool_use' in template_dict) {
                    chat_template = template_dict['tool_use'];
                } else if ('default' in template_dict) {
                    chat_template = template_dict['default'];
                } else {
                    throw Error(
                        `This model has multiple chat templates with no default specified! Please either pass a chat ` +
                            `template or the name of the template you wish to use to the 'chat_template' argument. Available ` +
                            `template names are ${Object.keys(template_dict).sort()}.`,
                    );
                }
            }
        } else if (chat_template === null) {
            // These are the cases when the model has a single template
            // priority: `chat_template` argument > `tokenizer.chat_template`
            if (this.chat_template) {
                chat_template = this.chat_template;
            } else {
                throw Error(
                    'Cannot use apply_chat_template() because tokenizer.chat_template is not set and no template ' +
                        'argument was passed! For information about writing templates and setting the ' +
                        'tokenizer.chat_template attribute, please see the documentation at ' +
                        'https://huggingface.co/docs/transformers/main/en/chat_templating',
                );
            }
        }
        return chat_template;
    }

    /**
     * Converts a list of message objects with `"role"` and `"content"` keys to a list of token
     * ids. This method is intended for use with chat models, and will read the tokenizer's chat_template attribute to
     * determine the format and control tokens to use when converting.
     *
     * See [here](https://huggingface.co/docs/transformers/chat_templating) for more information.
     *
     * **Example:** Applying a chat template to a conversation.
     *
     * ```javascript
     * import { AutoTokenizer } from "@huggingface/transformers";
     *
     * const tokenizer = await AutoTokenizer.from_pretrained("Xenova/mistral-tokenizer-v1");
     *
     * const chat = [
     *   { "role": "user", "content": "Hello, how are you?" },
     *   { "role": "assistant", "content": "I'm doing great. How can I help you today?" },
     *   { "role": "user", "content": "I'd like to show off how chat templating works!" },
     * ]
     *
     * const text = tokenizer.apply_chat_template(chat, { tokenize: false });
     * // "<s>[INST] Hello, how are you? [/INST]I'm doing great. How can I help you today?</s> [INST] I'd like to show off how chat templating works! [/INST]"
     *
     * const input_ids = tokenizer.apply_chat_template(chat, { tokenize: true, return_tensor: false });
     * // [1, 733, 16289, 28793, 22557, 28725, 910, 460, 368, 28804, 733, 28748, 16289, 28793, 28737, 28742, 28719, 2548, 1598, 28723, 1602, 541, 315, 1316, 368, 3154, 28804, 2, 28705, 733, 16289, 28793, 315, 28742, 28715, 737, 298, 1347, 805, 910, 10706, 5752, 1077, 3791, 28808, 733, 28748, 16289, 28793]
     * ```
     *
     * @param {Message[]} conversation A list of message objects with `"role"` and `"content"` keys,
     * representing the chat history so far.
     * @param {Object} options An optional object containing the following properties:
     * @param {string} [options.chat_template=null] A Jinja template to use for this conversion. If
     * this is not passed, the model's chat template will be used instead.
     * @param {Object[]} [options.tools=null]
     * A list of tools (callable functions) that will be accessible to the model. If the template does not
     * support function calling, this argument will have no effect. Each tool should be passed as a JSON Schema,
     * giving the name, description and argument types for the tool. See our
     * [chat templating guide](https://huggingface.co/docs/transformers/main/en/chat_templating#automated-function-conversion-for-tool-use)
     * for more information.
     * @param {Record<string, string>[]} [options.documents=null]
     * A list of dicts representing documents that will be accessible to the model if it is performing RAG
     * (retrieval-augmented generation). If the template does not support RAG, this argument will have no
     * effect. We recommend that each document should be a dict containing "title" and "text" keys. Please
     * see the RAG section of the [chat templating guide](https://huggingface.co/docs/transformers/main/en/chat_templating#arguments-for-RAG)
     * for examples of passing documents with chat templates.
     * @param {boolean} [options.add_generation_prompt=false] Whether to end the prompt with the token(s) that indicate
     * the start of an assistant message. This is useful when you want to generate a response from the model.
     * Note that this argument will be passed to the chat template, and so it must be supported in the
     * template for this argument to have any effect.
     * @param {boolean} [options.tokenize=true] Whether to tokenize the output. If false, the output will be a string.
     * @param {boolean} [options.padding=false] Whether to pad sequences to the maximum length. Has no effect if tokenize is false.
     * @param {boolean} [options.truncation=false] Whether to truncate sequences to the maximum length. Has no effect if tokenize is false.
     * @param {number} [options.max_length=null] Maximum length (in tokens) to use for padding or truncation. Has no effect if tokenize is false.
     * If not specified, the tokenizer's `max_length` attribute will be used as a default.
     * @param {boolean} [options.return_tensor=true] Whether to return the output as a Tensor or an Array. Has no effect if tokenize is false.
     * @param {boolean} [options.return_dict=true] Whether to return a dictionary with named outputs. Has no effect if tokenize is false.
     * @param {Object} [options.tokenizer_kwargs={}] Additional options to pass to the tokenizer.
     * @returns {string | Tensor | number[]| number[][]|BatchEncoding} The tokenized output.
     */
    apply_chat_template(
        conversation,
        {
            tools = null,
            documents = null,
            chat_template = null,
            add_generation_prompt = false,
            tokenize = true,
            padding = false,
            truncation = false,
            max_length = null,
            return_tensor = true,
            return_dict = true,
            tokenizer_kwargs = {},
            ...kwargs
        } = {},
    ) {
        chat_template = this.get_chat_template({ chat_template, tools });

        if (typeof chat_template !== 'string') {
            throw Error(`chat_template must be a string, but got ${typeof chat_template}`);
        }

        // Compilation function uses a cache to avoid recompiling the same template
        /** @type {import('@huggingface/jinja').Template} */
        let compiledTemplate = this._compiled_template_cache.get(chat_template);
        if (compiledTemplate === undefined) {
            compiledTemplate = new Template(chat_template);
            this._compiled_template_cache.set(chat_template, compiledTemplate);
        }

        const special_tokens_map = Object.create(null);
        for (const key of SPECIAL_TOKEN_ATTRIBUTES) {
            const value = getTokenFromConfig(this.config, key);
            if (value) {
                special_tokens_map[key] = value;
            }
        }

        const rendered = compiledTemplate.render({
            messages: conversation,
            add_generation_prompt,
            tools,
            documents,
            ...special_tokens_map,
            ...kwargs,
        });

        if (tokenize) {
            const out = this._call(rendered, {
                add_special_tokens: false,
                padding,
                truncation,
                max_length,
                return_tensor,
                ...tokenizer_kwargs,
            });
            return return_dict ? out : out.input_ids;
        }

        return rendered;
    }
}

/**
 * Helper function to build translation inputs for an `NllbTokenizer` or `M2M100Tokenizer`.
 * @param {PreTrainedTokenizer} self The tokenizer instance.
 * @param {string|string[]} raw_inputs The text to tokenize.
 * @param {Object} tokenizer_options Options to be sent to the tokenizer
 * @param {Object} generate_kwargs Generation options.
 * @returns {Object} Object to be passed to the model.
 */
export function _build_translation_inputs(self, raw_inputs, tokenizer_options, generate_kwargs) {
    if (!('language_codes' in self) || !Array.isArray(self.language_codes)) {
        throw new Error(
            'Tokenizer must have `language_codes` attribute set and it should be an array of language ids.',
        );
    }
    if (!('languageRegex' in self) || !(self.languageRegex instanceof RegExp)) {
        throw new Error('Tokenizer must have `languageRegex` attribute set and it should be a regular expression.');
    }
    if (!('lang_to_token' in self) || typeof self.lang_to_token !== 'function') {
        throw new Error('Tokenizer must have `lang_to_token` attribute set and it should be a function.');
    }
    const src_lang_token = generate_kwargs.src_lang;
    const tgt_lang_token = generate_kwargs.tgt_lang;

    // Check that the target language is valid:
    if (!self.language_codes.includes(tgt_lang_token)) {
        throw new Error(
            `Target language code "${tgt_lang_token}" is not valid. Must be one of: {${self.language_codes.join(', ')}}`,
        );
    }

    // Allow `src_lang` to be optional. If not set, we'll use the tokenizer's default.
    if (src_lang_token !== undefined) {
        // Check that the source language is valid:
        if (!self.language_codes.includes(src_lang_token)) {
            throw new Error(
                `Source language code "${src_lang_token}" is not valid. Must be one of: {${self.language_codes.join(', ')}}`,
            );
        }

        // In the same way as the Python library, we override the post-processor
        // to force the source language to be first:
        for (const item of self._tokenizer.post_processor.config.single) {
            if ('SpecialToken' in item && self.languageRegex.test(item.SpecialToken.id)) {
                item.SpecialToken.id = self.lang_to_token(src_lang_token);
                break;
            }
        }
        // TODO: Do the same for pair?
    }

    // Override the `forced_bos_token_id` to force the correct language
    generate_kwargs.forced_bos_token_id = self._tokenizer.token_to_id(self.lang_to_token(tgt_lang_token));

    return self._call(raw_inputs, tokenizer_options);
}
