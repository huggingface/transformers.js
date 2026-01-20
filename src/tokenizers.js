/**
 * @file Tokenizers are used to prepare textual inputs for a model.
 *
 * **Example:** Create an `AutoTokenizer` and use it to tokenize a sentence.
 * This will automatically detect the tokenizer type based on the tokenizer class defined in `tokenizer.json`.
 * ```javascript
 * import { AutoTokenizer } from '@huggingface/transformers';
 *
 * const tokenizer = await AutoTokenizer.from_pretrained('Xenova/bert-base-uncased');
 * const { input_ids } = await tokenizer('I love transformers!');
 * // Tensor {
 * //   data: BigInt64Array(6) [101n, 1045n, 2293n, 19081n, 999n, 102n],
 * //   dims: [1, 6],
 * //   type: 'int64',
 * //   size: 6,
 * // }
 * ```
 *
 * @module tokenizers
 */
import { Tokenizer, Decoder, MetaspacePreTokenizer } from '@huggingface/tokenizers';
import { Callable } from './utils/generic.js';

import { isIntegralNumber, mergeArrays, pick } from './utils/core.js';

import { getModelJSON } from './utils/hub.js';

import { max, round } from './utils/maths.js';
import { Tensor } from './utils/tensor.js';

import { Template } from '@huggingface/jinja';

import { WHISPER_LANGUAGE_MAPPING } from './models/whisper/common_whisper.js';

/**
 * @typedef {import('./utils/hub.js').PretrainedOptions} PretrainedTokenizerOptions
 */

/**
 * Loads a tokenizer from the specified path.
 * @param {string} pretrained_model_name_or_path The path to the tokenizer directory.
 * @param {PretrainedTokenizerOptions} options Additional options for loading the tokenizer.
 * @returns {Promise<any[]>} A promise that resolves with information about the loaded tokenizer.
 */
async function loadTokenizer(pretrained_model_name_or_path, options) {
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
function prepareTensorForDecode(tensor) {
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

export class TokenizersBackend extends PreTrainedTokenizer {}

/**
 * BertTokenizer is a class used to tokenize text for BERT models.
 * @extends PreTrainedTokenizer
 */
export class BertTokenizer extends PreTrainedTokenizer {
    return_token_type_ids = true;
}
/**
 * Albert tokenizer
 * @extends PreTrainedTokenizer
 */
export class AlbertTokenizer extends PreTrainedTokenizer {
    return_token_type_ids = true;
}
export class MobileBertTokenizer extends PreTrainedTokenizer {
    return_token_type_ids = true;
}
export class SqueezeBertTokenizer extends PreTrainedTokenizer {
    return_token_type_ids = true;
}
export class DebertaTokenizer extends PreTrainedTokenizer {
    return_token_type_ids = true;
}
export class DebertaV2Tokenizer extends PreTrainedTokenizer {
    return_token_type_ids = true;
}
export class HerbertTokenizer extends PreTrainedTokenizer {
    return_token_type_ids = true;
}
export class ConvBertTokenizer extends PreTrainedTokenizer {
    return_token_type_ids = true;
}
export class RoFormerTokenizer extends PreTrainedTokenizer {
    return_token_type_ids = true;
}
export class DistilBertTokenizer extends PreTrainedTokenizer {}
export class CamembertTokenizer extends PreTrainedTokenizer {}
export class XLMTokenizer extends PreTrainedTokenizer {
    return_token_type_ids = true;

    constructor(tokenizerJSON, tokenizerConfig) {
        super(tokenizerJSON, tokenizerConfig);
        console.warn(
            'WARNING: `XLMTokenizer` is not yet supported by Hugging Face\'s "fast" tokenizers library. Therefore, you may experience slightly inaccurate results.',
        );
    }
}
export class ElectraTokenizer extends PreTrainedTokenizer {
    return_token_type_ids = true;
}

export class T5Tokenizer extends PreTrainedTokenizer {}
export class GPT2Tokenizer extends PreTrainedTokenizer {}
export class BartTokenizer extends PreTrainedTokenizer {}
export class MBartTokenizer extends PreTrainedTokenizer {
    constructor(tokenizerJSON, tokenizerConfig) {
        super(tokenizerJSON, tokenizerConfig);

        this.languageRegex = /^[a-z]{2}_[A-Z]{2}$/;
        this.language_codes = this.all_special_tokens.filter((x) => this.languageRegex.test(x)).map((x) => x);
        this.lang_to_token = (x) => x; // Identity function
    }

    /**
     * Helper function to build translation inputs for an `MBartTokenizer`.
     * @param {string|string[]} raw_inputs The text to tokenize.
     * @param {Object} tokenizer_options Options to be sent to the tokenizer
     * @param {Object} generate_kwargs Generation options.
     * @returns {Object} Object to be passed to the model.
     */
    _build_translation_inputs(raw_inputs, tokenizer_options, generate_kwargs) {
        return _build_translation_inputs(this, raw_inputs, tokenizer_options, generate_kwargs);
    }
}
export class MBart50Tokenizer extends MBartTokenizer {} // NOTE: extends MBartTokenizer

export class RobertaTokenizer extends PreTrainedTokenizer {}

export class BloomTokenizer extends PreTrainedTokenizer {}

export class LlamaTokenizer extends PreTrainedTokenizer {
    padding_side = 'left';
}
export class CodeLlamaTokenizer extends PreTrainedTokenizer {}

export class XLMRobertaTokenizer extends PreTrainedTokenizer {}
export class MPNetTokenizer extends PreTrainedTokenizer {}

export class FalconTokenizer extends PreTrainedTokenizer {}

export class GPTNeoXTokenizer extends PreTrainedTokenizer {}

export class EsmTokenizer extends PreTrainedTokenizer {}

export class Qwen2Tokenizer extends PreTrainedTokenizer {}

export class GemmaTokenizer extends PreTrainedTokenizer {}

export class Grok1Tokenizer extends PreTrainedTokenizer {}

/**
 * Helper function to build translation inputs for an `NllbTokenizer` or `M2M100Tokenizer`.
 * @param {PreTrainedTokenizer} self The tokenizer instance.
 * @param {string|string[]} raw_inputs The text to tokenize.
 * @param {Object} tokenizer_options Options to be sent to the tokenizer
 * @param {Object} generate_kwargs Generation options.
 * @returns {Object} Object to be passed to the model.
 * @private
 */
function _build_translation_inputs(self, raw_inputs, tokenizer_options, generate_kwargs) {
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

/**
 * The NllbTokenizer class is used to tokenize text for NLLB ("No Language Left Behind") models.
 *
 * No Language Left Behind (NLLB) is a first-of-its-kind, AI breakthrough project
 * that open-sources models capable of delivering high-quality translations directly
 * between any pair of 200+ languages — including low-resource languages like Asturian,
 * Luganda, Urdu and more. It aims to help people communicate with anyone, anywhere,
 * regardless of their language preferences. For more information, check out their
 * [paper](https://huggingface.co/papers/2207.04672).
 *
 * For a list of supported languages (along with their language codes),
 * @see {@link https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200}
 */
export class NllbTokenizer extends PreTrainedTokenizer {
    constructor(tokenizerJSON, tokenizerConfig) {
        super(tokenizerJSON, tokenizerConfig);

        this.languageRegex = /^[a-z]{3}_[A-Z][a-z]{3}$/;
        this.language_codes = this.all_special_tokens.filter((x) => this.languageRegex.test(x));
        this.lang_to_token = (x) => x; // Identity function
    }

    /**
     * Helper function to build translation inputs for an `NllbTokenizer`.
     * @param {string|string[]} raw_inputs The text to tokenize.
     * @param {Object} tokenizer_options Options to be sent to the tokenizer
     * @param {Object} generate_kwargs Generation options.
     * @returns {Object} Object to be passed to the model.
     */
    _build_translation_inputs(raw_inputs, tokenizer_options, generate_kwargs) {
        return _build_translation_inputs(this, raw_inputs, tokenizer_options, generate_kwargs);
    }
}

/**
 * The M2M100Tokenizer class is used to tokenize text for M2M100 ("Many-to-Many") models.
 *
 * M2M100 is a multilingual encoder-decoder (seq-to-seq) model trained for Many-to-Many
 * multilingual translation. It was introduced in this [paper](https://huggingface.co/papers/2010.11125)
 * and first released in [this](https://github.com/pytorch/fairseq/tree/master/examples/m2m_100) repository.
 *
 * For a list of supported languages (along with their language codes),
 * @see {@link https://huggingface.co/facebook/m2m100_418M#languages-covered}
 */
export class M2M100Tokenizer extends PreTrainedTokenizer {
    constructor(tokenizerJSON, tokenizerConfig) {
        super(tokenizerJSON, tokenizerConfig);

        this.languageRegex = /^__[a-z]{2,3}__$/;
        this.language_codes = this.all_special_tokens
            .filter((x) => this.languageRegex.test(x))
            .map((x) => x.slice(2, -2));
        this.lang_to_token = (x) => `__${x}__`;
    }

    /**
     * Helper function to build translation inputs for an `M2M100Tokenizer`.
     * @param {string|string[]} raw_inputs The text to tokenize.
     * @param {Object} tokenizer_options Options to be sent to the tokenizer
     * @param {Object} generate_kwargs Generation options.
     * @returns {Object} Object to be passed to the model.
     */
    _build_translation_inputs(raw_inputs, tokenizer_options, generate_kwargs) {
        return _build_translation_inputs(this, raw_inputs, tokenizer_options, generate_kwargs);
    }
}

const PUNCTUATION_REGEX = '\\p{P}\\u0021-\\u002F\\u003A-\\u0040\\u005B-\\u0060\\u007B-\\u007E';
const PUNCTUATION_ONLY_REGEX = new RegExp(`^[${PUNCTUATION_REGEX}]+$`, 'gu');

/**
 * WhisperTokenizer tokenizer
 * @extends PreTrainedTokenizer
 */
export class WhisperTokenizer extends PreTrainedTokenizer {
    constructor(tokenizerJSON, tokenizerConfig) {
        super(tokenizerJSON, tokenizerConfig);
    }
    get timestamp_begin() {
        return this._tokenizer.token_to_id('<|notimestamps|>') + 1;
    }

    /**
     * Decodes automatic speech recognition (ASR) sequences.
     * @param {Array<{tokens: bigint[], token_timestamps?: number[], stride: number[]}>} sequences The sequences to decode.
     * @param {Object} options The options to use for decoding.
     * @returns {Array<string|{chunks?: undefined|Array<{language: string|null, timestamp: Array<number|null>, text: string}>}>} The decoded sequences.
     */
    _decode_asr(
        sequences,
        { return_timestamps = false, return_language = false, time_precision = null, force_full_sequences = true } = {},
    ) {
        // Set force_full_sequences=false if you want streaming
        // TODO add support for `return_language`

        // Internal method meant to only be used by asr pipeline.
        // Handles all the little quirks specific to whisper to handle
        // the various options not allowed in other seq2seq models

        // =========== Overview ============
        // - iterate over all outputs
        // - all tokens within output
        // - Each token can be
        //   - language token
        //   - special token
        //   - timestamp token
        //   - text token
        // - We accumulate the text tokens.
        // - We split on end timestamps
        // - Lots of complexity comes from stride and timestamps

        if (time_precision === null) {
            throw Error('Must specify time_precision');
        }
        let last_language = null;

        const returnWordTimestamps = return_timestamps === 'word';

        function new_chunk() {
            return { language: last_language, timestamp: [null, null], text: '' };
        }

        // Welcome to the state machine!
        const chunks = [];
        let chunk = new_chunk();
        let time_offset = 0.0;
        const timestamp_begin = this.timestamp_begin;
        // Whisper timestamp tokens start from 0.00 and go to timestamp 30.00 in 0.02 increments.
        // We can calculate the last time stamp token as timestamp_begin plus the number of tokens
        // tokens from 0.00 to 30.00 which is 1500.
        const total_timestamp_tokens = 1500; // (30.00 - 0.00) / 0.02
        const timestamp_end = timestamp_begin + total_timestamp_tokens;

        let previous_tokens = [];
        let previous_token_timestamps = [];

        let skip = false;
        let right_stride_start = null;

        const all_special_ids = new Set(this.all_special_ids);

        for (const output of sequences) {
            // NOTE: python version has batches, so it uses [0]
            const token_ids = output.tokens;
            const token_timestamps = returnWordTimestamps ? output.token_timestamps : null;

            // These keep track of timestamps within strides, which need
            // to be skipped and resolve all tokens in a single chunk.
            let last_timestamp = null;
            let first_timestamp = timestamp_begin;

            if ('stride' in output) {
                const [chunk_len, stride_left, stride_right] = output.stride;

                // Offset the timings to account for the other `model_outputs`.
                time_offset -= stride_left;
                right_stride_start = chunk_len - stride_right;

                // Keeping track of timestamps within strides
                // We're going to NOT split on those, and delay until we're
                // out of BOTH stride. Otherwise lots of issues occur and
                // corner cases
                if (stride_left) {
                    first_timestamp = stride_left / time_precision + timestamp_begin;
                }

                if (stride_right) {
                    for (let i = token_ids.length - 1; i >= 0; --i) {
                        const token = Number(token_ids[i]);
                        if (token >= timestamp_begin) {
                            // There can be several token in the right stride
                            // But the last one is ALWAYS going to be skipped
                            if (
                                last_timestamp !== null &&
                                (token - timestamp_begin) * time_precision < right_stride_start
                            ) {
                                break;
                            }
                            last_timestamp = token;
                        }
                    }
                }
            }

            let current_tokens = [];
            let current_token_timestamps = [];

            // - all tokens within output
            for (let i = 0; i < token_ids.length; ++i) {
                const token = Number(token_ids[i]);
                // 4 possible states for each token
                // - 1/ Language code
                // - 2/ all other special tokens (which we ignore)
                // - 3/ Timestamp
                // - 4/ Regular text

                if (all_special_ids.has(token)) {
                    const text = this.decode([token]);
                    const language = WHISPER_LANGUAGE_MAPPING.get(text.slice(2, -2));

                    if (language !== undefined) {
                        // 1/ Indeed some language
                        // TODO Handle when language is different from the previous
                        // one, and we cannot use timestamped tokens to create chunks
                        if (last_language !== null && language !== last_language && !return_timestamps) {
                            previous_tokens.push(current_tokens);
                            const resolved_tokens = this.findLongestCommonSequence(previous_tokens)[0];
                            const resolved_text = this.decode(resolved_tokens);
                            chunk.text = resolved_text;
                            chunks.push(chunk);

                            // Flush all our temporary context
                            previous_tokens = [];
                            current_tokens = [];
                            chunk = new_chunk();
                        }

                        last_language = chunk.language = language;
                    } else {
                        // 2/ This is a regular special token, ignoring it
                    }
                } else if (token >= timestamp_begin && token <= timestamp_end) {
                    // 3/ Timestamp token
                    const time = (token - timestamp_begin) * time_precision + time_offset;
                    const rounded_time = round(time, 2);

                    if (last_timestamp !== null && token >= last_timestamp) {
                        // Whisper outputted a timestamp token, but it falls within
                        // our stride, so we're going to skip it for the time being
                        // and resolve this later
                        // Skip is necessary because timestamp tokens always come
                        // by pair, so we need to skip the next one too (which would mark the start of another chunk).
                        skip = true;
                    } else if (skip || (previous_tokens.length > 0 && token < first_timestamp)) {
                        skip = false;
                    } else if (chunk.timestamp[0] === null) {
                        chunk.timestamp[0] = rounded_time;
                    } else {
                        // This is the end of the timestamp chunk
                        if (rounded_time === chunk.timestamp[0]) {
                            // This is a bug in timestamp token output
                            // where we're taking the duplicate token
                            // as a stop where it should be a start.
                            // This is an issue in the underlying model output
                            // Let's just skip it so it becomes de-factor a start agin
                        } else {
                            chunk.timestamp[1] = rounded_time;

                            // Handling merges
                            previous_tokens.push(current_tokens);

                            if (returnWordTimestamps) {
                                previous_token_timestamps.push(current_token_timestamps);
                            }
                            const [resolved_tokens, resolved_token_timestamps] = this.findLongestCommonSequence(
                                previous_tokens,
                                previous_token_timestamps,
                            );

                            const resolved_text = this.decode(resolved_tokens);
                            chunk.text = resolved_text;

                            if (returnWordTimestamps) {
                                chunk.words = this.collateWordTimestamps(
                                    resolved_tokens,
                                    resolved_token_timestamps,
                                    last_language,
                                );
                            }

                            chunks.push(chunk);

                            // Flush all our temporary context
                            previous_tokens = [];
                            current_tokens = [];
                            previous_token_timestamps = [];
                            current_token_timestamps = [];
                            chunk = new_chunk();
                        }
                    }
                } else {
                    // 4/ Regular token
                    // We just append to the list of all tokens so we can handle
                    // merges later and decode into text.
                    current_tokens.push(token);

                    if (returnWordTimestamps) {
                        let start_time = round(token_timestamps[i] + time_offset, 2);

                        let end_time;
                        if (i + 1 < token_timestamps.length) {
                            end_time = round(token_timestamps[i + 1] + time_offset, 2);

                            // Do not allow punctuation-only tokens to have a duration.
                            // This prevents long pauses from messing up the timestamps.
                            const decoded_text = this.decode([token]);
                            if (PUNCTUATION_ONLY_REGEX.test(decoded_text)) {
                                // Add `time_precision` to avoid overlapping timestamps
                                end_time = round(Math.min(start_time + time_precision, end_time), 2);
                            }
                        } else {
                            // should never happen
                            end_time = null;
                        }
                        current_token_timestamps.push([start_time, end_time]);
                    }
                }
            }

            if ('stride' in output) {
                const [chunk_len, stride_left, stride_right] = output.stride;
                time_offset += chunk_len - stride_right;
            }

            // Leftover tokens
            if (current_tokens.length > 0) {
                previous_tokens.push(current_tokens);
                if (returnWordTimestamps) {
                    previous_token_timestamps.push(current_token_timestamps);
                }
            } else if (previous_tokens.every((p) => p.length === 0)) {
                // Flushing previous tokens (END)"
                chunk = new_chunk();
                previous_tokens = [];
                current_tokens = [];
                previous_token_timestamps = [];
                current_token_timestamps = [];
            }
        }

        if (previous_tokens.length > 0) {
            if (force_full_sequences && return_timestamps) {
                // Last token should always be timestamps, so there shouldn't be
                // leftover
                throw new Error(
                    'Whisper did not predict an ending timestamp, which can happen if audio is cut off in the middle of a word. ' +
                        'Also make sure WhisperTimeStampLogitsProcessor was used during generation.',
                );
            }

            // Happens when we don't use timestamps
            const [resolved_tokens, resolved_token_timestamps] = this.findLongestCommonSequence(
                previous_tokens,
                previous_token_timestamps,
            );

            // Flushing previous tokens (FINAL)
            const resolved_text = this.decode(resolved_tokens);
            chunk.text = resolved_text;
            if (returnWordTimestamps) {
                chunk.words = this.collateWordTimestamps(resolved_tokens, resolved_token_timestamps, last_language);
            }
            chunks.push(chunk);
        }

        let optional = Object.create(null);

        // Preparing and cleaning up the pipeline output
        const full_text = chunks.map((chunk) => chunk.text).join('');
        if (return_timestamps || return_language) {
            for (let i = 0; i < chunks.length; ++i) {
                const chunk = chunks[i];
                if (!return_timestamps) {
                    delete chunk['timestamp'];
                }

                if (!return_language) {
                    delete chunk['language'];
                }
            }
            if (returnWordTimestamps) {
                const new_chunks = [];
                for (const chunk of chunks) {
                    for (const word of chunk.words) {
                        new_chunks.push(word);
                    }
                }
                optional = { chunks: new_chunks };
            } else {
                optional = { chunks: chunks };
            }
        }
        return [full_text, optional];
    }

    /**
     * Finds the longest common sequence among the provided sequences.
     * @param {number[][]} sequences An array of sequences of token ids to compare.
     * @returns {number[][]} The longest common sequence found.
     * @throws {Error} If there is a bug within the function.
     * @private
     */
    findLongestCommonSequence(sequences, token_timestamp_sequences = null) {
        // It would be much harder to do O(n) because of fault tolerance.
        // We actually have a really good property which is that the total sequence
        // MUST be those subsequences in order.
        // If token_timestamp_sequences is provided, will split those sequences in
        // exactly the same way.
        let leftSequence = sequences[0];
        let leftLength = leftSequence.length;
        let totalSequence = [];

        const use_token_timestamp_sequences =
            Array.isArray(token_timestamp_sequences) && token_timestamp_sequences.length > 0;
        let total_token_timestamp_sequence = use_token_timestamp_sequences ? [] : null;
        let left_token_timestamp_sequence = use_token_timestamp_sequences ? token_timestamp_sequences[0] : null;
        for (let i = 1; i < sequences.length; ++i) {
            const rightSequence = sequences[i];
            let max = 0.0;
            let maxIndices = [leftLength, leftLength, 0, 0];
            // Here we're sliding matches
            // [a, b, c, d]
            //          [c, d, f]
            // =        [c] == [d]

            // [a, b, c, d]
            //       [c, d, f]
            // =     [c, d] == [c, d]

            // [a, b, c, d]
            //    [c, d, f]

            // =  [b, c, d] == [c, d, f]

            // [a, b, c, d]
            // [c, d, f]

            // [a, b, c] == [c, d, f]

            // [a, b, c, d]
            // [d, f]

            // [a, b] == [d, f]

            // [a, b, c, d]
            // [f]

            // [a] == [f]

            const rightLength = rightSequence.length;
            for (let j = 1; j < leftLength + rightLength; ++j) {
                // Slightly convoluted because we don't want out of bound indices
                // This will be necessary for a small conflict resolution optimization
                // later
                const leftStart = Math.max(0, leftLength - j);
                const leftStop = Math.min(leftLength, leftLength + rightLength - j);
                const left = leftSequence.slice(leftStart, leftStop);
                const rightStart = Math.max(0, j - leftLength);
                const rightStop = Math.min(rightLength, j);
                const right = rightSequence.slice(rightStart, rightStop);
                if (left.length !== right.length) {
                    throw new Error(
                        'There is a bug within whisper `decode_asr` function, please report it. Dropping to prevent bad inference.',
                    );
                }

                let matches;
                if (use_token_timestamp_sequences) {
                    // Get length of longest subsequence of tokens that match
                    // and have timestamps that are in order
                    matches = left.filter(
                        (elem, idx) =>
                            elem === right[idx] &&
                            left_token_timestamp_sequence[leftStart + idx] <=
                                token_timestamp_sequences[i][rightStart + idx],
                    ).length;
                } else {
                    matches = left.filter((elem, idx) => elem === right[idx]).length;
                }

                // epsilon to favor long perfect matches
                const eps = j / 10000.0;
                const matching = matches / j + eps;
                if (matches > 1 && matching > max) {
                    max = matching;
                    maxIndices = [leftStart, leftStop, rightStart, rightStop];
                }
            }
            const [leftStart, leftStop, rightStart, rightStop] = maxIndices;
            const leftMid = Math.floor((leftStop + leftStart) / 2);
            const rightMid = Math.floor((rightStop + rightStart) / 2);
            totalSequence.push(...leftSequence.slice(0, leftMid));
            leftSequence = rightSequence.slice(rightMid);
            leftLength = leftSequence.length;

            if (use_token_timestamp_sequences) {
                total_token_timestamp_sequence.push(...left_token_timestamp_sequence.slice(0, leftMid));
                left_token_timestamp_sequence = token_timestamp_sequences[i].slice(rightMid);
            }
        }
        totalSequence.push(...leftSequence);

        if (use_token_timestamp_sequences) {
            total_token_timestamp_sequence.push(...left_token_timestamp_sequence);
            return [totalSequence, total_token_timestamp_sequence];
        } else {
            return [totalSequence, []];
        }
    }

    /** @private */
    collateWordTimestamps(tokens, token_timestamps, language) {
        const [words, _, token_indices] = this.combineTokensIntoWords(tokens, language);

        const timings = [];
        for (let i = 0; i < words.length; ++i) {
            const indices = token_indices[i];
            timings.push({
                text: words[i],
                timestamp: [token_timestamps[indices.at(0)][0], token_timestamps[indices.at(-1)][1]],
            });
        }
        return timings;
    }

    /**
     * Groups tokens by word. Returns a tuple containing a list of strings with the words,
     * and a list of `token_id` sequences with the tokens making up each word.
     * @param {number[]} tokens
     * @param {string} [language]
     * @param {string} prepend_punctionations
     * @param {string} append_punctuations
     *
     * @private
     */
    combineTokensIntoWords(
        tokens,
        language,
        prepend_punctionations = '"\'“¡¿([{-',
        append_punctuations = '"\'.。,，!！?？:：”)]}、',
    ) {
        language = language ?? 'english';

        let words, word_tokens, token_indices;

        if (['chinese', 'japanese', 'thai', 'lao', 'myanmar'].includes(language)) {
            // These languages don't typically use spaces.
            [words, word_tokens, token_indices] = this.splitTokensOnUnicode(tokens);
        } else {
            [words, word_tokens, token_indices] = this.splitTokensOnSpaces(tokens);
        }

        return this.mergePunctuations(words, word_tokens, token_indices, prepend_punctionations, append_punctuations);
    }

    /** @type {PreTrainedTokenizer['decode']} */
    decode(token_ids, decode_args) {
        let text;
        // @ts-ignore
        if (decode_args?.decode_with_timestamps) {
            if (token_ids instanceof Tensor) {
                token_ids = prepareTensorForDecode(token_ids);
            }
            text = this.decodeWithTimestamps(token_ids, decode_args);
        } else {
            text = super.decode(token_ids, decode_args);
        }
        // TODO: implement offsets
        // if (decode_args.output_offsets) {
        //     let offsets = this.computeOffsets
        // }
        return text;
    }

    /**
     * @param {number[]|bigint[]} token_ids List of token IDs to decode.
     * @param {Object} decode_args Optional arguments for decoding
     * @private
     */
    decodeWithTimestamps(token_ids, decode_args) {
        const time_precision = decode_args?.time_precision ?? 0.02;

        const timestamp_begin = this.all_special_ids.at(-1) + 1;
        /**@type {any[]} */
        let outputs = [[]];
        for (let token of token_ids) {
            token = Number(token);
            if (token >= timestamp_begin) {
                const timestamp = ((token - timestamp_begin) * time_precision).toFixed(2);
                outputs.push(`<|${timestamp}|>`);
                outputs.push([]);
            } else {
                outputs[outputs.length - 1].push(token);
            }
        }
        outputs = outputs.map((s) => (typeof s === 'string' ? s : super.decode(s, decode_args)));

        return outputs.join('');
    }

    /**
     * Combine tokens into words by splitting at any position where the tokens are decoded as valid unicode points.
     * @param {number[]} tokens
     * @returns {*}
     * @private
     */
    splitTokensOnUnicode(tokens) {
        const decoded_full = this.decode(tokens, {
            // @ts-ignore
            decode_with_timestamps: true,
        });
        const replacement_char = '\uFFFD';

        const words = [];
        const word_tokens = [];
        const token_indices = [];
        let current_tokens = [];
        let current_indices = [];
        let unicode_offset = 0;

        for (let token_idx = 0; token_idx < tokens.length; ++token_idx) {
            const token = tokens[token_idx];

            current_tokens.push(token);
            current_indices.push(token_idx);

            const decoded = this.decode(current_tokens, {
                // @ts-ignore
                decode_with_timestamps: true,
            });

            if (
                !decoded.includes(replacement_char) ||
                decoded_full[unicode_offset + decoded.indexOf(replacement_char)] === replacement_char
            ) {
                words.push(decoded);
                word_tokens.push(current_tokens);
                token_indices.push(current_indices);
                current_tokens = [];
                current_indices = [];
                unicode_offset += decoded.length;
            }
        }

        return [words, word_tokens, token_indices];
    }

    /**
     * Combine tokens into words by splitting at whitespace and punctuation tokens.
     * @param {number[]} tokens
     * @private
     */
    splitTokensOnSpaces(tokens) {
        const [subwords, subword_tokens_list, subword_indices_list] = this.splitTokensOnUnicode(tokens);

        const words = [];
        const word_tokens = [];
        const token_indices = [];

        for (let i = 0; i < subwords.length; ++i) {
            const subword = subwords[i];
            const subword_tokens = subword_tokens_list[i];
            const subword_indices = subword_indices_list[i];

            // @ts-ignore
            const special = subword_tokens[0] >= this._tokenizer.token_to_id('<|endoftext|>');
            const with_space = subword.startsWith(' ');
            const trimmed = subword.trim();
            const punctuation = PUNCTUATION_ONLY_REGEX.test(trimmed);

            if (special || with_space || punctuation || words.length === 0) {
                words.push(subword);
                word_tokens.push(subword_tokens);
                token_indices.push(subword_indices);
            } else {
                const ix = words.length - 1;
                words[ix] += subword;
                word_tokens[ix].push(...subword_tokens);
                token_indices[ix].push(...subword_indices);
            }
        }

        return [words, word_tokens, token_indices];
    }

    /**
     * Merges punctuation tokens with neighboring words.
     * @param {string[]} words
     * @param {number[][]} tokens
     * @param {number[][]} indices
     * @param {string} prepended
     * @param {string} appended
     * @private
     */
    mergePunctuations(words, tokens, indices, prepended, appended) {
        const newWords = structuredClone(words);
        const newTokens = structuredClone(tokens);
        const newIndices = structuredClone(indices);

        // prepend punctuations
        let i = newWords.length - 2;
        let j = newWords.length - 1;

        while (i >= 0) {
            if (newWords[i].startsWith(' ') && prepended.includes(newWords[i].trim())) {
                newWords[j] = newWords[i] + newWords[j];
                newTokens[j] = mergeArrays(newTokens[i], newTokens[j]);
                newIndices[j] = mergeArrays(newIndices[i], newIndices[j]);
                newWords[i] = '';
                newTokens[i] = [];
                newIndices[i] = [];
            } else {
                j = i;
            }
            --i;
        }

        // append punctuations
        i = 0;
        j = 1;
        while (j < newWords.length) {
            if (!newWords[i].endsWith(' ') && appended.includes(newWords[j])) {
                newWords[i] += newWords[j];
                newTokens[i] = mergeArrays(newTokens[i], newTokens[j]);
                newIndices[i] = mergeArrays(newIndices[i], newIndices[j]);
                newWords[j] = '';
                newTokens[j] = [];
                newIndices[j] = [];
            } else {
                i = j;
            }
            ++j;
        }

        return [
            newWords.filter((x) => x),
            newTokens.filter((x) => x.length > 0),
            newIndices.filter((x) => x.length > 0),
        ];
    }
}
export class CodeGenTokenizer extends PreTrainedTokenizer {}
export class CLIPTokenizer extends PreTrainedTokenizer {}
export class SiglipTokenizer extends PreTrainedTokenizer {}

/**
 * @todo This model is not yet supported by Hugging Face's "fast" tokenizers library (https://github.com/huggingface/tokenizers).
 * Therefore, this implementation (which is based on fast tokenizers) may produce slightly inaccurate results.
 */
export class MarianTokenizer extends PreTrainedTokenizer {
    /**
     * Create a new MarianTokenizer instance.
     * @param {Object} tokenizerJSON The JSON of the tokenizer.
     * @param {Object} tokenizerConfig The config of the tokenizer.
     */
    constructor(tokenizerJSON, tokenizerConfig) {
        super(tokenizerJSON, tokenizerConfig);

        this.languageRegex = /^(>>\w+<<)\s*/g;

        this.supported_language_codes = Array.from(this.get_vocab().keys()).filter((x) => this.languageRegex.test(x));

        console.warn(
            'WARNING: `MarianTokenizer` is not yet supported by Hugging Face\'s "fast" tokenizers library. Therefore, you may experience slightly inaccurate results.',
        );
    }

    /**
     * Encodes a single text. Overriding this method is necessary since the language codes
     * must be removed before encoding with sentencepiece model.
     * @see https://github.com/huggingface/transformers/blob/12d51db243a00726a548a43cc333390ebae731e3/src/transformers/models/marian/tokenization_marian.py#L204-L213
     *
     * @param {string|null} text The text to encode.
     * @returns {string[]|null} The encoded tokens.
     */
    _encode_text(text) {
        if (text === null) return null;

        // Check if text starts with language code:
        const [matchInfo, ...remainder] = text.trim().split(this.languageRegex);

        if (remainder.length === 0) {
            // No language code, encode normally
            return super._encode_text(matchInfo);
        } else if (remainder.length === 2) {
            // Text starts with language code, so we do not encode it with sentencepiece.
            const [language, text] = remainder;

            if (!this.supported_language_codes.includes(language)) {
                console.warn(
                    `Unsupported language code "${language}" detected, which may lead to unexpected behavior. Should be one of: ${JSON.stringify(this.supported_language_codes)}`,
                );
            }
            return mergeArrays([language], super._encode_text(text));
        }
    }
}

export class Wav2Vec2CTCTokenizer extends PreTrainedTokenizer {}

export class BlenderbotTokenizer extends PreTrainedTokenizer {}
export class BlenderbotSmallTokenizer extends PreTrainedTokenizer {}

export class SpeechT5Tokenizer extends PreTrainedTokenizer {}

export class NougatTokenizer extends PreTrainedTokenizer {}

// Custom decoder for VITS
class VitsDecoder extends Decoder {
    /** @type {Decoder['decode_chain']} */
    decode_chain(tokens) {
        let decoded = '';
        for (let i = 1; i < tokens.length; i += 2) {
            decoded += tokens[i];
        }
        return [decoded];
    }
}
export class VitsTokenizer extends PreTrainedTokenizer {
    constructor(tokenizerJSON, tokenizerConfig) {
        super(tokenizerJSON, tokenizerConfig);

        // Custom decoder function
        this._tokenizer.decoder = new VitsDecoder({});
    }
}

export class CohereTokenizer extends PreTrainedTokenizer {}

export class MgpstrTokenizer extends PreTrainedTokenizer {}

export class Ernie4_5_Tokenizer extends PreTrainedTokenizer {}

/**
 * Helper class which is used to instantiate pretrained tokenizers with the `from_pretrained` function.
 * The chosen tokenizer class is determined by the type specified in the tokenizer config.
 *
 * @example
 * const tokenizer = await AutoTokenizer.from_pretrained('Xenova/bert-base-uncased');
 */
export class AutoTokenizer {
    static TOKENIZER_CLASS_MAPPING = {
        T5Tokenizer,
        DistilBertTokenizer,
        CamembertTokenizer,
        DebertaTokenizer,
        DebertaV2Tokenizer,
        BertTokenizer,
        HerbertTokenizer,
        ConvBertTokenizer,
        RoFormerTokenizer,
        XLMTokenizer,
        ElectraTokenizer,
        MobileBertTokenizer,
        SqueezeBertTokenizer,
        AlbertTokenizer,
        GPT2Tokenizer,
        BartTokenizer,
        MBartTokenizer,
        MBart50Tokenizer,
        RobertaTokenizer,
        WhisperTokenizer,
        CodeGenTokenizer,
        CLIPTokenizer,
        SiglipTokenizer,
        MarianTokenizer,
        BloomTokenizer,
        NllbTokenizer,
        M2M100Tokenizer,
        LlamaTokenizer,
        CodeLlamaTokenizer,
        XLMRobertaTokenizer,
        MPNetTokenizer,
        FalconTokenizer,
        GPTNeoXTokenizer,
        EsmTokenizer,
        Wav2Vec2CTCTokenizer,
        BlenderbotTokenizer,
        BlenderbotSmallTokenizer,
        SpeechT5Tokenizer,
        NougatTokenizer,
        VitsTokenizer,
        Qwen2Tokenizer,
        GemmaTokenizer,
        Grok1Tokenizer,
        CohereTokenizer,
        MgpstrTokenizer,
        Ernie4_5_Tokenizer,

        // Base case:
        PreTrainedTokenizer,

        // As of transformers v5.0.0
        TokenizersBackend,
    };

    /**
     * Instantiate one of the tokenizer classes of the library from a pretrained model.
     *
     * The tokenizer class to instantiate is selected based on the `tokenizer_class` property of the config object
     * (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible)
     *
     * @param {string} pretrained_model_name_or_path The name or path of the pretrained model. Can be either:
     * - A string, the *model id* of a pretrained tokenizer hosted inside a model repo on huggingface.co.
     *   Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
     *   user or organization name, like `dbmdz/bert-base-german-cased`.
     * - A path to a *directory* containing tokenizer files, e.g., `./my_model_directory/`.
     * @param {PretrainedTokenizerOptions} options Additional options for loading the tokenizer.
     *
     * @returns {Promise<PreTrainedTokenizer>} A new instance of the PreTrainedTokenizer class.
     */
    static async from_pretrained(
        pretrained_model_name_or_path,
        { progress_callback = null, config = null, cache_dir = null, local_files_only = false, revision = 'main' } = {},
    ) {
        const [tokenizerJSON, tokenizerConfig] = await loadTokenizer(pretrained_model_name_or_path, {
            progress_callback,
            config,
            cache_dir,
            local_files_only,
            revision,
        });

        // Some tokenizers are saved with the "Fast" suffix, so we remove that if present.
        const tokenizerName = tokenizerConfig.tokenizer_class?.replace(/Fast$/, '') ?? 'PreTrainedTokenizer';

        let cls = this.TOKENIZER_CLASS_MAPPING[tokenizerName];
        if (!cls) {
            console.warn(`Unknown tokenizer class "${tokenizerName}", attempting to construct from base class.`);
            cls = PreTrainedTokenizer;
        }
        return new cls(tokenizerJSON, tokenizerConfig);
    }
}
