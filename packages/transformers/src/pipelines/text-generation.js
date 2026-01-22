import { Pipeline } from './_base.js';

import { Tensor } from '../utils/tensor.js';

/**
 * @typedef {import('./_base.js').TextPipelineConstructorArgs} TextPipelineConstructorArgs
 * @typedef {import('./_base.js').Disposable} Disposable
 * @typedef {import('../tokenization_utils.js').Message[]} Chat
 */

function isChat(x) {
    return Array.isArray(x) && x.every((x) => 'role' in x && 'content' in x);
}

/**
 * @typedef {Object} TextGenerationSingleString
 * @property {string} generated_text The generated text.
 * @typedef {TextGenerationSingleString[]} TextGenerationStringOutput
 *
 * @typedef {Object} TextGenerationSingleChat
 * @property {Chat} generated_text The generated chat.
 * @typedef {TextGenerationSingleChat[]} TextGenerationChatOutput
 *
 * @typedef {TextGenerationSingleString | TextGenerationSingleChat} TextGenerationSingle
 * @typedef {TextGenerationSingle[]} TextGenerationOutput
 *
 * @typedef {Object} TextGenerationSpecificParams Parameters specific to text-generation pipelines.
 * @property {boolean} [add_special_tokens] Whether or not to add special tokens when tokenizing the sequences.
 * @property {boolean} [return_full_text=true] If set to `false` only added text is returned, otherwise the full text is returned.
 * @property {Object} [tokenizer_encode_kwargs] Additional keyword arguments to pass along to the encoding step of the tokenizer.
 * If the text input is a chat, it is passed to `apply_chat_template`. Otherwise, it is passed to the tokenizer's call function.
 * @typedef {import('../generation/configuration_utils.js').GenerationConfig & TextGenerationSpecificParams} TextGenerationConfig
 *
 * @callback TextGenerationPipelineCallbackString
 * @param {string} texts One prompt to complete.
 * @param {Partial<TextGenerationConfig>} [options] Additional keyword arguments to pass along to the generate method of the model.
 * @returns {Promise<TextGenerationStringOutput>} An array containing the generated text(s).
 *
 * @callback TextGenerationPipelineCallbackChat
 * @param {Chat} texts One chat to complete.
 * @param {Partial<TextGenerationConfig>} [options] Additional keyword arguments to pass along to the generate method of the model.
 * @returns {Promise<TextGenerationChatOutput>} An array containing the generated chat(s).
 *
 * @callback TextGenerationPipelineCallbackStringBatched
 * @param {string[]} texts Several prompts to complete.
 * @param {Partial<TextGenerationConfig>} [options] Additional keyword arguments to pass along to the generate method of the model.
 * @returns {Promise<TextGenerationStringOutput[]>} An array of arrays, each containing the generated text(s) for the corresponding input.
 *
 * @callback TextGenerationPipelineCallbackChatBatched
 * @param {Chat[]} texts Several chats to complete.
 * @param {Partial<TextGenerationConfig>} [options] Additional keyword arguments to pass along to the generate method of the model.
 * @returns {Promise<TextGenerationChatOutput[]>} An array of arrays, each containing the generated chat(s) for the corresponding input.
 *
 * @typedef {TextGenerationPipelineCallbackString & TextGenerationPipelineCallbackChat & TextGenerationPipelineCallbackStringBatched & TextGenerationPipelineCallbackChatBatched} TextGenerationPipelineCallback
 *
 * @typedef {TextPipelineConstructorArgs & TextGenerationPipelineCallback & Disposable} TextGenerationPipelineType
 */

/**
 * Language generation pipeline using any `ModelWithLMHead` or `ModelForCausalLM`.
 * This pipeline predicts the words that will follow a specified text prompt.
 * NOTE: For the full list of generation parameters, see [`GenerationConfig`](./utils/generation#module_utils/generation.GenerationConfig).
 *
 * **Example:** Text generation with `HuggingFaceTB/SmolLM2-135M` (default settings).
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const generator = await pipeline('text-generation', 'onnx-community/SmolLM2-135M-ONNX');
 * const text = 'Once upon a time,';
 * const output = await generator(text, { max_new_tokens: 8 });
 * // [{ generated_text: 'Once upon a time, there was a little girl named Lily.' }]
 * ```
 *
 * **Example:** Chat completion with `onnx-community/Qwen3-0.6B-ONNX`.
 * ```javascript
 * import { pipeline, TextStreamer } from '@huggingface/transformers';
 *
 * // Create a text generation pipeline
 * const generator = await pipeline(
 *   'text-generation',
 *   'onnx-community/Qwen3-0.6B-ONNX',
 *   { dtype: 'q4f16' },
 * );
 *
 * // Define the list of messages
 * const messages = [
 *   { role: 'system', content: 'You are a helpful assistant.' },
 *   { role: 'user', content: 'Write me a poem about Machine Learning.' },
 * ];
 *
 * // Generate a response
 * const output = await generator(messages, {
 *   max_new_tokens: 512,
 *   do_sample: false,
 *   streamer: new TextStreamer(generator.tokenizer, { skip_prompt: true, skip_special_tokens: true }),
 * });
 * console.log(output[0].generated_text.at(-1)?.content);
 * ```
 */
export class TextGenerationPipeline
    extends /** @type {new (options: TextPipelineConstructorArgs) => TextGenerationPipelineType} */ (Pipeline)
{
    async _call(texts, generate_kwargs = {}) {
        let isBatched = false;
        let isChatInput = false;

        // By default, do not add special tokens, unless the tokenizer specifies otherwise
        let add_special_tokens =
            generate_kwargs.add_special_tokens ??
            (this.tokenizer.add_bos_token || this.tokenizer.add_eos_token) ??
            false;

        let tokenizer_kwargs = generate_kwargs.tokenizer_encode_kwargs;

        // Normalize inputs
        /** @type {string[]} */
        let inputs;
        if (typeof texts === 'string') {
            inputs = texts = [texts];
        } else if (Array.isArray(texts) && texts.every((x) => typeof x === 'string')) {
            isBatched = true;
            inputs = /** @type {string[]} */ (texts);
        } else {
            if (isChat(texts)) {
                texts = [/** @type {Chat} */ (texts)];
            } else if (Array.isArray(texts) && texts.every(isChat)) {
                isBatched = true;
            } else {
                throw new Error('Input must be a string, an array of strings, a Chat, or an array of Chats');
            }
            isChatInput = true;

            // If the input is a chat, we need to apply the chat template
            inputs = /** @type {string[]} */ (
                /** @type {Chat[]} */ (texts).map((x) =>
                    this.tokenizer.apply_chat_template(x, {
                        tokenize: false,
                        add_generation_prompt: true,
                        ...tokenizer_kwargs,
                    }),
                )
            );
            // Chat template handles these already
            add_special_tokens = false;
            tokenizer_kwargs = undefined;
        }

        // By default, return full text
        const return_full_text = isChatInput ? false : (generate_kwargs.return_full_text ?? true);

        this.tokenizer.padding_side = 'left';
        const text_inputs = this.tokenizer(inputs, {
            add_special_tokens,
            padding: true,
            truncation: true,
            ...tokenizer_kwargs,
        });

        const outputTokenIds = /** @type {Tensor} */ (
            await this.model.generate({
                ...text_inputs,
                ...generate_kwargs,
            })
        );

        const decoded = this.tokenizer.batch_decode(outputTokenIds, {
            skip_special_tokens: true,
        });

        let promptLengths;
        if (!return_full_text && text_inputs.input_ids.dims.at(-1) > 0) {
            promptLengths = this.tokenizer
                .batch_decode(text_inputs.input_ids, {
                    skip_special_tokens: true,
                })
                .map((x) => x.length);
        }

        /** @type {TextGenerationOutput[]} */
        const toReturn = Array.from({ length: texts.length }, (_) => []);
        for (let i = 0; i < decoded.length; ++i) {
            const textIndex = Math.floor((i / outputTokenIds.dims[0]) * texts.length);

            if (promptLengths) {
                // Trim the decoded text to only include the generated part
                decoded[i] = decoded[i].slice(promptLengths[textIndex]);
            }
            toReturn[textIndex].push(
                /** @type {TextGenerationSingle} */ ({
                    generated_text: isChatInput
                        ? [.../** @type {Chat[]} */ (texts)[textIndex], { role: 'assistant', content: decoded[i] }]
                        : decoded[i],
                }),
            );
        }
        return !isBatched && toReturn.length === 1 ? toReturn[0] : toReturn;
    }
}
