// JS doesn't support mixins, so we define some reused functions here, and allow "this" to be passed in
import { pick } from '../utils/core.js';
import { cat, full_like, ones, Tensor, toI64Tensor, zeros_like } from '../utils/tensor.js';
import { sessionRun } from './session.js';
import { getModelJSON } from '../utils/hub.js';
import { isONNXProxy } from '../backends/onnx.js';

/**
 * Perform forward pass on the seq2seq model (both encoder and decoder).
 * @param {Object} self The seq2seq model object.
 * @param {Object} model_inputs The input object for the model containing encoder and decoder inputs.
 * @returns {Promise<Seq2SeqLMOutput>} Promise that resolves with the output of the seq2seq model.
 * @private
 */
export async function seq2seqForward(self, model_inputs) {
    let { encoder_outputs, input_ids, decoder_input_ids, ...other_decoder_inputs } = model_inputs;
    // Encode if needed
    if (!encoder_outputs) {
        const encoder_inputs = pick(model_inputs, self.sessions['model'].inputNames);
        // Encoder outputs are not given, so we must compute them.
        encoder_outputs = (await encoderForward(self, encoder_inputs)).last_hidden_state;
    }

    other_decoder_inputs.input_ids = decoder_input_ids;
    other_decoder_inputs.encoder_hidden_states = encoder_outputs;

    if (self.sessions['decoder_model_merged'].inputNames.includes('encoder_attention_mask')) {
        other_decoder_inputs.encoder_attention_mask = model_inputs.attention_mask;
    }

    return await decoderForward(self, other_decoder_inputs, true);
}

/**
 * Forward pass of an encoder model.
 * @param {Object} self The encoder model.
 * @param {Object} model_inputs The input data to be used for the forward pass.
 * @returns {Promise<Object>} The model's outputs.
 * @private
 */
export async function encoderForward(self, model_inputs) {
    const session = self.sessions['model'];
    const encoderFeeds = pick(model_inputs, session.inputNames);

    if (session.inputNames.includes('inputs_embeds') && !encoderFeeds.inputs_embeds) {
        if (!model_inputs.input_ids) {
            throw new Error('Both `input_ids` and `inputs_embeds` are missing in the model inputs.');
        }
        encoderFeeds.inputs_embeds = await self.encode_text({ input_ids: model_inputs.input_ids });
    }
    if (session.inputNames.includes('token_type_ids') && !encoderFeeds.token_type_ids) {
        if (!encoderFeeds.input_ids) {
            throw new Error('Both `input_ids` and `token_type_ids` are missing in the model inputs.');
        }
        // Assign default `token_type_ids` (all zeroes) to the `encoderFeeds` if the model expects it,
        // but they weren't created by the tokenizer.
        encoderFeeds.token_type_ids = zeros_like(encoderFeeds.input_ids);
    }
    if (session.inputNames.includes('pixel_mask') && !encoderFeeds.pixel_mask) {
        if (!encoderFeeds.pixel_values) {
            throw new Error('Both `pixel_values` and `pixel_mask` are missing in the model inputs.');
        }
        // Assign default `pixel_mask` (all ones) to the `encoderFeeds` if the model expects it,
        // but they weren't created by the processor.
        const dims = encoderFeeds.pixel_values.dims;
        encoderFeeds.pixel_mask = ones([dims[0], dims[2], dims[3]]);
    }

    return await sessionRun(session, encoderFeeds);
}

export async function autoEncoderForward(self, model_inputs) {
    const encoded = await self.encode(model_inputs);
    const decoded = await self.decode(encoded);
    return decoded;
}

/**
 * Forward pass of a decoder model.
 * @param {Object} self The decoder model.
 * @param {Object} model_inputs The input data to be used for the forward pass.
 * @returns {Promise<Object>} The logits and past key values.
 * @private
 */
export async function decoderForward(self, model_inputs, is_encoder_decoder = false) {
    const session = self.sessions[is_encoder_decoder ? 'decoder_model_merged' : 'model'];

    const { past_key_values, ...new_model_inputs } = model_inputs;

    if (session.inputNames.includes('use_cache_branch')) {
        new_model_inputs.use_cache_branch = boolTensor(!!past_key_values);
    }
    if (
        session.inputNames.includes('position_ids') &&
        new_model_inputs.attention_mask &&
        !new_model_inputs.position_ids
    ) {
        // NOTE: Handle a special case for paligemma/gemma3 models, where positions are 1-indexed
        const start_index = ['paligemma', 'gemma3_text', 'gemma3'].includes(self.config.model_type) ? 1 : 0;
        new_model_inputs.position_ids = createPositionIds(new_model_inputs, past_key_values, start_index);
    }

    // Unpack the `past_key_values` object into model inputs
    self.addPastKeyValues(new_model_inputs, past_key_values);

    // Select only the inputs that are needed for the current session
    const fixed = pick(new_model_inputs, session.inputNames);
    return await sessionRun(session, fixed);
}

/**
 * Abstract forward pass function for image-text-to-text or audio-text-to-text models.
 * @param {Object} self The model object.
 * @param {Object} params Additional parameters.
 * @param {Function} [params.encode_function] The function to encode the modality values.
 * @param {Function} [params.merge_function] The function to merge the modality features with the input embeddings.
 * @param {string} [params.modality_input_name] The modality input name.
 * @param {string} [params.modality_output_name] The modality output name.
 * @param {Tensor} [params.input_ids=null]
 * @param {Tensor} [params.attention_mask=null]
 * @param {Tensor} [params.position_ids=null]
 * @param {Tensor} [params.inputs_embeds=null]
 * @param {Tensor} [params.past_key_values=null]
 * @param {Object} [params.generation_config=null]
 * @param {Object} [params.logits_processor=null]
 * @returns {Promise<Tensor>} The model's output tensor
 * @private
 */
export async function genericTextToTextForward(
    self,
    {
        // Generic parameters:
        encode_function,
        merge_function,
        modality_input_name,
        modality_output_name,

        // Produced by the tokenizer/processor:
        input_ids = null,
        attention_mask = null,

        // Used during generation:
        position_ids = null,
        inputs_embeds = null,
        past_key_values = null,

        // Generic generation parameters
        generation_config = null,
        logits_processor = null,

        // Additional parameters
        ...kwargs
    },
) {
    const modality_values = kwargs[modality_input_name];
    if (!inputs_embeds) {
        // 1. Extract the text embeddings.
        inputs_embeds = await self.encode_text({ input_ids, ...kwargs });

        // 2. Possibly, merge text and modality values
        if (modality_values && input_ids.dims[1] !== 1) {
            const modality_features = await encode_function({
                // Pass the modality values under its expected key.
                // The caller knows whether this is audio or image.
                [modality_input_name]: modality_values,
                ...kwargs,
            });
            ({ inputs_embeds, attention_mask } = merge_function({
                [modality_output_name]: modality_features,
                inputs_embeds,
                input_ids,
                attention_mask,
            }));
        } else if (past_key_values && modality_values && input_ids.dims[1] === 1) {
            // This branch handles the cache case.
            const target_length = input_ids.dims[1]; // always 1
            const past_length = Object.values(past_key_values)[0].dims.at(-2);

            attention_mask = cat(
                [
                    ones([input_ids.dims[0], past_length]),
                    attention_mask.slice(null, [attention_mask.dims[1] - target_length, attention_mask.dims[1]]),
                ],
                1,
            );
        }
    }

    if (!position_ids) {
        if (self.config.model_type === 'qwen2_vl') {
            // Special case for qwen2_vl models
            // @ts-ignore
            const { image_grid_thw, video_grid_thw } = kwargs;
            [position_ids] = self.get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask);
        }
    }

    // 3. Call the decoder forward using the updated inputs.
    const outputs = await decoderForward(
        self,
        {
            inputs_embeds,
            past_key_values,
            attention_mask,
            position_ids,
            generation_config,
            logits_processor,
        },
        true,
    );
    return outputs;
}

/**
 * Forward pass of an audio-text-to-text model.
 * @param {Object} self The audio-text-to-text model.
 * @param {Object} params The inputs for the audio-text-to-text forward pass.
 * @returns {Promise<Tensor>} The model's output tensor.
 * @private
 */
export async function audioTextToTextForward(self, params) {
    return await genericTextToTextForward(self, {
        ...params,
        modality_input_name: 'audio_values',
        modality_output_name: 'audio_features',
        encode_function: self.encode_audio.bind(self),
        merge_function: self._merge_input_ids_with_audio_features.bind(self),
    });
}

/**
 * Forward pass of an image-text-to-text model.
 * @param {Object} self The image-text-to-text model.
 * @param {Object} params The inputs for the image-text-to-text forward pass.
 * @returns {Promise<Tensor>} The model's output tensor.
 * @private
 */
export async function imageTextToTextForward(self, params) {
    return await genericTextToTextForward(self, {
        ...params,
        modality_input_name: 'pixel_values',
        modality_output_name: 'image_features',
        encode_function: self.encode_image.bind(self),
        merge_function: self._merge_input_ids_with_image_features.bind(self),
    });
}

/**
 * Helper function to perform the following:
 * ```python
 * x = attention_mask.long().cumsum(-1) - 1
 * x.masked_fill_(attention_mask == 0, 1)
 * ```
 * @param {Tensor} attention_mask
 * @returns {{data: BigInt64Array, dims: number[]}}
 */
export function cumsum_masked_fill(attention_mask, start_index = 0) {
    const [bz, seq_len] = attention_mask.dims;
    const attn_mask_data = attention_mask.data;

    const data = new BigInt64Array(attn_mask_data.length);
    for (let i = 0; i < bz; ++i) {
        const start = i * seq_len;
        let sum = BigInt(start_index);
        for (let j = 0; j < seq_len; ++j) {
            const index = start + j;
            if (attn_mask_data[index] === 0n) {
                data[index] = BigInt(1);
            } else {
                // === 1n
                data[index] = sum;
                sum += attn_mask_data[index];
            }
        }
    }
    return { data, dims: attention_mask.dims };
}

/**
 * If the model supports providing position_ids, we create position_ids on the fly for batch generation,
 * by computing the cumulative sum of the attention mask along the sequence length dimension.
 *
 * Equivalent to:
 * ```python
 * position_ids = attention_mask.long().cumsum(-1) - 1
 * position_ids.masked_fill_(attention_mask == 0, 1)
 * if past_key_values:
 *     position_ids = position_ids[:, -input_ids.shape[1] :]
 * ```
 */
export function createPositionIds(model_inputs, past_key_values = null, start_index = 0) {
    const { input_ids, inputs_embeds, attention_mask } = model_inputs;

    const { data, dims } = cumsum_masked_fill(attention_mask, start_index);
    let position_ids = new Tensor('int64', data, dims);
    if (past_key_values) {
        const offset = -(input_ids ?? inputs_embeds).dims.at(1);
        position_ids = position_ids.slice(null, [offset, null]);
    }
    return position_ids;
}

export function decoder_prepare_inputs_for_generation(self, input_ids, model_inputs, generation_config) {
    const past_length = model_inputs.past_key_values ? Object.values(model_inputs.past_key_values)[0].dims.at(-2) : 0;

    if (!model_inputs.attention_mask) {
        // If the attention mask is not provided, we attempt to infer based on provided inputs
        let dims;
        for (const key of ['input_ids', 'inputs_embeds', 'position_ids']) {
            if (model_inputs[key]) {
                dims = model_inputs[key].dims;
                break;
            }
        }
        if (!dims) {
            throw new Error('attention_mask is not provided, and unable to infer its shape from model inputs.');
        }
        model_inputs.attention_mask = ones([dims[0], past_length + dims[1]]);
    }

    if (model_inputs.past_key_values) {
        const { input_ids, attention_mask } = model_inputs;

        // Keep only the unprocessed tokens:
        // 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        // some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
        // input)
        if (attention_mask && attention_mask.dims[1] > input_ids.dims[1]) {
            // NOTE: not needed since we only pass the generated tokens to the next forward pass
            // const offset = -(attention_mask.dims[1] - past_length);
            // model_inputs.input_ids = input_ids.slice(null, [offset, null]);
        }
        // 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens.
        // We can discard input_ids based on the past_length.
        else if (past_length < input_ids.dims[1]) {
            // NOTE: Required for phi models.
            // See https://github.com/huggingface/transformers/issues/30809#issuecomment-2111918479 for more information.
            model_inputs.input_ids = input_ids.slice(null, [past_length, null]);
        }
        // 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
        else {
        }
    }

    return model_inputs;
}

export function encoder_decoder_prepare_inputs_for_generation(self, input_ids, model_inputs, generation_config) {
    if (model_inputs.past_key_values) {
        input_ids = input_ids.map((x) => [x.at(-1)]);
    }

    return {
        ...model_inputs,
        decoder_input_ids: toI64Tensor(input_ids),
    };
}

export function multimodal_text_to_text_prepare_inputs_for_generation(self, ...args) {
    if (self.config.is_encoder_decoder) {
        return encoder_decoder_prepare_inputs_for_generation(self, ...args);
    } else {
        return decoder_prepare_inputs_for_generation(self, ...args);
    }
}

export function multimodality_prepare_inputs_for_generation(self, input_ids, model_inputs, generation_config) {
    const has_past_key_values = !!model_inputs.past_key_values;

    if (generation_config.guidance_scale !== null && generation_config.guidance_scale > 1) {
        if (has_past_key_values) {
            model_inputs.input_ids = cat([model_inputs.input_ids, model_inputs.input_ids], 0);
            // NOTE: attention_mask handled in generation
        } else {
            model_inputs.input_ids = cat(
                [model_inputs.input_ids, full_like(model_inputs.input_ids, BigInt(generation_config.pad_token_id))],
                0,
            );
            model_inputs.attention_mask = cat(
                [model_inputs.attention_mask, full_like(model_inputs.attention_mask, 0n)],
                0,
            );
        }
    }

    if (has_past_key_values || !model_inputs.pixel_values) {
        model_inputs.pixel_values = full([0, 0, 3, 384, 384], 1.0);
    }

    if (has_past_key_values) {
        const num_img_tokens = 0;
        const num_text_tokens = 1;
        const has_image = num_img_tokens > 0 ? 1 : 0;

        const batch_size = 1;
        model_inputs.images_seq_mask = new Tensor(
            'bool',
            new Array(num_img_tokens + num_text_tokens).fill(true).fill(false, 0, num_text_tokens),
            [batch_size, num_img_tokens + num_text_tokens],
        );
        model_inputs.images_emb_mask = new Tensor('bool', new Array(num_img_tokens).fill(!!has_image), [
            batch_size,
            1,
            num_img_tokens,
        ]);
    }
    return model_inputs;
}

export function chatterbox_prepare_inputs_for_generation(self, input_ids, model_inputs, generation_config) {
    if (!model_inputs.position_ids && self.sessions['embed_tokens'].inputNames.includes('position_ids')) {
        // If position_ids are not provided, we create them on the fly using the position of the START_SPEECH_TOKEN
        const START_SPEECH_TOKEN = 6561;
        if (model_inputs.input_ids.dims[1] === 1) {
            const position_ids = Array.from(
                {
                    length: input_ids.length,
                },
                (_, i) => input_ids[i].length - input_ids[i].findLastIndex((x) => x == START_SPEECH_TOKEN) - 1,
            );
            model_inputs.position_ids = new Tensor('int64', position_ids, [input_ids.length, 1]);
        } else {
            const batched_input_ids = model_inputs.input_ids.tolist();
            const position_ids_list = batched_input_ids.map((ids) => {
                let position = 0;
                return ids.map((id) => (id >= START_SPEECH_TOKEN ? 0 : position++));
            });
            model_inputs.position_ids = new Tensor('int64', position_ids_list.flat(), model_inputs.input_ids.dims);
        }
    }
    if (model_inputs.input_ids.dims[1] === 1) {
        // We are in generation mode and no longer need the audio inputs
        delete model_inputs.audio_values;
        delete model_inputs.audio_features;
        delete model_inputs.audio_tokens;
        delete model_inputs.speaker_embeddings;
        delete model_inputs.speaker_features;
    }
    return decoder_prepare_inputs_for_generation(self, input_ids, model_inputs, generation_config);
}

/**
 * Validate model inputs
 * @param {Object} session The InferenceSession object that will be run.
 * @param {Object} inputs The inputs to check.
 * @returns {Record<string, Tensor>} The checked inputs.
 * @throws {Error} If any inputs are missing.
 * @private
 */
export function validateInputs(session, inputs) {
    /**
     * NOTE: Create either a shallow or deep copy based on `onnx.wasm.proxy`
     * @type {Record<string, Tensor>}
     */
    const checkedInputs = Object.create(null);
    const missingInputs = [];
    for (const inputName of session.inputNames) {
        const tensor = inputs[inputName];
        // Rare case where one of the model's input names corresponds to a built-in
        // object name (e.g., toString), which would cause a simple (!tensor) check to fail,
        // because it's not undefined but a function.
        if (!(tensor instanceof Tensor)) {
            missingInputs.push(inputName);
            continue;
        }
        // NOTE: When `env.wasm.proxy is true` the tensor is moved across the Worker
        // boundary, transferring ownership to the worker and invalidating the tensor.
        // So, in this case, we simply sacrifice a clone for it.
        checkedInputs[inputName] = isONNXProxy() ? tensor.clone() : tensor;
    }
    if (missingInputs.length > 0) {
        throw new Error(
            `An error occurred during model execution: "Missing the following inputs: ${missingInputs.join(', ')}.`,
        );
    }

    const numInputsProvided = Object.keys(inputs).length;
    const numInputsNeeded = session.inputNames.length;
    if (numInputsProvided > numInputsNeeded) {
        // No missing inputs, but too many inputs were provided.
        // Warn the user and ignore the extra inputs.
        let ignored = Object.keys(inputs).filter((inputName) => !session.inputNames.includes(inputName));
        console.warn(
            `WARNING: Too many inputs were provided (${numInputsProvided} > ${numInputsNeeded}). The following inputs will be ignored: "${ignored.join(', ')}".`,
        );
    }

    return checkedInputs;
}

export function default_merge_input_ids_with_features({
    modality_token_id,
    inputs_embeds,
    modality_features,
    input_ids,
    attention_mask,
}) {
    const token_positions = input_ids.tolist().map((ids) =>
        ids.reduce((acc, x, idx) => {
            if (x == modality_token_id) acc.push(idx);
            return acc;
        }, []),
    );
    const n_tokens = token_positions.reduce((acc, x) => acc + x.length, 0);
    const n_features = modality_features.dims[0];
    if (n_tokens !== n_features) {
        throw new Error(`Number of tokens and features do not match: tokens: ${n_tokens}, features ${n_features}`);
    }

    // Equivalent to performing a masked_scatter
    let img = 0;
    for (let i = 0; i < token_positions.length; ++i) {
        const tokens = token_positions[i];
        const embeds = inputs_embeds[i];
        for (let j = 0; j < tokens.length; ++j) {
            embeds[tokens[j]].data.set(modality_features[img++].data);
        }
    }
    return { inputs_embeds, attention_mask };
}

export function default_merge_input_ids_with_image_features({
    image_token_id,
    inputs_embeds,
    image_features,
    input_ids,
    attention_mask,
}) {
    return default_merge_input_ids_with_features({
        modality_token_id: image_token_id,
        inputs_embeds,
        modality_features: image_features,
        input_ids,
        attention_mask,
    });
}

export function default_merge_input_ids_with_audio_features({
    audio_token_id,
    inputs_embeds,
    audio_features,
    input_ids,
    attention_mask,
}) {
    return default_merge_input_ids_with_features({
        modality_token_id: audio_token_id,
        inputs_embeds,
        modality_features: audio_features,
        input_ids,
        attention_mask,
    });
}

/**
 * Helper function to load multiple optional configuration files
 * @param {string} pretrained_model_name_or_path The path to the directory containing the config file.
 * @param {Record<string, string>} names The names of the config files to load.
 * @param {import('./utils/hub.js').PretrainedModelOptions} options Additional options for loading the configs.
 * @returns {Promise<Record<string, any>>} A Promise that resolves to a dictionary of configuration objects.
 * @private
 */
export async function getOptionalConfigs(pretrained_model_name_or_path, names, options) {
    return Object.fromEntries(
        await Promise.all(
            Object.keys(names).map(async (name) => {
                const config = await getModelJSON(pretrained_model_name_or_path, names[name], false, options);
                return [name, config];
            }),
        ),
    );
}
