import { sessionRun } from '../session.js';
import { PreTrainedModel, decoder_forward, decoder_prepare_inputs_for_generation } from '../modeling_utils.js';
import { cat, ones, full, Tensor } from '../../utils/tensor.js';

const SILENCE_TOKEN = 4299n;
const START_SPEECH_TOKEN = 6561n;

export class ChatterboxPreTrainedModel extends PreTrainedModel {
    forward_params = [
        'input_ids',
        'inputs_embeds',
        'attention_mask',
        'position_ids',
        'audio_values',
        'exaggeration',
        'audio_features',
        'audio_tokens',
        'speaker_embeddings',
        'speaker_features',
        'past_key_values',
    ];
    main_input_name = 'input_ids';

    _return_dict_in_generate_keys = ['audio_tokens', 'speaker_embeddings', 'speaker_features'];
}
export class ChatterboxModel extends ChatterboxPreTrainedModel {
    /**
     * @param {Tensor} audio_values
     * @returns {Promise<{audio_features: Tensor, audio_tokens: Tensor, speaker_embeddings: Tensor, speaker_features: Tensor}>}
     */
    async encode_speech(audio_values) {
        return sessionRun(this.sessions['speech_encoder'], {
            audio_values,
        });
    }

    async forward({
        // Produced by the tokenizer/processor:
        input_ids = null,
        attention_mask = null,
        audio_values = null,
        exaggeration = null,

        // Used during generation:
        position_ids = null,
        inputs_embeds = null,
        past_key_values = null,

        // Generic generation parameters
        generation_config = null,
        logits_processor = null,

        // Speaker embeddings/features (useful for re-using pre-computed speaker data)
        audio_features = null, // float32[batch_size,sequence_length,1024]
        audio_tokens = null, // int64[batch_size,audio_sequence_length]
        speaker_embeddings = null, // float32[batch_size,192]
        speaker_features = null, // float32[batch_size,feature_dim,80]

        // TODO: needed?
        ...kwargs
    }) {
        let speech_encoder_outputs;
        if (!inputs_embeds) {
            const expected_inputs = this.sessions['embed_tokens'].inputNames;
            const embed_model_inputs = { input_ids };
            if (expected_inputs.includes('exaggeration')) {
                // Support the following types for exaggeration:
                // 1. null/undefined (no exaggeration): use the default of 0.5
                // 2. number: broadcast to (batch_size,)
                // 3. number[]: convert to Tensor of shape (batch_size,)
                // 4. Tensor of shape (batch_size, 1)
                if (!(exaggeration instanceof Tensor)) {
                    const batch_size = input_ids.dims[0];
                    if (exaggeration == null) {
                        exaggeration = full([batch_size], 0.5);
                    } else if (typeof exaggeration === 'number') {
                        exaggeration = full([batch_size], exaggeration);
                    } else if (Array.isArray(exaggeration)) {
                        exaggeration = new Tensor('float32', exaggeration, [batch_size]);
                    } else {
                        throw new Error('Unsupported type for `exaggeration` input');
                    }
                }
                embed_model_inputs.exaggeration = exaggeration;
            }
            if (expected_inputs.includes('position_ids')) {
                embed_model_inputs.position_ids = position_ids;
            }

            ({ inputs_embeds } = await sessionRun(this.sessions['embed_tokens'], embed_model_inputs));

            if (audio_features && audio_tokens && speaker_embeddings && speaker_features) {
                // Use pre-computed speech encoder outputs
                speech_encoder_outputs = { audio_features, audio_tokens, speaker_embeddings, speaker_features };
            }

            if (speech_encoder_outputs || audio_values) {
                speech_encoder_outputs ??= await this.encode_speech(audio_values);

                // Update LLM inputs
                inputs_embeds = cat([speech_encoder_outputs.audio_features, inputs_embeds], 1);
                attention_mask = ones([inputs_embeds.dims[0], inputs_embeds.dims[1]]);
            } else {
                const target_length = inputs_embeds.dims[1];
                if (!past_key_values || target_length !== 1) {
                    throw new Error('Incorrect state encountered during generation.');
                }
                const past_length = Object.values(past_key_values)[0].dims.at(-2);
                attention_mask = ones([inputs_embeds.dims[0], past_length + target_length]);
            }
        }

        const outputs = await decoder_forward(
            this,
            {
                inputs_embeds,
                past_key_values,
                attention_mask,
                generation_config,
                logits_processor,
            },
            false,
        );
        return {
            ...outputs,
            ...speech_encoder_outputs,
        };
    }

    prepare_inputs_for_generation(input_ids, model_inputs, generation_config) {
        if (!model_inputs.position_ids && this.sessions['embed_tokens'].inputNames.includes('position_ids')) {
            // If position_ids are not provided, we create them on the fly using the position of the START_SPEECH_TOKEN
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
        return decoder_prepare_inputs_for_generation(this, input_ids, model_inputs, generation_config);
    }

    /** @type {PreTrainedModel['generate']} */
    async generate(params) {
        const { sequences, audio_tokens, speaker_embeddings, speaker_features } = /** @type {any} */ (
            await super.generate({
                ...params,
                return_dict_in_generate: true,
            })
        );

        const new_tokens = sequences.slice(null, [
            params.input_ids.dims[1], // Exclude start of speech token
            -1, // Exclude end of speech token
        ]);

        const silence_tokens = full([new_tokens.dims[0], 3], SILENCE_TOKEN); // Add 3 silence tokens
        const speech_tokens = cat([audio_tokens, new_tokens, silence_tokens], 1);

        const { waveform } = await sessionRun(this.sessions['conditional_decoder'], {
            speech_tokens,
            speaker_features,
            speaker_embeddings,
        });
        return waveform;
    }
}
