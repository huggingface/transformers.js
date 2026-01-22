import {
    PreTrainedModel,
    decoder_forward,
    default_merge_input_ids_with_image_features,
    default_merge_input_ids_with_audio_features,
} from '../modeling_utils.js';
import { sessionRun } from '../session.js';

export class Gemma3nPreTrainedModel extends PreTrainedModel {
    forward_params = [
        'input_ids',
        'attention_mask',
        'inputs_embeds',
        'per_layer_inputs',

        'position_ids',
        'pixel_values',
        'input_features',
        'input_features_mask',
        'past_key_values',
    ];
}
export class Gemma3nForConditionalGeneration extends Gemma3nPreTrainedModel {
    async forward({
        // Produced by the tokenizer/processor:
        input_ids = null,
        attention_mask = null,
        pixel_values = null,
        input_features = null,
        input_features_mask = null,

        // Used during generation:
        position_ids = null,
        inputs_embeds = null,
        per_layer_inputs = null,
        past_key_values = null,

        // Generic generation parameters
        generation_config = null,
        logits_processor = null,

        // TODO: needed?
        ...kwargs
    }) {
        if (!inputs_embeds || !per_layer_inputs) {
            // 1. Extract the text embeddings.
            ({ inputs_embeds, per_layer_inputs } = await sessionRun(this.sessions['embed_tokens'], {
                input_ids,
            }));
            if (input_ids.dims[1] !== 1) {
                if (pixel_values) {
                    // Encode the image
                    const { image_features } = await sessionRun(this.sessions['vision_encoder'], {
                        pixel_values,
                    });
                    ({ inputs_embeds, attention_mask } = this._merge_input_ids_with_image_features({
                        image_features,
                        inputs_embeds,
                        input_ids,
                        attention_mask,
                    }));
                }

                if (input_features) {
                    // Encode the audio
                    const { audio_features } = await sessionRun(this.sessions['audio_encoder'], {
                        input_features,
                        input_features_mask,
                    });
                    ({ inputs_embeds, attention_mask } = this._merge_input_ids_with_audio_features({
                        audio_features,
                        inputs_embeds,
                        input_ids,
                        attention_mask,
                    }));
                }
            }
        }

        const outputs = await decoder_forward(
            this,
            {
                inputs_embeds,
                per_layer_inputs,
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

    _merge_input_ids_with_image_features(kwargs) {
        const vision_hidden_size = kwargs.image_features.dims.at(-1);
        const reshaped_image_hidden_states = kwargs.image_features.view(-1, vision_hidden_size);
        return default_merge_input_ids_with_image_features({
            // @ts-ignore
            image_token_id: this.config.image_token_id,
            ...kwargs,
            image_features: reshaped_image_hidden_states,
        });
    }
    _merge_input_ids_with_audio_features(kwargs) {
        const audio_hidden_size = kwargs.audio_features.dims.at(-1);
        const reshaped_audio_features = kwargs.audio_features.view(-1, audio_hidden_size);

        return default_merge_input_ids_with_audio_features({
            // @ts-ignore
            audio_token_id: this.config.audio_token_id,
            ...kwargs,
            audio_features: reshaped_audio_features,
        });
    }
}
