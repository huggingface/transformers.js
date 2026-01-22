import { PreTrainedModel, default_merge_input_ids_with_audio_features } from '../modeling_utils.js';

export class UltravoxPreTrainedModel extends PreTrainedModel {
    forward_params = ['input_ids', 'attention_mask', 'position_ids', 'audio_values', 'past_key_values'];
}

export class UltravoxModel extends UltravoxPreTrainedModel {
    _merge_input_ids_with_audio_features(kwargs) {
        const audio_hidden_size = kwargs.audio_features.dims.at(-1);
        const reshaped_audio_features = kwargs.audio_features.view(-1, audio_hidden_size);

        return default_merge_input_ids_with_audio_features({
            // @ts-ignore
            audio_token_id: this.config.ignore_index ?? this.config.audio_token_id,
            ...kwargs,
            audio_features: reshaped_audio_features,
        });
    }
}

export class VoxtralForConditionalGeneration extends UltravoxModel {}
