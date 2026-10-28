import { PreTrainedModel, default_merge_input_ids_with_audio_features } from '../modeling_utils.js';

export class UltravoxPreTrainedModel extends PreTrainedModel {
    forward_params = ['input_ids', 'attention_mask', 'position_ids', 'audio_values', 'past_key_values'];
}

export class UltravoxModel extends UltravoxPreTrainedModel {
    _multimodal_generation_config() {
        if (this.config.model_type !== 'ultravox') return null;
        return {
            modality_input_names: ['audio_values'],
            modality_output_name: 'audio_features',
            encode_function: this.encode_audio.bind(this),
            merge_function: this._merge_input_ids_with_audio_features.bind(this),
            batch_axes: { position_ids: 0 },
        };
    }

    _merge_input_ids_with_audio_features(kwargs) {
        const audio_hidden_size = kwargs.audio_features.dims.at(-1);
        const reshaped_audio_features = kwargs.audio_features.view(-1, audio_hidden_size);

        // This export has no per-example audio frame lengths. A padded ragged
        // batch cannot be scattered using only a total feature count.
        // @ts-ignore
        const audio_token_id = this.config.ignore_index ?? this.config.audio_token_id ?? this.config.audio_token_index;
        const counts = kwargs.input_ids.tolist().map((ids) => ids.filter((id) => id == audio_token_id).length);
        const features_per_sample = reshaped_audio_features.dims[0] / counts.length;
        if (counts.some((count) => count !== features_per_sample)) {
            throw new Error(
                'Ultravox requires equal audio feature counts per sample; variable-length padded audio is unsupported.',
            );
        }
        return default_merge_input_ids_with_audio_features({
            audio_token_id,
            ...kwargs,
            audio_features: reshaped_audio_features,
        });
    }
}
