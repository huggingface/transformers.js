import { PreTrainedModel, default_merge_input_ids_with_image_features } from '../modeling_utils.js';

export class PaliGemmaPreTrainedModel extends PreTrainedModel {
    forward_params = [
        'input_ids',
        // 'inputs_embeds',
        'attention_mask',
        'pixel_values',
        'position_ids',
        'past_key_values',
    ];
}

export class PaliGemmaForConditionalGeneration extends PaliGemmaPreTrainedModel {
    _merge_input_ids_with_image_features(kwargs) {
        const vision_hidden_size = kwargs.image_features.dims.at(-1);
        const reshaped_image_hidden_states = kwargs.image_features.view(-1, vision_hidden_size);

        return default_merge_input_ids_with_image_features({
            // @ts-ignore
            image_token_id: this.config.image_token_index,
            ...kwargs,
            image_features: reshaped_image_hidden_states,
        });
    }
}
