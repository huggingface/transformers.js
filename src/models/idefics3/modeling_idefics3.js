import { PreTrainedModel, default_merge_input_ids_with_image_features } from '../modeling_utils.js';
import { sessionRun } from '../session.js';

export class Idefics3PreTrainedModel extends PreTrainedModel {
    forward_params = [
        'input_ids',
        'attention_mask',
        'pixel_values',
        'pixel_attention_mask',
        'position_ids',
        'past_key_values',
    ];
}

/**
 * The Idefics3 model which consists of a vision backbone and a language model.
 */
export class Idefics3ForConditionalGeneration extends Idefics3PreTrainedModel {
    async encode_image({ pixel_values, pixel_attention_mask }) {
        const features = (await sessionRun(this.sessions['vision_encoder'], { pixel_values, pixel_attention_mask }))
            .image_features;
        return features;
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
}

/**
 * The SmolVLM Model with a language modeling head.
 * It is made up a SigLIP vision encoder, with a language modeling head on top.
 */
export class SmolVLMForConditionalGeneration extends Idefics3ForConditionalGeneration {}
