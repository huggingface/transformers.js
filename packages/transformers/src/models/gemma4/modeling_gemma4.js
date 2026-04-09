import { Gemma3nForConditionalGeneration } from '../gemma3n/modeling_gemma3n.js';
import { sessionRun } from '../session.js';

export class Gemma4ForConditionalGeneration extends Gemma3nForConditionalGeneration {
    forward_params = [
        'input_ids',
        'attention_mask',
        'inputs_embeds',
        'per_layer_inputs',

        'position_ids',
        'pixel_values',
        'image_position_ids',
        'input_features',
        'input_features_mask',
        'past_key_values',
    ];

    _encode_vision(kwargs) {
        return sessionRun(this.sessions['vision_encoder'], {
            pixel_values: kwargs.pixel_values,
            pixel_position_ids: kwargs.image_position_ids,
        });
    }
}

export class Gemma4ForCausalLM extends Gemma4ForConditionalGeneration {}
