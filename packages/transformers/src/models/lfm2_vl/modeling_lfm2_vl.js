import { LlavaForConditionalGeneration } from '../llava/modeling_llava.js';
import { sessionRun } from '../session.js';

export class Lfm2VlForConditionalGeneration extends LlavaForConditionalGeneration {
    forward_params = [
        'input_ids',
        'attention_mask',
        'pixel_values',
        'pixel_attention_mask',
        'spatial_shapes',
        'position_ids',
        'past_key_values',
    ];

    async encode_image({ pixel_values, pixel_attention_mask, spatial_shapes }) {
        return (await sessionRun(this.sessions['vision_encoder'], {
            pixel_values, pixel_attention_mask, spatial_shapes,
        })).image_features;
    }
}
