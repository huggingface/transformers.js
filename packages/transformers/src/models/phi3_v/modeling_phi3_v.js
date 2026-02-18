import { PreTrainedModel, decoder_forward } from '../modeling_utils.js';
import { sessionRun } from '../session.js';
import { Tensor } from '../../utils/tensor.js';

export class Phi3VPreTrainedModel extends PreTrainedModel {
    forward_params = [
        'input_ids',
        'inputs_embeds',
        'attention_mask',
        'position_ids',
        'pixel_values',
        'image_sizes',
        'past_key_values',
    ];
}
export class Phi3VForCausalLM extends Phi3VPreTrainedModel {
    async forward({
        // Produced by the tokenizer/processor:
        input_ids = null,
        attention_mask = null,
        pixel_values = null,
        image_sizes = null,

        // Used during generation:
        position_ids = null,
        inputs_embeds = null,
        past_key_values = null,

        // Generic generation parameters
        generation_config = null,
        logits_processor = null,

        // TODO: needed?
        ...kwargs
    }) {
        if (!inputs_embeds) {
            let image_features;
            if (pixel_values && input_ids.dims[1] !== 1) {
                if (!image_sizes) {
                    throw new Error('`image_sizes` must be provided when `pixel_values` is provided.');
                }

                // Encode the image
                ({ image_features } = await sessionRun(this.sessions['vision_encoder'], {
                    pixel_values,
                    image_sizes,
                }));
            } else {
                const hidden_size = this.config.normalized_config.hidden_size;
                image_features = new Tensor('float32', [], [0, hidden_size]);
            }

            ({ inputs_embeds } = await sessionRun(this.sessions['prepare_inputs_embeds'], {
                input_ids,
                image_features,
            }));
        }

        const outputs = await decoder_forward(
            this,
            {
                inputs_embeds,
                past_key_values,
                attention_mask,
                position_ids,
                generation_config,
                logits_processor,
            },
            false,
        );
        return outputs;
    }
}
