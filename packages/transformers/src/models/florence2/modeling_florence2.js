import { PreTrainedModel, encoder_forward, decoder_forward } from '../modeling_utils.js';
import { cat, ones } from '../../utils/tensor.js';

export class Florence2PreTrainedModel extends PreTrainedModel {
    forward_params = [
        // Encoder inputs
        'input_ids',
        'inputs_embeds',
        'attention_mask',
        'pixel_values',

        // Decoder inputs
        'encoder_outputs',
        'decoder_input_ids',
        'decoder_inputs_embeds',
        'decoder_attention_mask',
        'past_key_values',
    ];
    main_input_name = 'inputs_embeds';
}

export class Florence2ForConditionalGeneration extends Florence2PreTrainedModel {
    _merge_input_ids_with_image_features({ inputs_embeds, image_features, input_ids, attention_mask }) {
        return {
            inputs_embeds: cat(
                [
                    image_features, // image embeds
                    inputs_embeds, // task prefix embeds
                ],
                1,
            ),
            attention_mask: cat(
                [
                    ones(image_features.dims.slice(0, 2)), // image attention mask
                    attention_mask, // task prefix attention mask
                ],
                1,
            ),
        };
    }

    async _prepare_inputs_embeds({ input_ids, pixel_values, inputs_embeds, attention_mask }) {
        if (!input_ids && !pixel_values) {
            throw new Error('Either `input_ids` or `pixel_values` should be provided.');
        }

        // 1. Possibly, extract the input embeddings
        let text_features, image_features;
        if (input_ids) {
            text_features = await this.encode_text({ input_ids });
        }
        if (pixel_values) {
            image_features = await this.encode_image({ pixel_values });
        }

        // 2. Possibly, merge text and images
        if (text_features && image_features) {
            ({ inputs_embeds, attention_mask } = this._merge_input_ids_with_image_features({
                inputs_embeds: text_features,
                image_features,
                input_ids,
                attention_mask,
            }));
        } else {
            inputs_embeds = text_features || image_features;
        }

        return { inputs_embeds, attention_mask };
    }

    async forward({
        input_ids,
        pixel_values,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        encoder_outputs,
        past_key_values,

        inputs_embeds,
        decoder_inputs_embeds,
    }) {
        if (!inputs_embeds) {
            ({ inputs_embeds, attention_mask } = await this._prepare_inputs_embeds({
                input_ids,
                pixel_values,
                inputs_embeds,
                attention_mask,
            }));
        }

        if (!encoder_outputs) {
            // Must compute encoder outputs
            let { last_hidden_state } = await encoder_forward(this, { inputs_embeds, attention_mask });
            encoder_outputs = last_hidden_state;
        }

        if (!decoder_inputs_embeds) {
            if (!decoder_input_ids) {
                throw new Error('Either `decoder_input_ids` or `decoder_inputs_embeds` should be provided.');
            }
            decoder_inputs_embeds = await this.encode_text({ input_ids: decoder_input_ids });
        }

        const decoderFeeds = {
            inputs_embeds: decoder_inputs_embeds,
            attention_mask: decoder_attention_mask,
            encoder_attention_mask: attention_mask,
            encoder_hidden_states: encoder_outputs,
            past_key_values,
        };
        return await decoder_forward(this, decoderFeeds, true);
    }
}
