import { PreTrainedModel } from '../modeling_utils.js';

/**
 * Vision Encoder-Decoder model based on OpenAI's GPT architecture for image captioning and other vision tasks
 */
export class VisionEncoderDecoderModel extends PreTrainedModel {
    main_input_name = 'pixel_values';
    forward_params = [
        // Encoder inputs
        'pixel_values',

        // Decoder inpputs
        'decoder_input_ids',
        'encoder_hidden_states',
        'past_key_values',
    ];
}
