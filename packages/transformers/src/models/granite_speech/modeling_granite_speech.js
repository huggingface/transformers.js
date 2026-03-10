import { UltravoxModel } from '../ultravox/modeling_ultravox.js';

export class GraniteSpeechForConditionalGeneration extends UltravoxModel {
    forward_params = ['input_ids', 'attention_mask', 'input_features', 'past_key_values'];
}
