import { PreTrainedModel, decoder_prepare_inputs_for_generation } from '../modeling_utils.js';
import { ones_like, zeros_like } from '../../utils/tensor.js';

export class HrmTextPreTrainedModel extends PreTrainedModel {}
export class HrmTextModel extends HrmTextPreTrainedModel {}
export class HrmTextForCausalLM extends HrmTextPreTrainedModel {
    forward_params = ['input_ids', 'attention_mask', 'token_type_ids', 'past_key_values'];

    prepare_inputs_for_generation(input_ids, model_inputs, generation_config) {
        const prepared = decoder_prepare_inputs_for_generation(this, input_ids, model_inputs, generation_config);

        // HRM-Text can be pretrained as a PrefixLM.
        // token_type_ids=1 marks tokens inside the bidirectional prefix block,
        // and 0 marks autoregressively generated tokens.
        // Check cached tokens.
        const has_past_tokens = (prepared.past_key_values?.get_seq_length() ?? 0) > 0;
        prepared.token_type_ids = (has_past_tokens ? zeros_like : ones_like)(prepared.input_ids);
        return prepared;
    }
}
