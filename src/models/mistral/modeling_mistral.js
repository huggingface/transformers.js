import { PreTrainedModel } from '../_base/pre-trained-model.js';

/**
 * The bare Mistral Model outputting raw hidden-states without any specific head on top.
 */
export class MistralPreTrainedModel extends PreTrainedModel {}

export class MistralModel extends MistralPreTrainedModel {}

export class MistralForCausalLM extends MistralPreTrainedModel {}
