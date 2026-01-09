import { PreTrainedModel } from '../_base/pre-trained-model.js';

/**
 * The bare Gemma Model outputting raw hidden-states without any specific head on top.
 */
export class GemmaPreTrainedModel extends PreTrainedModel {}

/**
 * The bare Gemma Model outputting raw hidden-states without any specific head on top.
 */
export class GemmaModel extends GemmaPreTrainedModel {}

export class GemmaForCausalLM extends GemmaPreTrainedModel {}
