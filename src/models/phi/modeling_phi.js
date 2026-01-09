import { PreTrainedModel } from '../_base/pre-trained-model.js';

export class PhiPreTrainedModel extends PreTrainedModel {}

/**
 * The bare Phi Model outputting raw hidden-states without any specific head on top.
 */
export class PhiModel extends PhiPreTrainedModel {}

export class PhiForCausalLM extends PhiPreTrainedModel {}
