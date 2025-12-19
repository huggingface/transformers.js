import { PreTrainedModel } from '../pre-trained-model.js';

/**
 * The bare Falcon Model outputting raw hidden-states without any specific head on top.
 */
export class FalconPreTrainedModel extends PreTrainedModel {}

export class FalconModel extends FalconPreTrainedModel {}

export class FalconForCausalLM extends FalconPreTrainedModel {}
