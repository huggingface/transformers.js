import { PreTrainedModel } from '../_base/pre-trained-model.js';

/**
 * The bare Starcoder2 Model outputting raw hidden-states without any specific head on top.
 */
export class Starcoder2PreTrainedModel extends PreTrainedModel {}

export class Starcoder2Model extends Starcoder2PreTrainedModel {}

export class Starcoder2ForCausalLM extends Starcoder2PreTrainedModel {}
