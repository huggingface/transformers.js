import { PreTrainedModel } from '../modeling_utils.js';

/**
 * The bare Gemma2 Model outputting raw hidden-states without any specific head on top.
 */
export class Gemma2PreTrainedModel extends PreTrainedModel {}

/**
 * The bare Gemma2 Model outputting raw hidden-states without any specific head on top.
 */
export class Gemma2Model extends Gemma2PreTrainedModel {}

export class Gemma2ForCausalLM extends Gemma2PreTrainedModel {}
