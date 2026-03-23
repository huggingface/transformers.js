import { PreTrainedModel } from '../modeling_utils.js';

export class BlenderbotPreTrainedModel extends PreTrainedModel {}

/**
 * The bare Blenderbot Model outputting raw hidden-states without any specific head on top.
 */
export class BlenderbotModel extends BlenderbotPreTrainedModel {}

/**
 * The Blenderbot Model with a language modeling head. Can be used for summarization.
 */
export class BlenderbotForConditionalGeneration extends BlenderbotPreTrainedModel {}
