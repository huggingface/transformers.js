import { PreTrainedModel } from '../modeling_utils.js';

export class BlenderbotSmallPreTrainedModel extends PreTrainedModel {}

/**
 * The bare BlenderbotSmall Model outputting raw hidden-states without any specific head on top.
 */
export class BlenderbotSmallModel extends BlenderbotSmallPreTrainedModel {}

/**
 * The BlenderbotSmall Model with a language modeling head. Can be used for summarization.
 */
export class BlenderbotSmallForConditionalGeneration extends BlenderbotSmallPreTrainedModel {}
