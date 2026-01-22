import { PreTrainedModel } from '../modeling_utils.js';

export class GPT2PreTrainedModel extends PreTrainedModel {}

export class GPT2Model extends GPT2PreTrainedModel {}

/**
 * GPT-2 language model head on top of the GPT-2 base model. This model is suitable for text generation tasks.
 */
export class GPT2LMHeadModel extends GPT2PreTrainedModel {}
// export class GPT2ForSequenceClassification extends GPT2PreTrainedModel {
// TODO
