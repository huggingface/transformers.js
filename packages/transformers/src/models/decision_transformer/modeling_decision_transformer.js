import { PreTrainedModel } from '../modeling_utils.js';

export class DecisionTransformerPreTrainedModel extends PreTrainedModel {}

/**
 * The model builds upon the GPT2 architecture to perform autoregressive prediction of actions in an offline RL setting.
 * Refer to the paper for more details: https://huggingface.co/papers/2106.01345
 */
export class DecisionTransformerModel extends DecisionTransformerPreTrainedModel {}
