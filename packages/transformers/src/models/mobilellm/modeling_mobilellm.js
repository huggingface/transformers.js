import { PreTrainedModel } from '../modeling_utils.js';

export class MobileLLMPreTrainedModel extends PreTrainedModel {}
export class MobileLLMModel extends MobileLLMPreTrainedModel {}
export class MobileLLMForCausalLM extends MobileLLMPreTrainedModel {}
