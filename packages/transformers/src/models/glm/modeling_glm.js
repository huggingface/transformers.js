import { PreTrainedModel } from '../modeling_utils.js';

export class GlmPreTrainedModel extends PreTrainedModel {}
export class GlmModel extends GlmPreTrainedModel {}
export class GlmForCausalLM extends GlmPreTrainedModel {}
