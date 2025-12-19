import { PreTrainedModel } from '../models.js';

export class NanoChatPreTrainedModel extends PreTrainedModel {}
export class NanoChatModel extends NanoChatPreTrainedModel {}
export class NanoChatForCausalLM extends NanoChatPreTrainedModel {}
