import { PreTrainedModel } from '../models.js';

/**
 * The bare Cohere Model outputting raw hidden-states without any specific head on top.
 */
export class CoherePreTrainedModel extends PreTrainedModel {}
export class CohereModel extends CoherePreTrainedModel {}

export class CohereForCausalLM extends CoherePreTrainedModel {}
