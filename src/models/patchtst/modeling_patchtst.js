import { PreTrainedModel } from '../_base/pre-trained-model.js';

export class PatchTSTPreTrainedModel extends PreTrainedModel {}

/**
 * The bare PatchTST Model outputting raw hidden-states without any specific head.
 */
export class PatchTSTModel extends PatchTSTPreTrainedModel {}

/**
 * The PatchTST for prediction model.
 */
export class PatchTSTForPrediction extends PatchTSTPreTrainedModel {}
