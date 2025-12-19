import { PreTrainedModel } from '../pre-trained-model.js';

export class PatchTSMixerPreTrainedModel extends PreTrainedModel {}

/**
 * The bare PatchTSMixer Model outputting raw hidden-states without any specific head.
 */
export class PatchTSMixerModel extends PatchTSMixerPreTrainedModel {}

/**
 * The PatchTSMixer for prediction model.
 */
export class PatchTSMixerForPrediction extends PatchTSMixerPreTrainedModel {}
