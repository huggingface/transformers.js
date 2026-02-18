import { PreTrainedModel } from '../modeling_utils.js';

export class ASTPreTrainedModel extends PreTrainedModel {}

/**
 * The bare AST Model transformer outputting raw hidden-states without any specific head on top.
 */
export class ASTModel extends ASTPreTrainedModel {}

/**
 * Audio Spectrogram Transformer model with an audio classification head on top
 * (a linear layer on top of the pooled output) e.g. for datasets like AudioSet, Speech Commands v2.
 */
export class ASTForAudioClassification extends ASTPreTrainedModel {}
