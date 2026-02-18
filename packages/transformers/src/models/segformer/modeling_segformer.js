import { PreTrainedModel } from '../modeling_utils.js';

export class SegformerPreTrainedModel extends PreTrainedModel {}

/**
 * The bare SegFormer encoder (Mix-Transformer) outputting raw hidden-states without any specific head on top.
 */
export class SegformerModel extends SegformerPreTrainedModel {}

/**
 * SegFormer Model transformer with an image classification head on top (a linear layer on top of the final hidden states) e.g. for ImageNet.
 */
export class SegformerForImageClassification extends SegformerPreTrainedModel {}

/**
 * SegFormer Model transformer with an all-MLP decode head on top e.g. for ADE20k, CityScapes.
 */
export class SegformerForSemanticSegmentation extends SegformerPreTrainedModel {}
