import { PreTrainedModel } from '../modeling_utils.js';

export class DepthAnythingPreTrainedModel extends PreTrainedModel {}

/**
 * Depth Anything Model with a depth estimation head on top (consisting of 3 convolutional layers) e.g. for KITTI, NYUv2.
 */
export class DepthAnythingForDepthEstimation extends DepthAnythingPreTrainedModel {}
