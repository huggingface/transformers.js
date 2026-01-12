import { PreTrainedModel } from '../modeling_utils.js';

export class VitPosePreTrainedModel extends PreTrainedModel {}

/**
 * The VitPose model with a pose estimation head on top.
 */
export class VitPoseForPoseEstimation extends VitPosePreTrainedModel {}
