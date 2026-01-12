import { PreTrainedModel } from '../modeling_utils.js';
import { SequenceClassifierOutput } from '../modeling_outputs.js';

export class MobileNetV3PreTrainedModel extends PreTrainedModel {}

/**
 * The bare MobileNetV3 model outputting raw hidden-states without any specific head on top.
 */
export class MobileNetV3Model extends MobileNetV3PreTrainedModel {}

/**
 * MobileNetV3 model with an image classification head on top (a linear layer on top of the pooled features),
 * e.g. for ImageNet.
 */
export class MobileNetV3ForImageClassification extends MobileNetV3PreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
export class MobileNetV3ForSemanticSegmentation extends MobileNetV3PreTrainedModel {}
