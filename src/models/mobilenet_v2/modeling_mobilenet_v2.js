import { PreTrainedModel } from '../_base/pre-trained-model.js';
import { SequenceClassifierOutput } from '../_base/output.js';

export class MobileNetV2PreTrainedModel extends PreTrainedModel {}

/**
 * The bare MobileNetV2 model outputting raw hidden-states without any specific head on top.
 */
export class MobileNetV2Model extends MobileNetV2PreTrainedModel {}

/**
 * MobileNetV2 model with an image classification head on top (a linear layer on top of the pooled features),
 * e.g. for ImageNet.
 */
export class MobileNetV2ForImageClassification extends MobileNetV2PreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
export class MobileNetV2ForSemanticSegmentation extends MobileNetV2PreTrainedModel {}
