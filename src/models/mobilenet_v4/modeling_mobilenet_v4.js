import { PreTrainedModel } from '../modeling_utils.js';
import { SequenceClassifierOutput } from '../modeling_outputs.js';

export class MobileNetV4PreTrainedModel extends PreTrainedModel {}

/**
 * The bare MobileNetV4 model outputting raw hidden-states without any specific head on top.
 */
export class MobileNetV4Model extends MobileNetV4PreTrainedModel {}

/**
 * MobileNetV4 model with an image classification head on top (a linear layer on top of the pooled features),
 * e.g. for ImageNet.
 */
export class MobileNetV4ForImageClassification extends MobileNetV4PreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
export class MobileNetV4ForSemanticSegmentation extends MobileNetV4PreTrainedModel {}
