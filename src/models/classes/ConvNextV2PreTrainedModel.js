import { PreTrainedModel } from '../models.js';
import { SequenceClassifierOutput } from './_base.js';

export class ConvNextV2PreTrainedModel extends PreTrainedModel {}

/**
 * The bare ConvNextV2 model outputting raw features without any specific head on top.
 */
export class ConvNextV2Model extends ConvNextV2PreTrainedModel {}

/**
 * ConvNextV2 Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for ImageNet.
 */
export class ConvNextV2ForImageClassification extends ConvNextV2PreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
