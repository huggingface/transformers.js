import { PreTrainedModel } from '../_base/pre-trained-model.js';
import { SequenceClassifierOutput } from '../_base/output.js';

export class EfficientNetPreTrainedModel extends PreTrainedModel {}

/**
 * The bare EfficientNet model outputting raw features without any specific head on top.
 */
export class EfficientNetModel extends EfficientNetPreTrainedModel {}

/**
 * EfficientNet Model with an image classification head on top (a linear layer on top of the pooled features).
 */
export class EfficientNetForImageClassification extends EfficientNetPreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
