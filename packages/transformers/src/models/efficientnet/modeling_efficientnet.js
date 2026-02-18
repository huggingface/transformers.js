import { PreTrainedModel } from '../modeling_utils.js';
import { SequenceClassifierOutput } from '../modeling_outputs.js';

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
