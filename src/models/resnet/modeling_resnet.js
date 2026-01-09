import { PreTrainedModel } from '../_base/pre-trained-model.js';
import { SequenceClassifierOutput } from '../_base/output.js';

/**
 * An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained models.
 */
export class ResNetPreTrainedModel extends PreTrainedModel {}

/**
 * The bare ResNet model outputting raw features without any specific head on top.
 */
export class ResNetModel extends ResNetPreTrainedModel {}

/**
 * ResNet Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for ImageNet.
 */
export class ResNetForImageClassification extends ResNetPreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
