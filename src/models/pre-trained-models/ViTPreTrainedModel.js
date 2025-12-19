import { PreTrainedModel } from '../pre-trained-model.js';
import { SequenceClassifierOutput } from '../output.js';

export class ViTPreTrainedModel extends PreTrainedModel {}
export class ViTModel extends ViTPreTrainedModel {}
export class ViTForImageClassification extends ViTPreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
