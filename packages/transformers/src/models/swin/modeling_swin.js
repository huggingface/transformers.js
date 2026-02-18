import { PreTrainedModel } from '../modeling_utils.js';
import { SequenceClassifierOutput } from '../modeling_outputs.js';

export class SwinPreTrainedModel extends PreTrainedModel {}
export class SwinModel extends SwinPreTrainedModel {}
export class SwinForImageClassification extends SwinPreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
export class SwinForSemanticSegmentation extends SwinPreTrainedModel {}
