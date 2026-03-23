import { PreTrainedModel } from '../modeling_utils.js';
import { SequenceClassifierOutput } from '../modeling_outputs.js';

export class FastViTPreTrainedModel extends PreTrainedModel {}
export class FastViTModel extends FastViTPreTrainedModel {}
export class FastViTForImageClassification extends FastViTPreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
