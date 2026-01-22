import { PreTrainedModel } from '../modeling_utils.js';
import { SequenceClassifierOutput } from '../modeling_outputs.js';

export class MobileViTV2PreTrainedModel extends PreTrainedModel {}
export class MobileViTV2Model extends MobileViTV2PreTrainedModel {}
export class MobileViTV2ForImageClassification extends MobileViTV2PreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
// TODO: MobileViTV2ForSemanticSegmentation
