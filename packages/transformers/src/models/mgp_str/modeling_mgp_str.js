import { PreTrainedModel } from '../modeling_utils.js';
import { ModelOutput } from '../modeling_outputs.js';

export class MgpstrModelOutput extends ModelOutput {
    constructor({ char_logits, bpe_logits, wp_logits }) {
        super();
        this.char_logits = char_logits;
        this.bpe_logits = bpe_logits;
        this.wp_logits = wp_logits;
    }

    get logits() {
        return [this.char_logits, this.bpe_logits, this.wp_logits];
    }
}

export class MgpstrPreTrainedModel extends PreTrainedModel {}

/**
 * MGP-STR Model transformer with three classification heads on top
 * (three A^3 modules and three linear layer on top of the transformer encoder output) for scene text recognition (STR).
 */
export class MgpstrForSceneTextRecognition extends MgpstrPreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new MgpstrModelOutput(await super._call(model_inputs));
    }
}
