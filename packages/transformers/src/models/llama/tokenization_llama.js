import { PreTrainedTokenizer } from '../../tokenization_utils.js';

export class LlamaTokenizer extends PreTrainedTokenizer {
    /** @override */
    padding_side = 'left';
}
