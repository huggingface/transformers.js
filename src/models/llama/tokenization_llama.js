import { PreTrainedTokenizer } from '../../tokenization_utils.js';

export class LlamaTokenizer extends PreTrainedTokenizer {
    padding_side = 'left';
}
