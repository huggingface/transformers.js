import { PreTrainedTokenizer } from '../../base/tokenization_utils.js';

export class LlamaTokenizer extends PreTrainedTokenizer {
    padding_side = 'left';
}
