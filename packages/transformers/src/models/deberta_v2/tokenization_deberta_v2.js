import { PreTrainedTokenizer } from '../../tokenization_utils.js';

export class DebertaV2Tokenizer extends PreTrainedTokenizer {
    /** @override */
    return_token_type_ids = true;
}
