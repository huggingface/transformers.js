import { PreTrainedTokenizer } from '../../tokenization_utils.js';

export class HerbertTokenizer extends PreTrainedTokenizer {
    /** @override */
    return_token_type_ids = true;
}
