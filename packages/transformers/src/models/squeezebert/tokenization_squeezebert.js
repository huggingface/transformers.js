import { PreTrainedTokenizer } from '../../tokenization_utils.js';

export class SqueezeBertTokenizer extends PreTrainedTokenizer {
    /** @override */
    return_token_type_ids = true;
}
