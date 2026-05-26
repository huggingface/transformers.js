import { PreTrainedTokenizer } from '../../tokenization_utils.js';

export class ConvBertTokenizer extends PreTrainedTokenizer {
    /** @override */
    return_token_type_ids = true;
}
