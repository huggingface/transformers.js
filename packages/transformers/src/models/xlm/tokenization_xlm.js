import { PreTrainedTokenizer } from '../../tokenization_utils.js';
import { logger } from '../../utils/logger.js';

export class XLMTokenizer extends PreTrainedTokenizer {
    return_token_type_ids = true;

    constructor(tokenizerJSON, tokenizerConfig) {
        super(tokenizerJSON, tokenizerConfig);
        logger.warn(
            'WARNING: `XLMTokenizer` is not yet supported by Hugging Face\'s "fast" tokenizers library. Therefore, you may experience slightly inaccurate results.',
        );
    }
}
