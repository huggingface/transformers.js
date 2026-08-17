import { PreTrainedTokenizer, loadTokenizer } from '../../tokenization_utils.js';
import { Qwen2Tokenizer } from '../qwen2/tokenization_qwen2.js';
import { logger } from '../../utils/logger.js';

const TOKENIZERS = { Qwen2Tokenizer };

export class AutoTokenizer {
    static async from_pretrained(
        pretrained_model_name_or_path,
        { progress_callback = null, config = null, cache_dir = null, local_files_only = false, revision = 'main' } = {},
    ) {
        const [tokenizerJSON, tokenizerConfig] = await loadTokenizer(pretrained_model_name_or_path, {
            progress_callback,
            config,
            cache_dir,
            local_files_only,
            revision,
        });
        const tokenizerName = tokenizerConfig.tokenizer_class?.replace(/Fast$/, '') ?? 'PreTrainedTokenizer';
        const cls = TOKENIZERS[tokenizerName] ?? PreTrainedTokenizer;
        if (!(tokenizerName in TOKENIZERS) && tokenizerName !== 'PreTrainedTokenizer') {
            logger.warn(`Unknown tokenizer class "${tokenizerName}", attempting to construct from base class.`);
        }
        return new cls(tokenizerJSON, tokenizerConfig);
    }
}
