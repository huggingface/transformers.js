/**
 * @file Tokenizers are used to prepare textual inputs for a model.
 *
 * **Example:** Create an `AutoTokenizer` and use it to tokenize a sentence.
 * This will automatically detect the tokenizer type based on the tokenizer class defined in `tokenizer.json`.
 * ```javascript
 * import { AutoTokenizer } from '@huggingface/transformers';
 *
 * const tokenizer = await AutoTokenizer.from_pretrained('Xenova/bert-base-uncased');
 * const { input_ids } = await tokenizer('I love transformers!');
 * // Tensor {
 * //   data: BigInt64Array(6) [101n, 1045n, 2293n, 19081n, 999n, 102n],
 * //   dims: [1, 6],
 * //   type: 'int64',
 * //   size: 6,
 * // }
 * ```
 *
 * @module tokenizers
 */
import { PreTrainedTokenizer } from './base/tokenization_utils.js';

export { PreTrainedTokenizer };
