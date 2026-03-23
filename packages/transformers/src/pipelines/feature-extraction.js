import { Pipeline } from './_base.js';

import { Tensor, mean_pooling, quantize_embeddings } from '../utils/tensor.js';

/**
 * @typedef {import('./_base.js').TextPipelineConstructorArgs} TextPipelineConstructorArgs
 * @typedef {import('./_base.js').Disposable} Disposable
 */

/**
 * @typedef {Object} FeatureExtractionPipelineOptions Parameters specific to feature extraction pipelines.
 * @property {'none'|'mean'|'cls'|'first_token'|'eos'|'last_token'} [pooling="none"] The pooling method to use.
 * @property {boolean} [normalize=false] Whether or not to normalize the embeddings in the last dimension.
 * @property {boolean} [quantize=false] Whether or not to quantize the embeddings.
 * @property {'binary'|'ubinary'} [precision='binary'] The precision to use for quantization.
 *
 * @callback FeatureExtractionPipelineCallback Extract the features of the input(s).
 * @param {string|string[]} texts One or several texts (or one list of texts) to get the features of.
 * @param {FeatureExtractionPipelineOptions} [options] The options to use for feature extraction.
 * @returns {Promise<Tensor>} The features computed by the model.
 *
 * @typedef {TextPipelineConstructorArgs & FeatureExtractionPipelineCallback & Disposable} FeatureExtractionPipelineType
 */

/**
 * Feature extraction pipeline using no model head. This pipeline extracts the hidden
 * states from the base transformer, which can be used as features in downstream tasks.
 *
 * **Example:** Run feature extraction using `onnx-community/all-MiniLM-L6-v2-ONNX` (without pooling or normalization).
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const extractor = await pipeline('feature-extraction', 'onnx-community/all-MiniLM-L6-v2-ONNX');
 * const output = await extractor('This is a simple test.');
 * // Tensor {
 * //   type: 'float32',
 * //   data: Float32Array [0.2157987803220749, -0.09140099585056305, ...],
 * //   dims: [1, 8, 384]
 * // }
 *
 * // You can convert this Tensor to a nested JavaScript array using `.tolist()`:
 * console.log(output.tolist());
 * ```
 *
 * **Example:** Run feature extraction using `onnx-community/all-MiniLM-L6-v2-ONNX` (with pooling and normalization).
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const extractor = await pipeline('feature-extraction', 'onnx-community/all-MiniLM-L6-v2-ONNX');
 * const output = await extractor('This is a simple test.', { pooling: 'mean', normalize: true });
 * // Tensor {
 * //   type: 'float32',
 * //   data: Float32Array [0.09528215229511261, -0.024730168282985687, ...],
 * //   dims: [1, 384]
 * // }
 *
 * // You can convert this Tensor to a nested JavaScript array using `.tolist()`:
 * console.log(output.tolist());
 * ```
 *
 * **Example:** Run feature extraction using `onnx-community/all-MiniLM-L6-v2-ONNX` models (with pooling and binary quantization).
 * ```javascript
 * const extractor = await pipeline('feature-extraction', 'onnx-community/all-MiniLM-L6-v2-ONNX');
 * const output = await extractor('This is a simple test.', { pooling: 'mean', quantize: true, precision: 'binary' });
 * // Tensor {
 * //   type: 'int8',
 * //   data: Int8Array [49, 108, 25, ...],
 * //   dims: [1, 48]
 * // }
 *
 * // You can convert this Tensor to a nested JavaScript array using `.tolist()`:
 * console.log(output.tolist());
 * ```
 */
export class FeatureExtractionPipeline
    extends /** @type {new (options: TextPipelineConstructorArgs) => FeatureExtractionPipelineType} */ (Pipeline)
{
    /** @type {FeatureExtractionPipelineCallback} */
    async _call(
        texts,
        {
            pooling = /** @type {'none'} */ ('none'),
            normalize = false,
            quantize = false,
            precision = /** @type {'binary'} */ ('binary'),
        } = {},
    ) {
        // Run tokenization
        const model_inputs = this.tokenizer(texts, {
            padding: true,
            truncation: true,
        });

        // Run model
        const outputs = await this.model(model_inputs);

        // TODO: Provide warning to the user that they might be using model which was not exported
        // specifically for feature extraction
        // console.log(this.model.config)
        // console.log(outputs)

        /** @type {Tensor} */
        let result = outputs.last_hidden_state ?? outputs.logits ?? outputs.token_embeddings;

        switch (pooling) {
            case 'none':
                // Skip pooling
                break;
            case 'mean':
                result = mean_pooling(result, model_inputs.attention_mask);
                break;
            case 'first_token':
            case 'cls':
                result = result.slice(null, 0);
                break;
            case 'last_token':
            case 'eos':
                result = result.slice(null, -1);
                break;
            default:
                throw Error(`Pooling method '${pooling}' not supported.`);
        }

        if (normalize) {
            result = result.normalize(2, -1);
        }

        if (quantize) {
            result = quantize_embeddings(result, precision);
        }

        return result;
    }
}
