import { Pipeline, prepareImages } from './_base.js';

import { Tensor, topk } from '../utils/tensor.js';
import { softmax } from '../utils/maths.js';

/**
 * @typedef {import('./_base.js').ImagePipelineConstructorArgs} ImagePipelineConstructorArgs
 * @typedef {import('./_base.js').Disposable} Disposable
 * @typedef {import('./_base.js').ImagePipelineInputs} ImagePipelineInputs
 */

/**
 * @typedef {Object} ImageClassificationSingle
 * @property {string} label The label identified by the model.
 * @property {number} score The score attributed by the model for that label.
 * @typedef {ImageClassificationSingle[]} ImageClassificationOutput
 *
 * @typedef {Object} ImageClassificationPipelineOptions Parameters specific to image classification pipelines.
 * @property {number} [top_k=1] The number of top labels that will be returned by the pipeline.
 *
 * @callback ImageClassificationPipelineCallback Assign labels to the image(s) passed as inputs.
 * @param {ImagePipelineInputs} images The input images(s) to be classified.
 * @param {ImageClassificationPipelineOptions} [options] The options to use for image classification.
 * @returns {Promise<ImageClassificationOutput|ImageClassificationOutput[]>} An array or object containing the predicted labels and scores.
 *
 * @typedef {ImagePipelineConstructorArgs & ImageClassificationPipelineCallback & Disposable} ImageClassificationPipelineType
 */

/**
 * Image classification pipeline using any `AutoModelForImageClassification`.
 * This pipeline predicts the class of an image.
 *
 * **Example:** Classify an image.
 * ```javascript
 * const classifier = await pipeline('image-classification', 'Xenova/vit-base-patch16-224');
 * const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/tiger.jpg';
 * const output = await classifier(url);
 * // [
 * //   { label: 'tiger, Panthera tigris', score: 0.632695734500885 },
 * // ]
 * ```
 *
 * **Example:** Classify an image and return top `n` classes.
 * ```javascript
 * const classifier = await pipeline('image-classification', 'Xenova/vit-base-patch16-224');
 * const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/tiger.jpg';
 * const output = await classifier(url, { top_k: 3 });
 * // [
 * //   { label: 'tiger, Panthera tigris', score: 0.632695734500885 },
 * //   { label: 'tiger cat', score: 0.3634825646877289 },
 * //   { label: 'lion, king of beasts, Panthera leo', score: 0.00045060308184474707 },
 * // ]
 * ```
 *
 * **Example:** Classify an image and return all classes.
 * ```javascript
 * const classifier = await pipeline('image-classification', 'Xenova/vit-base-patch16-224');
 * const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/tiger.jpg';
 * const output = await classifier(url, { top_k: 0 });
 * // [
 * //   { label: 'tiger, Panthera tigris', score: 0.632695734500885 },
 * //   { label: 'tiger cat', score: 0.3634825646877289 },
 * //   { label: 'lion, king of beasts, Panthera leo', score: 0.00045060308184474707 },
 * //   { label: 'jaguar, panther, Panthera onca, Felis onca', score: 0.00035465499968267977 },
 * //   ...
 * // ]
 * ```
 */
export class ImageClassificationPipeline
    extends /** @type {new (options: ImagePipelineConstructorArgs) => ImageClassificationPipelineType} */ (Pipeline)
{
    /**
     * Create a new ImageClassificationPipeline.
     * @param {ImagePipelineConstructorArgs} options An object used to instantiate the pipeline.
     */
    constructor(options) {
        super(options);
    }

    /** @type {ImageClassificationPipelineCallback} */
    async _call(images, { top_k = 5 } = {}) {
        const preparedImages = await prepareImages(images);

        const { pixel_values } = await this.processor(preparedImages);
        const output = await this.model({ pixel_values });

        // @ts-expect-error TS2339
        const id2label = this.model.config.id2label;

        /** @type {ImageClassificationOutput[]} */
        const toReturn = [];
        for (const batch of output.logits) {
            const scores = await topk(new Tensor('float32', softmax(batch.data), batch.dims), top_k);

            const values = scores[0].tolist();
            const indices = scores[1].tolist();

            const vals = indices.map((x, i) => ({
                label: /** @type {string} */ (id2label ? id2label[x] : `LABEL_${x}`),
                score: /** @type {number} */ (values[i]),
            }));
            toReturn.push(vals);
        }

        return Array.isArray(images) ? toReturn : toReturn[0];
    }
}
