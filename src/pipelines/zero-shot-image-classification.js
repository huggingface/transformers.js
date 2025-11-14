import { Pipeline, prepareImages } from './_base.js';

import { softmax } from '../utils/maths.js';

/**
 * @typedef {import('./_base.js').TextImagePipelineConstructorArgs} TextImagePipelineConstructorArgs
 * @typedef {import('./_base.js').Disposable} Disposable
 * @typedef {import('./_base.js').ImagePipelineInputs} ImagePipelineInputs
 */

/**
 * @typedef {Object} ZeroShotImageClassificationOutput
 * @property {string} label The label identified by the model. It is one of the suggested `candidate_label`.
 * @property {number} score The score attributed by the model for that label (between 0 and 1).
 *
 * @typedef {Object} ZeroShotImageClassificationPipelineOptions Parameters specific to zero-shot image classification pipelines.
 * @property {string} [hypothesis_template="This is a photo of {}"] The sentence used in conjunction with `candidate_labels`
 * to attempt the image classification by replacing the placeholder with the candidate_labels.
 * Then likelihood is estimated by using `logits_per_image`.
 *
 * @callback ZeroShotImageClassificationPipelineCallback Assign labels to the image(s) passed as inputs.
 * @param {ImagePipelineInputs} images The input images.
 * @param {string[]} candidate_labels The candidate labels for this image.
 * @param {ZeroShotImageClassificationPipelineOptions} [options] The options to use for zero-shot image classification.
 * @returns {Promise<ZeroShotImageClassificationOutput[]|ZeroShotImageClassificationOutput[][]>} An array of objects containing the predicted labels and scores.
 *
 * @typedef {TextImagePipelineConstructorArgs & ZeroShotImageClassificationPipelineCallback & Disposable} ZeroShotImageClassificationPipelineType
 */

/**
 * Zero shot image classification pipeline. This pipeline predicts the class of
 * an image when you provide an image and a set of `candidate_labels`.
 *
 * **Example:** Zero shot image classification w/ `Xenova/clip-vit-base-patch32`.
 * ```javascript
 * const classifier = await pipeline('zero-shot-image-classification', 'Xenova/clip-vit-base-patch32');
 * const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/tiger.jpg';
 * const output = await classifier(url, ['tiger', 'horse', 'dog']);
 * // [
 * //   { score: 0.9993917942047119, label: 'tiger' },
 * //   { score: 0.0003519294841680676, label: 'horse' },
 * //   { score: 0.0002562698791734874, label: 'dog' }
 * // ]
 * ```
 */
export class ZeroShotImageClassificationPipeline
    extends /** @type {new (options: TextImagePipelineConstructorArgs) => ZeroShotImageClassificationPipelineType} */ (
        Pipeline
    )
{
    /**
     * Create a new ZeroShotImageClassificationPipeline.
     * @param {TextImagePipelineConstructorArgs} options An object used to instantiate the pipeline.
     */
    constructor(options) {
        super(options);
    }

    /** @type {ZeroShotImageClassificationPipelineCallback} */
    async _call(images, candidate_labels, { hypothesis_template = 'This is a photo of {}' } = {}) {
        const isBatched = Array.isArray(images);
        const preparedImages = await prepareImages(images);

        // Insert label into hypothesis template
        const texts = candidate_labels.map((x) => hypothesis_template.replace('{}', x));

        // Run tokenization
        const text_inputs = this.tokenizer(texts, {
            padding: this.model.config.model_type === 'siglip' ? 'max_length' : true,
            truncation: true,
        });

        // Run processor
        const { pixel_values } = await this.processor(preparedImages);

        // Run model with both text and pixel inputs
        const output = await this.model({ ...text_inputs, pixel_values });

        const function_to_apply =
            this.model.config.model_type === 'siglip'
                ? (batch) => batch.sigmoid().data
                : (batch) => softmax(batch.data);

        // Compare each image with each candidate label
        const toReturn = [];
        for (const batch of output.logits_per_image) {
            // Compute softmax per image
            const probs = function_to_apply(batch);

            const result = [...probs].map((x, i) => ({
                score: x,
                label: candidate_labels[i],
            }));
            result.sort((a, b) => b.score - a.score); // sort by score in descending order
            toReturn.push(result);
        }

        return isBatched ? toReturn : toReturn[0];
    }
}
