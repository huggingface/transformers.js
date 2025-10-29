import { Pipeline, prepareImages } from './_base.js';

import { Tensor } from '../utils/tensor.js';

/**
 * @typedef {import('./_base.js').TextImagePipelineConstructorArgs} TextImagePipelineConstructorArgs
 * @typedef {import('./_base.js').Disposable} Disposable
 * @typedef {import('./_base.js').ImagePipelineInputs} ImagePipelineInputs
 */

/**
 * @typedef {Object} ImageToTextSingle
 * @property {string} generated_text The generated text.
 * @typedef {ImageToTextSingle[]} ImageToTextOutput
 *
 * @callback ImageToTextPipelineCallback Assign labels to the image(s) passed as inputs.
 * @param {ImagePipelineInputs} texts The images to be captioned.
 * @param {Partial<import('../generation/configuration_utils.js').GenerationConfig>} [options] Additional keyword arguments to pass along to the generate method of the model.
 * @returns {Promise<ImageToTextOutput|ImageToTextOutput[]>} An object (or array of objects) containing the generated text(s).
 *
 * @typedef {TextImagePipelineConstructorArgs & ImageToTextPipelineCallback & Disposable} ImageToTextPipelineType
 */

/**
 * Image To Text pipeline using a `AutoModelForVision2Seq`. This pipeline predicts a caption for a given image.
 *
 * **Example:** Generate a caption for an image w/ `Xenova/vit-gpt2-image-captioning`.
 * ```javascript
 * const captioner = await pipeline('image-to-text', 'Xenova/vit-gpt2-image-captioning');
 * const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg';
 * const output = await captioner(url);
 * // [{ generated_text: 'a cat laying on a couch with another cat' }]
 * ```
 *
 * **Example:** Optical Character Recognition (OCR) w/ `Xenova/trocr-small-handwritten`.
 * ```javascript
 * const captioner = await pipeline('image-to-text', 'Xenova/trocr-small-handwritten');
 * const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/handwriting.jpg';
 * const output = await captioner(url);
 * // [{ generated_text: 'Mr. Brown commented icily.' }]
 * ```
 */
export class ImageToTextPipeline
    extends /** @type {new (options: TextImagePipelineConstructorArgs) => ImageToTextPipelineType} */ (Pipeline)
{
    /**
     * Create a new ImageToTextPipeline.
     * @param {TextImagePipelineConstructorArgs} options An object used to instantiate the pipeline.
     */
    constructor(options) {
        super(options);
    }

    /** @type {ImageToTextPipelineCallback} */
    async _call(images, generate_kwargs = {}) {
        const isBatched = Array.isArray(images);
        const preparedImages = await prepareImages(images);

        const { pixel_values } = await this.processor(preparedImages);

        const toReturn = [];
        for (const batch of pixel_values) {
            batch.dims = [1, ...batch.dims];
            const output = await this.model.generate({ inputs: batch, ...generate_kwargs });
            const decoded = this.tokenizer
                .batch_decode(/** @type {Tensor} */ (output), {
                    skip_special_tokens: true,
                })
                .map((x) => ({ generated_text: x.trim() }));
            toReturn.push(decoded);
        }

        return isBatched ? toReturn : toReturn[0];
    }
}
