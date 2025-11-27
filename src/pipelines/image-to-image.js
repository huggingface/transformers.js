import { Pipeline, prepareImages } from './_base.js';
import { RawImage } from '../utils/image.js';

/**
 * @typedef {import('./_base.js').ImagePipelineConstructorArgs} ImagePipelineConstructorArgs
 * @typedef {import('./_base.js').Disposable} Disposable
 * @typedef {import('./_base.js').ImagePipelineInputs} ImagePipelineInputs
 */

/**
 * @callback ImageToImagePipelineCallback Transform the image(s) passed as inputs.
 * @param {ImagePipelineInputs} images The images to transform.
 * @returns {Promise<RawImage|RawImage[]>} The transformed image or list of images.
 *
 * @typedef {ImagePipelineConstructorArgs & ImageToImagePipelineCallback & Disposable} ImageToImagePipelineType
 */

/**
 * Image to Image pipeline using any `AutoModelForImageToImage`. This pipeline generates an image based on a previous image input.
 *
 * **Example:** Super-resolution w/ `Xenova/swin2SR-classical-sr-x2-64`
 * ```javascript
 * const upscaler = await pipeline('image-to-image', 'Xenova/swin2SR-classical-sr-x2-64');
 * const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/butterfly.jpg';
 * const output = await upscaler(url);
 * // RawImage {
 * //   data: Uint8Array(786432) [ 41, 31, 24,  43, ... ],
 * //   width: 512,
 * //   height: 512,
 * //   channels: 3
 * // }
 * ```
 */
export class ImageToImagePipeline
    extends /** @type {new (options: ImagePipelineConstructorArgs) => ImageToImagePipelineType} */ (Pipeline)
{
    /**
     * Create a new ImageToImagePipeline.
     * @param {ImagePipelineConstructorArgs} options An object used to instantiate the pipeline.
     */
    constructor(options) {
        super(options);
    }

    /** @type {ImageToImagePipelineCallback} */
    async _call(images) {
        const preparedImages = await prepareImages(images);
        const inputs = await this.processor(preparedImages);
        const outputs = await this.model(inputs);

        /** @type {RawImage[]} */
        const toReturn = [];
        for (const batch of outputs.reconstruction) {
            const output = batch.squeeze().clamp_(0, 1).mul_(255).round_().to('uint8');
            toReturn.push(RawImage.fromTensor(output));
        }

        return toReturn.length > 1 ? toReturn : toReturn[0];
    }
}
