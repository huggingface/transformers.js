import { ImageSegmentationPipeline } from './image-segmentation.js';
import { prepareImages } from './_base.js';
import { RawImage } from '../utils/image.js';

/**
 * @typedef {import('./_base.js').ImagePipelineConstructorArgs} ImagePipelineConstructorArgs
 * @typedef {import('./_base.js').Disposable} Disposable
 * @typedef {import('./_base.js').ImagePipelineInputs} ImagePipelineInputs
 */

/**
 * @typedef {Object} BackgroundRemovalPipelineOptions Parameters specific to image segmentation pipelines.
 *
 * @callback BackgroundRemovalPipelineCallback Segment the input images.
 * @param {ImagePipelineInputs} images The input images.
 * @param {BackgroundRemovalPipelineOptions} [options] The options to use for image segmentation.
 * @returns {Promise<RawImage[]>} The images with the background removed.
 *
 * @typedef {ImagePipelineConstructorArgs & BackgroundRemovalPipelineCallback & Disposable} BackgroundRemovalPipelineType
 */

/**
 * Background removal pipeline using certain `AutoModelForXXXSegmentation`.
 * This pipeline removes the backgrounds of images.
 *
 * **Example:** Perform background removal with `Xenova/modnet`.
 * ```javascript
 * const segmenter = await pipeline('background-removal', 'Xenova/modnet');
 * const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/portrait-of-woman_small.jpg';
 * const output = await segmenter(url);
 * // [
 * //   RawImage { data: Uint8ClampedArray(648000) [ ... ], width: 360, height: 450, channels: 4 }
 * // ]
 * ```
 */
export class BackgroundRemovalPipeline
    extends /** @type {new (options: ImagePipelineConstructorArgs) => BackgroundRemovalPipelineType} */ (
        /** @type {any} */ (ImageSegmentationPipeline)
    )
{
    /**
     * Create a new BackgroundRemovalPipeline.
     * @param {ImagePipelineConstructorArgs} options An object used to instantiate the pipeline.
     */
    constructor(options) {
        super(options);
    }

    /** @type {BackgroundRemovalPipelineCallback} */
    async _call(images, options = {}) {
        const isBatched = Array.isArray(images);

        if (isBatched && images.length !== 1) {
            throw Error('Background removal pipeline currently only supports a batch size of 1.');
        }

        const preparedImages = await prepareImages(images);

        // @ts-expect-error TS2339
        const masks = await super._call(images, options);
        const result = preparedImages.map((img, i) => {
            const cloned = img.clone();
            cloned.putAlpha(masks[i].mask);
            return cloned;
        });

        return result;
    }
}
