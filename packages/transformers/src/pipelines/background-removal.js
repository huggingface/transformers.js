import { ImageSegmentationPipeline } from './image-segmentation.js';
import { prepareImages } from './_base.js';
import { RawImage } from '../utils/image.js';

/**
 * @typedef {import('./_base.js').ImagePipelineConstructorArgs} ImagePipelineConstructorArgs
 * @typedef {import('./_base.js').Disposable} Disposable
 * @typedef {import('./_base.js').ImageInput} ImageInput
 */

/**
 * @typedef {Object} BackgroundRemovalPipelineOptions Parameters specific to background removal pipelines.
 *
 * @typedef {ImagePipelineConstructorArgs & BackgroundRemovalPipelineCallback & Disposable} BackgroundRemovalPipelineType
 */

/**
 * @template T
 * @typedef {T extends ImageInput[] ? RawImage[] : RawImage} BackgroundRemovalPipelineResult
 */

/**
 * @typedef {<T extends ImageInput | ImageInput[]>(images: T, options?: BackgroundRemovalPipelineOptions) => Promise<BackgroundRemovalPipelineResult<T>>} BackgroundRemovalPipelineCallback
 */

/**
 * Background removal pipeline using certain `AutoModelForXXXSegmentation`.
 * This pipeline removes the backgrounds of images.
 *
 * **Example:** Perform background removal with `Xenova/modnet`.
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const segmenter = await pipeline('background-removal', 'Xenova/modnet');
 * const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/portrait-of-woman_small.jpg';
 * const output = await segmenter(url);
 * // RawImage { data: Uint8ClampedArray(648000) [ ... ], width: 360, height: 450, channels: 4 }
 * ```
 */
export class BackgroundRemovalPipeline
    extends /** @type {new (options: ImagePipelineConstructorArgs) => BackgroundRemovalPipelineType} */ (
        /** @type {any} */ (ImageSegmentationPipeline)
    )
{
    async _call(images, options = {}) {
        const preparedImages = await prepareImages(images);

        // @ts-expect-error TS2339
        const masks = await super._call(images, options);
        const result = preparedImages.map((img, i) => {
            const cloned = img.clone();
            cloned.putAlpha(masks[i].mask);
            return cloned;
        });

        return Array.isArray(images) ? result : result[0];
    }
}
