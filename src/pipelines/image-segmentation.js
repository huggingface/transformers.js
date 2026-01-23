import { Pipeline, prepareImages } from './_base.js';

import { RawImage } from '../utils/image.js';

/**
 * @typedef {import('./_base.js').ImagePipelineConstructorArgs} ImagePipelineConstructorArgs
 * @typedef {import('./_base.js').Disposable} Disposable
 * @typedef {import('./_base.js').ImagePipelineInputs} ImagePipelineInputs
 */

const SUBTASKS_MAPPING = {
    // Mapping of subtasks to their corresponding post-processing function names.
    panoptic: 'post_process_panoptic_segmentation',
    instance: 'post_process_instance_segmentation',
    semantic: 'post_process_semantic_segmentation',
};

/**
 * @typedef {Object} ImageSegmentationOutputSingle
 * @property {string|null} label The label of the segment.
 * @property {number|null} score The score of the segment.
 * @property {RawImage} mask The mask of the segment.
 *
 * @typedef {ImageSegmentationOutputSingle[]} ImageSegmentationOutput
 *
 * @typedef {Object} ImageSegmentationPipelineOptions Parameters specific to image segmentation pipelines.
 * @property {number} [threshold=0.5] Probability threshold to filter out predicted masks.
 * @property {number} [mask_threshold=0.5] Threshold to use when turning the predicted masks into binary values.
 * @property {number} [overlap_mask_area_threshold=0.8] Mask overlap threshold to eliminate small, disconnected segments.
 * @property {null|string} [subtask=null] Segmentation task to be performed. One of [`panoptic`, `instance`, and `semantic`],
 * depending on model capabilities. If not set, the pipeline will attempt to resolve (in that order).
 * @property {number[]} [label_ids_to_fuse=null] List of label ids to fuse. If not set, do not fuse any labels.
 * @property {number[][]} [target_sizes=null] List of target sizes for the input images. If not set, use the original image sizes.
 *
 * @callback ImageSegmentationPipelineCallback Segment the input images.
 * @param {ImagePipelineInputs} images The input images.
 * @param {ImageSegmentationPipelineOptions} [options] The options to use for image segmentation.
 * @returns {Promise<ImageSegmentationOutput>} The annotated segments.
 *
 * @typedef {ImagePipelineConstructorArgs & ImageSegmentationPipelineCallback & Disposable} ImageSegmentationPipelineType
 */

/**
 * Image segmentation pipeline using any `AutoModelForXXXSegmentation`.
 * This pipeline predicts masks of objects and their classes.
 *
 * **Example:** Perform image segmentation with `Xenova/detr-resnet-50-panoptic`.
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const segmenter = await pipeline('image-segmentation', 'Xenova/detr-resnet-50-panoptic');
 * const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg';
 * const output = await segmenter(url);
 * // [
 * //   { label: 'remote', score: 0.9984649419784546, mask: RawImage { ... } },
 * //   { label: 'cat', score: 0.9994316101074219, mask: RawImage { ... } }
 * // ]
 * ```
 */
export class ImageSegmentationPipeline
    extends /** @type {new (options: ImagePipelineConstructorArgs) => ImageSegmentationPipelineType} */ (Pipeline)
{
    /** @type {ImageSegmentationPipelineCallback} */
    async _call(
        images,
        {
            threshold = 0.5,
            mask_threshold = 0.5,
            overlap_mask_area_threshold = 0.8,
            label_ids_to_fuse = null,
            target_sizes = null,
            subtask = null,
        } = {},
    ) {
        const isBatched = Array.isArray(images);

        if (isBatched && images.length !== 1) {
            throw Error('Image segmentation pipeline currently only supports a batch size of 1.');
        }

        const preparedImages = await prepareImages(images);
        const imageSizes = preparedImages.map((x) => [x.height, x.width]);

        const inputs = await this.processor(preparedImages);

        const { inputNames, outputNames } = this.model.sessions['model'];
        if (!inputNames.includes('pixel_values')) {
            if (inputNames.length !== 1) {
                throw Error(`Expected a single input name, but got ${inputNames.length} inputs: ${inputNames}.`);
            }

            const newName = inputNames[0];
            if (newName in inputs) {
                throw Error(`Input name ${newName} already exists in the inputs.`);
            }
            // To ensure compatibility with certain background-removal models,
            // we may need to perform a mapping of input to output names
            inputs[newName] = inputs.pixel_values;
        }

        const output = await this.model(inputs);

        let fn = null;
        if (subtask !== null) {
            fn = SUBTASKS_MAPPING[subtask];
        } else if (this.processor.image_processor) {
            for (const [task, func] of Object.entries(SUBTASKS_MAPPING)) {
                if (func in this.processor.image_processor) {
                    fn = this.processor.image_processor[func].bind(this.processor.image_processor);
                    subtask = task;
                    break;
                }
            }
        }

        // @ts-expect-error TS2339
        const id2label = this.model.config.id2label;

        /** @type {ImageSegmentationOutput} */
        const annotation = [];
        if (!subtask) {
            // We define an epsilon to safeguard against numerical/precision issues when detecting
            // the normalization mode of the output (i.e., sigmoid already applied, or not).
            // See https://github.com/microsoft/onnxruntime/issues/23943 for more information.
            const epsilon = 1e-5;

            // Perform standard image segmentation
            const result = output[outputNames[0]];
            for (let i = 0; i < imageSizes.length; ++i) {
                const size = imageSizes[i];
                const item = result[i];
                if (item.data.some((x) => x < -epsilon || x > 1 + epsilon)) {
                    item.sigmoid_();
                }
                const mask = await RawImage.fromTensor(item.mul_(255).to('uint8')).resize(size[1], size[0]);
                annotation.push({
                    label: null,
                    score: null,
                    mask,
                });
            }
        } else if (subtask === 'panoptic' || subtask === 'instance') {
            const processed = fn(
                output,
                threshold,
                mask_threshold,
                overlap_mask_area_threshold,
                label_ids_to_fuse,
                target_sizes ?? imageSizes, // TODO FIX?
            )[0];

            const segmentation = processed.segmentation;

            for (const segment of processed.segments_info) {
                const maskData = new Uint8ClampedArray(segmentation.data.length);
                for (let i = 0; i < segmentation.data.length; ++i) {
                    if (segmentation.data[i] === segment.id) {
                        maskData[i] = 255;
                    }
                }

                const mask = new RawImage(maskData, segmentation.dims[1], segmentation.dims[0], 1);

                annotation.push({
                    score: segment.score,
                    label: id2label[segment.label_id],
                    mask: mask,
                });
            }
        } else if (subtask === 'semantic') {
            const { segmentation, labels } = fn(output, target_sizes ?? imageSizes)[0];

            for (const label of labels) {
                const maskData = new Uint8ClampedArray(segmentation.data.length);
                for (let i = 0; i < segmentation.data.length; ++i) {
                    if (segmentation.data[i] === label) {
                        maskData[i] = 255;
                    }
                }

                const mask = new RawImage(maskData, segmentation.dims[1], segmentation.dims[0], 1);

                annotation.push({
                    score: null,
                    label: id2label[label],
                    mask: mask,
                });
            }
        } else {
            throw Error(`Subtask ${subtask} not supported.`);
        }

        return annotation;
    }
}
