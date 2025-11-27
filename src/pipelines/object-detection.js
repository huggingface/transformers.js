import { Pipeline, prepareImages, get_bounding_box } from './_base.js';

/**
 * @typedef {import('./_base.js').ImagePipelineConstructorArgs} ImagePipelineConstructorArgs
 * @typedef {import('./_base.js').Disposable} Disposable
 * @typedef {import('./_base.js').ImagePipelineInputs} ImagePipelineInputs
 * @typedef {import('./_base.js').BoundingBox} BoundingBox
 */

/**
 * @typedef {Object} ObjectDetectionPipelineSingle
 * @property {string} label The class label identified by the model.
 * @property {number} score The score attributed by the model for that label.
 * @property {BoundingBox} box The bounding box of detected object in image's original size, or as a percentage if `percentage` is set to true.
 * @typedef {ObjectDetectionPipelineSingle[]} ObjectDetectionPipelineOutput
 *
 * @typedef {Object} ObjectDetectionPipelineOptions Parameters specific to object detection pipelines.
 * @property {number} [threshold=0.9] The threshold used to filter boxes by score.
 * @property {boolean} [percentage=false] Whether to return the boxes coordinates in percentage (true) or in pixels (false).
 *
 * @callback ObjectDetectionPipelineCallback Detect objects (bounding boxes & classes) in the image(s) passed as inputs.
 * @param {ImagePipelineInputs} images The input images.
 * @param {ObjectDetectionPipelineOptions} [options] The options to use for object detection.
 * @returns {Promise<ObjectDetectionPipelineOutput|ObjectDetectionPipelineOutput[]>} A list of objects or a list of list of objects.
 *
 * @typedef {ImagePipelineConstructorArgs & ObjectDetectionPipelineCallback & Disposable} ObjectDetectionPipelineType
 */

/**
 * Object detection pipeline using any `AutoModelForObjectDetection`.
 * This pipeline predicts bounding boxes of objects and their classes.
 *
 * **Example:** Run object-detection with `Xenova/detr-resnet-50`.
 * ```javascript
 * const detector = await pipeline('object-detection', 'Xenova/detr-resnet-50');
 * const img = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg';
 * const output = await detector(img, { threshold: 0.9 });
 * // [{
 * //   score: 0.9976370930671692,
 * //   label: "remote",
 * //   box: { xmin: 31, ymin: 68, xmax: 190, ymax: 118 }
 * // },
 * // ...
 * // {
 * //   score: 0.9984092116355896,
 * //   label: "cat",
 * //   box: { xmin: 331, ymin: 19, xmax: 649, ymax: 371 }
 * // }]
 * ```
 */
export class ObjectDetectionPipeline
    extends /** @type {new (options: ImagePipelineConstructorArgs) => ObjectDetectionPipelineType} */ (Pipeline)
{
    /**
     * Create a new ObjectDetectionPipeline.
     * @param {ImagePipelineConstructorArgs} options An object used to instantiate the pipeline.
     */
    constructor(options) {
        super(options);
    }

    /** @type {ObjectDetectionPipelineCallback} */
    async _call(images, { threshold = 0.9, percentage = false } = {}) {
        const isBatched = Array.isArray(images);

        if (isBatched && images.length !== 1) {
            throw Error('Object detection pipeline currently only supports a batch size of 1.');
        }
        const preparedImages = await prepareImages(images);

        const imageSizes = percentage ? null : preparedImages.map((x) => [x.height, x.width]);

        const { pixel_values, pixel_mask } = await this.processor(preparedImages);
        const output = await this.model({ pixel_values, pixel_mask });

        // @ts-ignore
        const processed = this.processor.image_processor.post_process_object_detection(output, threshold, imageSizes);

        // Add labels
        // @ts-expect-error TS2339
        const id2label = this.model.config.id2label;

        // Format output
        /** @type {ObjectDetectionPipelineOutput[]} */
        const result = processed.map((batch) =>
            batch.boxes.map((box, i) => ({
                score: batch.scores[i],
                label: id2label[batch.classes[i]],
                box: get_bounding_box(box, !percentage),
            })),
        );

        return isBatched ? result : result[0];
    }
}
