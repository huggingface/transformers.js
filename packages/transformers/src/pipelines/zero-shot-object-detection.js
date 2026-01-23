import { Pipeline, prepareImages, get_bounding_box } from './_base.js';

/**
 * @typedef {import('./_base.js').TextImagePipelineConstructorArgs} TextImagePipelineConstructorArgs
 * @typedef {import('./_base.js').Disposable} Disposable
 * @typedef {import('./_base.js').ImageInput} ImageInput
 * @typedef {import('./_base.js').BoundingBox} BoundingBox
 */

/**
 * @typedef {Object} ZeroShotObjectDetectionOutputSingle
 * @property {string} label Text query corresponding to the found object.
 * @property {number} score Score corresponding to the object (between 0 and 1).
 * @property {BoundingBox} box Bounding box of the detected object in image's original size, or as a percentage if `percentage` is set to true.
 *
 * @typedef {ZeroShotObjectDetectionOutputSingle[]} ZeroShotObjectDetectionOutput
 *
 * @typedef {Object} ZeroShotObjectDetectionPipelineOptions Parameters specific to zero-shot object detection pipelines.
 * @property {number} [threshold=0.1] The probability necessary to make a prediction.
 * @property {number} [top_k=null] The number of top predictions that will be returned by the pipeline.
 * If the provided number is `null` or higher than the number of predictions available, it will default
 * to the number of predictions.
 * @property {boolean} [percentage=false] Whether to return the boxes coordinates in percentage (true) or in pixels (false).
 *
 * @callback ZeroShotObjectDetectionPipelineCallbackSingle Detect objects (bounding boxes & classes) in the image(s) passed as inputs.
 * @param {ImageInput} images The input images.
 * @param {string[]} candidate_labels What the model should recognize in the image.
 * @param {ZeroShotObjectDetectionPipelineOptions} [options] The options to use for zero-shot object detection.
 * @returns {Promise<ZeroShotObjectDetectionOutput>} An array of objects containing the predicted labels, scores, and bounding boxes.
 *
 * @callback ZeroShotObjectDetectionPipelineCallbackBatch Detect objects (bounding boxes & classes) in the image(s) passed as inputs.
 * @param {ImageInput[]} images The input images.
 * @param {string[]} candidate_labels What the model should recognize in the image.
 * @param {ZeroShotObjectDetectionPipelineOptions} [options] The options to use for zero-shot object detection.
 * @returns {Promise<ZeroShotObjectDetectionOutput[]>} An array of objects containing the predicted labels, scores, and bounding boxes.
 *
 * @typedef {ZeroShotObjectDetectionPipelineCallbackSingle & ZeroShotObjectDetectionPipelineCallbackBatch} ZeroShotObjectDetectionPipelineCallback
 *
 * @typedef {TextImagePipelineConstructorArgs & ZeroShotObjectDetectionPipelineCallback & Disposable} ZeroShotObjectDetectionPipelineType
 */

/**
 * Zero-shot object detection pipeline. This pipeline predicts bounding boxes of
 * objects when you provide an image and a set of `candidate_labels`.
 *
 * **Example:** Zero-shot object detection w/ `Xenova/owlvit-base-patch32`.
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const detector = await pipeline('zero-shot-object-detection', 'Xenova/owlvit-base-patch32');
 * const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/astronaut.png';
 * const candidate_labels = ['human face', 'rocket', 'helmet', 'american flag'];
 * const output = await detector(url, candidate_labels);
 * // [
 * //   {
 * //     score: 0.24392342567443848,
 * //     label: 'human face',
 * //     box: { xmin: 180, ymin: 67, xmax: 274, ymax: 175 }
 * //   },
 * //   {
 * //     score: 0.15129457414150238,
 * //     label: 'american flag',
 * //     box: { xmin: 0, ymin: 4, xmax: 106, ymax: 513 }
 * //   },
 * //   {
 * //     score: 0.13649864494800568,
 * //     label: 'helmet',
 * //     box: { xmin: 277, ymin: 337, xmax: 511, ymax: 511 }
 * //   },
 * //   {
 * //     score: 0.10262022167444229,
 * //     label: 'rocket',
 * //     box: { xmin: 352, ymin: -1, xmax: 463, ymax: 287 }
 * //   }
 * // ]
 * ```
 *
 * **Example:** Zero-shot object detection w/ `Xenova/owlvit-base-patch32` (returning top 4 matches and setting a threshold).
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const detector = await pipeline('zero-shot-object-detection', 'Xenova/owlvit-base-patch32');
 * const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/beach.png';
 * const candidate_labels = ['hat', 'book', 'sunglasses', 'camera'];
 * const output = await detector(url, candidate_labels, { top_k: 4, threshold: 0.05 });
 * // [
 * //   {
 * //     score: 0.1606510728597641,
 * //     label: 'sunglasses',
 * //     box: { xmin: 347, ymin: 229, xmax: 429, ymax: 264 }
 * //   },
 * //   {
 * //     score: 0.08935828506946564,
 * //     label: 'hat',
 * //     box: { xmin: 38, ymin: 174, xmax: 258, ymax: 364 }
 * //   },
 * //   {
 * //     score: 0.08530698716640472,
 * //     label: 'camera',
 * //     box: { xmin: 187, ymin: 350, xmax: 260, ymax: 411 }
 * //   },
 * //   {
 * //     score: 0.08349756896495819,
 * //     label: 'book',
 * //     box: { xmin: 261, ymin: 280, xmax: 494, ymax: 425 }
 * //   }
 * // ]
 * ```
 */
export class ZeroShotObjectDetectionPipeline
    extends /** @type {new (options: TextImagePipelineConstructorArgs) => ZeroShotObjectDetectionPipelineType} */ (
        Pipeline
    )
{
    async _call(images, candidate_labels, { threshold = 0.1, top_k = null, percentage = false } = {}) {
        const isBatched = Array.isArray(images);
        const preparedImages = await prepareImages(images);

        // Run tokenization
        const text_inputs = this.tokenizer(candidate_labels, {
            padding: true,
            truncation: true,
        });

        // Run processor
        const model_inputs = await this.processor(preparedImages);

        // Since non-maximum suppression is performed for exporting, we need to
        // process each image separately. For more information, see:
        // https://github.com/huggingface/optimum/blob/e3b7efb1257c011db907ef40ab340e795cc5684c/optimum/exporters/onnx/model_configs.py#L1028-L1032
        const toReturn = [];
        for (let i = 0; i < preparedImages.length; ++i) {
            const image = preparedImages[i];
            const imageSize = percentage ? null : [[image.height, image.width]];
            const pixel_values = model_inputs.pixel_values[i].unsqueeze_(0);

            // Run model with both text and pixel inputs
            const output = await this.model({ ...text_inputs, pixel_values });

            let result;
            if ('post_process_grounded_object_detection' in this.processor) {
                // @ts-ignore
                const processed = this.processor.post_process_grounded_object_detection(output, text_inputs.input_ids, {
                    // TODO: support separate threshold values
                    box_threshold: threshold,
                    text_threshold: threshold,
                    target_sizes: imageSize,
                })[0];
                result = processed.boxes.map((box, i) => ({
                    score: processed.scores[i],
                    label: processed.labels[i],
                    box: get_bounding_box(box, !percentage),
                }));
            } else {
                // @ts-ignore
                const processed = this.processor.image_processor.post_process_object_detection(
                    output,
                    threshold,
                    imageSize,
                    true,
                )[0];
                result = processed.boxes.map((box, i) => ({
                    score: processed.scores[i],
                    label: candidate_labels[processed.classes[i]],
                    box: get_bounding_box(box, !percentage),
                }));
            }
            result.sort((a, b) => b.score - a.score);

            if (top_k !== null) {
                result = result.slice(0, top_k);
            }
            toReturn.push(result);
        }

        return isBatched ? toReturn : toReturn[0];
    }
}
