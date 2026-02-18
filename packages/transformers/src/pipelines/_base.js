import { PreTrainedTokenizer } from '../tokenization_utils.js';
import { PreTrainedModel } from '../models/modeling_utils.js';
import { Processor } from '../processing_utils.js';

import { Callable } from '../utils/generic.js';

import { read_audio } from '../utils/audio.js';
import { RawImage } from '../utils/image.js';

/**
 * @typedef {string | RawImage | URL | Blob | HTMLCanvasElement | OffscreenCanvas} ImageInput
 * @typedef {ImageInput|ImageInput[]} ImagePipelineInputs
 */

/**
 * Prepare images for further tasks.
 * @param {ImagePipelineInputs} images images to prepare.
 * @returns {Promise<RawImage[]>} returns processed images.
 */
export async function prepareImages(images) {
    if (!Array.isArray(images)) {
        images = [images];
    }

    // Possibly convert any non-images to images
    return await Promise.all(images.map((x) => RawImage.read(x)));
}

/**
 * @typedef {string | URL | Float32Array | Float64Array} AudioInput
 * @typedef {AudioInput|AudioInput[]} AudioPipelineInputs
 */

/**
 * Prepare audios for further tasks.
 * @param {AudioPipelineInputs} audios audios to prepare.
 * @param {number} sampling_rate sampling rate of the audios.
 * @returns {Promise<Float32Array[]>} The preprocessed audio data.
 */
export async function prepareAudios(audios, sampling_rate) {
    if (!Array.isArray(audios)) {
        audios = [audios];
    }

    return await Promise.all(
        audios.map((x) => {
            if (typeof x === 'string' || x instanceof URL) {
                return read_audio(x, sampling_rate);
            } else if (x instanceof Float64Array) {
                return new Float32Array(x);
            }
            return x;
        }),
    );
}

/**
 * @typedef {Object} BoundingBox
 * @property {number} xmin The minimum x coordinate of the bounding box.
 * @property {number} ymin The minimum y coordinate of the bounding box.
 * @property {number} xmax The maximum x coordinate of the bounding box.
 * @property {number} ymax The maximum y coordinate of the bounding box.
 */

/**
 * Helper function to convert list [xmin, xmax, ymin, ymax] into object { "xmin": xmin, ... }
 * @param {number[]} box The bounding box as a list.
 * @param {boolean} asInteger Whether to cast to integers.
 * @returns {BoundingBox} The bounding box as an object.
 * @private
 */
export function get_bounding_box(box, asInteger) {
    if (asInteger) {
        box = box.map((x) => x | 0);
    }
    const [xmin, ymin, xmax, ymax] = box;

    return { xmin, ymin, xmax, ymax };
}

/**
 * @callback DisposeType Disposes the item.
 * @returns {Promise<void>} A promise that resolves when the item has been disposed.
 *
 * @typedef {Object} Disposable
 * @property {DisposeType} dispose A promise that resolves when the pipeline has been disposed.
 */

/**
 * The Pipeline class is the class from which all pipelines inherit.
 * Refer to this class for methods shared across different pipelines.
 */
export class Pipeline extends Callable {
    /**
     * Create a new Pipeline.
     * @param {Object} options An object containing the following properties:
     * @param {string} [options.task] The task of the pipeline. Useful for specifying subtasks.
     * @param {PreTrainedModel} [options.model] The model used by the pipeline.
     * @param {PreTrainedTokenizer} [options.tokenizer=null] The tokenizer used by the pipeline (if any).
     * @param {Processor} [options.processor=null] The processor used by the pipeline (if any).
     */
    constructor({ task, model, tokenizer = null, processor = null }) {
        super();
        this.task = task;
        this.model = model;
        this.tokenizer = tokenizer;
        this.processor = processor;
    }

    /** @type {DisposeType} */
    async dispose() {
        await this.model.dispose();
    }
}

/**
 * @typedef {Object} ModelTokenizerConstructorArgs
 * @property {string} task The task of the pipeline. Useful for specifying subtasks.
 * @property {PreTrainedModel} model The model used by the pipeline.
 * @property {PreTrainedTokenizer} tokenizer The tokenizer used by the pipeline.
 *
 * @typedef {ModelTokenizerConstructorArgs} TextPipelineConstructorArgs An object used to instantiate a text-based pipeline.
 */

/**
 * @typedef {Object} ModelProcessorConstructorArgs
 * @property {string} task The task of the pipeline. Useful for specifying subtasks.
 * @property {PreTrainedModel} model The model used by the pipeline.
 * @property {Processor} processor The processor used by the pipeline.
 *
 * @typedef {ModelProcessorConstructorArgs} AudioPipelineConstructorArgs An object used to instantiate an audio-based pipeline.
 * @typedef {ModelProcessorConstructorArgs} ImagePipelineConstructorArgs An object used to instantiate an image-based pipeline.
 */

/**
 * @typedef {Object} ModelTokenizerProcessorConstructorArgs
 * @property {string} task The task of the pipeline. Useful for specifying subtasks.
 * @property {PreTrainedModel} model The model used by the pipeline.
 * @property {PreTrainedTokenizer} tokenizer The tokenizer used by the pipeline.
 * @property {Processor} processor The processor used by the pipeline.
 *
 * @typedef {ModelTokenizerProcessorConstructorArgs} TextAudioPipelineConstructorArgs An object used to instantiate a text- and audio-based pipeline.
 * @typedef {ModelTokenizerProcessorConstructorArgs} TextImagePipelineConstructorArgs An object used to instantiate a text- and image-based pipeline.
 */
