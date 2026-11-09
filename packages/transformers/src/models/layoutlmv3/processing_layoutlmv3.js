import { Processor } from '../../processing_utils.js';
import { AutoImageProcessor } from '../auto/image_processing_auto.js';
import { AutoTokenizer } from '../auto/tokenization_auto.js';

/**
 * Processor for LayoutLMv3. LayoutLMv3 requires both document images and
 * word-level bounding boxes. Since Transformers.js cannot run Tesseract OCR in
 * the browser, words and normalized boxes must be supplied by the caller.
 *
 * Use `LayoutLMv3Processor.from_pretrained` directly, since LayoutLMv3
 * repositories do not set `processor_class` in `preprocessor_config.json`.
 *
 * Example:
 * ```javascript
 * const processor = await LayoutLMv3Processor.from_pretrained('onnx-community/layoutlmv3-base-ONNX');
 * const image = await load_image('document.png');
 * const inputs = await processor(image, ['Hello', 'world'], {
 *   boxes: [[100, 100, 200, 200], [300, 300, 400, 400]],
 * });
 * ```
 */
export class LayoutLMv3Processor extends Processor {
    static image_processor_class = AutoImageProcessor;
    static tokenizer_class = AutoTokenizer;

    /**
     * The names of the model inputs produced by this processor.
     * @returns {string[]} The model input names.
     */
    get model_input_names() {
        return ['input_ids', 'bbox', 'attention_mask', 'pixel_values'];
    }

    /**
     * Preprocess a document image together with pre-tokenized words and boxes.
     *
     * @param {import('../../utils/image.js').RawImage|import('../../utils/image.js').RawImage[]} images The image or images to preprocess.
     * @param {string|string[]|string[][]|null} [text=null] The words of the document, or the words of each document in a batch.
     * @param {Object} [options] Additional tokenization and preprocessing options.
     * @param {number[][]|number[][][]|null} [options.boxes=null] The normalized bounding boxes corresponding to the words.
     * @param {number[]|number[][]|null} [options.word_labels=null] The labels corresponding to the words.
     * @returns {Promise<Object>} The model inputs, including `pixel_values`.
     */
    async _call(images, text = null, options = {}) {
        const { boxes = null, word_labels = null, ...tokenizer_options } = options;

        if (text === null || text === undefined) {
            throw new Error(
                'LayoutLMv3Processor requires `text` and `boxes` because OCR is not supported in Transformers.js.',
            );
        }
        if (boxes === null) {
            throw new Error('LayoutLMv3Processor requires `boxes` because OCR is not supported in Transformers.js.');
        }

        // Split a raw string into words, matching the word-level input expected
        // by the tokenizer. For reliable alignment with OCR boxes, pass an
        // array of words instead.
        if (typeof text === 'string') {
            text = text.trim() === '' ? [] : text.trim().split(/\s+/);
        }

        const image_inputs = await this.image_processor(images, options);
        // @ts-expect-error The LayoutLMv3 tokenizer accepts word-level boxes.
        const tokenizer_inputs = this.tokenizer(text, {
            boxes,
            word_labels,
            ...tokenizer_options,
        });

        return {
            ...tokenizer_inputs,
            pixel_values: image_inputs.pixel_values,
        };
    }
}
