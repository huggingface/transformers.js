import { Processor } from '../../processing_utils.js';
import { AutoImageProcessor } from '../auto/image_processing_auto.js';
import { AutoTokenizer } from '../auto/tokenization_auto.js';
import { RawImage } from '../../utils/image.js';

export class Qwen2VLProcessor extends Processor {
    static image_processor_class = AutoImageProcessor;
    static tokenizer_class = AutoTokenizer;
    static image_token = '<|image_pad|>';

    /**
     *
     * @param {string|string[]} text
     * @param {RawImage|RawImage[]} images
     * @param  {...any} args
     * @returns {Promise<any>}
     */
    async _call(text, images = null, ...args) {
        if (!Array.isArray(text)) {
            text = [text];
        }

        let image_inputs, image_grid_thw;

        if (images) {
            image_inputs = await this.image_processor(images);
            image_grid_thw = image_inputs.image_grid_thw;
        }

        if (image_grid_thw) {
            // @ts-expect-error TS2551
            let merge_length = this.image_processor.config.merge_size ** 2;
            let index = 0;

            const image_token = /** @type {typeof Qwen2VLProcessor} */ (this.constructor).image_token;
            const image_grid_thw_list = image_grid_thw.tolist();
            text = text.map((t) => {
                while (t.includes(image_token)) {
                    const prod = Number(image_grid_thw_list[index++].reduce((a, b) => a * b, 1n));
                    t = t.replace(image_token, '<|placeholder|>'.repeat(Math.floor(prod / merge_length)));
                }
                return t.replaceAll('<|placeholder|>', image_token);
            });
        }

        const text_inputs = this.tokenizer(text);

        return {
            ...text_inputs,
            ...image_inputs,
        };
    }
}
