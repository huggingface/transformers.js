import { Processor } from '../../processing_utils.js';
import { AutoImageProcessor } from '../auto/image_processing_auto.js';
import { AutoTokenizer } from '../auto/tokenization_auto.js';

export class Lfm3VlProcessor extends Processor {
    static tokenizer_class = AutoTokenizer;
    static image_processor_class = AutoImageProcessor;

    /**
     * @param {import('../../utils/image.js').RawImage|import('../../utils/image.js').RawImage[]|null} images
     * @param {string|string[]|null} [text]
     * @param {Record<string, any>} [kwargs]
     */
    async _call(images, text = null, kwargs = {}) {
        const image_inputs = images ? await this.image_processor(images, kwargs) : {};

        if (text !== null) {
            if (!Array.isArray(text)) text = [text];
            const image_token = this.tokenizer.config.image_token ?? '<|image|>';
            const grids = image_inputs.image_grid_thw?.tolist() ?? [];
            const num_placeholders = text.reduce((count, sample) => count + sample.split(image_token).length - 1, 0);
            if (num_placeholders !== grids.length) {
                throw new Error(
                    `Found ${num_placeholders} ${image_token} tokens in the text but received ${grids.length} images.`,
                );
            }
            const merge_size = /** @type {Record<string, any>} */ (this.image_processor.config).merge_size;
            let index = 0;
            text = text.map((sample) =>
                sample.replaceAll(image_token, () => {
                    const num_patches = Number(grids[index++].reduce((a, b) => a * b, 1n));
                    return image_token.repeat(num_patches / merge_size ** 2);
                }),
            );
        }

        return {
            ...image_inputs,
            ...(text !== null ? this.tokenizer(text, kwargs) : {}),
        };
    }
}
