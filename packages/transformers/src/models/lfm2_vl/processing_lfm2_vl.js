import { Processor } from '../../processing_utils.js';
import { AutoImageProcessor } from '../auto/image_processing_auto.js';
import { AutoTokenizer } from '../auto/tokenization_auto.js';

/**
 * @typedef {import('../../utils/image.js').RawImage} RawImage
 */

export class Lfm2VlProcessor extends Processor {
    static tokenizer_class = AutoTokenizer;
    static image_processor_class = AutoImageProcessor;

    /**
     * @param {RawImage|RawImage[]} images
     * @param {string|string[]|null} [text]
     * @param {Record<string, any>} [kwargs]
     */
    async _call(images, text = null, kwargs = {}) {
        const { image_rows, image_cols, image_sizes, ...image_inputs } = await this.image_processor(images, {
            ...kwargs,
            return_row_col_info: true,
        });

        if (text) {
            const image_token = this.config.image_token ?? '<image>';
            const {
                tile_size = 512,
                downsample_factor = 2,
                encoder_patch_size = 16,
                use_thumbnail = true,
            } = /** @type {Record<string, any>} */ (this.image_processor.config);

            const ds = (/** @type {number} */ s) => Math.ceil(Math.floor(s / encoder_patch_size) / downsample_factor);
            const tokens_per_tile = ds(tile_size) ** 2;
            const image_start = this.config.image_start_token ?? '<|image_start|>';
            const image_end = this.config.image_end_token ?? '<|image_end|>';
            const thumbnail_token = this.config.image_thumbnail ?? '<|img_thumbnail|>';

            if (!Array.isArray(text)) text = [text];

            let image_idx = 0;
            text = text.map((sample) => {
                const parts = sample.split(image_token);
                return (
                    parts[0] +
                    parts
                        .slice(1)
                        .map((part) => {
                            const idx = image_idx++;
                            const [h, w] = image_sizes[idx];
                            const rows = image_rows[idx],
                                cols = image_cols[idx];
                            const tokens_for_image = ds(h) * ds(w);

                            let expanded = image_start;
                            if (rows > 1 || cols > 1) {
                                const tile_str = image_token.repeat(tokens_per_tile);
                                for (let r = 0; r < rows; ++r)
                                    for (let c = 0; c < cols; ++c)
                                        expanded += `<|img_row_${r + 1}_col_${c + 1}|>` + tile_str;
                                if (use_thumbnail) expanded += thumbnail_token + image_token.repeat(tokens_for_image);
                            } else {
                                expanded += image_token.repeat(tokens_for_image);
                            }
                            return expanded + image_end + part;
                        })
                        .join('')
                );
            });
        }

        return {
            ...image_inputs,
            ...(text ? this.tokenizer(text, kwargs) : {}),
        };
    }
}
