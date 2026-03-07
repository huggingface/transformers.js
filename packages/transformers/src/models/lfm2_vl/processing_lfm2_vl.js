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
     * Compute the number of tokens for a single tile.
     * @param {number} tile_size
     * @param {number} encoder_patch_size
     * @param {number} downsample_factor
     * @returns {number}
     */
    _compute_tokens_per_tile(tile_size, encoder_patch_size, downsample_factor) {
        const downsampled = Math.ceil(Math.floor(tile_size / encoder_patch_size) / downsample_factor);
        return downsampled * downsampled;
    }

    /**
     * Compute the number of tokens for a resized image (thumbnail or single-tile).
     * @param {number[]} image_size [height, width]
     * @param {number} encoder_patch_size
     * @param {number} downsample_factor
     * @returns {number}
     */
    _compute_tokens_for_image(image_size, encoder_patch_size, downsample_factor) {
        const patches_h = Math.ceil(Math.floor(image_size[0] / encoder_patch_size) / downsample_factor);
        const patches_w = Math.ceil(Math.floor(image_size[1] / encoder_patch_size) / downsample_factor);
        return patches_h * patches_w;
    }

    /**
     * Build the expanded token string for a single image.
     * @param {number} rows
     * @param {number} cols
     * @param {number} tokens_per_tile
     * @param {number} tokens_for_image
     * @param {string} image_token
     * @param {boolean} use_thumbnail
     * @returns {string}
     */
    _build_image_tokens(rows, cols, tokens_per_tile, tokens_for_image, image_token, use_thumbnail) {
        const image_start_token = this.config.image_start_token ?? '<|image_start|>';
        const image_end_token = this.config.image_end_token ?? '<|image_end|>';
        const image_thumbnail_token = this.config.image_thumbnail ?? '<|img_thumbnail|>';

        const parts = [image_start_token];

        if (rows > 1 || cols > 1) {
            const tile_tokens = image_token.repeat(tokens_per_tile);
            for (let row = 0; row < rows; ++row) {
                for (let col = 0; col < cols; ++col) {
                    parts.push(`<|img_row_${row + 1}_col_${col + 1}|>`);
                    parts.push(tile_tokens);
                }
            }
            if (use_thumbnail) {
                parts.push(image_thumbnail_token);
                parts.push(image_token.repeat(tokens_for_image));
            }
        } else {
            parts.push(image_token.repeat(tokens_for_image));
        }

        parts.push(image_end_token);
        return parts.join('');
    }

    /**
     * @param {RawImage|RawImage[]} images
     * @param {string|string[]|null} [text]
     * @param {Record<string, any>} [kwargs]
     */
    async _call(images, text = null, kwargs = {}) {
        const image_inputs = await this.image_processor(images, {
            ...kwargs,
            return_row_col_info: true,
        });

        const image_rows = image_inputs.image_rows;
        const image_cols = image_inputs.image_cols;
        const image_sizes = image_inputs.image_sizes;

        delete image_inputs.image_rows;
        delete image_inputs.image_cols;
        delete image_inputs.image_sizes;

        if (text) {
            const image_token = this.config.image_token ?? '<image>';
            const {
                tile_size = 512,
                downsample_factor = 2,
                encoder_patch_size = 16,
                use_thumbnail = true,
            } = /** @type {Record<string, any>} */ (this.image_processor.config);

            const tokens_per_tile = this._compute_tokens_per_tile(tile_size, encoder_patch_size, downsample_factor);

            if (!Array.isArray(text)) {
                text = [text];
            }

            let image_idx = 0;
            const expanded_text = [];
            for (const sample of text) {
                const parts = sample.split(image_token);
                const result = [parts[0]];
                for (let i = 1; i < parts.length; ++i) {
                    const tokens_for_image = this._compute_tokens_for_image(
                        image_sizes[image_idx],
                        encoder_patch_size,
                        downsample_factor,
                    );
                    result.push(
                        this._build_image_tokens(
                            image_rows[image_idx],
                            image_cols[image_idx],
                            tokens_per_tile,
                            tokens_for_image,
                            image_token,
                            use_thumbnail,
                        ),
                    );
                    result.push(parts[i]);
                    image_idx++;
                }
                expanded_text.push(result.join(''));
            }
            text = expanded_text;
        }

        return {
            ...image_inputs,
            ...(text ? this.tokenizer(text, kwargs) : {}),
        };
    }
}
