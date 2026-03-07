import { Processor } from '../../processing_utils.js';
import { AutoImageProcessor } from '../auto/image_processing_auto.js';
import { AutoTokenizer } from '../auto/tokenization_auto.js';

export class Lfm2VlProcessor extends Processor {
    static tokenizer_class = AutoTokenizer;
    static image_processor_class = AutoImageProcessor;

    /**
     * @typedef {import('../../utils/image.js').RawImage} RawImage
     */

    /**
     * Compute the number of tokens for a single tile.
     */
    _compute_tokens_per_tile(tile_size, encoder_patch_size, downsample_factor) {
        const num_patches = Math.floor(tile_size / encoder_patch_size);
        const downsampled = Math.ceil(num_patches / downsample_factor);
        return downsampled * downsampled;
    }

    /**
     * Compute the number of tokens for a resized image (thumbnail or single-tile).
     */
    _compute_tokens_for_image(image_size, encoder_patch_size, downsample_factor) {
        const [image_height, image_width] = image_size;
        const patches_h = Math.ceil(Math.floor(image_height / encoder_patch_size) / downsample_factor);
        const patches_w = Math.ceil(Math.floor(image_width / encoder_patch_size) / downsample_factor);
        return patches_h * patches_w;
    }

    /**
     * Build the expanded token string for a single image.
     */
    _build_image_tokens(rows, cols, tokens_per_tile, tokens_for_image, image_token, use_thumbnail) {
        const parts = [];

        const image_start_token = this.config.image_start_token ?? '<|image_start|>';
        const image_end_token = this.config.image_end_token ?? '<|image_end|>';
        const image_thumbnail_token = this.config.image_thumbnail ?? '<|img_thumbnail|>';

        parts.push(image_start_token);

        const is_multi_tile = rows > 1 || cols > 1;
        if (is_multi_tile) {
            for (let row = 0; row < rows; ++row) {
                for (let col = 0; col < cols; ++col) {
                    parts.push(`<|img_row_${row + 1}_col_${col + 1}|>`);
                    parts.push(image_token.repeat(tokens_per_tile));
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

    async _call(/** @type {RawImage|RawImage[]} */ images, text = null, kwargs = {}) {
        const image_inputs = await this.image_processor(images, {
            ...kwargs,
            return_row_col_info: true,
        });

        const image_rows = image_inputs.image_rows;
        const image_cols = image_inputs.image_cols;
        const image_sizes = image_inputs.image_sizes;

        // Clean up non-tensor fields before returning
        delete image_inputs.image_rows;
        delete image_inputs.image_cols;
        delete image_inputs.image_sizes;

        // Expand text with image token placeholders
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
                    const rows = image_rows[image_idx];
                    const cols = image_cols[image_idx];
                    const size = image_sizes[image_idx];
                    image_idx++;

                    const tokens_for_image = this._compute_tokens_for_image(size, encoder_patch_size, downsample_factor);
                    const expanded = this._build_image_tokens(
                        rows, cols, tokens_per_tile, tokens_for_image,
                        image_token, use_thumbnail,
                    );
                    result.push(expanded);
                    result.push(parts[i]);
                }
                expanded_text.push(result.join(''));
            }
            text = expanded_text;
        }

        const text_inputs = text ? this.tokenizer(text, kwargs) : {};

        return {
            ...image_inputs,
            ...text_inputs,
        };
    }
}
