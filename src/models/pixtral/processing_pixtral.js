
import { Processor } from "../../base/processing_utils.js";
import { AutoImageProcessor } from "../auto/image_processing_auto.js";
import { AutoTokenizer } from "../../tokenizers.js";

export class PixtralProcessor extends Processor {
    static tokenizer_class = AutoTokenizer
    static image_processor_class = AutoImageProcessor
    static uses_processor_config = true;

    /**
     * @typedef {import('../../utils/image.js').RawImage} RawImage
     */

    // `images` is required, `text` is optional
    async _call(/** @type {RawImage|RawImage[]} */ images, text = null, kwargs = {}) {
        
        const image_inputs = await this.image_processor(images, kwargs);

        if (text) {
            const [height, width] = image_inputs.pixel_values.dims.slice(-2);

            const { image_token, image_break_token, image_end_token, patch_size, spatial_merge_size } = this.config;
            const real_patch_size = patch_size * spatial_merge_size;
            const num_height_tokens = Math.floor(height / real_patch_size);
            const num_width_tokens = Math.floor(width / real_patch_size);

            text = structuredClone(text); // Avoid modifying the original text input
            if (!Array.isArray(text)) {
                text = [text];
            }
            for (let i = 0; i < text.length; ++i) {
                const width_tokens = image_token.repeat(num_width_tokens);
                const row = width_tokens + image_break_token;
                const finalRow = width_tokens + image_end_token;
                const full = row.repeat(num_height_tokens - 1) + finalRow;
                text[i] = text[i].replace(image_token, full);
            }
        }
        
        const text_inputs = text ? this.tokenizer(text, kwargs) : {};

        return {
            ...image_inputs,
            ...text_inputs,
        }
    }
}
