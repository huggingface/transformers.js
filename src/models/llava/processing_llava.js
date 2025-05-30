
import { Processor } from "../../base/processing_utils.js";
import { AutoImageProcessor } from "../auto/image_processing_auto.js";
import { AutoTokenizer } from "../../tokenizers.js";

export class LlavaProcessor extends Processor {
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

            const {image_token, patch_size, num_additional_image_tokens} = this.config;
            const num_image_tokens = Math.floor(
                height / patch_size
            ) * Math.floor(width / patch_size) + num_additional_image_tokens;

            text = structuredClone(text); // Avoid modifying the original text input
            if (!Array.isArray(text)) {
                text = [text];
            }
            for (let i = 0; i < text.length; ++i) {
                text[i] = text[i].replace(image_token, image_token.repeat(num_image_tokens));
            }
        }
        
        const text_inputs = text ? this.tokenizer(text, kwargs) : {};

        return {
            ...image_inputs,
            ...text_inputs,
        }
    }
}
