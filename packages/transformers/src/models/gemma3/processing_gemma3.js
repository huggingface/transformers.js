import { Processor } from '../../processing_utils.js';
import { AutoImageProcessor } from '../auto/image_processing_auto.js';
import { AutoTokenizer } from '../auto/tokenization_auto.js';

export class Gemma3Processor extends Processor {
    static tokenizer_class = AutoTokenizer;
    static image_processor_class = AutoImageProcessor;
    static uses_processor_config = true;
    static uses_chat_template_file = true;

    constructor(config, components, chat_template) {
        super(config, components, chat_template);
        this.image_seq_length = this.config.image_seq_length;

        const { boi_token, image_token, eoi_token } = this.tokenizer.config;
        this.boi_token = boi_token;
        this.image_token = image_token;
        this.eoi_token = eoi_token;
        const image_tokens_expanded = image_token.repeat(this.image_seq_length);
        this.full_image_sequence = `\n\n${boi_token}${image_tokens_expanded}${eoi_token}\n\n`;
    }

    /**
     * @param {string|string[]} text
     * @param {import('../../utils/image.js').RawImage|import('../../utils/image.js').RawImage[]} [images]
     * @param {Object} [options]
     */
    async _call(text, images = null, options = {}) {
        if (typeof text === 'string') {
            text = [text];
        }

        let image_inputs;
        if (images) {
            image_inputs = await this.image_processor(images, options);
            text = text.map((prompt) => prompt.replaceAll(this.boi_token, this.full_image_sequence));
        }

        const text_inputs = this.tokenizer(text, options);
        return {
            ...text_inputs,
            ...image_inputs,
        };
    }
}
