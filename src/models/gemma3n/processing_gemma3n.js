
import { Processor } from "../../base/processing_utils.js";
import { AutoImageProcessor } from "../auto/image_processing_auto.js";
import { AutoFeatureExtractor } from "../auto/feature_extraction_auto.js";
import { AutoTokenizer } from "../../tokenizers.js";
import { RawImage } from "../../utils/image.js";
import { RawAudio } from "../../utils/audio.js";

export class Gemma3nProcessor extends Processor {
    static image_processor_class = AutoImageProcessor;
    static feature_extractor_class = AutoFeatureExtractor;
    static tokenizer_class = AutoTokenizer;
    static uses_processor_config = true;
    static uses_chat_template_file = true;

    constructor(config, components, chat_template) {
        super(config, components, chat_template);
        this.audio_seq_length = this.config.audio_seq_length;
        this.image_seq_length = this.config.image_seq_length;

        const {
            // Audio tokens
            audio_token_id, boa_token, audio_token, eoa_token,

            // Image tokens
            image_token_id, boi_token, image_token, eoi_token
        } = this.tokenizer.config;

        this.audio_token_id = audio_token_id
        this.boa_token = boa_token
        this.audio_token = audio_token
        const audio_tokens_expanded = audio_token.repeat(this.audio_seq_length);
        this.full_audio_sequence = `\n\n${boa_token}${audio_tokens_expanded}${eoa_token}\n\n`

        this.image_token_id = image_token_id
        this.boi_token = boi_token
        this.image_token = image_token
        const image_tokens_expanded = image_token.repeat(this.image_seq_length);
        this.full_image_sequence = `\n\n${boi_token}${image_tokens_expanded}${eoi_token}\n\n`
    }

    /**
     * 
     * @param {string|string[]} text 
     * @param {RawImage|RawImage[]|RawImage[][]} images
     * @param {RawAudio|RawAudio[]|RawAudio[][]} audio
     * @returns {Promise<any>}
     */
    async _call(text, images = null, audio = null, options = {}) {

        if (typeof text === 'string') {
            text = [text];
        }

        let audio_inputs;
        if (audio) {
            audio_inputs = await this.feature_extractor(audio, options);

            text = text.map(prompt => prompt.replaceAll(this.audio_token, this.full_audio_sequence));
        }
        let image_inputs;
        if (images) {
            image_inputs = await this.image_processor(images, options);
            text = text.map(prompt => prompt.replaceAll(this.image_token, this.full_image_sequence));
        }

        let text_inputs = this.tokenizer(text, options);
        return {
            ...text_inputs,
            ...image_inputs,
            ...audio_inputs,
        }
    }
}
