import { Processor } from '../../processing_utils.js';
import { AutoTokenizer } from '../auto/tokenization_auto.js';
import { Gemma4ImageProcessor } from './image_processing_gemma4.js';
import { Gemma4AudioFeatureExtractor } from './feature_extraction_gemma4.js';
import { PROCESSOR_NAME, CHAT_TEMPLATE_NAME } from '../../utils/constants.js';
import { getModelJSON, getModelText } from '../../utils/hub.js';

export class Gemma4Processor extends Processor {
    static uses_processor_config = true;
    static uses_chat_template_file = true;

    constructor(config, components, chat_template) {
        super(config, components, chat_template);

        this.audio_ms_per_token = this.config.audio_ms_per_token ?? 40;
        this.audio_seq_length = this.config.audio_seq_length ?? 750;
        this.image_seq_length = this.config.image_seq_length ?? 280;

        const { audio_token, boa_token, eoa_token, image_token, boi_token, eoi_token } = this.tokenizer.config;
        this.audio_token = audio_token;
        this.boa_token = boa_token;
        this.eoa_token = eoa_token;
        this.image_token = image_token;
        this.boi_token = boi_token;
        this.eoi_token = eoi_token;
    }

    static async from_pretrained(pretrained_model_name_or_path, options = {}) {
        const [config, tokenizer, chat_template] = await Promise.all([
            getModelJSON(pretrained_model_name_or_path, PROCESSOR_NAME, true, options),
            AutoTokenizer.from_pretrained(pretrained_model_name_or_path, options),
            getModelText(pretrained_model_name_or_path, CHAT_TEMPLATE_NAME, false, options),
        ]);

        const components = { tokenizer };
        if (config.image_processor) {
            components.image_processor = new Gemma4ImageProcessor(config.image_processor);
        }
        if (config.feature_extractor) {
            components.feature_extractor = new Gemma4AudioFeatureExtractor(config.feature_extractor);
        }

        return new this(config, components, chat_template);
    }

    /**
     * Compute the number of audio soft tokens for a single waveform.
     * Replicates the audio encoder's sequence-length arithmetic:
     * mel framing → two SSCP conv layers (kernel=3, stride=2) → cap.
     * @param {number} num_samples
     * @param {number} sampling_rate
     * @returns {number}
     */
    _compute_audio_num_tokens(num_samples, sampling_rate) {
        const frame_length = Math.round((sampling_rate * 20.0) / 1000.0);
        const hop_length = Math.round((sampling_rate * 10.0) / 1000.0);

        const pad_left = Math.floor(frame_length / 2);
        let t = Math.floor((num_samples + pad_left - frame_length - 1) / hop_length) + 1;
        if (t <= 0) return 0;

        // Two SSCP conv layers: kernel=3, stride=2, pad=1+1
        for (let i = 0; i < 2; ++i) t = Math.floor((t - 1) / 2) + 1;

        return Math.min(t, this.audio_seq_length);
    }

    async _call(text, images = null, audio = null, options = {}) {
        if (typeof text === 'string') {
            text = [text];
        }

        let image_inputs;
        if (images) {
            image_inputs = await this.image_processor(images, options);
            const counts = image_inputs.num_soft_tokens_per_image;
            let i = 0;
            text = text.map((t) =>
                t.replaceAll(
                    this.image_token,
                    () => `\n\n${this.boi_token}${this.image_token.repeat(counts[i++])}${this.eoi_token}\n\n`,
                ),
            );
        }

        let audio_inputs;
        if (audio) {
            const audio_array = Array.isArray(audio) ? audio : [audio];
            audio_inputs = await this.feature_extractor(audio_array[0], options);

            const sampling_rate = this.feature_extractor.config.sampling_rate ?? 16000;
            let i = 0;
            text = text.map((t) =>
                t.replaceAll(
                    this.audio_token,
                    () =>
                        `\n\n${this.boa_token}${this.audio_token.repeat(this._compute_audio_num_tokens(audio_array[i++].length, sampling_rate))}${this.eoa_token}\n\n`,
                ),
            );
        }

        return {
            ...this.tokenizer(text, options),
            ...image_inputs,
            ...audio_inputs,
        };
    }
}
