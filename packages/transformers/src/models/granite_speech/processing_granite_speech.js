import { AutoFeatureExtractor } from '../auto/feature_extraction_auto.js';
import { AutoTokenizer } from '../auto/tokenization_auto.js';
import { Processor } from '../../processing_utils.js';
import { Tensor } from '../../utils/tensor.js';

export class GraniteSpeechProcessor extends Processor {
    static tokenizer_class = AutoTokenizer;
    static feature_extractor_class = AutoFeatureExtractor;
    static uses_processor_config = true;

    /**
     * Compute the number of audio tokens for a given raw audio length.
     * @param {number} audioLength Raw audio sample count.
     * @returns {number} Number of projector output tokens.
     */
    _get_num_audio_features(audioLength) {
        const { hop_length } = this.feature_extractor.config.melspec_kwargs;
        const { projector_window_size, projector_downsample_rate } = this.feature_extractor.config;
        const effective_window_size = Math.floor(projector_window_size / projector_downsample_rate);
        const mel_length = Math.floor(audioLength / hop_length) + 1;
        const encoder_length = Math.floor(mel_length / 2);
        const nblocks = Math.ceil(encoder_length / projector_window_size);
        return nblocks * effective_window_size;
    }

    /**
     * @param {string} text The text input to process.
     * @param {Float32Array} audio The audio input to process.
     */
    async _call(text, audio = null, kwargs = {}) {
        if (Array.isArray(text)) {
            throw new Error('Batched inputs are not supported yet.');
        }

        let audio_inputs = {};
        if (audio) {
            const { input_features } = await this.feature_extractor(audio);
            audio_inputs['input_features'] = input_features;

            // Compute audio embed sizes and mask in the processor
            const audio_embed_size = this._get_num_audio_features(audio.length);
            const mask_data = new Uint8Array(audio_embed_size).fill(1);
            audio_inputs['input_features_mask'] = new Tensor('bool', mask_data, [1, audio_embed_size]);

            const audio_token = this.config.audio_token ?? '<|audio|>';
            if (!text.includes(audio_token)) {
                throw new Error(`The input text does not contain the audio token ${audio_token}.`);
            }
            text = text.replaceAll(audio_token, audio_token.repeat(audio_embed_size));
        }

        const text_inputs = this.tokenizer(text, {
            add_special_tokens: false,
            ...kwargs,
        });

        return {
            ...text_inputs,
            ...audio_inputs,
        };
    }
}
