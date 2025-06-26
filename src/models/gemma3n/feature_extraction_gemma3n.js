import { FeatureExtractor, validate_audio_inputs } from '../../base/feature_extraction_utils.js';
import { full, Tensor } from '../../utils/tensor.js';
import { mel_filter_bank, spectrogram, window_function } from '../../utils/audio.js';

export class Gemma3nAudioFeatureExtractor extends FeatureExtractor {

    constructor(config) {
        super(config);

        const {
            fft_length, feature_size, min_frequency, max_frequency, sampling_rate, frame_length
        } = this.config;

        const mel_filters = mel_filter_bank(
            Math.floor(1 + fft_length / 2), // num_frequency_bins
            feature_size, // num_mel_filters
            min_frequency, // min_frequency
            max_frequency, // max_frequency
            sampling_rate, // sampling_rate
            null, // norm
            "htk", // mel_scale
            false, // triangularize_in_mel_space
        );
        this.mel_filters = mel_filters;

        this.window = window_function(frame_length, 'hann')
    }

    /**
     * Computes the log-Mel spectrogram of the provided audio waveform.
     * @param {Float32Array|Float64Array} waveform The audio waveform to process.
     * @param {number} max_length The maximum number of frames to return.
     * @returns {Promise<Tensor>} An object containing the log-Mel spectrogram data as a Float32Array and its dimensions as an array of numbers.
     */
    async _extract_fbank_features(waveform, max_length) {
        // NOTE: We don't pad/truncate since that is passed in as `max_num_frames`
        return spectrogram(
            waveform,
            this.window, // window
            this.config.frame_length, // frame_length
            this.config.hop_length, // hop_length
            {
                fft_length: this.config.fft_length,
                center: false,
                onesided: true,
                preemphasis: this.config.preemphasis,
                preemphasis_htk_flavor: this.config.preemphasis_htk_flavor,
                mel_filters: this.mel_filters,
                log_mel: 'log',
                mel_floor: this.config.mel_floor,
                remove_dc_offset: false,

                // Custom
                transpose: true,
            }
        )
    }

    /**
     * Asynchronously extracts features from a given audio using the provided configuration.
     * @param {Float32Array|Float64Array} audio The audio data as a Float32Array/Float64Array.
     * @param {Object} options Optional parameters for feature extraction.
     * @param {number} [options.max_length=480_000] If provided, defines the maximum length of the audio to allow.
     * Audio longer than this will be truncated if `truncation=True`.
     * @param {boolean} [options.truncation=true] Whether or not to truncate audio above `max_length`.
     * @param {boolean} [options.padding=true] Whether to pad the sequence to a multiple of `pad_to_multiple_of`.
     * @param {number} [options.pad_to_multiple_of=128] The number to pad the sequence to a multiple of.
     * @returns {Promise<{ input_features: Tensor, input_features_mask: Tensor }>} A Promise resolving to an object containing the extracted input features and attention masks as Tensors.
     */
    async _call(audio, {
        max_length = 480_000,
        truncation=true,
        padding = true,
        pad_to_multiple_of = 128,
    } = {}) {
        validate_audio_inputs(audio, 'Gemma3nAudioFeatureExtractor');
        if (truncation && audio.length > max_length) {
            audio = audio.slice(0, max_length);
        }
        if (padding && audio.length % pad_to_multiple_of !== 0) {
            const padding_length = pad_to_multiple_of - (audio.length % pad_to_multiple_of);
            const padded_audio = new Float64Array(audio.length + padding_length);
            padded_audio.set(audio);
            if (this.config.padding_value !== 0) {
                padded_audio.fill(this.config.padding_value, audio.length);
            }
            audio = padded_audio;
        }

        const features = await this._extract_fbank_features(audio, this.config.max_length);
        const padded_attention_mask = full([1, features.dims[0]], true);
        return {
            input_features: features.unsqueeze_(0),
            input_features_mask: padded_attention_mask,
        }
    }
}
