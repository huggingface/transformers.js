import { FeatureExtractor, validate_audio_inputs } from '../../feature_extraction_utils.js';
import { Tensor, full, matmul, add } from '../../utils/tensor.js';
import { mel_filter_bank, spectrogram, window_function, spectrogram_kernel } from '../../utils/audio.js';

export class WeSpeakerFeatureExtractor extends FeatureExtractor {
    constructor(config) {
        super(config);

        const sampling_rate = this.config.sampling_rate;
        const mel_filters = mel_filter_bank(
            257, // num_frequency_bins
            this.config.num_mel_bins, // num_mel_filters
            20, // min_frequency
            Math.floor(sampling_rate / 2), // max_frequency
            sampling_rate, // sampling_rate
            null, // norm
            'kaldi', // mel_scale
            true, // triangularize_in_mel_space
        );
        this.mel_filters = mel_filters;

        // Kaldi compliance: the waveform is scaled to the range of 16-bit signed integers (32768 == 2 ** 15).
        // Everything before the power spectrum is linear, so the scale is applied to the window instead.
        this.window = window_function(400, 'hamming', {
            periodic: false,
        }).map((x) => x * 32768);
        this.min_num_frames = this.config.min_num_frames;

        this.spectrogram_kernel = spectrogram_kernel(this.window, 160, {
            fft_length: 512,
            preemphasis: 0.97,
            remove_dc_offset: true,
        });
    }

    /**
     * Computes the log-Mel spectrogram of the provided audio waveform.
     * @param {Float32Array|Float64Array} waveform The audio waveform to process.
     * @returns {Promise<Tensor>} An object containing the log-Mel spectrogram data as a Float32Array and its dimensions as an array of numbers.
     */
    async _extract_fbank_features(waveform) {
        return spectrogram(
            waveform,
            this.window, // window
            400, // frame_length
            160, // hop_length
            {
                fft_length: 512,
                power: 2.0,
                center: false,
                preemphasis: 0.97,
                mel_filters: this.mel_filters,
                kernel: this.spectrogram_kernel,
                log_mel: 'log',
                mel_floor: 1.192092955078125e-7,
                remove_dc_offset: true,

                // Custom
                transpose: true,
                min_num_frames: this.min_num_frames,
            },
        );
    }

    /**
     * Asynchronously extracts features from a given audio using the provided configuration.
     * @param {Float32Array|Float64Array} audio The audio data as a Float32Array/Float64Array.
     * @returns {Promise<{ input_features: Tensor }>} A Promise resolving to an object containing the extracted input features as a Tensor.
     */
    async _call(audio) {
        validate_audio_inputs(audio, 'WeSpeakerFeatureExtractor');

        let features = await this._extract_fbank_features(audio);

        if (this.config.fbank_centering_span === null) {
            // center features with global average ([1, frames] @ [frames, mel] gives the negated mean)
            const num_frames = features.dims[0];
            features = await add(features, await matmul(full([1, num_frames], -1 / num_frames), features));
        }

        return {
            input_features: features.unsqueeze_(0),
        };
    }
}
