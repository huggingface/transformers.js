import { FeatureExtractor, validate_audio_inputs } from '../../feature_extraction_utils.js';
import { Tensor } from '../../utils/tensor.js';
import { mel_filter_bank, window_function } from '../../utils/audio.js';
import { FFT } from '../../utils/maths.js';

export class Gemma4AudioFeatureExtractor extends FeatureExtractor {
    constructor(config) {
        super(config);

        const { fft_length, feature_size, min_frequency, max_frequency, sampling_rate, frame_length } = this.config;

        this.mel_filters = mel_filter_bank(
            Math.floor(1 + fft_length / 2),
            feature_size,
            min_frequency,
            max_frequency,
            sampling_rate,
            null,
            'htk',
            false,
        );
        this.window = window_function(frame_length, 'hann');
    }

    /**
     * Compute log-Mel spectrogram matching Python's `_extract_spectrogram` exactly.
     *
     * Key differences from the shared `spectrogram()` utility:
     *   1. Semicausal time padding: prepend frame_length // 2 zeros
     *   2. Frame count uses unfold(size=frame_length+1)
     *   3. Mel projection done in float64 (matching numpy's auto-promotion)
     *
     * @param {Float32Array|Float64Array} waveform
     * @param {number} original_length Length of real audio before pad-to-multiple.
     * @returns {{ features: Tensor, mask: Tensor }}
     */
    _extract_fbank_features(waveform, original_length) {
        const { frame_length, hop_length, fft_length, mel_floor, feature_size } = this.config;
        const num_frequency_bins = Math.floor(fft_length / 2) + 1;

        // 1. Prepend semicausal padding
        const pad_left = Math.floor(frame_length / 2);
        // @ts-expect-error ts(2351)
        const padded = new waveform.constructor(pad_left + waveform.length);
        padded.set(waveform, pad_left);

        // 2. Compute frame count matching Python's unfold(size=frame_length+1, step=hop_length)
        const frame_size_for_unfold = frame_length + 1;
        const num_frames = Math.floor((padded.length - frame_size_for_unfold) / hop_length) + 1;

        // 3. STFT: window + FFT per frame, compute magnitude, mel project, log — all in float64
        const fft = new FFT(fft_length);
        const inputBuffer = new Float64Array(fft_length);
        const outputBuffer = new Float64Array(fft.outputBufferSize);
        const mel_data = new Float32Array(num_frames * feature_size);

        for (let i = 0; i < num_frames; ++i) {
            const offset = i * hop_length;

            // Load frame and apply window
            inputBuffer.fill(0);
            for (let j = 0; j < frame_length; ++j) {
                inputBuffer[j] = padded[offset + j] * this.window[j];
            }

            fft.realTransform(outputBuffer, inputBuffer);

            // Magnitude → mel projection → log (all in float64)
            for (let m = 0; m < feature_size; ++m) {
                const filter = this.mel_filters[m];
                let sum = 0;
                for (let f = 0; f < num_frequency_bins; ++f) {
                    const f2 = f << 1;
                    const mag = Math.sqrt(outputBuffer[f2] ** 2 + outputBuffer[f2 + 1] ** 2);
                    sum += mag * filter[f];
                }
                mel_data[i * feature_size + m] = Math.log(sum + mel_floor);
            }
        }

        // 4. Build frame-aware attention mask
        const sample_mask = new Uint8Array(padded.length);
        sample_mask.fill(1, pad_left, pad_left + original_length);

        const frame_mask = new Uint8Array(num_frames);
        for (let i = 0; i < num_frames; ++i) {
            frame_mask[i] = sample_mask[i * hop_length + frame_size_for_unfold - 1] ? 1 : 0;
        }

        // 5. Zero out features for invalid frames (matching Python's speech * mask[..., None])
        for (let i = 0; i < num_frames; ++i) {
            if (!frame_mask[i]) {
                mel_data.fill(0, i * feature_size, (i + 1) * feature_size);
            }
        }

        return {
            features: new Tensor('float32', mel_data, [num_frames, feature_size]),
            mask: new Tensor('bool', frame_mask, [num_frames]),
        };
    }

    /**
     * @param {Float32Array|Float64Array} audio
     * @param {Object} options
     * @returns {Promise<{ input_features: Tensor, input_features_mask: Tensor }>}
     */
    async _call(audio, { max_length = 480_000, truncation = true, padding = true, pad_to_multiple_of = 128 } = {}) {
        validate_audio_inputs(audio, 'Gemma4AudioFeatureExtractor');

        if (truncation && audio.length > max_length) {
            audio = audio.slice(0, max_length);
        }

        const original_length = audio.length;

        if (padding && audio.length % pad_to_multiple_of !== 0) {
            const padding_length = pad_to_multiple_of - (audio.length % pad_to_multiple_of);
            const padded_audio = new Float64Array(audio.length + padding_length);
            padded_audio.set(audio);
            audio = padded_audio;
        }

        const { features, mask } = this._extract_fbank_features(audio, original_length);

        return {
            input_features: features.unsqueeze_(0),
            input_features_mask: mask.unsqueeze_(0),
        };
    }
}
