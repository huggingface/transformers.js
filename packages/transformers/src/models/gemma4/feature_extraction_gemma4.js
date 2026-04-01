import { validate_audio_inputs } from '../../feature_extraction_utils.js';
import { Tensor } from '../../utils/tensor.js';
import { spectrogram } from '../../utils/audio.js';
import { Gemma3nAudioFeatureExtractor } from '../gemma3n/feature_extraction_gemma3n.js';

export class Gemma4AudioFeatureExtractor extends Gemma3nAudioFeatureExtractor {
    /**
     * @override
     * Gemma4 uses semicausal padding, unfold(frame_length+1) framing, and
     * additive mel_floor — all controlled via flags on the shared spectrogram().
     */
    async _extract_fbank_features(waveform, max_length) {
        const { frame_length, hop_length, fft_length } = this.config;

        // Compute frame count matching Python's unfold(size=frame_length+1, step=hop_length)
        const pad_left = Math.floor(frame_length / 2);
        const padded_length = waveform.length + pad_left; // after semicausal padding
        const num_frames = Math.floor((padded_length - (frame_length + 1)) / hop_length) + 1;

        return spectrogram(
            waveform,
            this.window,
            frame_length,
            hop_length,
            {
                fft_length,
                center: true,
                pad_mode: 'semicausal',
                onesided: true,
                preemphasis: this.config.preemphasis,
                preemphasis_htk_flavor: this.config.preemphasis_htk_flavor,
                mel_filters: this.mel_filters,
                log_mel: 'log',
                mel_floor: this.config.mel_floor,
                mel_floor_mode: 'add',
                remove_dc_offset: false,
                transpose: true,
                max_num_frames: num_frames,
            },
        );
    }

    /**
     * @override
     * Tracks original audio length to build a frame-aware attention mask
     * and zeros out features for invalid (padded) frames.
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

        const features = await this._extract_fbank_features(audio, this.config.max_length);
        const num_frames = features.dims[0];
        const num_features = features.dims[1];

        // Build frame-aware mask: a frame is valid only when all its samples are real audio.
        // After semicausal padding (pad_left zeros prepended), real audio spans
        // [pad_left, pad_left + original_length) in the padded waveform.
        const { frame_length, hop_length } = this.config;
        const pad_left = Math.floor(frame_length / 2);
        const frame_size_for_unfold = frame_length + 1;

        const sample_mask = new Uint8Array(audio.length + pad_left);
        sample_mask.fill(1, pad_left, pad_left + original_length);

        const frame_mask = new Uint8Array(num_frames);
        for (let i = 0; i < num_frames; ++i) {
            frame_mask[i] = sample_mask[i * hop_length + frame_size_for_unfold - 1] ? 1 : 0;
        }

        // Zero out features for invalid frames (matching Python's speech * mask[..., None])
        const feat_data = /** @type {Float32Array} */ (features.data);
        for (let i = 0; i < num_frames; ++i) {
            if (!frame_mask[i]) {
                feat_data.fill(0, i * num_features, (i + 1) * num_features);
            }
        }

        return {
            input_features: features.unsqueeze_(0),
            input_features_mask: new Tensor('bool', frame_mask, [1, num_frames]),
        };
    }
}
