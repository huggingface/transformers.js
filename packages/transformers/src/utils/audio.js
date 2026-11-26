/**
 * @file Audio I/O helpers.
 *
 * Decode audio files and URLs into the `Float32Array` pipelines expect,
 * and wrap generated waveforms in `RawAudio` for playback or saving.
 *
 * @module utils/audio
 */

import { getFile } from './hub.js';
import { saveBlob } from './io.js';
import { Tensor, matmul, conv1d, pow, clamp, add, mul, log, amax, pad } from './tensor.js';
import { logger } from './logger.js';

/**
 * Helper function to load audio from a path/URL.
 * @param {string|URL} url The path/URL to load the audio from.
 * @param {number} sampling_rate The sampling rate to use when decoding the audio.
 * @returns {Promise<Float32Array>} The decoded audio as a `Float32Array`.
 */
export async function load_audio(url, sampling_rate) {
    if (typeof AudioContext === 'undefined') {
        // Running in node or an environment without AudioContext
        throw Error(
            'Unable to load audio from path/URL since `AudioContext` is not available in your environment. ' +
                'Instead, audio data should be passed directly to the pipeline/processor. ' +
                'For more information and some example code, see https://huggingface.co/docs/transformers.js/guides/node-audio-processing.',
        );
    }

    const response = await (await getFile(url)).arrayBuffer();
    const audioCTX = new AudioContext({ sampleRate: sampling_rate });
    if (typeof sampling_rate === 'undefined') {
        logger.warn(`No sampling rate provided, using default of ${audioCTX.sampleRate}Hz.`);
    }
    const decoded = await audioCTX.decodeAudioData(response);

    /** @type {Float32Array} */
    let audio;

    // We now replicate HuggingFace's `ffmpeg_read` method:
    if (decoded.numberOfChannels === 2) {
        // When downmixing a stereo audio file to mono using the -ac 1 option in FFmpeg,
        // the audio signal is summed across both channels to create a single mono channel.
        // However, if the audio is at full scale (i.e. the highest possible volume level),
        // the summing of the two channels can cause the audio signal to clip or distort.

        // To prevent this clipping, FFmpeg applies a scaling factor of 1/sqrt(2) (~ 0.707)
        // to the audio signal before summing the two channels. This scaling factor ensures
        // that the combined audio signal will not exceed the maximum possible level, even
        // if both channels are at full scale.

        // After applying this scaling factor, the audio signal from both channels is summed
        // to create a single mono channel. It's worth noting that this scaling factor is
        // only applied when downmixing stereo audio to mono using the -ac 1 option in FFmpeg.
        // If you're using a different downmixing method, or if you're not downmixing the
        // audio at all, this scaling factor may not be needed.
        const SCALING_FACTOR = Math.sqrt(2);

        const left = decoded.getChannelData(0);
        const right = decoded.getChannelData(1);

        audio = new Float32Array(left.length);
        for (let i = 0; i < decoded.length; ++i) {
            audio[i] = (SCALING_FACTOR * (left[i] + right[i])) / 2;
        }
    } else {
        // If the audio is not stereo, we can just use the first channel:
        audio = decoded.getChannelData(0);
    }

    return audio;
}

/**
 * @deprecated Use {@link load_audio} instead.
 * @internal
 */
export async function read_audio(url, sampling_rate) {
    return await load_audio(url, sampling_rate);
}

/**
 * Helper function to generate windows that are special cases of the generalized cosine window.
 * See https://www.mathworks.com/help/signal/ug/generalized-cosine-windows.html for more information.
 * @param {number} M Number of points in the output window. If zero or less, an empty array is returned.
 * @param {number} a_0 Offset for the generalized cosine window.
 * @returns {Float64Array} The generated window.
 */
function generalized_cosine_window(M, a_0) {
    if (M < 1) {
        return new Float64Array();
    }
    if (M === 1) {
        return new Float64Array([1]);
    }

    const a_1 = 1 - a_0;
    const factor = (2 * Math.PI) / (M - 1);

    const cos_vals = new Float64Array(M);
    for (let i = 0; i < M; ++i) {
        cos_vals[i] = a_0 - a_1 * Math.cos(i * factor);
    }
    return cos_vals;
}

/**
 * Generates a Hanning window of length M.
 * See https://numpy.org/doc/stable/reference/generated/numpy.hanning.html for more information.
 *
 * @param {number} M The length of the Hanning window to generate.
 * @returns {Float64Array} The generated Hanning window.
 */
export function hanning(M) {
    return generalized_cosine_window(M, 0.5);
}

/**
 * Generates a Hamming window of length M.
 * See https://numpy.org/doc/stable/reference/generated/numpy.hamming.html for more information.
 *
 * @param {number} M The length of the Hamming window to generate.
 * @returns {Float64Array} The generated Hamming window.
 */
export function hamming(M) {
    return generalized_cosine_window(M, 0.54);
}

const HERTZ_TO_MEL_MAPPING = {
    htk: (/** @type {number} */ freq) => 2595.0 * Math.log10(1.0 + freq / 700.0),
    kaldi: (/** @type {number} */ freq) => 1127.0 * Math.log(1.0 + freq / 700.0),
    slaney: (/** @type {number} */ freq, min_log_hertz = 1000.0, min_log_mel = 15.0, logstep = 27.0 / Math.log(6.4)) =>
        freq >= min_log_hertz ? min_log_mel + Math.log(freq / min_log_hertz) * logstep : (3.0 * freq) / 200.0,
};

/**
 * @template {Float32Array|Float64Array|number} T
 * @param {T} freq
 * @param {keyof typeof HERTZ_TO_MEL_MAPPING} [mel_scale]
 * @returns {T}
 */
function hertz_to_mel(freq, mel_scale = 'htk') {
    const fn = HERTZ_TO_MEL_MAPPING[mel_scale];
    if (!fn) {
        throw new Error('mel_scale should be one of "htk", "slaney" or "kaldi".');
    }

    // @ts-expect-error ts(2322)
    return typeof freq === 'number' ? fn(freq) : freq.map((x) => fn(x));
}

const MEL_TO_HERTZ_MAPPING = {
    htk: (/** @type {number} */ mels) => 700.0 * (10.0 ** (mels / 2595.0) - 1.0),
    kaldi: (/** @type {number} */ mels) => 700.0 * (Math.exp(mels / 1127.0) - 1.0),
    slaney: (/** @type {number} */ mels, min_log_hertz = 1000.0, min_log_mel = 15.0, logstep = Math.log(6.4) / 27.0) =>
        mels >= min_log_mel ? min_log_hertz * Math.exp(logstep * (mels - min_log_mel)) : (200.0 * mels) / 3.0,
};

/**
 * @template {Float32Array|Float64Array|number} T
 * @param {T} mels
 * @param {keyof typeof MEL_TO_HERTZ_MAPPING} [mel_scale]
 * @returns {T}
 */
function mel_to_hertz(mels, mel_scale = 'htk') {
    const fn = MEL_TO_HERTZ_MAPPING[mel_scale];
    if (!fn) {
        throw new Error('mel_scale should be one of "htk", "slaney" or "kaldi".');
    }

    // @ts-expect-error ts(2322)
    return typeof mels === 'number' ? fn(mels) : mels.map((x) => fn(x));
}

/**
 * Creates a triangular filter bank.
 *
 * Adapted from torchaudio and librosa.
 *
 * @param {Float64Array} fft_freqs Discrete frequencies of the FFT bins in Hz, of shape `(num_frequency_bins,)`.
 * @param {Float64Array} filter_freqs Center frequencies of the triangular filters to create, in Hz, of shape `(num_mel_filters,)`.
 * @returns {number[][]} of shape `(num_frequency_bins, num_mel_filters)`.
 */
function _create_triangular_filter_bank(fft_freqs, filter_freqs) {
    const filter_diff = Float64Array.from(
        { length: filter_freqs.length - 1 },
        (_, i) => filter_freqs[i + 1] - filter_freqs[i],
    );

    const slopes = Array.from(
        {
            length: fft_freqs.length,
        },
        () => new Array(filter_freqs.length),
    );

    for (let j = 0; j < fft_freqs.length; ++j) {
        const slope = slopes[j];
        for (let i = 0; i < filter_freqs.length; ++i) {
            slope[i] = filter_freqs[i] - fft_freqs[j];
        }
    }

    const numFreqs = filter_freqs.length - 2;
    const ret = Array.from({ length: numFreqs }, () => new Array(fft_freqs.length));

    for (let j = 0; j < fft_freqs.length; ++j) {
        // 201
        const slope = slopes[j];
        for (let i = 0; i < numFreqs; ++i) {
            // 80
            const down = -slope[i] / filter_diff[i];
            const up = slope[i + 2] / filter_diff[i + 1];
            ret[i][j] = Math.max(0, Math.min(down, up));
        }
    }
    return ret;
}

/**
 * Return evenly spaced numbers over a specified interval.
 * @param {number} start The starting value of the sequence.
 * @param {number} end The end value of the sequence.
 * @param {number} num Number of samples to generate.
 * @returns `num` evenly spaced samples, calculated over the interval `[start, stop]`.
 */
function linspace(start, end, num) {
    const step = (end - start) / (num - 1);
    return Float64Array.from({ length: num }, (_, i) => start + step * i);
}

/**
 * Creates a frequency bin conversion matrix used to obtain a mel spectrogram. This is called a *mel filter bank*, and
 * various implementation exist, which differ in the number of filters, the shape of the filters, the way the filters
 * are spaced, the bandwidth of the filters, and the manner in which the spectrum is warped. The goal of these
 * features is to approximate the non-linear human perception of the variation in pitch with respect to the frequency.
 * @param {number} num_frequency_bins Number of frequency bins (should be the same as `n_fft // 2 + 1`
 * where `n_fft` is the size of the Fourier Transform used to compute the spectrogram).
 * @param {number} num_mel_filters Number of mel filters to generate.
 * @param {number} min_frequency Lowest frequency of interest in Hz.
 * @param {number} max_frequency Highest frequency of interest in Hz. This should not exceed `sampling_rate / 2`.
 * @param {number} sampling_rate Sample rate of the audio waveform.
 * @param {string|null} [norm] If `"slaney"`, divide the triangular mel weights by the width of the mel band (area normalization).
 * @param {keyof typeof MEL_TO_HERTZ_MAPPING} [mel_scale] The mel frequency scale to use, `"htk"` or `"slaney"`.
 * @param {boolean} [triangularize_in_mel_space] If this option is enabled, the triangular filter is applied in mel space rather than frequency space.
 * This should be set to `true` in order to get the same results as `torchaudio` when computing mel filters.
 * @returns {number[][]} Triangular filter bank matrix, which is a 2D array of shape (`num_frequency_bins`, `num_mel_filters`).
 * This is a projection matrix to go from a spectrogram to a mel spectrogram.
 */
export function mel_filter_bank(
    num_frequency_bins,
    num_mel_filters,
    min_frequency,
    max_frequency,
    sampling_rate,
    norm = null,
    mel_scale = 'htk',
    triangularize_in_mel_space = false,
) {
    if (norm !== null && norm !== 'slaney') {
        throw new Error('norm must be one of null or "slaney"');
    }

    if (num_frequency_bins < 2) {
        throw new Error(`Require num_frequency_bins: ${num_frequency_bins} >= 2`);
    }

    if (min_frequency > max_frequency) {
        throw new Error(`Require min_frequency: ${min_frequency} <= max_frequency: ${max_frequency}`);
    }

    const mel_min = hertz_to_mel(min_frequency, mel_scale);
    const mel_max = hertz_to_mel(max_frequency, mel_scale);
    const mel_freqs = linspace(mel_min, mel_max, num_mel_filters + 2);

    let filter_freqs = mel_to_hertz(mel_freqs, mel_scale);
    let fft_freqs; // frequencies of FFT bins in Hz

    if (triangularize_in_mel_space) {
        const fft_bin_width = sampling_rate / ((num_frequency_bins - 1) * 2);
        fft_freqs = hertz_to_mel(
            Float64Array.from({ length: num_frequency_bins }, (_, i) => i * fft_bin_width),
            mel_scale,
        );
        filter_freqs = mel_freqs;
    } else {
        fft_freqs = linspace(0, Math.floor(sampling_rate / 2), num_frequency_bins);
    }

    const mel_filters = _create_triangular_filter_bank(fft_freqs, filter_freqs);

    if (norm !== null && norm === 'slaney') {
        // Slaney-style mel is scaled to be approx constant energy per channel
        for (let i = 0; i < num_mel_filters; ++i) {
            const filter = mel_filters[i];
            const enorm = 2.0 / (filter_freqs[i + 2] - filter_freqs[i]);
            for (let j = 0; j < num_frequency_bins; ++j) {
                // Apply this enorm to all frequency bins
                filter[j] *= enorm;
            }
        }
    }

    // TODO warn if there is a zero row

    return mel_filters;
}

/**
 * Pads a signal with reflected copies of itself on both ends, excluding the edge samples. Padding longer than the
 * signal keeps reflecting (the signal is extended periodically with period `2 * (length - 1)`).
 * @param {Float32Array} signal The signal to pad (at least 2 samples).
 * @param {number} left The amount of padding to add to the left.
 * @param {number} right The amount of padding to add to the right.
 * @returns {Promise<Float32Array>} The padded signal.
 */
async function reflect_pad(signal, left, right) {
    const period = signal.length - 1;
    if (period < 1 && (left > 0 || right > 0)) {
        throw new Error('Reflect padding needs a signal of at least 2 samples.');
    }
    let padded = new Tensor('float32', signal, [signal.length]);
    while (left > 0 || right > 0) {
        // ONNX reflect padding is limited to `length - 1` per side. Padding by a multiple of `period` keeps each edge
        // on a point of symmetry of the periodic extension, so repeated reflection continues it correctly.
        const room = Math.floor((padded.dims[0] - 1) / period) * period;
        const l = Math.min(left, room);
        const r = Math.min(right, room);
        padded = await pad(padded, [l, r], { mode: 'reflect' });
        left -= l;
        right -= r;
    }
    return /** @type {Float32Array} */ (padded.data);
}

/**
 * Helper function to compute `amplitude_to_db` and `power_to_db`.
 * @param {Tensor} spectrogram
 * @param {number} factor
 * @param {number} reference
 * @param {number} min_value
 * @param {number} db_range
 * @returns {Promise<Tensor>}
 */
async function _db_conversion_helper(spectrogram, factor, reference, min_value, db_range) {
    if (reference <= 0) {
        throw new Error('reference must be greater than zero');
    }

    if (min_value <= 0) {
        throw new Error('min_value must be greater than zero');
    }

    reference = Math.max(min_value, reference);

    // factor * log10(max(min_value, x) - log10(reference))
    const logReference = Math.log10(reference);
    let result = await clamp(spectrogram, min_value);
    if (logReference !== 0) {
        result = await add(result, -logReference);
    }
    result = await mul(await log(result), factor * Math.LOG10E);

    if (db_range !== null) {
        if (db_range <= 0) {
            throw new Error('db_range must be greater than zero');
        }
        result = await clamp(result, Number((await amax(result)).item()) - db_range);
    }

    return result;
}

/**
 * Converts an amplitude spectrogram to the decibel scale. This computes `20 * log10(spectrogram / reference)`,
 * using basic logarithm properties for numerical stability.
 *
 * The motivation behind applying the log function on the (mel) spectrogram is that humans do not hear loudness on a
 * linear scale. Generally to double the perceived volume of a sound we need to put 8 times as much energy into it.
 * This means that large variations in energy may not sound all that different if the sound is loud to begin with.
 * This compression operation makes the (mel) spectrogram features match more closely what humans actually hear.
 *
 * @param {Tensor} spectrogram The input amplitude (mel) spectrogram.
 * @param {number} [reference=1.0] Sets the input spectrogram value that corresponds to 0 dB.
 * For example, use `np.max(spectrogram)` to set the loudest part to 0 dB. Must be greater than zero.
 * @param {number} [min_value=1e-5] The spectrogram will be clipped to this minimum value before conversion to decibels,
 * to avoid taking `log(0)`. The default of `1e-5` corresponds to a minimum of -100 dB. Must be greater than zero.
 * @param {number} [db_range=null] Sets the maximum dynamic range in decibels. For example, if `db_range = 80`, the
 * difference between the peak value and the smallest value will never be more than 80 dB. Must be greater than zero.
 * @returns {Promise<Tensor>} The spectrogram in decibels.
 */
function amplitude_to_db(spectrogram, reference = 1.0, min_value = 1e-5, db_range = null) {
    return _db_conversion_helper(spectrogram, 20.0, reference, min_value, db_range);
}

/**
 * Converts a power spectrogram to the decibel scale. This computes `10 * log10(spectrogram / reference)`,
 * using basic logarithm properties for numerical stability.
 *
 * The motivation behind applying the log function on the (mel) spectrogram is that humans do not hear loudness on a
 * linear scale. Generally to double the perceived volume of a sound we need to put 8 times as much energy into it.
 * This means that large variations in energy may not sound all that different if the sound is loud to begin with.
 * This compression operation makes the (mel) spectrogram features match more closely what humans actually hear.
 *
 * Based on the implementation of `librosa.power_to_db`.
 *
 * @param {Tensor} spectrogram The input power (mel) spectrogram. Note that a power spectrogram has the amplitudes squared!
 * @param {number} [reference=1.0] Sets the input spectrogram value that corresponds to 0 dB.
 * For example, use `np.max(spectrogram)` to set the loudest part to 0 dB. Must be greater than zero.
 * @param {number} [min_value=1e-10] The spectrogram will be clipped to this minimum value before conversion to decibels,
 * to avoid taking `log(0)`. The default of `1e-10` corresponds to a minimum of -100 dB. Must be greater than zero.
 * @param {number} [db_range=null] Sets the maximum dynamic range in decibels. For example, if `db_range = 80`, the
 * difference between the peak value and the smallest value will never be more than 80 dB. Must be greater than zero.
 * @returns {Promise<Tensor>} The spectrogram in decibels.
 */
function power_to_db(spectrogram, reference = 1.0, min_value = 1e-10, db_range = null) {
    return _db_conversion_helper(spectrogram, 10.0, reference, min_value, db_range);
}

/**
 * Frames per pass through the ONNX Runtime ops in `compute_mel_spectrogram`. Bounds their intermediate
 * tensors (a few MB for Whisper's 400-sample frames) regardless of the audio length.
 */
const SPECTROGRAM_CHUNK_FRAMES = 1024;

/**
 * Builds the linear operator that `spectrogram()` applies to one frame before taking magnitudes:
 * DC offset removal and preemphasis over the frame's first `length` (real) samples, the window, and the DFT.
 * Each step is linear, so they combine into one `[frame_length, 2 * num_frequency_bins]` matrix `M`
 * with `[real, imag] = frame @ M` (the imaginary part's sign is irrelevant for magnitudes).
 *
 * @param {Float32Array|Float64Array} window The window, of length `frame_length`.
 * @param {number} fft_length The FFT length.
 * @param {number} num_frequency_bins The number of frequency bins.
 * @param {number} length The number of real (not zero-padded) samples in the frame.
 * @param {boolean|null} remove_dc_offset Whether to subtract the frame's mean.
 * @param {number|null} preemphasis The preemphasis coefficient, or `null` for none.
 * @returns {Float64Array} `M` in row-major order.
 */
function frame_operator(window, fft_length, num_frequency_bins, length, remove_dc_offset, preemphasis) {
    const frame_length = window.length;
    const width = 2 * num_frequency_bins;
    const rows = Math.min(length, frame_length);
    const cos = new Float64Array(fft_length);
    const sin = new Float64Array(fft_length);
    for (let m = 0; m < fft_length; ++m) {
        cos[m] = Math.cos((2 * Math.PI * m) / fft_length);
        sin[m] = Math.sin((2 * Math.PI * m) / fft_length);
    }

    // Window and DFT: M[n, k] = window[n] * cos(2πkn / fft_length), M[n, bins + k] = window[n] * sin(...).
    // Rows past `length` stay zero: those samples are zero padding.
    const M = new Float64Array(frame_length * width);
    for (let n = 0; n < rows; ++n) {
        const row = n * width;
        for (let k = 0; k < num_frequency_bins; ++k) {
            const m = (k * n) % fft_length;
            M[row + k] = window[n] * cos[m];
            M[row + num_frequency_bins + k] = window[n] * sin[m];
        }
    }

    // Preemphasis y[j] = x[j] - p * x[j - 1], y[0] = (1 - p) * x[0], applied before the window:
    // M <- P^T M, i.e. row i becomes P[i, i] * M[i] - p * M[i + 1].
    if (preemphasis !== null) {
        for (let i = 0; i < rows; ++i) {
            const row = i * width;
            const scale = i === 0 ? 1 - preemphasis : 1;
            const has_next = i + 1 < rows;
            for (let k = 0; k < width; ++k) {
                M[row + k] = scale * M[row + k] - (has_next ? preemphasis * M[row + width + k] : 0);
            }
        }
    }

    // DC offset removal x - mean(x), applied first: M <- (I - 11^T / length) M over the real rows.
    if (remove_dc_offset && rows > 0) {
        const mean = new Float64Array(width);
        for (let i = 0; i < rows; ++i) {
            for (let k = 0; k < width; ++k) mean[k] += M[i * width + k];
        }
        for (let k = 0; k < width; ++k) mean[k] /= rows;
        for (let i = 0; i < rows; ++i) {
            for (let k = 0; k < width; ++k) M[i * width + k] -= mean[k];
        }
    }
    return M;
}

/**
 * @typedef {Object} SpectrogramKernel The precomputed per-frame work of `spectrogram()`, from `spectrogram_kernel()`.
 * @property {Float32Array|Float64Array} window The window it was built for.
 * @property {number} hop_length The hop length it was built for.
 * @property {number} fft_length The FFT length it was built for.
 * @property {number} num_frequency_bins The number of frequency bins.
 * @property {number|null} preemphasis The preemphasis coefficient it was built for.
 * @property {boolean} remove_dc_offset Whether it removes the DC offset.
 * @property {Tensor} weight The polyphase `conv1d` kernel, of shape `[2 * num_frequency_bins, hop_length, taps]`.
 * @property {number} offset The first frame sample the kernel covers.
 */

/**
 * Builds the `conv1d` kernel that `spectrogram()` uses to compute every frame's DFT in ONNX Runtime.
 *
 * DC offset removal, preemphasis, the window and the DFT are linear, so they combine into one per-frame matrix
 * (see `frame_operator`). It is rearranged into a polyphase kernel: convolving it (stride 1) with the signal viewed as
 * `[blocks, hop_length]` computes one frame per block, so the hop length is a shape instead of a stride. Leading and
 * trailing all-zero rows (e.g. a window zero-padded to `frame_length`) are dropped.
 *
 * Building it takes `O(frame_length * fft_length)` work in JS, so feature extractors build it once, like their window
 * and mel filters, and pass it to `spectrogram()` as `options.kernel`.
 *
 * @param {Float32Array|Float64Array} window The window, of length `frame_length`.
 * @param {number} hop_length The stride between successive frames in samples.
 * @param {Object} [options]
 * @param {number} [options.fft_length=null] The size of the FFT buffer in samples. If `null`, uses `window.length`.
 * @param {boolean} [options.onesided=true] Whether to compute only the positive frequencies.
 * @param {number} [options.preemphasis=null] The preemphasis coefficient, or `null` for none.
 * @param {boolean} [options.remove_dc_offset=null] Whether to subtract each frame's mean.
 * @returns {SpectrogramKernel} The kernel, to pass to `spectrogram()` with the same window and settings.
 */
export function spectrogram_kernel(
    window,
    hop_length,
    { fft_length = null, onesided = true, preemphasis = null, remove_dc_offset = null } = {},
) {
    const frame_length = window.length;
    fft_length ??= frame_length;
    preemphasis ??= null;
    const num_frequency_bins = onesided ? Math.floor(fft_length / 2) + 1 : fft_length;
    const width = 2 * num_frequency_bins;
    const M = frame_operator(window, fft_length, num_frequency_bins, frame_length, remove_dc_offset, preemphasis);

    const is_zero_row = (/** @type {number} */ n) => M.subarray(n * width, (n + 1) * width).every((v) => v === 0);
    let first = 0;
    while (first < frame_length - 1 && is_zero_row(first)) ++first;
    let last = frame_length - 1;
    while (last > first && is_zero_row(last)) --last;
    const length = last - first + 1;

    const taps = Math.ceil(length / hop_length);
    const data = new Float32Array(width * hop_length * taps);
    for (let o = 0; o < width; ++o) {
        for (let c = 0; c < hop_length; ++c) {
            for (let j = 0; j < taps; ++j) {
                const n = j * hop_length + c;
                if (n < length) data[(o * hop_length + c) * taps + j] = M[(first + n) * width + o];
            }
        }
    }
    return {
        window,
        hop_length,
        fft_length,
        num_frequency_bins,
        preemphasis,
        remove_dc_offset: !!remove_dc_offset,
        weight: new Tensor('float32', data, [width, hop_length, taps]),
        offset: first,
    };
}

/**
 * Builds the mel filters for a `matmul` with the DFT outputs: `[num_mel_filters, columns]`, or its transpose
 * `[columns, num_mel_filters]` if `transposed` (to multiply as `x.T @ filters`). With `duplicated`, the filters are
 * repeated for the real and imaginary parts (`columns = 2 * bins`), so the product also sums them.
 * @param {number[][]} mel_filters The mel filters, of shape `[num_mel_filters, bins]`.
 * @param {boolean} duplicated Whether to repeat the filters for the real and imaginary parts.
 * @param {boolean} transposed Whether to return the transpose.
 * @returns {Tensor}
 */
function mel_matrix(mel_filters, duplicated, transposed) {
    const num_mel_filters = mel_filters.length;
    const bins = mel_filters[0].length;
    const columns = duplicated ? 2 * bins : bins;
    const data = new Float32Array(num_mel_filters * columns);
    for (let m = 0; m < num_mel_filters; ++m) {
        const filter = mel_filters[m];
        for (let k = 0; k < columns; ++k) {
            data[transposed ? k * num_mel_filters + m : m * columns + k] = filter[k % bins];
        }
    }
    return new Tensor('float32', data, transposed ? [columns, num_mel_filters] : [num_mel_filters, columns]);
}

/**
 * Builds the `[bins, 2 * bins]` matrix that sums the real and imaginary parts of the DFT outputs.
 * @param {number} bins
 * @returns {Tensor}
 */
function complex_sum_matrix(bins) {
    const data = new Float32Array(bins * 2 * bins);
    for (let k = 0; k < bins; ++k) {
        data[k * 2 * bins + k] = 1;
        data[k * 2 * bins + bins + k] = 1;
    }
    return new Tensor('float32', data, [bins, 2 * bins]);
}

/**
 * Computes `mel_filters @ |STFT(waveform)|^power` with ONNX Runtime ops, so it stays fast on JS engines without a JIT
 * (e.g. Hermes in React Native), where a JS FFT per frame takes seconds per 30 s of audio.
 *
 * A polyphase `conv1d` with the kernel from `spectrogram_kernel()` computes every frame's DFT, then `pow` and
 * `matmul` give the mel spectrogram. Frames that only partly overlap the signal (when `min_num_frames` asks for more
 * frames) need DC offset removal and preemphasis over fewer samples, so they use their own operators, in one batched
 * `matmul`.
 *
 * @param {Float32Array|Float64Array} waveform The (already padded) waveform.
 * @param {SpectrogramKernel} kernel The kernel from `spectrogram_kernel()`.
 * @param {Object} options
 * @param {number} options.num_frames Number of frames to compute.
 * @param {number} options.total_frames Number of output frames; frames past `num_frames` are zero.
 * @param {number} options.power
 * @param {number[][]} options.mel_filters
 * @param {boolean} options.transpose Whether to return `[total_frames, num_mel_filters]`.
 * @returns {Promise<Tensor>} The mel spectrogram of shape `[num_mel_filters, total_frames]`,
 * or `[total_frames, num_mel_filters]` if `transpose`.
 */
async function compute_mel_spectrogram(waveform, kernel, { num_frames, total_frames, power, mel_filters, transpose }) {
    const { window, hop_length, fft_length, num_frequency_bins, preemphasis, remove_dc_offset, weight, offset } =
        kernel;
    const frame_length = window.length;
    const width = 2 * num_frequency_bins;
    const num_mel_filters = mel_filters.length;
    const signal = waveform instanceof Float32Array ? waveform : Float32Array.from(waveform);
    const output = new Float32Array(num_mel_filters * total_frames);

    // For `power = 2`, one matmul sums the squared real and imaginary parts and projects them onto the mel filters.
    // Otherwise, sum them first, then take `|X|^power = (|X|^2)^(power / 2)`, then project.
    const squared_power = power === 2;
    const complex_sum = squared_power ? null : complex_sum_matrix(num_frequency_bins);
    const filters = mel_matrix(mel_filters, squared_power, transpose);

    /**
     * Writes `mel_filters @ |X|^power` for consecutive frames, given the squared DFT outputs.
     * @param {Float32Array} squared The squared DFT outputs, of shape `[2 * bins, count]`.
     * @param {number} start Index of the first frame.
     * @param {number} count Number of frames.
     */
    const write_mel = async (squared, start, count) => {
        let x = new Tensor('float32', squared, [width, count]);
        if (!squared_power) {
            x = await pow(await matmul(complex_sum, x), power / 2);
        }
        if (transpose) {
            // [count, num_mel_filters] rows are contiguous in the output.
            const mel = await matmul(x, filters, { transpose_a: true });
            output.set(/** @type {Float32Array} */ (mel.data), start * num_mel_filters);
        } else {
            const data = /** @type {Float32Array} */ ((await matmul(filters, x)).data);
            for (let m = 0; m < num_mel_filters; ++m) {
                output.set(data.subarray(m * count, (m + 1) * count), m * total_frames + start);
            }
        }
    };

    const taps = weight.dims[2];

    for (let start = 0; start < num_frames; start += SPECTROGRAM_CHUNK_FRAMES) {
        const count = Math.min(SPECTROGRAM_CHUNK_FRAMES, num_frames - start);
        // Frame `start + i` covers blocks `start + i` to `start + i + taps - 1` of `hop_length` samples,
        // counted from sample `offset` of the first frame.
        const blocks = count + taps - 1;
        const begin = start * hop_length + offset;
        const end = begin + blocks * hop_length;
        let chunk;
        if (end <= signal.length) {
            chunk = signal.subarray(begin, end);
        } else {
            chunk = new Float32Array(end - begin);
            if (begin < signal.length) chunk.set(signal.subarray(begin));
        }
        const spectrum = await conv1d(new Tensor('float32', chunk, [1, blocks, hop_length]), weight, {
            channels_last: true,
        });
        await write_mel(/** @type {Float32Array} */ ((await pow(spectrum, 2)).data), start, count);
    }

    // Frames that only partly overlap the signal, if DC offset removal or preemphasis depend on that.
    if (remove_dc_offset || preemphasis !== null) {
        const first = Math.max(0, Math.floor((signal.length - frame_length) / hop_length) + 1);
        const last = Math.min(num_frames, Math.ceil(signal.length / hop_length));
        const partial = Math.max(0, last - first);
        if (partial > 0) {
            const frames = new Float32Array(partial * frame_length);
            const operators = new Float32Array(partial * frame_length * width);
            for (let i = 0; i < partial; ++i) {
                const frame_start = (first + i) * hop_length;
                const length = signal.length - frame_start;
                frames.set(signal.subarray(frame_start), i * frame_length);
                operators.set(
                    frame_operator(window, fft_length, num_frequency_bins, length, remove_dc_offset, preemphasis),
                    i * frame_length * width,
                );
            }
            const spectrum = await matmul(
                new Tensor('float32', frames, [partial, 1, frame_length]),
                new Tensor('float32', operators, [partial, frame_length, width]),
            );
            const squared = /** @type {Float32Array} */ ((await pow(spectrum, 2)).data);
            // [partial, 1, 2 * bins] -> [2 * bins, partial]
            const transposed = new Float32Array(width * partial);
            for (let i = 0; i < partial; ++i) {
                for (let k = 0; k < width; ++k) transposed[k * partial + i] = squared[i * width + k];
            }
            await write_mel(transposed, first, partial);
        }
    }
    return new Tensor('float32', output, transpose ? [total_frames, num_mel_filters] : [num_mel_filters, total_frames]);
}

/**
 * Calculates a spectrogram over one waveform using the Short-Time Fourier Transform.
 *
 * This function can create the following kinds of spectrograms:
 *   - amplitude spectrogram (`power = 1.0`)
 *   - power spectrogram (`power = 2.0`)
 *   - complex-valued spectrogram (`power = None`)
 *   - log spectrogram (use `log_mel` argument)
 *   - mel spectrogram (provide `mel_filters`)
 *   - log-mel spectrogram (provide `mel_filters` and `log_mel`)
 *
 * In this implementation, the window is assumed to be zero-padded to have the same size as the analysis frame.
 * A padded window can be obtained from `window_function()`. The FFT input buffer may be larger than the analysis frame,
 * typically the next power of two.
 *
 * @param {Float32Array|Float64Array} waveform The input waveform of shape `(length,)`. This must be a single real-valued, mono waveform.
 * @param {Float32Array|Float64Array} window The windowing function to apply of shape `(frame_length,)`, including zero-padding if necessary. The actual window length may be
 * shorter than `frame_length`, but we're assuming the array has already been zero-padded.
 * @param {number} frame_length The length of the analysis frames in samples (a.k.a., `fft_length`).
 * @param {number} hop_length The stride between successive analysis frames in samples.
 * @param {Object} options
 * @param {number} [options.fft_length=null] The size of the FFT buffer in samples. This determines how many frequency bins the spectrogram will have.
 * For optimal speed, this should be a power of two. If `null`, uses `frame_length`.
 * @param {number} [options.power=1.0] If 1.0, returns the amplitude spectrogram. If 2.0, returns the power spectrogram. If `null`, returns complex numbers.
 * @param {boolean} [options.center=true] Whether to pad the waveform so that frame `t` is centered around time `t * hop_length`. If `false`, frame
 * `t` will start at time `t * hop_length`.
 * @param {string} [options.pad_mode="reflect"] Padding mode used when `center` is `true`. Possible values are: `"constant"` (pad with zeros),
 * `"edge"` (pad with edge values), `"reflect"` (pads with mirrored values).
 * @param {boolean} [options.onesided=true] If `true`, only computes the positive frequencies and returns a spectrogram containing `fft_length // 2 + 1`
 * frequency bins. If `false`, also computes the negative frequencies and returns `fft_length` frequency bins.
 * @param {number} [options.preemphasis=null] Coefficient for a low-pass filter that applies pre-emphasis before the DFT.
 * @param {boolean} [options.preemphasis_htk_flavor=true] Whether to apply the pre-emphasis filter in the HTK flavor.
 * @param {number[][]} [options.mel_filters=null] The mel filter bank of shape `(num_freq_bins, num_mel_filters)`.
 * If supplied, applies this filter bank to create a mel spectrogram.
 * @param {number} [options.mel_floor=1e-10] Minimum value of mel frequency banks.
 * @param {string} [options.log_mel=null] How to convert the spectrogram to log scale. Possible options are:
 * `null` (don't convert), `"log"` (take the natural logarithm), `"log10"` (take the base-10 logarithm), `"dB"` (convert to decibels),
 * `"log10_max_norm"` (take `log10`, then apply `(max(x, maxVal - 8) + 4) / 4` normalization, where `maxVal` is computed from data or given by `max_log_mel`).
 * Can only be used when `power` is not `null`.
 * @param {number} [options.max_log_mel=null] When `log_mel` is `"log10_max_norm"`, use this fixed value as the max instead of computing from data.
 * @param {number} [options.reference=1.0] Sets the input spectrogram value that corresponds to 0 dB. For example, use `max(spectrogram)[0]` to set
 * the loudest part to 0 dB. Must be greater than zero.
 * @param {number} [options.min_value=1e-10] The spectrogram will be clipped to this minimum value before conversion to decibels, to avoid taking `log(0)`.
 * For a power spectrogram, the default of `1e-10` corresponds to a minimum of -100 dB. For an amplitude spectrogram, the value `1e-5` corresponds to -100 dB.
 * Must be greater than zero.
 * @param {number} [options.db_range=null] Sets the maximum dynamic range in decibels. For example, if `db_range = 80`, the difference between the
 * peak value and the smallest value will never be more than 80 dB. Must be greater than zero.
 * @param {boolean} [options.remove_dc_offset=null] Subtract mean from waveform on each frame, applied before pre-emphasis. This should be set to `true` in
 * order to get the same results as `torchaudio.compliance.kaldi.fbank` when computing mel filters.
 * @param {number} [options.max_num_frames=null] If provided, limits the number of frames to compute to this value.
 * @param {number} [options.min_num_frames=null] If provided, ensures the number of frames to compute is at least this value.
 * @param {boolean} [options.do_pad=true] If `true`, pads the output spectrogram to have `max_num_frames` frames.
 * @param {boolean} [options.transpose=false] If `true`, the returned spectrogram will have shape `(num_frames, num_frequency_bins/num_mel_filters)`. If `false`, the returned spectrogram will have shape `(num_frequency_bins/num_mel_filters, num_frames)`.
 * @param {number} [options.mel_offset=0] Offset to add to the mel spectrogram to avoid taking the log of zero.
 * @param {SpectrogramKernel} [options.kernel=null] The kernel from `spectrogram_kernel()` for this `window`, `hop_length`,
 * `fft_length`, `onesided`, `preemphasis` and `remove_dc_offset`. Feature extractors build it once and pass it here;
 * if `null`, it is built on every call.
 * @param {string} [options.mel_floor_mode="clamp"] If `mel_offset` is provided, this option determines how to apply it. If `"clamp"`, the mel spectrogram will be clamped to have a minimum value of `mel_offset`. If `"add"`, `mel_offset` will be added to all values of the mel spectrogram.
 * @returns {Promise<Tensor>} Spectrogram of shape `(num_frequency_bins, length)` (regular spectrogram) or shape `(num_mel_filters, length)` (mel spectrogram).
 */
export async function spectrogram(
    waveform,
    window,
    frame_length,
    hop_length,
    {
        fft_length = null,
        power = 1.0,
        center = true,
        pad_mode = 'reflect',
        onesided = true,
        preemphasis = null,
        preemphasis_htk_flavor = true,
        mel_filters = null,
        mel_floor = 1e-10,
        log_mel = null,
        max_log_mel = null,
        reference = 1.0,
        min_value = 1e-10,
        db_range = null,
        remove_dc_offset = null,

        // Custom parameters for efficiency reasons
        min_num_frames = null,
        max_num_frames = null,
        do_pad = true,
        transpose = false,
        mel_offset = 0,
        mel_floor_mode = 'clamp',
        kernel = null,
    } = {},
) {
    const window_length = window.length;
    if (fft_length === null) {
        fft_length = frame_length;
    }
    if (frame_length > fft_length) {
        throw Error(`frame_length (${frame_length}) may not be larger than fft_length (${fft_length})`);
    }

    if (window_length !== frame_length) {
        throw new Error(`Length of the window (${window_length}) must equal frame_length (${frame_length})`);
    }

    if (hop_length <= 0) {
        throw new Error('hop_length must be greater than zero');
    }

    if (power === null && mel_filters !== null) {
        throw new Error(
            'You have provided `mel_filters` but `power` is `None`. Mel spectrogram computation is not yet supported for complex-valued spectrogram. ' +
                'Specify `power` to fix this issue.',
        );
    }

    if (!preemphasis_htk_flavor) {
        throw new Error('`preemphasis_htk_flavor=false` is not currently supported.');
    }

    let signal = waveform instanceof Float32Array ? waveform : new Float32Array(waveform);
    if (center) {
        const padding = Math.floor(frame_length / 2);
        switch (pad_mode) {
            case 'reflect':
                signal = await reflect_pad(signal, padding, padding);
                break;
            case 'constant':
                signal = /** @type {Float32Array} */ (
                    (await pad(new Tensor('float32', signal, [signal.length]), [padding, padding])).data
                );
                break;
            case 'semicausal':
                // Prepend padding zeros only (no right padding)
                signal = /** @type {Float32Array} */ (
                    (await pad(new Tensor('float32', signal, [signal.length]), [padding, 0])).data
                );
                break;
            default:
                throw new Error(`pad_mode="${pad_mode}" not implemented yet.`);
        }
    }

    // split waveform into frames of frame_length size
    let num_frames = Math.floor(1 + Math.floor((signal.length - frame_length) / hop_length));
    if (min_num_frames !== null && num_frames < min_num_frames) {
        num_frames = min_num_frames;
    }
    const num_frequency_bins = onesided ? Math.floor(fft_length / 2) + 1 : fft_length;

    let d1 = num_frames;
    let d1Max = num_frames;

    // If maximum number of frames is provided, we must either pad or truncate
    if (max_num_frames !== null) {
        if (max_num_frames > num_frames) {
            // input is too short, so we pad
            if (do_pad) {
                d1Max = max_num_frames;
            }
        } else {
            // input is too long, so we truncate
            d1Max = d1 = max_num_frames;
        }
    }

    // TODO: What if `mel_filters` is null?
    const num_mel_filters = mel_filters.length;
    if (kernel === null) {
        kernel = spectrogram_kernel(window, hop_length, { fft_length, onesided, preemphasis, remove_dc_offset });
    } else if (
        kernel.window !== window ||
        kernel.hop_length !== hop_length ||
        kernel.fft_length !== fft_length ||
        kernel.num_frequency_bins !== num_frequency_bins ||
        kernel.preemphasis !== (preemphasis ?? null) ||
        kernel.remove_dc_offset !== !!remove_dc_offset
    ) {
        throw new Error('`kernel` was built for a different window or settings than this `spectrogram()` call.');
    }
    let mel_spec = await compute_mel_spectrogram(signal, kernel, {
        num_frames: d1,
        total_frames: d1Max,
        power,
        mel_filters,
        transpose,
    });

    if (mel_floor_mode === 'add') {
        mel_spec = await add(mel_spec, mel_offset + mel_floor);
    } else {
        mel_spec = await clamp(mel_spec, mel_floor);
        if (mel_offset !== 0) {
            mel_spec = await add(mel_spec, mel_offset);
        }
    }

    if (power !== null && log_mel !== null) {
        const mel_spec_data = /** @type {Float32Array} */ (mel_spec.data);
        // The log is taken over the first `d1 * num_mel_filters` values (`dB` and the max use all of them).
        const o = Math.min(mel_spec_data.length, d1 * num_mel_filters);
        const head = new Tensor('float32', mel_spec_data.subarray(0, o), [o]);
        switch (log_mel) {
            case 'log':
                mel_spec_data.set(/** @type {Float32Array} */ ((await log(head)).data));
                break;
            case 'log10':
                mel_spec_data.set(/** @type {Float32Array} */ ((await mul(await log(head), Math.LOG10E)).data));
                break;
            case 'log10_max_norm': {
                mel_spec_data.set(/** @type {Float32Array} */ ((await mul(await log(head), Math.LOG10E)).data));
                const logMax = max_log_mel ?? Number((await amax(mel_spec)).item());
                // (max(x, logMax - 8) + 4) / 4
                const normalized = await add(await mul(await clamp(head, logMax - 8.0), 0.25), 1.0);
                mel_spec_data.set(/** @type {Float32Array} */ (normalized.data));
                break;
            }
            case 'dB':
                if (power === 1.0) {
                    mel_spec = await amplitude_to_db(mel_spec, reference, min_value, db_range);
                } else if (power === 2.0) {
                    mel_spec = await power_to_db(mel_spec, reference, min_value, db_range);
                } else {
                    throw new Error(`Cannot use log_mel option '${log_mel}' with power ${power}`);
                }
                break;
            default:
                throw new Error(
                    `log_mel must be one of null, 'log', 'log10', 'log10_max_norm', or 'dB'. Got '${log_mel}'`,
                );
        }
    }

    return mel_spec;
}

/**
 * Returns an array containing the specified window.
 * @param {number} window_length The length of the window in samples.
 * @param {string} name The name of the window function.
 * @param {Object} options Additional options.
 * @param {boolean} [options.periodic=true] Whether the window is periodic or symmetric.
 * @param {number} [options.frame_length=null] The length of the analysis frames in samples.
 * Provide a value for `frame_length` if the window is smaller than the frame length, so that it will be zero-padded.
 * @param {boolean} [options.center=true] Whether to center the window inside the FFT buffer. Only used when `frame_length` is provided.
 * @returns {Float64Array} The window of shape `(window_length,)` or `(frame_length,)`.
 */
export function window_function(window_length, name, { periodic = true, frame_length = null, center = true } = {}) {
    const length = periodic ? window_length + 1 : window_length;
    let window;
    switch (name) {
        case 'boxcar':
            window = new Float64Array(length).fill(1.0);
            break;
        case 'hann':
        case 'hann_window':
            window = hanning(length);
            break;
        case 'hamming':
            window = hamming(length);
            break;
        case 'povey':
            window = hanning(length).map((x) => Math.pow(x, 0.85));
            break;
        default:
            throw new Error(`Unknown window type ${name}.`);
    }
    if (periodic) {
        window = window.subarray(0, window_length);
    }
    if (frame_length === null || window_length === frame_length) {
        return window;
    }
    if (window_length > frame_length) {
        throw new Error(
            `Length of the window (${window_length}) may not be larger than frame_length (${frame_length})`,
        );
    }

    const padded = new Float64Array(frame_length);
    const offset = center ? Math.floor((frame_length - window_length) / 2) : 0;
    padded.set(window, offset);
    return padded;
}

/**
 * Efficiently encode audio data to a WAV file.
 * WAV file specs : https://en.wikipedia.org/wiki/WAV#WAV_File_header
 *
 * Adapted from https://www.npmjs.com/package/audiobuffer-to-wav
 * @param {Float32Array[]} chunks The audio samples.
 * @param {number} rate The sample rate.
 * @returns {Blob} The WAV file as a Blob.
 */
function encodeWAV(chunks, rate) {
    const totalLength = chunks.reduce((acc, chunk) => acc + chunk.length, 0);

    const buffer = new ArrayBuffer(44);
    const view = new DataView(buffer);

    /* RIFF identifier */
    writeString(view, 0, 'RIFF');
    /* RIFF chunk length */
    view.setUint32(4, 36 + totalLength * 4, true);
    /* RIFF type */
    writeString(view, 8, 'WAVE');
    /* format chunk identifier */
    writeString(view, 12, 'fmt ');
    /* format chunk length */
    view.setUint32(16, 16, true);
    /* sample format (raw) */
    view.setUint16(20, 3, true);
    /* channel count */
    view.setUint16(22, 1, true);
    /* sample rate */
    view.setUint32(24, rate, true);
    /* byte rate (sample rate * block align) */
    view.setUint32(28, rate * 4, true);
    /* block align (channel count * bytes per sample) */
    view.setUint16(32, 4, true);
    /* bits per sample */
    view.setUint16(34, 32, true);
    /* data chunk identifier */
    writeString(view, 36, 'data');
    /* data chunk length */
    view.setUint32(40, totalLength * 4, true);

    // SharedArrayBuffer cannot back a BlobPart, so only ArrayBuffer-backed chunks get a zero-copy view.
    return new Blob(
        [
            buffer,
            ...chunks.map((chunk) =>
                chunk.buffer instanceof ArrayBuffer
                    ? new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.byteLength)
                    : chunk.slice(),
            ),
        ],
        { type: 'audio/wav' },
    );
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; ++i) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

/**
 * An audio buffer paired with its sampling rate.
 *
 * **Example:**
 * ```javascript
 * import { RawAudio } from '@huggingface/transformers';
 * const samples = new Float32Array(16000); // 1 second of silence @ 16 kHz
 * const audio = new RawAudio(samples, 16000);
 * const blob = audio.toBlob(); // WAV blob for upload or playback
 * await audio.save('out.wav'); // Saves the audio as a WAV file named 'out.wav'
 * ```
 */
export class RawAudio {
    /**
     * Create a new `RawAudio` object.
     * @param {Float32Array|Float32Array[]} audio Audio data, either as a single `Float32Array` chunk or multiple `Float32Array` chunks.
     * @param {number} sampling_rate Sampling rate of the audio data
     */
    constructor(audio, sampling_rate) {
        this.audio = audio;
        this.sampling_rate = sampling_rate;
    }

    /**
     * Get the audio data, accumulating all chunks if necessary.
     * @returns {Float32Array} The audio data.
     */
    get data() {
        if (Array.isArray(this.audio)) {
            if (this.audio.length === 0) {
                return new Float32Array(0);
            }
            if (this.audio.length === 1) {
                return this.audio[0];
            }
            // Concatenate all chunks into a single Float32Array
            const totalLength = this.audio.reduce((acc, chunk) => acc + chunk.length, 0);
            const result = new Float32Array(totalLength);
            let offset = 0;
            for (const chunk of this.audio) {
                result.set(chunk, offset);
                offset += chunk.length;
            }
            return result;
        } else {
            return this.audio;
        }
    }

    /**
     * Convert the audio to a blob.
     * @returns {Blob}
     */
    toBlob() {
        let audio = this.audio;
        if (audio instanceof Float32Array) {
            audio = [audio]; // Ensure audio is an array of chunks
        }
        return encodeWAV(audio, this.sampling_rate);
    }

    /**
     * Save the audio to a wav file.
     * @param {string} path
     * @returns {Promise<void>}
     */
    async save(path) {
        return saveBlob(path, this.toBlob());
    }
}
