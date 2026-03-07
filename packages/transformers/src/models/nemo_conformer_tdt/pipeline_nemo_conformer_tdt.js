import { Tensor } from '../../utils/tensor.js';
import {
    buildWordChunks,
    buildNemoSegmentChunks,
    buildNemoWindowSpecs,
    mergeNemoWindowResults,
} from './utils_nemo_conformer_tdt.js';

const NEMO_AUTO_WINDOW_THRESHOLD_S = 180;
const NEMO_AUTO_CHUNK_LENGTH_S = 90;
const NEMO_AUTO_STRIDE_LENGTH_S = 10;

function validateNemoAudio(audio, index) {
    if (!(audio instanceof Float32Array || audio instanceof Float64Array)) {
        throw new TypeError(
            `Nemo Conformer TDT pipeline expected audio at index ${index} to be Float32Array or Float64Array.`,
        );
    }
    if (audio.length === 0) {
        throw new Error(`Nemo Conformer TDT pipeline expected non-empty audio at index ${index}.`);
    }
    for (let i = 0; i < audio.length; ++i) {
        if (!Number.isFinite(audio[i])) {
            throw new Error(
                `Nemo Conformer TDT pipeline expected finite audio samples; found ${audio[i]} at index ${index}:${i}.`,
            );
        }
    }
}

/**
 * Run the ASR pipeline adapter for Nemo Conformer TDT models.
 * Keeps the public contract task-shaped while delegating rich outputs to `model.transcribe()`.
 *
 * @param {{
 *   model: any,
 *   processor: any,
 *   tokenizer: any,
 *   audio: Float32Array|Float64Array|Array<Float32Array|Float64Array>,
 *   kwargs: Record<string, any>,
 *   prepareAudios: (audio: any[], sampling_rate: number) => Promise<(Float32Array|Float64Array)[]>,
 * }} options
 */
export async function runNemoConformerTDTPipeline({
    model,
    processor,
    tokenizer,
    audio,
    kwargs,
    prepareAudios,
}) {
    if (typeof model?.transcribe !== 'function') {
        throw new Error('Nemo Conformer TDT model does not expose a `transcribe` method.');
    }
    if (!processor) {
        throw new Error('Nemo Conformer TDT pipeline requires a processor.');
    }
    if (!tokenizer) {
        throw new Error('Nemo Conformer TDT pipeline requires a tokenizer.');
    }
    if (!processor.feature_extractor?.config?.sampling_rate) {
        throw new Error(
            'Nemo Conformer TDT pipeline requires `processor.feature_extractor.config.sampling_rate` to prepare audio.',
        );
    }

    const return_timestamps = kwargs.return_timestamps ?? false;
    const wantWordTimestamps = return_timestamps === 'word';
    const wantTimestampChunks = return_timestamps === true || wantWordTimestamps;
    const requested_chunk_length_s = kwargs.chunk_length_s ?? 0;
    const requested_stride_length_s = kwargs.stride_length_s ?? null;

    const single = !Array.isArray(audio);
    const batchedAudio = single ? [audio] : audio;
    const sampling_rate = processor.feature_extractor.config.sampling_rate;
    const preparedAudios = await prepareAudios(batchedAudio, sampling_rate);
    for (let i = 0; i < preparedAudios.length; ++i) {
        validateNemoAudio(preparedAudios[i], i);
    }

    const featureCache = /** @type {{ max_entries: number, max_size_mb: number }|null|undefined} */ (
        /** @type {any} */ (processor.feature_extractor)?.feature_cache
    );
    const cacheOwnsTensors = !!(
        featureCache &&
        featureCache.max_entries > 0 &&
        featureCache.max_size_mb > 0
    );

    const runNemoTranscribe = async (windowAudio, decodeOptions) => {
        const inputs = await processor(windowAudio);
        try {
            return await model.transcribe(inputs, decodeOptions);
        } finally {
            if (!cacheOwnsTensors) {
                const seen = new Set();
                for (const value of Object.values(inputs ?? {})) {
                    if (value instanceof Tensor && !seen.has(value)) {
                        value.dispose();
                        seen.add(value);
                    }
                }
            }
        }
    };

    const toReturn = [];
    for (const aud of preparedAudios) {
        const audio_duration_s = aud.length / sampling_rate;
        const autoWindowing = requested_chunk_length_s <= 0 && audio_duration_s > NEMO_AUTO_WINDOW_THRESHOLD_S;
        const chunk_length_s =
            requested_chunk_length_s > 0
                ? requested_chunk_length_s
                : autoWindowing
                  ? NEMO_AUTO_CHUNK_LENGTH_S
                  : 0;
        const stride_length_s =
            requested_chunk_length_s > 0
                ? requested_stride_length_s
                : autoWindowing
                  ? NEMO_AUTO_STRIDE_LENGTH_S
                  : null;

        if (chunk_length_s > 0) {
            const windows = buildNemoWindowSpecs(aud, sampling_rate, chunk_length_s, stride_length_s);
            const windowResults = [];
            for (const window of windows) {
                const output = await runNemoTranscribe(window.audio, {
                    tokenizer,
                    return_timestamps: true,
                    return_words: true,
                    return_tokens: true,
                    return_metrics: false,
                    timeOffset: window.start_s,
                });
                windowResults.push({ window, output });
            }

            const merged = mergeNemoWindowResults(tokenizer, windowResults);
            const result = { text: merged.text || windowResults.map((x) => x.output.text ?? '').join(' ').trim() };
            if (wantWordTimestamps) {
                result.chunks = buildWordChunks(merged.words);
            } else if (wantTimestampChunks) {
                result.chunks = buildNemoSegmentChunks(merged.words, merged.utterance_timestamp, result.text);
            }
            toReturn.push(result);
            continue;
        }

        const output = await runNemoTranscribe(aud, {
            tokenizer,
            return_timestamps: wantTimestampChunks,
            return_words: wantTimestampChunks,
            return_metrics: false,
        });

        const result = { text: output.text ?? '' };
        if (wantWordTimestamps) {
            result.chunks = buildWordChunks(output.words ?? []);
        } else if (wantTimestampChunks) {
            result.chunks = buildNemoSegmentChunks(output.words ?? [], output.utterance_timestamp ?? null, result.text);
        }
        toReturn.push(result);
    }

    return single ? toReturn[0] : toReturn;
}
