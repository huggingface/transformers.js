import { Tensor } from '../../utils/tensor.js';
import { NEMO_FEATURE_OUTPUT_OWNERSHIP, NEMO_FEATURE_OUTPUT_RELEASE } from './feature_extraction_nemo_conformer_tdt.js';
import {
    buildWordChunks,
    buildNemoSegmentChunks,
    joinTimedWords,
    partitionNemoWordsIntoSegments,
} from './transducer_segment_offsets.js';
import { dedupeMergedWords } from './transducer_window_merge.js';

const NEMO_AUTO_WINDOW_THRESHOLD_S = 180;
const NEMO_MIN_CHUNK_LENGTH_S = 20;
const NEMO_MAX_CHUNK_LENGTH_S = 180;
const NEMO_AUTO_CHUNK_LENGTH_S = 90;
const NEMO_AUTO_WINDOW_FALLBACK_OVERLAP_S = 10;
const NEMO_AUTO_WINDOW_EPSILON_S = 1e-6;
const NEMO_SEGMENT_DEDUP_TOLERANCE_S = 0.15;
const NEMO_CURSOR_MIN_ADVANCE_S = 1.0;
const NEMO_CURSOR_GAP_THRESHOLD_S = 0.2;
const NEMO_CURSOR_SNAP_WINDOW_S = 0.5;

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

function disposeNemoPipelineInputs(inputs) {
    const seen = new Set();
    for (const value of Object.values(inputs ?? {})) {
        if (value instanceof Tensor && !seen.has(value)) {
            value.dispose();
            seen.add(value);
        }
    }
}

function releaseNemoPipelineInputs(inputs) {
    const release = inputs?.[NEMO_FEATURE_OUTPUT_RELEASE];
    if (typeof release === 'function') {
        release();
    }
}

function normalizeNemoChunkLengthS(value) {
    const num = Number(value);
    if (!Number.isFinite(num) || num <= 0) {
        return 0;
    }
    return Math.max(NEMO_MIN_CHUNK_LENGTH_S, Math.min(NEMO_MAX_CHUNK_LENGTH_S, num));
}

function flattenNemoSegmentWords(segments) {
    return segments.flatMap((segment) => segment.words);
}

function mergePendingAndCurrentNemoWords(pendingWords, currentWords) {
    const normalizedPendingWords = Array.isArray(pendingWords) ? pendingWords : [];
    const normalizedCurrentWords = Array.isArray(currentWords) ? currentWords : [];

    if (normalizedPendingWords.length === 0) {
        return dedupeMergedWords(normalizedCurrentWords);
    }
    if (normalizedCurrentWords.length === 0) {
        return dedupeMergedWords(normalizedPendingWords);
    }

    const pendingStart = normalizedPendingWords[0].startTime;
    const currentStart = normalizedCurrentWords[0].startTime;
    if (currentStart <= pendingStart + NEMO_AUTO_WINDOW_EPSILON_S) {
        return dedupeMergedWords(normalizedCurrentWords);
    }

    return dedupeMergedWords([...normalizedPendingWords, ...normalizedCurrentWords]);
}

function normalizeNemoSegmentText(text) {
    return String(text ?? '')
        .normalize('NFKC')
        .replace(/[“”]/g, '"')
        .replace(/[‘’]/g, "'")
        .replace(/\s+/g, ' ')
        .trim()
        .toLowerCase();
}

function isDuplicateFinalizedNemoSegment(finalizedSegments, segment) {
    const normalized = normalizeNemoSegmentText(segment.text);
    if (!normalized) {
        return false;
    }

    return finalizedSegments.some(
        (candidate) =>
            normalizeNemoSegmentText(candidate.text) === normalized &&
            Math.abs(candidate.timestamp[1] - segment.timestamp[1]) < NEMO_SEGMENT_DEDUP_TOLERANCE_S,
    );
}

function appendFinalizedNemoSegment(finalizedSegments, segment) {
    if (isDuplicateFinalizedNemoSegment(finalizedSegments, segment)) {
        return;
    }
    finalizedSegments.push(segment);
}

function relocateNemoCursorToNearbyGap(target_s, words) {
    let best = target_s;
    let bestDist = NEMO_CURSOR_SNAP_WINDOW_S + 1;

    for (let i = 0; i < words.length - 1; ++i) {
        const current = words[i];
        const next = words[i + 1];
        const gapStart = current.endTime;
        const gapEnd = next.startTime;
        const gap = gapEnd - gapStart;
        if (gap < NEMO_CURSOR_GAP_THRESHOLD_S) {
            continue;
        }

        for (const candidate of [gapStart, gapEnd]) {
            if (candidate + NEMO_AUTO_WINDOW_EPSILON_S < target_s) {
                continue;
            }
            const dist = candidate - target_s;
            if (dist <= NEMO_CURSOR_SNAP_WINDOW_S && dist < bestDist) {
                best = candidate;
                bestDist = dist;
            }
        }
    }

    return best;
}

async function runNemoAutoSentenceWindowing({ audio, sampling_rate, chunk_length_s, tokenizer, runNemoTranscribe }) {
    const audio_duration_s = audio.length / sampling_rate;
    const fallback_overlap_s = Math.min(NEMO_AUTO_WINDOW_FALLBACK_OVERLAP_S, Math.max(0, chunk_length_s - 1));
    const fallback_advance_s = Math.max(1, chunk_length_s - fallback_overlap_s);
    const maxWindows = Math.max(
        4,
        Math.ceil(Math.max(0, audio_duration_s - chunk_length_s) / NEMO_CURSOR_MIN_ADVANCE_S) + 2,
    );

    /** @type {Array<{ words: Array<{ text: string, startTime: number, endTime: number, confidence?: number }>, text: string, timestamp: [number, number] }>} */
    const finalizedSegments = [];
    /** @type {Array<{ text: string, startTime: number, endTime: number, confidence?: number }>} */
    let pendingWords = [];
    let lastTextFallback = '';
    let start_s = 0;
    let shouldMergePending = false;

    for (
        let windowIndex = 0;
        windowIndex < maxWindows && start_s < audio_duration_s - NEMO_AUTO_WINDOW_EPSILON_S;
        ++windowIndex
    ) {
        const end_s = Math.min(audio_duration_s, start_s + chunk_length_s);
        const start_sample = Math.max(0, Math.min(audio.length - 1, Math.floor(start_s * sampling_rate)));
        const end_sample = Math.max(start_sample + 1, Math.min(audio.length, Math.ceil(end_s * sampling_rate)));
        const windowAudio = audio.subarray(start_sample, end_sample);
        const is_last_window = end_s >= audio_duration_s - NEMO_AUTO_WINDOW_EPSILON_S;

        const output = await runNemoTranscribe(windowAudio, {
            tokenizer,
            returnTimestamps: true,
            returnWords: true,
            returnMetrics: false,
            timeOffset: start_s,
        });
        lastTextFallback = output.text ?? lastTextFallback;

        const currentWords = Array.isArray(output.words) ? output.words : [];
        const windowWords = shouldMergePending
            ? mergePendingAndCurrentNemoWords(pendingWords, currentWords)
            : dedupeMergedWords(currentWords);
        const segments = partitionNemoWordsIntoSegments(windowWords);

        if (is_last_window) {
            for (const segment of segments) {
                appendFinalizedNemoSegment(finalizedSegments, segment);
            }
            pendingWords = [];
            break;
        }

        if (segments.length > 1) {
            const pendingSegment = segments[segments.length - 1];
            const pendingStart_s = pendingSegment.timestamp[0];
            if (pendingStart_s >= start_s + NEMO_CURSOR_MIN_ADVANCE_S - NEMO_AUTO_WINDOW_EPSILON_S) {
                const readySegments = segments.slice(0, -1);
                for (const segment of readySegments) {
                    appendFinalizedNemoSegment(finalizedSegments, segment);
                }

                pendingWords = dedupeMergedWords(pendingSegment.words);
                const next_start_s = Math.min(
                    audio_duration_s,
                    relocateNemoCursorToNearbyGap(pendingStart_s, windowWords),
                );
                shouldMergePending = next_start_s > pendingStart_s + NEMO_AUTO_WINDOW_EPSILON_S;
                if (next_start_s > start_s + NEMO_AUTO_WINDOW_EPSILON_S) {
                    start_s = next_start_s;
                    continue;
                }
            }
        }

        pendingWords = windowWords;
        shouldMergePending = true;

        const fallback_start_s = Math.min(audio_duration_s, start_s + fallback_advance_s);
        if (fallback_start_s <= start_s + NEMO_AUTO_WINDOW_EPSILON_S) {
            break;
        }
        start_s = fallback_start_s;
    }

    const words = dedupeMergedWords([...flattenNemoSegmentWords(finalizedSegments), ...pendingWords]);
    const text = words.length > 0 ? joinTimedWords(words) : String(lastTextFallback ?? '').trim();
    const utteranceTimestamp =
        words.length > 0
            ? /** @type {[number, number]} */ ([words[0].startTime, words[words.length - 1].endTime])
            : null;

    return {
        text,
        words,
        utteranceTimestamp,
        chunks: buildNemoSegmentChunks(words, utteranceTimestamp, text),
    };
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
export async function runNemoConformerTDTPipeline({ model, processor, tokenizer, audio, kwargs, prepareAudios }) {
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
    const requested_chunk_length_s = normalizeNemoChunkLengthS(kwargs.chunk_length_s ?? 0);

    const single = !Array.isArray(audio);
    const batchedAudio = single ? [audio] : audio;
    const sampling_rate = processor.feature_extractor.config.sampling_rate;
    const preparedAudios = await prepareAudios(batchedAudio, sampling_rate);
    for (let i = 0; i < preparedAudios.length; ++i) {
        validateNemoAudio(preparedAudios[i], i);
    }

    const runNemoTranscribe = async (windowAudio, decodeOptions) => {
        const inputs = await processor(windowAudio);
        const cacheOwnsTensors = inputs?.[NEMO_FEATURE_OUTPUT_OWNERSHIP] === true;
        try {
            return await model.transcribe(inputs, decodeOptions);
        } finally {
            if (cacheOwnsTensors) {
                releaseNemoPipelineInputs(inputs);
            } else {
                disposeNemoPipelineInputs(inputs);
            }
        }
    };

    const toReturn = [];
    for (const aud of preparedAudios) {
        const audio_duration_s = aud.length / sampling_rate;
        const autoWindowing = requested_chunk_length_s <= 0 && audio_duration_s > NEMO_AUTO_WINDOW_THRESHOLD_S;
        const chunk_length_s =
            requested_chunk_length_s > 0 ? requested_chunk_length_s : autoWindowing ? NEMO_AUTO_CHUNK_LENGTH_S : 0;
        const useSentenceWindowing = chunk_length_s > 0;

        if (useSentenceWindowing) {
            const merged = await runNemoAutoSentenceWindowing({
                audio: aud,
                sampling_rate,
                chunk_length_s,
                tokenizer,
                runNemoTranscribe,
            });
            const result = { text: merged.text };
            if (wantWordTimestamps) {
                result.chunks = buildWordChunks(merged.words);
            } else if (wantTimestampChunks) {
                result.chunks = merged.chunks;
            }
            toReturn.push(result);
            continue;
        }

        const output = await runNemoTranscribe(aud, {
            tokenizer,
            returnTimestamps: wantTimestampChunks,
            returnWords: wantTimestampChunks,
            returnMetrics: false,
        });

        const result = { text: output.text ?? '' };
        if (wantWordTimestamps) {
            result.chunks = buildWordChunks(output.words ?? []);
        } else if (wantTimestampChunks) {
            result.chunks = buildNemoSegmentChunks(output.words ?? [], output.utteranceTimestamp ?? null, result.text);
        }
        toReturn.push(result);
    }

    return single ? toReturn[0] : toReturn;
}
