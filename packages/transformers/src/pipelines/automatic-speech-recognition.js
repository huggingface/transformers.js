import { Pipeline, prepareAudios } from './_base.js';

import { Tensor } from '../utils/tensor.js';
import { max, round } from '../utils/maths.js';
import { logger } from '../utils/logger.js';

/**
 * @typedef {import('./_base.js').TextAudioPipelineConstructorArgs} TextAudioPipelineConstructorArgs
 * @typedef {import('./_base.js').Disposable} Disposable
 * @typedef {import('./_base.js').AudioInput} AudioInput
 */

/**
 * @typedef {Object} Chunk
 * @property {[number, number]} timestamp The start and end timestamp of the chunk in seconds.
 * @property {string} text The recognized text.
 */

/**
 * @typedef {'utterance' | 'word' | 'token' | 'all'} TimestampGranularity
 */

/**
 * @typedef {Object} AutomaticSpeechRecognitionOutput
 * @property {string} text The recognized text.
 * @property {Chunk[]} [chunks] When using `return_timestamps`, the `chunks` will become a list
 * containing all the various text chunks identified by the model.
 * @property {Chunk[]} [tokens] Optional token-level timestamp chunks for models that support them.
 * @property {[number, number]} [utterance] Optional utterance-level timestamp span.
 *
 * @typedef {Object} AutomaticSpeechRecognitionSpecificParams Parameters specific to automatic-speech-recognition pipelines.
 * @property {boolean|'word'} [return_timestamps] Whether to return timestamps or not. Default is `false`.
 * @property {TimestampGranularity} [timestamp_granularity] Granularity used when `return_timestamps` is enabled for Parakeet TDT models. Default is `'word'`.
 * @property {number} [chunk_length_s] The length of audio chunks to process in seconds. Default is 0 (no chunking).
 * @property {number} [stride_length_s] The length of overlap between consecutive audio chunks in seconds. If not provided, defaults to `chunk_length_s / 6`.
 * @property {boolean} [force_full_sequences] Whether to force outputting full sequences or not. Default is `false`.
 * @property {string} [language] The source language. Default is `null`, meaning it should be auto-detected. Use this to potentially improve performance if the source language is known.
 * @property {string} [task] The task to perform. Default is `null`, meaning it should be auto-detected.
 * @property {number} [num_frames] The number of frames in the input audio.
 * @typedef {import('../generation/parameters.js').GenerationFunctionParameters & AutomaticSpeechRecognitionSpecificParams} AutomaticSpeechRecognitionConfig
 *
 * @callback AutomaticSpeechRecognitionPipelineCallbackSingle Transcribe the audio sequence given as inputs to text.
 * @param {AudioInput} audio The input audio file(s) to be transcribed. The input is either:
 * - `string` or `URL` that is the filename/URL of the audio file, the file will be read at the processor's sampling rate
 * to get the waveform using the [`AudioContext`](https://developer.mozilla.org/en-US/docs/Web/API/AudioContext) API.
 * If `AudioContext` is not available, you should pass the raw waveform in as a Float32Array of shape `(n, )`.
 * - `Float32Array` or `Float64Array` of shape `(n, )`, representing the raw audio at the correct sampling rate (no further check will be done).
 * @param {Partial<AutomaticSpeechRecognitionConfig>} [options] Additional keyword arguments to pass along to the generate method of the model.
 * @returns {Promise<AutomaticSpeechRecognitionOutput>} An object containing the transcription text and optionally timestamps if `return_timestamps` is `true`.
 *
 * @callback AutomaticSpeechRecognitionPipelineCallbackBatch Transcribe the audio sequences given as inputs to text.
 * @param {AudioInput[]} audio The input audio file(s) to be transcribed. Each entry is either:
 * - `string` or `URL` that is the filename/URL of the audio file, the file will be read at the processor's sampling rate
 * to get the waveform using the [`AudioContext`](https://developer.mozilla.org/en-US/docs/Web/API/AudioContext) API.
 * If `AudioContext` is not available, you should pass the raw waveform in as a Float32Array of shape `(n, )`.
 * - `Float32Array` or `Float64Array` of shape `(n, )`, representing the raw audio at the correct sampling rate (no further check will be done).
 * @param {Partial<AutomaticSpeechRecognitionConfig>} [options] Additional keyword arguments to pass along to the generate method of the model.
 * @returns {Promise<AutomaticSpeechRecognitionOutput[]>} An object containing the transcription text and optionally timestamps if `return_timestamps` is `true`.
 *
 * @typedef {AutomaticSpeechRecognitionPipelineCallbackSingle & AutomaticSpeechRecognitionPipelineCallbackBatch} AutomaticSpeechRecognitionPipelineCallback
 *
 * @typedef {TextAudioPipelineConstructorArgs & AutomaticSpeechRecognitionPipelineCallback & Disposable} AutomaticSpeechRecognitionPipelineType
 */

/**
 * Pipeline that aims at extracting spoken text contained within some audio.
 *
 * **Example:** Transcribe English.
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny.en');
 * const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav';
 * const output = await transcriber(url);
 * // { text: " And so my fellow Americans ask not what your country can do for you, ask what you can do for your country." }
 * ```
 *
 * **Example:** Transcribe English w/ timestamps.
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny.en');
 * const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav';
 * const output = await transcriber(url, { return_timestamps: true });
 * // {
 * //   text: " And so my fellow Americans ask not what your country can do for you, ask what you can do for your country."
 * //   chunks: [
 * //     { timestamp: [0, 8],  text: " And so my fellow Americans ask not what your country can do for you" }
 * //     { timestamp: [8, 11], text: " ask what you can do for your country." }
 * //   ]
 * // }
 * ```
 *
 * **Example:** Transcribe English w/ word-level timestamps.
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny.en');
 * const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav';
 * const output = await transcriber(url, { return_timestamps: 'word' });
 * // {
 * //   "text": " And so my fellow Americans ask not what your country can do for you ask what you can do for your country.",
 * //   "chunks": [
 * //     { "text": " And", "timestamp": [0, 0.78] },
 * //     { "text": " so", "timestamp": [0.78, 1.06] },
 * //     { "text": " my", "timestamp": [1.06, 1.46] },
 * //     ...
 * //     { "text": " for", "timestamp": [9.72, 9.92] },
 * //     { "text": " your", "timestamp": [9.92, 10.22] },
 * //     { "text": " country.", "timestamp": [10.22, 13.5] }
 * //   ]
 * // }
 * ```
 *
 * **Example:** Transcribe French.
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-small');
 * const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/french-audio.mp3';
 * const output = await transcriber(url, { language: 'french', task: 'transcribe' });
 * // { text: " J'adore, j'aime, je n'aime pas, je déteste." }
 * ```
 *
 * **Example:** Translate French to English.
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-small');
 * const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/french-audio.mp3';
 * const output = await transcriber(url, { language: 'french', task: 'translate' });
 * // { text: " I love, I like, I don't like, I hate." }
 * ```
 *
 * **Example:** Transcribe/translate audio longer than 30 seconds.
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny.en');
 * const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/ted_60.wav';
 * const output = await transcriber(url, { chunk_length_s: 30, stride_length_s: 5 });
 * // { text: " So in college, I was a government major, which means [...] So I'd start off light and I'd bump it up" }
 * ```
 */
export class AutomaticSpeechRecognitionPipeline
    extends /** @type {new (options: TextAudioPipelineConstructorArgs) => AutomaticSpeechRecognitionPipelineType} */ (
        Pipeline
    )
{
    async _call(audio, kwargs = {}) {
        switch (this.model.config.model_type) {
            case 'whisper':
            case 'lite-whisper':
                return this._call_whisper(audio, kwargs);
            case 'wav2vec2':
            case 'wav2vec2-bert':
            case 'unispeech':
            case 'unispeech-sat':
            case 'hubert':
            case 'parakeet_ctc':
                return this._call_wav2vec2(audio, kwargs);
            case 'nemo-conformer-tdt':
                return this._call_nemo_conformer_tdt(audio, kwargs);
            case 'moonshine':
                return this._call_moonshine(audio, kwargs);
            default:
                throw new Error(
                    `AutomaticSpeechRecognitionPipeline does not support model type '${this.model.config.model_type}'.`,
                );
        }
    }

    async _call_wav2vec2(audio, kwargs) {
        // TODO use kwargs

        if (kwargs.language) {
            logger.warn('`language` parameter is not yet supported for `wav2vec2` models, defaulting to "English".');
        }
        if (kwargs.task) {
            logger.warn('`task` parameter is not yet supported for `wav2vec2` models, defaulting to "transcribe".');
        }

        const single = !Array.isArray(audio);
        const batchedAudio = single ? [audio] : audio;

        const sampling_rate = this.processor.feature_extractor.config.sampling_rate;
        const preparedAudios = await prepareAudios(batchedAudio, sampling_rate);

        const toReturn = [];
        for (const aud of preparedAudios) {
            const inputs = await this.processor(aud);
            const output = await this.model(inputs);
            const logits = output.logits[0];

            const predicted_ids = [];
            for (const item of logits) {
                predicted_ids.push(max(item.data)[1]);
            }
            const predicted_sentences = this.tokenizer.decode(predicted_ids, { skip_special_tokens: true }).trim();
            toReturn.push({ text: predicted_sentences });
        }
        return single ? toReturn[0] : toReturn;
    }

    async _call_whisper(audio, kwargs) {
        const return_timestamps = kwargs.return_timestamps ?? false;
        const chunk_length_s = kwargs.chunk_length_s ?? 0;
        const force_full_sequences = kwargs.force_full_sequences ?? false;
        let stride_length_s = kwargs.stride_length_s ?? null;

        const generation_config = { ...kwargs };

        if (return_timestamps === 'word') {
            generation_config['return_token_timestamps'] = true;
            generation_config['return_timestamps'] = false; // Do not predict timestamp tokens
        }

        const single = !Array.isArray(audio);
        const batchedAudio = single ? [audio] : audio;
        const feature_extractor_config = this.processor.feature_extractor.config;

        // @ts-expect-error TS2339
        const time_precision = feature_extractor_config.chunk_length / this.model.config.max_source_positions;
        const hop_length = feature_extractor_config.hop_length;

        const sampling_rate = feature_extractor_config.sampling_rate;
        const preparedAudios = await prepareAudios(batchedAudio, sampling_rate);

        const toReturn = [];
        for (const aud of preparedAudios) {
            /** @type {{stride: number[], input_features: Tensor, is_last: boolean, tokens?: bigint[], token_timestamps?: number[]}[]} */
            let chunks = [];
            if (chunk_length_s > 0) {
                if (stride_length_s === null) {
                    stride_length_s = chunk_length_s / 6;
                } else if (chunk_length_s <= stride_length_s) {
                    throw Error('`chunk_length_s` must be larger than `stride_length_s`.');
                }

                // TODO support different stride_length_s (for left and right)

                const window = sampling_rate * chunk_length_s;
                const stride = sampling_rate * stride_length_s;
                const jump = window - 2 * stride;
                let offset = 0;

                // Create subarrays of audio with overlaps
                while (true) {
                    const offset_end = offset + window;
                    const subarr = aud.subarray(offset, offset_end);
                    const feature = await this.processor(subarr);

                    const is_first = offset === 0;
                    const is_last = offset_end >= aud.length;
                    chunks.push({
                        stride: [subarr.length, is_first ? 0 : stride, is_last ? 0 : stride],
                        input_features: feature.input_features,
                        is_last,
                    });
                    if (is_last) break;
                    offset += jump;
                }
            } else {
                chunks = [
                    {
                        stride: [aud.length, 0, 0],
                        input_features: (await this.processor(aud)).input_features,
                        is_last: true,
                    },
                ];
            }

            // Generate for each set of input features
            for (const chunk of chunks) {
                generation_config.num_frames = Math.floor(chunk.stride[0] / hop_length);

                // NOTE: doing sequentially for now
                const data = await this.model.generate({
                    inputs: chunk.input_features,
                    ...generation_config,
                });

                // TODO: Right now we only get top beam
                if (return_timestamps === 'word') {
                    // @ts-expect-error TS2339
                    chunk.tokens = data.sequences.tolist()[0];
                    // @ts-expect-error TS2339
                    chunk.token_timestamps = data.token_timestamps
                        .tolist()[0]
                        .map((/** @type {number} */ x) => round(x, 2));
                } else {
                    chunk.tokens = /** @type {Tensor} */ (data)[0].tolist();
                }

                // convert stride to seconds
                chunk.stride = chunk.stride.map((x) => x / sampling_rate);
            }

            // Merge text chunks
            // @ts-ignore
            const [full_text, optional] = this.tokenizer._decode_asr(chunks, {
                time_precision,
                return_timestamps,
                force_full_sequences,
            });

            toReturn.push({ text: full_text, ...optional });
        }
        return single ? toReturn[0] : toReturn;
    }

    /**
     * @param {any} return_timestamps
     * @param {any} timestamp_granularity
     * @returns {TimestampGranularity|null}
     */
    _normalizeNemoConformerTimestampGranularity(return_timestamps, timestamp_granularity) {
        if (!return_timestamps) {
            return null;
        }

        const granularity = timestamp_granularity ?? 'word';
        const allowed = ['utterance', 'word', 'token', 'all'];
        if (!allowed.includes(granularity)) {
            throw new Error(
                `Invalid \`timestamp_granularity\`: "${granularity}". Expected one of: ${allowed.join(', ')}.`,
            );
        }
        return /** @type {TimestampGranularity} */ (granularity);
    }

    /**
     * @param {any} result
     * @param {TimestampGranularity|null} granularity
     * @returns {AutomaticSpeechRecognitionOutput}
     */
    _formatNemoConformerTDTResult(result, granularity) {
        const text = result.text ?? '';
        if (!granularity) {
            return { text };
        }

        const wordChunks = (result.word_timestamps ?? []).map((item) => ({
            text: item.text,
            timestamp: item.timestamp,
        }));
        const tokenChunks = (result.token_timestamps ?? []).map((timestamp, index) => {
            const tokenId = result.token_ids?.[index];
            const decodedToken =
                tokenId == null
                    ? ''
                    : (this.tokenizer?.decode([tokenId], {
                          skip_special_tokens: true,
                          clean_up_tokenization_spaces: false,
                      }) ?? '');
            return {
                text: decodedToken || (tokenId == null ? '' : `${tokenId}`),
                timestamp,
            };
        });
        const utterance = result.utterance_timestamp;

        if (granularity === 'utterance') {
            if (!utterance) {
                return { text, chunks: [] };
            }
            return {
                text,
                chunks: [{ text, timestamp: utterance }],
            };
        }

        if (granularity === 'word') {
            return { text, chunks: wordChunks };
        }

        if (granularity === 'token') {
            return { text, chunks: tokenChunks };
        }

        return {
            text,
            chunks: wordChunks,
            tokens: tokenChunks,
            ...(utterance ? { utterance } : {}),
        };
    }

    /**
     * Nemo Conformer TDT ASR output rules:
     * - `return_timestamps=false`: `{ text }`
     * - `timestamp_granularity='utterance'`: `chunks` contains a single utterance span
     * - `timestamp_granularity='word'`: `chunks` contains word-level spans
     * - `timestamp_granularity='token'`: `chunks` contains token-level spans
     * - `timestamp_granularity='all'`: returns `chunks` (word), `tokens`, and `utterance`
     */
    async _call_nemo_conformer_tdt(audio, kwargs) {
        if (typeof /** @type {any} */ (this.model).transcribe !== 'function') {
            throw new Error('Nemo Conformer TDT model does not expose a `transcribe` method.');
        }
        if (!this.processor) {
            throw new Error('Nemo Conformer TDT pipeline requires a processor.');
        }
        if (!this.tokenizer) {
            throw new Error('Nemo Conformer TDT pipeline requires a tokenizer.');
        }
        if (!this.processor.feature_extractor?.config?.sampling_rate) {
            throw new Error(
                'Nemo Conformer TDT pipeline requires `processor.feature_extractor.config.sampling_rate` to prepare audio.',
            );
        }

        const return_timestamps = kwargs.return_timestamps ?? false;
        const withTimestamps = return_timestamps !== false;
        const granularity = this._normalizeNemoConformerTimestampGranularity(
            withTimestamps,
            kwargs.timestamp_granularity,
        );

        const decodeOptions = {
            tokenizer: this.tokenizer,
            return_token_timestamps: granularity === 'token' || granularity === 'all',
            return_word_timestamps: granularity === 'word' || granularity === 'all',
            return_utterance_timestamp: granularity === 'utterance' || granularity === 'all',
        };

        const single = !Array.isArray(audio);
        const batchedAudio = single ? [audio] : audio;
        const sampling_rate = this.processor.feature_extractor.config.sampling_rate;
        const preparedAudios = await prepareAudios(batchedAudio, sampling_rate);

        const toReturn = [];
        for (const aud of preparedAudios) {
            const inputs = await this.processor(aud);
            const output = await /** @type {any} */ (this.model).transcribe(inputs, decodeOptions);
            toReturn.push(this._formatNemoConformerTDTResult(output, granularity));
        }

        return single ? toReturn[0] : toReturn;
    }

    async _call_moonshine(audio, kwargs) {
        const single = !Array.isArray(audio);
        const batchedAudio = single ? [audio] : audio;
        const sampling_rate = this.processor.feature_extractor.config.sampling_rate;
        const preparedAudios = await prepareAudios(batchedAudio, sampling_rate);
        const toReturn = [];
        for (const aud of preparedAudios) {
            const inputs = await this.processor(aud);

            // According to the [paper](https://huggingface.co/papers/2410.15608):
            // "We use greedy decoding, with a heuristic limit of 6 output tokens
            // per second of audio to avoid repeated output sequences."
            const max_new_tokens = Math.floor(aud.length / sampling_rate) * 6;
            const outputs = await this.model.generate({ max_new_tokens, ...kwargs, ...inputs });

            const text = this.processor.batch_decode(/** @type {Tensor} */ (outputs), { skip_special_tokens: true })[0];
            toReturn.push({ text });
        }
        return single ? toReturn[0] : toReturn;
    }
}
