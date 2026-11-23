import { Tensor, std_mean } from '../../utils/tensor.js';
import { PreTrainedModel } from '../modeling_utils.js';
import { WhisperGenerationConfig } from './generation_whisper.js';
import { whisper_language_to_code } from './common_whisper.js';
import { prepareTensorForDecode } from '../../tokenization_utils.js';
import {
    LogitsProcessorList,
    SuppressTokensAtBeginLogitsProcessor,
    WhisperTimeStampLogitsProcessor,
} from '../../generation/logits_process.js';
import { medianFilter, dynamic_time_warping } from '../../utils/maths.js';
import { ModelOutput } from '../modeling_outputs.js';
import { logger } from '../../utils/logger.js';

export class WhisperPreTrainedModel extends PreTrainedModel {
    requires_attention_mask = false;
    main_input_name = 'input_features';
    forward_params = [
        'input_features',
        'attention_mask',
        'decoder_input_ids',
        'decoder_attention_mask',
        'past_key_values',
    ];
}

/**
 * WhisperModel class for training Whisper models without a language model head.
 */
export class WhisperModel extends WhisperPreTrainedModel {}

/**
 * WhisperForConditionalGeneration class for generating conditional outputs from Whisper models.
 */
export class WhisperForConditionalGeneration extends WhisperPreTrainedModel {
    _prepare_generation_config(generation_config, kwargs) {
        return /** @type {WhisperGenerationConfig} */ (
            super._prepare_generation_config(generation_config, kwargs, WhisperGenerationConfig)
        );
    }

    /**
     *
     * @param {WhisperGenerationConfig} generation_config
     */
    _retrieve_init_tokens(generation_config) {
        // prefix tokens are of the form:
        //  - Multilingual: <|startoftranscript|> <|lang_id|> <|task|> [<|notimestamps|>]
        //  - English-only: <|startoftranscript|> [<|notimestamps|>]

        // 1. Handle <|startoftranscript|> token
        const init_tokens = [generation_config.decoder_start_token_id];

        // 2. Handle <|lang_id|> and <|task> tokens
        let language = generation_config.language;
        const task = generation_config.task;
        if (generation_config.is_multilingual) {
            if (!language) {
                // TODO: Implement language detection
                logger.warn('No language specified - defaulting to English (en).');
                language = 'en';
            }

            // Add language token
            const language_code = whisper_language_to_code(language);
            const language_token = `<|${language_code}|>`;
            init_tokens.push(generation_config.lang_to_id[language_token]);

            // Add task token
            // NOTE: Defaults to 'transcribe' if no task is specified
            init_tokens.push(generation_config.task_to_id[task ?? 'transcribe']);
        } else if (language || task) {
            throw new Error(
                'Cannot specify `task` or `language` for an English-only model. If the model is intended to be multilingual, pass `is_multilingual=true` to generate, or update the generation config.',
            );
        }

        // 3. Handle <|notimestamps|> token
        if (
            !generation_config.return_timestamps &&
            generation_config.no_timestamps_token_id &&
            init_tokens.at(-1) !== generation_config.no_timestamps_token_id
        ) {
            init_tokens.push(generation_config.no_timestamps_token_id);
        } else if (
            generation_config.return_timestamps &&
            init_tokens.at(-1) === generation_config.no_timestamps_token_id
        ) {
            logger.warn(
                '<|notimestamps|> prompt token is removed from generation_config since `return_timestamps` is set to `true`.',
            );
            init_tokens.pop();
        }

        // let's make sure we don't pass `null` tokens as prompt tokens
        return init_tokens.filter((token) => token != null);
    }

    /**
     * Transcribes or translates log-mel input features to a sequence of auto-regressively generated token ids.
     * @param {import('./generation_whisper.js').WhisperGenerationFunctionParameters} options
     * @returns {Promise<ModelOutput|Tensor>} The output of the model, which can contain the generated token ids, attentions, and scores.
     */
    async generate({
        inputs = null,
        generation_config = null,
        logits_processor = null,
        stopping_criteria = null,

        // Whisper-specific options (passed to kwargs)
        // prompt_ids = null,
        // language = null,
        // task = null,

        ...kwargs
    }) {
        generation_config = this._prepare_generation_config(generation_config, kwargs);
        const public_attentions = generation_config.output_attentions;

        const init_tokens =
            kwargs.decoder_input_ids instanceof Tensor
                ? prepareTensorForDecode(kwargs.decoder_input_ids)
                : (kwargs.decoder_input_ids ?? this._retrieve_init_tokens(generation_config));

        const prefix = /** @type {number[]} */ (Array.isArray(init_tokens[0]) ? init_tokens[0] : init_tokens);
        const prefix_length = prefix.length;
        const batch_size = /** @type {Tensor} */ (inputs ?? kwargs.input_features)?.dims[0] ?? 1;
        const decoder_input_ids = Array.isArray(init_tokens[0])
            ? /** @type {number[][]} */ (init_tokens)
            : Array.from({ length: batch_size }, () => [...prefix]);

        if (generation_config.return_timestamps) {
            logits_processor ??= new LogitsProcessorList();
            logits_processor.push(new WhisperTimeStampLogitsProcessor(generation_config, prefix));
        }

        if (generation_config.begin_suppress_tokens) {
            logits_processor ??= new LogitsProcessorList();
            logits_processor.push(
                new SuppressTokensAtBeginLogitsProcessor(generation_config.begin_suppress_tokens, prefix_length),
            );
        }

        if (generation_config.return_token_timestamps) {
            if (!generation_config.alignment_heads) {
                throw new Error(
                    'Model generation config has no `alignment_heads`, token-level timestamps not available. ' +
                        'See https://gist.github.com/hollance/42e32852f24243b748ae6bc1f985b13a on how to add this property to the generation config.',
                );
            }

            if (generation_config.task === 'translate') {
                logger.warn("Token-level timestamps may not be reliable for task 'translate'.");
            }

            kwargs.output_attentions = true;
            kwargs.return_dict_in_generate = true;
            generation_config.output_attentions = true;
            generation_config.return_dict_in_generate = true;
        }

        // For timestamp mode, use seek-based sequential generation.
        // This matches Python's WhisperForConditionalGeneration.generate() which uses a seek loop
        // to handle audio that the model doesn't fully transcribe in a single pass.
        // Skip the seek loop when max_new_tokens is explicitly set (e.g., for prefix token tests).
        if (generation_config.return_timestamps && !generation_config.max_new_tokens) {
            if (
                /** @type {Tensor} */ (inputs ?? kwargs.input_features)?.dims[0] !== 1 ||
                generation_config.num_return_sequences !== 1
            ) {
                throw new Error(
                    'Whisper seek generation supports only a single input and a single returned sequence. Set max_new_tokens for short-form batch/multiple returns.',
                );
            }
            if (public_attentions)
                throw new Error(
                    'Whisper seek generation does not expose public per-step attentions; use short-form generation.',
                );
            return this._generate_with_seek({
                inputs: inputs ?? kwargs.input_features,
                generation_config,
                logits_processor,
                init_tokens: prefix,
                kwargs,
            });
        }

        const outputs = await super.generate({
            inputs,
            generation_config,
            logits_processor,
            ...kwargs,
            decoder_input_ids,
        });

        if (generation_config.return_token_timestamps) {
            let aligned = false;
            outputs['num_beams'] = generation_config.num_beams;
            try {
                for (const step of outputs['cross_attentions'] ?? [])
                    for (const tensor of step) await tensor.ort_tensor.getData(false);
                outputs['token_timestamps'] = this._extract_token_timestamps(
                    // @ts-expect-error TS2345
                    outputs,
                    generation_config.alignment_heads,
                    Array.isArray(generation_config.num_frames)
                        ? generation_config.num_frames.map((x) => Math.floor(x / 2))
                        : generation_config.num_frames == null
                          ? null
                          : Math.floor(generation_config.num_frames / 2),
                    0.02,
                    prefix_length,
                );
                aligned = true;
            } finally {
                if (!public_attentions || !aligned) {
                    for (const key of ['cross_attentions', 'decoder_attentions', 'encoder_attentions']) {
                        for (const step of outputs[key] ?? []) for (const tensor of step) tensor.dispose();
                        delete outputs[key];
                    }
                }
                if (!aligned) {
                    await outputs['past_key_values']?.dispose();
                    for (const value of Object.values(outputs)) if (value instanceof Tensor) value.dispose();
                }
            }
        }

        return outputs;
    }

    /**
     * Generates with a seek loop for timestamp mode, re-encoding and generating
     * for each segment until all audio frames are consumed.
     * This matches Python's WhisperForConditionalGeneration.generate() behavior.
     * @private
     */
    async _generate_with_seek({ inputs, generation_config, logits_processor, init_tokens, kwargs }) {
        const timestamp_begin = generation_config.no_timestamps_token_id + 1;
        const eos_token_id = Array.isArray(generation_config.eos_token_id)
            ? generation_config.eos_token_id[0]
            : generation_config.eos_token_id;
        const return_token_timestamps = generation_config.return_token_timestamps;

        // input_features shape: [batch=1, n_mels, total_frames]
        const input_features = inputs ?? kwargs.input_features;
        const total_frames = Math.min(
            input_features.dims[2],
            (Array.isArray(generation_config.num_frames)
                ? generation_config.num_frames[0]
                : generation_config.num_frames) ?? Infinity,
        );

        // The encoder downsamples by input_stride (=2 for whisper), so:
        //   num_segment_frames = input_stride * max_source_positions = 3000 mel frames per segment
        // Timestamp token T maps to mel frame position T * input_stride
        const input_stride = 2;
        // @ts-expect-error ts(2339)
        const max_source_positions = /** @type {number} */ (this.config.max_source_positions);
        const num_segment_frames = input_stride * max_source_positions;

        let seek = 0;
        const allTokens = [];
        const allTokenTimestamps = [];

        while (seek < total_frames) {
            // Slice input features for this segment
            const seek_end = Math.min(seek + num_segment_frames, total_frames);
            // Keep the feature extractor's normalized silence padding. Valid audio
            // frames bound seek/alignment, but replacing existing padded features
            // with zeros changes the encoder input (zero is not log-mel silence).
            const segment_input = input_features.slice(null, null, [
                seek,
                Math.min(seek + num_segment_frames, input_features.dims[2]),
            ]);

            // Pad to full segment size if needed (whisper expects fixed-length input)
            let segment_features;
            const segment_frames = segment_input.dims[2];
            if (segment_frames < num_segment_frames) {
                const n_mels = input_features.dims[1];
                const padded_data = new Float32Array(n_mels * num_segment_frames);
                const numeric_segment = segment_input.to('float32');
                const src = /** @type {Float32Array} */ (numeric_segment.data);
                // Copy each mel band row separately to handle the stride difference
                for (let m = 0; m < n_mels; ++m) {
                    padded_data.set(src.subarray(m * segment_frames, (m + 1) * segment_frames), m * num_segment_frames);
                }
                if (numeric_segment !== segment_input) numeric_segment.dispose();
                segment_features = new Tensor('float32', padded_data, [1, n_mels, num_segment_frames]);
            } else {
                segment_features = segment_input;
            }

            // Reset logits processor begin_index for each segment
            if (logits_processor) {
                for (const proc of logits_processor) {
                    if ('begin_index' in proc) {
                        proc.begin_index = init_tokens.length;
                    }
                }
            }

            /** @type {any} */
            let outputs;
            try {
                outputs = await super.generate({
                    generation_config,
                    logits_processor,
                    ...kwargs,
                    decoder_input_ids: init_tokens,
                    input_features: segment_features,
                });
            } finally {
                segment_features.dispose();
                if (segment_features !== segment_input) segment_input.dispose();
            }

            try {
                // Extract tokens (skip init_tokens prefix)
                const raw_sequence = outputs.sequences ?? outputs;
                const generated_tokens = raw_sequence[0].tolist().map(Number).slice(init_tokens.length);

                // Remove trailing EOS
                if (generated_tokens.length > 0 && generated_tokens.at(-1) === eos_token_id) {
                    generated_tokens.pop();
                }

                if (generated_tokens.length === 0) {
                    // No tokens generated — skip the rest of the audio
                    break;
                }

                // Determine seek advancement using the same logic as Python's _retrieve_segment:
                // 1. Find consecutive timestamp token pairs (segment boundaries)
                // 2. If the sequence ends with a single timestamp (no speech after it),
                //    consume all remaining frames in this segment
                // 3. Otherwise, seek to the last complete segment boundary
                const is_timestamp = generated_tokens.map((t) => t >= timestamp_begin);

                // Check for single_timestamp_ending: last token is timestamp, second-to-last is not
                const single_timestamp_ending =
                    generated_tokens.length >= 2 &&
                    is_timestamp[generated_tokens.length - 1] &&
                    !is_timestamp[generated_tokens.length - 2];

                // Find consecutive timestamp pairs (segment boundaries)
                const segment_boundary_indices = [];
                for (let i = 0; i < generated_tokens.length - 1; ++i) {
                    if (is_timestamp[i] && is_timestamp[i + 1]) {
                        segment_boundary_indices.push(i + 1); // index of the second token in the pair
                    }
                }

                let segment_offset;
                let tokens_to_keep = generated_tokens.length;
                if (segment_boundary_indices.length > 0) {
                    if (single_timestamp_ending) {
                        // Ends with a single timestamp after the last pair — no more speech
                        segment_offset = seek_end - seek;
                    } else {
                        // Ends mid-segment — seek to the last pair's end timestamp
                        // Discard tokens after the last pair (they're from an incomplete segment)
                        // Keep up to the first token of the last pair (the end-of-segment timestamp),
                        // excluding the second token (the start-of-next-segment marker)
                        const last_boundary = segment_boundary_indices.at(-1);
                        const last_ts_pos = generated_tokens[last_boundary - 1] - timestamp_begin;
                        segment_offset = last_ts_pos * input_stride;
                        tokens_to_keep = last_boundary;
                    }
                } else {
                    // No consecutive pairs found — consume entire segment
                    segment_offset = seek_end - seek;
                }

                // Align only the retained segment, before normalization and DTW.
                // Every retained token uses the query that predicted it; do not
                // include the next segment token in normalization.
                let seek_token_timestamps;
                if (return_token_timestamps) {
                    outputs.num_beams = generation_config.num_beams;
                    const valid_length = Math.min(
                        Number(outputs.sequence_lengths?.data[0] ?? raw_sequence.dims[1]),
                        init_tokens.length + tokens_to_keep,
                    );
                    outputs.sequence_lengths?.dispose();
                    outputs.sequence_lengths = new Tensor('int64', [BigInt(valid_length)], [1]);
                    for (const step of outputs.cross_attentions ?? [])
                        for (const tensor of step) await tensor.ort_tensor.getData(false);
                    outputs.token_timestamps = this._extract_token_timestamps(
                        outputs,
                        generation_config.alignment_heads,
                        Math.floor(segment_offset / input_stride),
                        0.02,
                        init_tokens.length,
                    );
                    const time_offset = (seek / input_stride) * 0.02;
                    seek_token_timestamps = outputs.token_timestamps[0]
                        .tolist()
                        .slice(init_tokens.length)
                        .map((t) => t + time_offset);
                }

                // Offset timestamp tokens by the current seek position so they're
                // monotonically increasing across segments. Cap at the maximum valid
                // timestamp token (30.00s = 1500 positions) to stay within the token vocab.
                const timestamp_offset = Math.floor(seek / input_stride);
                const max_timestamp_token = timestamp_begin + 1500;
                for (let i = 0; i < tokens_to_keep; ++i) {
                    if (generated_tokens[i] >= timestamp_begin) {
                        generated_tokens[i] = Math.min(generated_tokens[i] + timestamp_offset, max_timestamp_token);
                    }
                }

                allTokens.push(...generated_tokens.slice(0, tokens_to_keep));
                if (seek_token_timestamps) {
                    allTokenTimestamps.push(...seek_token_timestamps.slice(0, tokens_to_keep));
                }
                if (segment_offset <= 0 || segment_offset > seek_end - seek)
                    throw new Error('Whisper returned an invalid seek offset.');
                seek += segment_offset;
            } finally {
                for (const key of ['cross_attentions', 'decoder_attentions', 'encoder_attentions']) {
                    for (const step of outputs[key] ?? []) for (const tensor of step) tensor.dispose();
                }
                await outputs.past_key_values?.dispose();
                if (outputs instanceof Tensor) outputs.dispose();
                else for (const value of Object.values(outputs)) if (value instanceof Tensor) value.dispose();
            }
        }

        // Add EOS back
        allTokens.push(eos_token_id);

        // Reconstruct output
        const full_sequence = [...init_tokens, ...allTokens];
        if (return_token_timestamps) {
            // Return dict format with sequences and token_timestamps
            const sequences = new Tensor('int64', full_sequence.map(BigInt), [1, full_sequence.length]);
            // Pad token_timestamps to match full_sequence (init_tokens get 0.0)
            const full_timestamps = [
                ...new Array(init_tokens.length).fill(0),
                ...allTokenTimestamps,
                allTokenTimestamps.at(-1) ?? 0,
            ];
            const token_timestamps = new Tensor('float32', new Float32Array(full_timestamps), [
                1,
                full_timestamps.length,
            ]);
            return { sequences, token_timestamps };
        }
        return new Tensor('int64', full_sequence.map(BigInt), [1, full_sequence.length]);
    }

    /**
     * Calculates token-level timestamps using the encoder-decoder cross-attentions and
     * dynamic time-warping (DTW) to map each output token to a position in the input audio.
     * If `num_frames` is specified, the encoder-decoder cross-attentions will be cropped before applying DTW.
     * @param {Object} generate_outputs Outputs generated by the model
     * @param {Tensor[][]} generate_outputs.cross_attentions The cross attentions output by the model
     * @param {Tensor} [generate_outputs.beam_indices] Global parent row for each generated token; padded with -1.
     * @param {Tensor} [generate_outputs.sequence_lengths] Valid sequence lengths including the prefix and EOS.
     * @param {number} [generate_outputs.num_beams] Beam count per input.
     * @param {Tensor} generate_outputs.sequences The sequences output by the model
     * @param {number[][]} alignment_heads Alignment heads of the model
     * @param {number|number[]} [num_frames=null] Valid encoder frames (20 ms each), per input or shared.
     * @param {number} [time_precision=0.02] Precision of the timestamps in seconds
     * @param {number} [num_input_ids=0] Number of decoder input ids (prefix tokens) to skip in DTW
     * @returns {Tensor} tensor containing the timestamps in seconds for each predicted token
     */
    _extract_token_timestamps(
        generate_outputs,
        alignment_heads,
        num_frames = null,
        time_precision = 0.02,
        num_input_ids = 0,
    ) {
        if (!generate_outputs.cross_attentions) {
            throw new Error(
                'Model outputs must contain cross attentions to extract timestamps. ' +
                    'This is most likely because the model was not exported with `output_attentions=True`.',
            );
        }
        if (num_frames == null) {
            logger.warn(
                '`num_frames` has not been set, meaning the entire audio will be analyzed. ' +
                    'This may lead to inaccurate token-level timestamps for short audios (< 30 seconds).',
            );
        }

        // @ts-expect-error TS2339
        let median_filter_width = this.config.median_filter_width;
        if (median_filter_width === undefined) {
            logger.warn('Model config has no `median_filter_width`, using default value of 7.');
            median_filter_width = 7;
        }

        if (!alignment_heads.length) throw new Error('Timestamp alignment requires at least one attention head.');
        const steps = generate_outputs.cross_attentions;
        const shape = generate_outputs.sequences.dims;
        const timestamps = new Tensor('float32', new Float32Array(shape[0] * shape[1]), shape);
        const ancestry = generate_outputs.beam_indices?.tolist();
        const lengths = generate_outputs.sequence_lengths?.tolist();
        for (let result = 0; result < shape[0]; ++result) {
            const valid = lengths ? Number(lengths[result]) : shape[1];
            const generated = valid - num_input_ids;
            // Each forward predicts one generated token, including EOS. The last
            // prefill query predicts the first token (the convention in PR #1809).
            const queries = Math.max(0, generated);
            if (!queries) continue;
            const firstRow = ancestry ? Number(ancestry[result][0]) : result;
            const inputIndex = Math.floor(firstRow / (generate_outputs.num_beams ?? 1));
            const requestedFrames = Array.isArray(num_frames) ? num_frames[inputIndex] : num_frames;
            const frames = Math.min(steps[0][alignment_heads[0][0]].dims[3], requestedFrames ?? Infinity);
            if (!Number.isInteger(frames) || frames <= 0)
                throw new Error('Timestamp encoder frame count must be positive.');
            const matrix = Array.from({ length: queries }, () => new Float32Array(frames));
            for (const [layer, head] of alignment_heads) {
                const weights = [];
                for (let step = 0; step < queries; ++step) {
                    const source = steps[step]?.[layer];
                    if (!source || head >= source.dims[1])
                        throw new Error('Missing timestamp alignment attention head or step.');
                    const row = ancestry ? Number(ancestry[result][step]) : result;
                    if (row < 0 || row >= source.dims[0]) throw new Error('Invalid timestamp beam ancestry.');
                    // Preserve original prefill rows for normalization. Uncached later
                    // forwards repeat the prefix: take only their final query.
                    const firstQuery = step === 0 ? 0 : source.dims[2] - 1;
                    for (let query = firstQuery; query < source.dims[2]; ++query) {
                        const offset = ((row * source.dims[1] + head) * source.dims[2] + query) * source.dims[3];
                        const selected = new Tensor(source.type, source.data.slice(offset, offset + frames), [frames]);
                        const numeric = selected.to('float32');
                        weights.push(Float32Array.from(numeric.data));
                        if (numeric !== selected) numeric.dispose();
                        selected.dispose();
                    }
                }
                // Normalize before cropping prompt rows, matching PR #1809's
                // single-beam algorithm while excluding other hypotheses and padding.
                const allWeights = new Tensor('float32', Float32Array.from(weights.flatMap((row) => Array.from(row))), [
                    weights.length,
                    frames,
                ]);
                const [std, average] = std_mean(allWeights, 0, 0);
                for (const row of weights) {
                    for (let frame = 0; frame < frames; ++frame) {
                        row[frame] = std.data[frame] > 0 ? (row[frame] - average.data[frame]) / std.data[frame] : 0;
                    }
                }
                allWeights.dispose();
                std.dispose();
                average.dispose();
                const skipped = weights.length - queries;
                for (let token = 0; token < queries; ++token) {
                    // Python leaves recordings too short for reflection padding unchanged.
                    const filtered =
                        frames <= Math.floor(median_filter_width / 2)
                            ? weights[skipped + token]
                            : medianFilter(weights[skipped + token], median_filter_width);
                    for (let frame = 0; frame < frames; ++frame) matrix[token][frame] += filtered[frame];
                }
            }
            const [text, times] = dynamic_time_warping(
                matrix.map((row) => Array.from(row, (value) => -Math.fround(value / alignment_heads.length))),
            );
            let last = 0;
            for (let i = 0; i < text.length; ++i) {
                if (i === 0 || text[i] !== text[i - 1]) {
                    last = times[i] * time_precision;
                    timestamps.data[result * shape[1] + num_input_ids + text[i]] = last;
                }
            }
        }

        return timestamps;
    }
}

export class LiteWhisperForConditionalGeneration extends WhisperForConditionalGeneration {}
