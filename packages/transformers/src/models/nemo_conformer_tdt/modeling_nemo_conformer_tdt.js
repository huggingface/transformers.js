import { AutoConfig } from '../../configs.js';
import { Tensor } from '../../utils/tensor.js';
import {
    PreTrainedModel,
    MODEL_TYPES,
    MODEL_TYPE_MAPPING,
    MODEL_NAME_TO_CLASS_MAPPING,
    MODEL_CLASS_TO_NAME_MAPPING,
} from '../modeling_utils.js';
import { constructSessions, sessionRun } from '../session.js';
import { buildTransducerDetailedOutputs, decodeTransducerText } from './transducer_text.js';

const NEMO_CONFORMER_TDT_MODEL_TYPE = 'nemo-conformer-tdt';

const DEFAULT_TRANSDUCER_IO = Object.freeze({
    encoder_output: 'outputs',
    decoder_encoder: 'encoder_outputs',
    decoder_token: 'targets',
    decoder_token_length: 'target_length',
    decoder_state_1: 'input_states_1',
    decoder_state_2: 'input_states_2',
    decoder_output: 'outputs',
    decoder_output_state_1: 'output_states_1',
    decoder_output_state_2: 'output_states_2',
});

function argmax(values, offset = 0, length = values.length - offset) {
    let maxIndex = offset;
    let maxValue = Number.NEGATIVE_INFINITY;
    const end = offset + length;
    for (let i = offset; i < end; ++i) {
        const v = values[i];
        if (v > maxValue) {
            maxValue = v;
            maxIndex = i;
        }
    }
    return maxIndex;
}

function toInt(value) {
    return typeof value === 'bigint' ? Number(value) : value;
}

function nowMs() {
    return typeof performance !== 'undefined' && typeof performance.now === 'function' ? performance.now() : Date.now();
}

function roundMetric(value, digits = 2) {
    if (!Number.isFinite(value)) return 0;
    const factor = 10 ** digits;
    return Math.round(value * factor) / factor;
}

function roundTs(value) {
    return Math.round(value * 1000) / 1000;
}

/**
 * @param {import('../../utils/tensor.js').Tensor['data']} logits
 * @param {number} tokenId
 * @param {number} vocabSize
 * @returns {{ confidence: number, logProb: number }}
 */
function confidenceFromLogits(logits, tokenId, vocabSize) {
    let maxLogit = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < vocabSize; ++i) {
        if (logits[i] > maxLogit) {
            maxLogit = logits[i];
        }
    }

    let expSum = 0;
    for (let i = 0; i < vocabSize; ++i) {
        expSum += Math.exp(logits[i] - maxLogit);
    }
    const logSumExp = maxLogit + Math.log(expSum);
    const logProb = logits[tokenId] - logSumExp;
    return {
        confidence: Math.exp(logProb),
        logProb,
    };
}

function resolveTransducerConfig(config, sessions) {
    const transducerConfig = config['transformers.js_config']?.transducer;
    if (!transducerConfig) {
        throw new Error(
            'Missing `transformers.js_config.transducer` in config.json for nemo-conformer-tdt. See external model repo contract.',
        );
    }

    const decoderConfig = transducerConfig.decoder ?? {};
    const numLayers = decoderConfig.num_layers;
    const hiddenSize = decoderConfig.hidden_size;

    if (!Number.isInteger(numLayers) || numLayers <= 0) {
        throw new Error('Invalid `transformers.js_config.transducer.decoder.num_layers`: expected a positive integer.');
    }
    if (!Number.isInteger(hiddenSize) || hiddenSize <= 0) {
        throw new Error(
            'Invalid `transformers.js_config.transducer.decoder.hidden_size`: expected a positive integer.',
        );
    }

    const io = {
        ...DEFAULT_TRANSDUCER_IO,
        ...(transducerConfig.io ?? {}),
    };
    const requiredDecoderInputs = [
        io.decoder_encoder,
        io.decoder_token,
        io.decoder_token_length,
        io.decoder_state_1,
        io.decoder_state_2,
    ];
    if (new Set(requiredDecoderInputs).size !== requiredDecoderInputs.length) {
        throw new Error(
            'Invalid `transformers.js_config.transducer.io`: decoder input names must be distinct ' +
                '(decoder_encoder, decoder_token, decoder_token_length, decoder_state_1, decoder_state_2).',
        );
    }
    const requiredDecoderOutputs = [io.decoder_output, io.decoder_output_state_1, io.decoder_output_state_2];
    if (new Set(requiredDecoderOutputs).size !== requiredDecoderOutputs.length) {
        throw new Error(
            'Invalid `transformers.js_config.transducer.io`: decoder output names must be distinct ' +
                '(decoder_output, decoder_output_state_1, decoder_output_state_2).',
        );
    }

    const decoderSession = sessions?.decoder_model_merged;
    if (!decoderSession) {
        throw new Error('Missing required session `decoder_model_merged` for Nemo Conformer TDT.');
    }

    const decoderInputNames = decoderSession.inputNames ?? [];
    const decoderOutputNames = decoderSession.outputNames ?? [];
    const missingDecoderInputs = [
        io.decoder_encoder,
        io.decoder_token,
        io.decoder_token_length,
        io.decoder_state_1,
        io.decoder_state_2,
    ].filter((name) => !decoderInputNames.includes(name));

    if (missingDecoderInputs.length > 0) {
        throw new Error(
            `Nemo Conformer TDT decoder session is missing expected inputs: ${missingDecoderInputs.join(', ')}. ` +
                'Override I/O names via `transformers.js_config.transducer.io` if your export uses different names.',
        );
    }
    const missingDecoderOutputs = [io.decoder_output, io.decoder_output_state_1, io.decoder_output_state_2].filter(
        (name) => !decoderOutputNames.includes(name),
    );
    if (missingDecoderOutputs.length > 0) {
        throw new Error(
            `Nemo Conformer TDT decoder session is missing expected outputs: ${missingDecoderOutputs.join(', ')}. ` +
                'Override I/O names via `transformers.js_config.transducer.io` if your export uses different names.',
        );
    }

    const encoderSession = sessions?.encoder_model;
    if (!encoderSession) {
        throw new Error('Missing required session `encoder_model` for Nemo Conformer TDT.');
    }
    if (!(encoderSession.outputNames ?? []).includes(io.encoder_output)) {
        throw new Error(
            `Nemo Conformer TDT encoder session is missing expected output: ${io.encoder_output}. ` +
                'Override `transformers.js_config.transducer.io.encoder_output` if your export uses a different name.',
        );
    }

    const maxSymbolsPerStep = transducerConfig.max_symbols_per_step ?? 10;
    const subsamplingFactor = transducerConfig.subsampling_factor ?? 8;
    const frameShiftS = transducerConfig.frame_shift_s ?? 0.01;
    const blankTokenId = transducerConfig.blank_token_id ?? 0;
    const encoderOutputLayout = transducerConfig.encoder_output_layout;
    const encoderInputLayout = transducerConfig.encoder_input_layout ?? 'BTF';
    const encoderFrameLayout = transducerConfig.encoder_frame_layout ?? 'BD1';
    const decoderTokenDType = transducerConfig.decoder_token_dtype ?? 'int32';
    const decoderTokenLengthDType = transducerConfig.decoder_token_length_dtype ?? 'int32';

    if (!Number.isInteger(blankTokenId) || blankTokenId < 0) {
        throw new Error('Invalid `transformers.js_config.transducer.blank_token_id`: expected a non-negative integer.');
    }
    if (!Number.isInteger(maxSymbolsPerStep) || maxSymbolsPerStep <= 0) {
        throw new Error(
            'Invalid `transformers.js_config.transducer.max_symbols_per_step`: expected a positive integer.',
        );
    }
    if (!Number.isFinite(subsamplingFactor) || subsamplingFactor <= 0) {
        throw new Error('Invalid `transformers.js_config.transducer.subsampling_factor`: expected a positive number.');
    }
    if (!Number.isFinite(frameShiftS) || frameShiftS <= 0) {
        throw new Error('Invalid `transformers.js_config.transducer.frame_shift_s`: expected a positive number.');
    }
    if (encoderOutputLayout !== 'BDT' && encoderOutputLayout !== 'BTD') {
        throw new Error('Invalid `transformers.js_config.transducer.encoder_output_layout`: expected "BDT" or "BTD".');
    }
    if (encoderInputLayout !== 'BTF' && encoderInputLayout !== 'BFT') {
        throw new Error('Invalid `transformers.js_config.transducer.encoder_input_layout`: expected "BTF" or "BFT".');
    }
    if (encoderFrameLayout !== 'BD1' && encoderFrameLayout !== 'B1D') {
        throw new Error('Invalid `transformers.js_config.transducer.encoder_frame_layout`: expected "BD1" or "B1D".');
    }
    if (!['int32', 'int64'].includes(decoderTokenDType)) {
        throw new Error(
            'Invalid `transformers.js_config.transducer.decoder_token_dtype`: expected "int32" or "int64".',
        );
    }
    if (!['int32', 'int64'].includes(decoderTokenLengthDType)) {
        throw new Error(
            'Invalid `transformers.js_config.transducer.decoder_token_length_dtype`: expected "int32" or "int64".',
        );
    }

    return {
        blank_token_id: blankTokenId,
        max_symbols_per_step: maxSymbolsPerStep,
        subsampling_factor: subsamplingFactor,
        frame_shift_s: frameShiftS,
        vocab_size: transducerConfig.vocab_size ?? config.vocab_size ?? null,
        duration_start_index: transducerConfig.duration_start_index ?? null,
        encoder_input_layout: encoderInputLayout,
        encoder_output_layout: encoderOutputLayout,
        encoder_frame_layout: encoderFrameLayout,
        decoder_token_dtype: decoderTokenDType,
        decoder_token_length_dtype: decoderTokenLengthDType,
        decoder: {
            num_layers: numLayers,
            hidden_size: hiddenSize,
        },
        io,
    };
}

export class NemoConformerTDTPreTrainedModel extends PreTrainedModel {
    main_input_name = 'input_features';
    forward_params = ['input_features', 'attention_mask'];

    constructor(config, sessions, configs) {
        super(config, sessions, configs);
        this.transducer = resolveTransducerConfig(config, sessions);
    }

    static supports(model_type) {
        return model_type === NEMO_CONFORMER_TDT_MODEL_TYPE;
    }

    /**
     * Load Nemo Conformer TDT sessions using v4 canonical ONNX filenames.
     * @type {typeof PreTrainedModel.from_pretrained}
     */
    static async from_pretrained(
        pretrained_model_name_or_path,
        {
            progress_callback = null,
            config = null,
            cache_dir = null,
            local_files_only = false,
            revision = 'main',
            model_file_name = null,
            subfolder = 'onnx',
            device = null,
            dtype = null,
            use_external_data_format = null,
            session_options = {},
        } = {},
    ) {
        const options = {
            progress_callback,
            config,
            cache_dir,
            local_files_only,
            revision,
            model_file_name,
            subfolder,
            device,
            dtype,
            use_external_data_format,
            session_options,
        };

        config = options.config = await AutoConfig.from_pretrained(pretrained_model_name_or_path, options);
        if (config.model_type !== NEMO_CONFORMER_TDT_MODEL_TYPE) {
            throw new Error(`Unsupported model type: ${config.model_type}`);
        }

        if (options.model_file_name && options.model_file_name !== 'encoder_model') {
            throw new Error(
                'NemoConformerForTDT does not support `model_file_name` override. ' +
                    'Expected canonical files: `encoder_model{suffix}.onnx` and `decoder_model_merged{suffix}.onnx`.',
            );
        }

        let sessions;
        try {
            sessions = await constructSessions(
                pretrained_model_name_or_path,
                {
                    encoder_model: 'encoder_model',
                    decoder_model_merged: 'decoder_model_merged',
                },
                options,
                'decoder_model_merged',
            );
        } catch (error) {
            const reason = error?.message ?? String(error);
            throw new Error(
                'Failed to load Nemo Conformer TDT sessions. Expected canonical v4 files under `onnx/`: ' +
                    '`encoder_model{suffix}.onnx` and `decoder_model_merged{suffix}.onnx`. ' +
                    `Original error: ${reason}`,
            );
        }

        return new this(config, sessions, {});
    }
}

export class NemoConformerForTDT extends NemoConformerTDTPreTrainedModel {
    async _runEncoder(feeds) {
        return await sessionRun(this.sessions.encoder_model, feeds);
    }

    async _runDecoder(feeds) {
        return await sessionRun(this.sessions.decoder_model_merged, feeds);
    }

    _disposeDecoderState(state, keepState = null) {
        if (!state) return;
        if (state.state1 instanceof Tensor && state.state1 !== keepState?.state1) {
            state.state1.dispose();
        }
        if (state.state2 instanceof Tensor && state.state2 !== keepState?.state2) {
            state.state2.dispose();
        }
    }

    _getEncoderOutput(outputs) {
        const name = this.transducer.io.encoder_output;
        const out = outputs?.[name];
        if (!(out instanceof Tensor)) {
            const available = outputs && typeof outputs === 'object' ? Object.keys(outputs).join(', ') : '(none)';
            throw new Error(
                `Nemo Conformer TDT encoder output "${name}" was not returned by the session. ` +
                    `Available outputs: ${available}.`,
            );
        }
        return out;
    }

    _getEncoderFrameCount(encoderOutput) {
        if (encoderOutput.dims.length !== 3 || encoderOutput.dims[0] !== 1) {
            throw new Error(
                `Nemo Conformer TDT expected encoder output dims [1, D, T] or [1, T, D], got [${encoderOutput.dims.join(', ')}].`,
            );
        }
        const layout = this.transducer.encoder_output_layout;
        if (layout === 'BDT') {
            return encoderOutput.dims[2];
        }
        if (layout === 'BTD') {
            return encoderOutput.dims[1];
        }
        throw new Error(
            `Unsupported encoder output layout "${layout}". Use 'BDT' or 'BTD' in transformers.js_config.transducer.`,
        );
    }

    _getFrameData(encoderOutput, frameIndex, reusableFrame) {
        const layout = this.transducer.encoder_output_layout;
        if (encoderOutput.type !== 'float32') {
            throw new Error(`Nemo Conformer TDT expected encoder output type "float32", got "${encoderOutput.type}".`);
        }
        const data = /** @type {Float32Array} */ (encoderOutput.data);

        if (layout === 'BDT') {
            const D = encoderOutput.dims[1];
            const T = encoderOutput.dims[2];
            const frame = reusableFrame && reusableFrame.length === D ? reusableFrame : new Float32Array(D);
            for (let d = 0; d < D; ++d) {
                frame[d] = data[d * T + frameIndex];
            }
            return frame;
        }

        if (layout === 'BTD') {
            const D = encoderOutput.dims[2];
            const offset = frameIndex * D;
            return data.subarray(offset, offset + D);
        }

        throw new Error(
            `Unsupported encoder output layout "${layout}". Use 'BDT' or 'BTD' in transformers.js_config.transducer.`,
        );
    }

    _createFrameTensor(frameData) {
        const layout = this.transducer.encoder_frame_layout;
        if (layout === 'BD1') {
            return new Tensor('float32', frameData, [1, frameData.length, 1]);
        } else if (layout === 'B1D') {
            return new Tensor('float32', frameData, [1, 1, frameData.length]);
        }
        throw new Error(
            `Unsupported encoder frame layout "${layout}". Use 'BD1' or 'B1D' in transformers.js_config.transducer.`,
        );
    }

    _buildEncoderFeeds(model_inputs) {
        const encoderSession = this.sessions.encoder_model;
        const feeds = {};
        const disposables = [];
        const inputFeatures = model_inputs.input_features;

        if (!(inputFeatures instanceof Tensor)) {
            throw new Error(
                'NemoConformerForTDT.transcribe expected `model_inputs.input_features` as a Tensor from the processor.',
            );
        }

        const missingInputs = [];
        let preparedEncoderInput = null;
        const getPreparedEncoderInput = () => {
            if (preparedEncoderInput) {
                return preparedEncoderInput;
            }

            const layout = this.transducer.encoder_input_layout;
            if (layout === 'BTF') {
                preparedEncoderInput = inputFeatures;
            } else if (layout === 'BFT') {
                preparedEncoderInput = inputFeatures.transpose(0, 2, 1);
                disposables.push(preparedEncoderInput);
            } else {
                throw new Error(
                    `Unsupported encoder input layout "${layout}". Use 'BTF' or 'BFT' in transformers.js_config.transducer.`,
                );
            }
            return preparedEncoderInput;
        };
        for (const name of encoderSession.inputNames) {
            if (name === 'input_features' || name === 'audio_signal') {
                feeds[name] = getPreparedEncoderInput();
                continue;
            }

            if (model_inputs[name] instanceof Tensor) {
                feeds[name] = model_inputs[name];
                continue;
            }

            if (name === 'length') {
                let length = null;
                const attentionMask = model_inputs.attention_mask;
                if (attentionMask instanceof Tensor) {
                    const maskData = attentionMask.data;
                    let sum = 0;
                    for (let i = 0; i < maskData.length; ++i) {
                        sum += toInt(maskData[i]);
                    }
                    length = sum;
                } else {
                    length = inputFeatures.dims[1];
                }
                if (!Number.isInteger(length) || length < 0) {
                    throw new Error(
                        `Nemo Conformer TDT expected a non-negative integer encoder length, got: ${length}.`,
                    );
                }
                const lengthTensor = new Tensor('int64', BigInt64Array.from([BigInt(length)]), [1]);
                disposables.push(lengthTensor);
                feeds[name] = lengthTensor;
                continue;
            }

            missingInputs.push(name);
        }

        if (missingInputs.length > 0) {
            for (const tensor of disposables) {
                tensor.dispose();
            }
            throw new Error(
                `Nemo Conformer TDT encoder session expects additional inputs that are not available: ${missingInputs.join(', ')}.`,
            );
        }

        return { feeds, disposables };
    }

    _resolveVocabSize(tokenizer) {
        if (Number.isInteger(this.transducer.vocab_size)) {
            return this.transducer.vocab_size;
        }

        if (tokenizer?.get_vocab) {
            const vocab = tokenizer.get_vocab();
            if (vocab instanceof Map) {
                let maxId = -1;
                for (const id of vocab.values()) {
                    const numericId = Number(id);
                    if (Number.isInteger(numericId) && numericId >= 0) {
                        maxId = Math.max(maxId, numericId);
                    }
                }
                if (maxId >= 0) {
                    return maxId + 1;
                }
            } else if (Array.isArray(vocab)) {
                if (vocab.length > 0) {
                    return vocab.length;
                }
            } else if (vocab && typeof vocab === 'object') {
                let maxId = -1;
                for (const id of Object.values(vocab)) {
                    const numericId = Number(id);
                    if (Number.isInteger(numericId) && numericId >= 0) {
                        maxId = Math.max(maxId, numericId);
                    }
                }
                if (maxId >= 0) {
                    return maxId + 1;
                }
            }
        }

        throw new Error(
            'Unable to resolve vocabulary size for Nemo Conformer TDT. Set `vocab_size` in config.json or provide tokenizer with a vocab.',
        );
    }

    _validateRuntimeConfig(vocabSize) {
        if (!Number.isInteger(vocabSize) || vocabSize <= 0) {
            throw new Error(`Invalid Nemo Conformer TDT config: vocab_size=${vocabSize} must be a positive integer.`);
        }
        if (this.transducer.blank_token_id >= vocabSize) {
            throw new Error(
                `Invalid Nemo Conformer TDT config: blank_token_id=${this.transducer.blank_token_id} must be < vocab_size=${vocabSize}.`,
            );
        }
        const durationStart = this.transducer.duration_start_index ?? vocabSize;
        if (!Number.isInteger(durationStart) || durationStart < vocabSize) {
            throw new Error(
                `Invalid Nemo Conformer TDT config: duration_start_index=${durationStart} must be an integer >= vocab_size=${vocabSize}.`,
            );
        }
    }

    /**
     * Transcribe model-ready features using TDT decoding.
     *
     * - `returnTimestamps: false` → `{ text, isFinal }` (+ metrics if `returnMetrics`)
     * - `returnTimestamps: true`  → adds `utteranceTimestamp` and grouped `confidence`
     * - `returnWords: true` (requires `returnTimestamps`) → adds `words` list
     * - `returnTokens: true` (requires `returnTimestamps`) → adds `tokens` list
     * - `returnMetrics` is independent and can be combined with either level.
     * - Debug flags (`returnFrameConfidences`, `returnFrameIndices`, `returnLogProbs`, `returnTdtSteps`) are independent.
     * - Legacy snake_case aliases (`return_timestamps`, `return_words`, `return_tokens`, `return_metrics`) are accepted.
     *
     * @param {Object} model_inputs Processor outputs (must include `input_features`).
     * @param {Object} [decode_options]
     * @param {any} [decode_options.tokenizer] Tokenizer for text reconstruction and word boundaries.
     * @param {boolean} [decode_options.returnTimestamps=true] Include utterance-level timestamps and confidence aggregates.
     * @param {boolean} [decode_options.return_timestamps] Legacy alias for `returnTimestamps`.
     * @param {boolean} [decode_options.returnWords=false] Include word-level list (requires `returnTimestamps`).
     * @param {boolean} [decode_options.return_words] Legacy alias for `returnWords`.
     * @param {boolean} [decode_options.returnTokens=false] Include token-level list (requires `returnTimestamps`).
     * @param {boolean} [decode_options.return_tokens] Legacy alias for `returnTokens`.
     * @param {boolean} [decode_options.returnMetrics=false] Include encoding/decoding timing metrics.
     * @param {boolean} [decode_options.return_metrics] Legacy alias for `returnMetrics`.
     * @param {boolean} [decode_options.returnFrameConfidences=false] Include per-frame confidence scores in `confidence`.
     * @param {boolean} [decode_options.returnFrameIndices=false] Include per-token encoder frame indices.
     * @param {boolean} [decode_options.returnLogProbs=false] Include per-token log probabilities.
     * @param {boolean} [decode_options.returnTdtSteps=false] Include raw TDT duration steps.
     * @param {number} [decode_options.timeOffset=0] Offset added to all timestamps (seconds).
     * @returns {Promise<{
     *  text: string,
     *  isFinal: boolean,
     *  utteranceTimestamp?: [number, number],
     *  words?: Array<{ text: string, startTime: number, endTime: number, confidence?: number }>,
     *  tokens?: Array<{ id: number, token: string, rawToken: string, isWordStart: boolean, startTime: number, endTime: number, confidence?: number }>,
     *  confidence?: { utterance?: number|null, wordAverage?: number|null, frames?: number[]|null, frameAverage?: number|null, averageLogProb?: number|null },
     *  metrics?: { preprocessMs: number, encodeMs: number, decodeMs: number, tokenizeMs: number, totalMs: number, rtf: number, rtfX: number },
     *  debug?: { frameIndices?: number[] | null, logProbs?: number[] | null, tdtSteps?: number[] | null },
     * }>}
     */
    async transcribe(model_inputs, decode_options = {}) {
        const {
            tokenizer = null,
            returnTimestamps: returnTimestampsOption,
            return_timestamps: legacyReturnTimestamps,
            returnWords: returnWordsOption,
            return_words: legacyReturnWords,
            returnTokens: returnTokensOption,
            return_tokens: legacyReturnTokens,
            returnMetrics: returnMetricsOption,
            return_metrics: legacyReturnMetrics,
            returnFrameConfidences = false,
            returnFrameIndices = false,
            returnLogProbs = false,
            returnTdtSteps = false,
            timeOffset = 0,
        } = decode_options;
        const returnTimestamps = returnTimestampsOption ?? legacyReturnTimestamps ?? true;
        const returnWords = returnWordsOption ?? legacyReturnWords ?? false;
        const returnTokens = returnTokensOption ?? legacyReturnTokens ?? false;
        const returnMetrics = returnMetricsOption ?? legacyReturnMetrics ?? false;

        if (!Number.isFinite(timeOffset)) {
            throw new Error('NemoConformerForTDT.transcribe expected `timeOffset` to be a finite number.');
        }
        const totalStart = nowMs();
        const io = this.transducer.io;
        const vocabSize = this._resolveVocabSize(tokenizer);
        this._validateRuntimeConfig(vocabSize);

        const { feeds: encoderFeeds, disposables } = this._buildEncoderFeeds(model_inputs);
        let encoderOutputs;
        const encodeStart = nowMs();
        try {
            encoderOutputs = await this._runEncoder(encoderFeeds);
        } finally {
            for (const tensor of disposables) {
                tensor.dispose();
            }
        }
        const encodeMs = nowMs() - encodeStart;

        let frameCount = 0;
        let encoderOutput = null;
        const frameTime = this.transducer.subsampling_factor * this.transducer.frame_shift_s;

        const numLayers = this.transducer.decoder.num_layers;
        const hiddenSize = this.transducer.decoder.hidden_size;
        const blankId = this.transducer.blank_token_id;
        const maxSymbolsPerStep = this.transducer.max_symbols_per_step;

        const needConfidences = !!returnTimestamps;

        /** @type {number[]} */
        const tokenIds = [];
        /** @type {[number, number][]} */
        const tokenTimestamps = [];
        /** @type {number[] | null} */
        const tokenConfidences = needConfidences ? [] : null;
        /** @type {Map<number, { sum: number, count: number }> | null} */
        const frameConfidenceStats = returnFrameConfidences ? new Map() : null;
        /** @type {number[] | null} */
        const frameIndices = returnFrameIndices ? [] : null;
        /** @type {number[] | null} */
        const logProbs = returnLogProbs || needConfidences ? [] : null;
        /** @type {number[] | null} */
        const tdtSteps = returnTdtSteps ? [] : null;

        let decoderState;
        let targetLengthTensor;
        let reusableFrame = null;

        let emittedOnFrame = 0;
        const decodeStart = nowMs();

        try {
            encoderOutput = this._getEncoderOutput(encoderOutputs);
            frameCount = this._getEncoderFrameCount(encoderOutput);
            decoderState = {
                state1: new Tensor('float32', new Float32Array(numLayers * hiddenSize), [numLayers, 1, hiddenSize]),
                state2: new Tensor('float32', new Float32Array(numLayers * hiddenSize), [numLayers, 1, hiddenSize]),
            };

            targetLengthTensor =
                this.transducer.decoder_token_length_dtype === 'int64'
                    ? new Tensor('int64', BigInt64Array.from([1n]), [1])
                    : new Tensor('int32', new Int32Array([1]), [1]);

            for (let frameIndex = 0; frameIndex < frameCount; ) {
                const frameData = this._getFrameData(encoderOutput, frameIndex, reusableFrame);
                if (this.transducer.encoder_output_layout === 'BDT') {
                    reusableFrame = frameData;
                }
                const frameTensor = this._createFrameTensor(frameData);
                const prevTokenId = tokenIds.length > 0 ? tokenIds[tokenIds.length - 1] : blankId;
                const tokenTensor =
                    this.transducer.decoder_token_dtype === 'int64'
                        ? new Tensor('int64', BigInt64Array.from([BigInt(prevTokenId)]), [1, 1])
                        : new Tensor('int32', new Int32Array([prevTokenId]), [1, 1]);

                const decoderFeeds = {
                    [io.decoder_encoder]: frameTensor,
                    [io.decoder_token]: tokenTensor,
                    [io.decoder_token_length]: targetLengthTensor,
                    [io.decoder_state_1]: decoderState.state1,
                    [io.decoder_state_2]: decoderState.state2,
                };

                let decoderOutput;
                try {
                    decoderOutput = await this._runDecoder(decoderFeeds);
                } finally {
                    tokenTensor.dispose();
                    frameTensor.dispose();
                }

                const logits = decoderOutput[io.decoder_output];
                const outputState1 = decoderOutput[io.decoder_output_state_1];
                const outputState2 = decoderOutput[io.decoder_output_state_2];
                const seenDecoderTensors = new Set();
                for (const value of Object.values(decoderOutput)) {
                    if (!(value instanceof Tensor) || seenDecoderTensors.has(value)) continue;
                    seenDecoderTensors.add(value);
                    if (value === logits || value === outputState1 || value === outputState2) {
                        continue;
                    }
                    value.dispose();
                }
                if (!(logits instanceof Tensor)) {
                    this._disposeDecoderState(
                        {
                            state1: outputState1,
                            state2: outputState2,
                        },
                        decoderState,
                    );
                    throw new Error(
                        `Nemo Conformer TDT decoder output "${io.decoder_output}" was not returned by the session.`,
                    );
                }
                if (!(outputState1 instanceof Tensor) || !(outputState2 instanceof Tensor)) {
                    logits.dispose();
                    this._disposeDecoderState(
                        {
                            state1: outputState1,
                            state2: outputState2,
                        },
                        decoderState,
                    );
                    throw new Error(
                        `Nemo Conformer TDT decoder state outputs "${io.decoder_output_state_1}" and "${io.decoder_output_state_2}" were not returned by the session.`,
                    );
                }
                const logitsData = logits.data;
                if (logitsData.length < vocabSize) {
                    logits.dispose();
                    this._disposeDecoderState(
                        {
                            state1: outputState1,
                            state2: outputState2,
                        },
                        decoderState,
                    );
                    throw new Error(
                        `Nemo Conformer TDT decoder output is too small (${logitsData.length}) for vocab_size=${vocabSize}.`,
                    );
                }
                const tokenId = argmax(logitsData, 0, vocabSize);
                const durationStart = this.transducer.duration_start_index ?? vocabSize;
                const hasDurationLogits = logitsData.length > durationStart;
                if (this.transducer.duration_start_index != null && !hasDurationLogits) {
                    logits.dispose();
                    this._disposeDecoderState(
                        {
                            state1: outputState1,
                            state2: outputState2,
                        },
                        decoderState,
                    );
                    throw new Error(
                        `Nemo Conformer TDT decoder output is missing duration logits: expected values beyond index ${durationStart - 1}, got length=${logitsData.length}.`,
                    );
                }
                const step = hasDurationLogits
                    ? argmax(logitsData, durationStart, logitsData.length - durationStart) - durationStart
                    : 0;
                if (tdtSteps) {
                    tdtSteps.push(step);
                }

                const maybeConfidence =
                    needConfidences || returnLogProbs || returnFrameConfidences
                        ? confidenceFromLogits(logitsData, tokenId, vocabSize)
                        : null;
                if (frameConfidenceStats && maybeConfidence) {
                    const stats = frameConfidenceStats.get(frameIndex);
                    if (stats) {
                        stats.sum += maybeConfidence.confidence;
                        stats.count += 1;
                    } else {
                        frameConfidenceStats.set(frameIndex, { sum: maybeConfidence.confidence, count: 1 });
                    }
                }

                const newState = {
                    state1: outputState1,
                    state2: outputState2,
                };

                if (tokenId !== blankId) {
                    this._disposeDecoderState(decoderState, newState);
                    decoderState = newState;

                    tokenIds.push(tokenId);
                    // TDT duration convention: step=0 means "stay on current frame" (duration index 0 = no advance).
                    // We still associate the token with this frame, so durationFrames is at least 1.
                    const durationFrames = Math.max(1, step);
                    const endFrame = Math.min(frameCount, frameIndex + durationFrames);
                    tokenTimestamps.push([
                        roundTs(frameIndex * frameTime + timeOffset),
                        roundTs(endFrame * frameTime + timeOffset),
                    ]);
                    if (tokenConfidences && maybeConfidence) {
                        tokenConfidences.push(maybeConfidence.confidence);
                    }
                    if (frameIndices) {
                        frameIndices.push(frameIndex);
                    }
                    if (logProbs && maybeConfidence) {
                        logProbs.push(maybeConfidence.logProb);
                    }
                    emittedOnFrame += 1;
                } else {
                    this._disposeDecoderState(newState, decoderState);
                }

                logits.dispose();

                if (step > 0) {
                    frameIndex += step;
                    emittedOnFrame = 0;
                } else if (tokenId === blankId || emittedOnFrame >= maxSymbolsPerStep) {
                    frameIndex += 1;
                    emittedOnFrame = 0;
                }
            }
        } finally {
            if (targetLengthTensor) targetLengthTensor.dispose();
            if (decoderState) this._disposeDecoderState(decoderState);
            if (encoderOutputs && typeof encoderOutputs === 'object') {
                const seen = new Set();
                for (const value of Object.values(encoderOutputs)) {
                    if (value instanceof Tensor && !seen.has(value)) {
                        value.dispose();
                        seen.add(value);
                    }
                }
            }
        }
        const decodeMs = nowMs() - decodeStart;

        const tokenizeStart = nowMs();
        const text = decodeTransducerText(tokenizer, tokenIds);
        const needDetailed = returnTimestamps && (returnWords || returnTokens);
        const detailed = needDetailed
            ? buildTransducerDetailedOutputs(tokenizer, tokenIds, tokenTimestamps, tokenConfidences)
            : null;
        const tokenizeMs = nowMs() - tokenizeStart;

        /** @type {any} */
        const result = { text, isFinal: true };
        const utteranceConfidence =
            tokenConfidences && tokenConfidences.length > 0
                ? roundMetric(tokenConfidences.reduce((a, b) => a + b, 0) / tokenConfidences.length, 6)
                : null;
        const utteranceTimestamp =
            tokenTimestamps.length > 0
                ? /** @type {[number, number]} */ ([
                      tokenTimestamps[0][0],
                      tokenTimestamps[tokenTimestamps.length - 1][1],
                  ])
                : /** @type {[number, number]} */ ([roundTs(timeOffset), roundTs(frameCount * frameTime + timeOffset)]);
        const averageLogProb =
            logProbs && logProbs.length > 0
                ? roundMetric(logProbs.reduce((a, b) => a + b, 0) / logProbs.length, 6)
                : null;

        if (returnTimestamps) {
            result.utteranceTimestamp = utteranceTimestamp;

            if (detailed) {
                if (returnWords) result.words = detailed.words;
                if (returnTokens) result.tokens = detailed.tokens;
            }

            result.confidence = {
                utterance: utteranceConfidence,
                wordAverage: detailed?.wordAverage != null ? roundMetric(detailed.wordAverage, 6) : null,
                averageLogProb,
            };
        }

        // Frame confidences are independent of return_timestamps — emit whenever requested.
        if (returnFrameConfidences && frameConfidenceStats && frameConfidenceStats.size > 0) {
            const frameConfidences = [];
            for (const { sum, count } of frameConfidenceStats.values()) {
                frameConfidences.push(sum / count);
            }
            result.confidence = {
                ...(result.confidence ?? {}),
                frames: frameConfidences,
                frameAverage: roundMetric(frameConfidences.reduce((a, b) => a + b, 0) / frameConfidences.length, 6),
            };
        }

        if (!returnTimestamps && averageLogProb != null) {
            result.confidence = {
                ...(result.confidence ?? {}),
                averageLogProb,
            };
        }

        const debug = {};
        if (returnFrameIndices) {
            debug.frameIndices = frameIndices;
        }
        if (returnLogProbs) {
            debug.logProbs = logProbs;
        }
        if (returnTdtSteps) {
            debug.tdtSteps = tdtSteps;
        }
        if (Object.keys(debug).length > 0) {
            result.debug = debug;
        }

        if (returnMetrics) {
            const totalMs = nowMs() - totalStart;
            const utteranceDuration = result.utteranceTimestamp
                ? Math.max(result.utteranceTimestamp[1] - result.utteranceTimestamp[0], 1e-8)
                : Math.max(frameCount * frameTime, 1e-8);
            const rtf = totalMs / 1000 / utteranceDuration;
            result.metrics = {
                preprocessMs: 0.0,
                encodeMs: roundMetric(encodeMs, 2),
                decodeMs: roundMetric(decodeMs, 2),
                tokenizeMs: roundMetric(tokenizeMs, 2),
                totalMs: roundMetric(totalMs, 2),
                rtf: roundMetric(rtf, 4),
                rtfX: roundMetric(1 / Math.max(rtf, 1e-8), 2),
            };
        }

        return result;
    }

    /**
     * Runs TDT transcription when called directly.
     * @param {Object} model_inputs
     */
    async _call(model_inputs) {
        return await this.transcribe(model_inputs);
    }
}

MODEL_TYPE_MAPPING.set('nemo-conformer-tdt', MODEL_TYPES.NemoConformerTDT);
MODEL_TYPE_MAPPING.set('NemoConformerForTDT', MODEL_TYPES.NemoConformerTDT);
MODEL_NAME_TO_CLASS_MAPPING.set('NemoConformerTDTPreTrainedModel', NemoConformerTDTPreTrainedModel);
MODEL_NAME_TO_CLASS_MAPPING.set('NemoConformerForTDT', NemoConformerForTDT);
MODEL_CLASS_TO_NAME_MAPPING.set(NemoConformerTDTPreTrainedModel, 'NemoConformerTDTPreTrainedModel');
MODEL_CLASS_TO_NAME_MAPPING.set(NemoConformerForTDT, 'NemoConformerForTDT');
