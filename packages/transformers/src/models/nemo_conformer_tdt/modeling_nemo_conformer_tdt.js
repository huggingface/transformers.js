import { AutoConfig } from '../../configs.js';
import { Tensor } from '../../utils/tensor.js';
import { PreTrainedModel } from '../modeling_utils.js';
import { constructSessions, sessionRun } from '../session.js';
import { buildTransducerWordTimestamps, decodeTransducerText } from './transducer_text.js';

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

function inferEncoderOutputLayout(outputTensor) {
    if (outputTensor.dims.length !== 3 || outputTensor.dims[0] !== 1) {
        throw new Error(
            `Nemo Conformer TDT expected encoder output dims [1, D, T] or [1, T, D], got [${outputTensor.dims.join(', ')}].`,
        );
    }

    // Heuristic fallback: in most Nemo exports D > T.
    return outputTensor.dims[1] >= outputTensor.dims[2] ? 'BDT' : 'BTD';
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
        encoder_input_layout: transducerConfig.encoder_input_layout ?? 'BTF',
        encoder_output_layout: transducerConfig.encoder_output_layout ?? null,
        encoder_frame_layout: transducerConfig.encoder_frame_layout ?? 'BD1',
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
        if (state.state1 && state.state1 !== keepState?.state1) {
            state.state1.dispose();
        }
        if (state.state2 && state.state2 !== keepState?.state2) {
            state.state2.dispose();
        }
    }

    _getEncoderOutput(outputs) {
        const name = this.transducer.io.encoder_output;
        return outputs[name] ?? Object.values(outputs)[0];
    }

    _encoderOutputToFrames(encoderOutput) {
        const layout = this.transducer.encoder_output_layout ?? inferEncoderOutputLayout(encoderOutput);
        const dims = encoderOutput.dims;
        const data = encoderOutput.data;
        const frames = [];

        if (layout === 'BDT') {
            const D = dims[1];
            const T = dims[2];
            for (let t = 0; t < T; ++t) {
                const frame = new Float32Array(D);
                for (let d = 0; d < D; ++d) {
                    frame[d] = data[d * T + t];
                }
                frames.push(frame);
            }
            return frames;
        }

        if (layout === 'BTD') {
            const T = dims[1];
            const D = dims[2];
            for (let t = 0; t < T; ++t) {
                const offset = t * D;
                frames.push(new Float32Array(data.subarray(offset, offset + D)));
            }
            return frames;
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
        for (const name of encoderSession.inputNames) {
            if (model_inputs[name] instanceof Tensor) {
                feeds[name] = model_inputs[name];
                continue;
            }

            if (name === 'input_features') {
                feeds[name] = inputFeatures;
                continue;
            }

            if (name === 'audio_signal') {
                const layout = this.transducer.encoder_input_layout;
                if (layout === 'BTF') {
                    feeds[name] = inputFeatures;
                } else if (layout === 'BFT') {
                    const transposed = inputFeatures.transpose(0, 2, 1);
                    disposables.push(transposed);
                    feeds[name] = transposed;
                } else {
                    throw new Error(
                        `Unsupported encoder input layout "${layout}". Use 'BTF' or 'BFT' in transformers.js_config.transducer.`,
                    );
                }
                continue;
            }

            if (name === 'length') {
                let length = null;
                const attentionMask = model_inputs.attention_mask;
                if (attentionMask instanceof Tensor) {
                    const mask = attentionMask.tolist();
                    length = mask[0].reduce((acc, x) => acc + toInt(x), 0);
                } else {
                    length = inputFeatures.dims[1];
                }
                const lengthTensor = new Tensor('int64', BigInt64Array.from([BigInt(length)]), [1]);
                disposables.push(lengthTensor);
                feeds[name] = lengthTensor;
                continue;
            }

            missingInputs.push(name);
        }

        if (missingInputs.length > 0) {
            throw new Error(
                `Nemo Conformer TDT encoder session expects additional inputs that are not available: ${missingInputs.join(', ')}.`,
            );
        }

        return { feeds, disposables };
    }

    _resolveVocabSize(tokenizer) {
        if (Number.isInteger(this.transducer.vocab_size) && this.transducer.vocab_size > 0) {
            return this.transducer.vocab_size;
        }

        if (tokenizer?.get_vocab) {
            const size = Object.keys(tokenizer.get_vocab()).length;
            if (size > 0) {
                return size;
            }
        }

        throw new Error(
            'Unable to resolve vocabulary size for Nemo Conformer TDT. Set `vocab_size` in config.json or provide tokenizer with a vocab.',
        );
    }

    _validateRuntimeConfig(vocabSize) {
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
     * @param {Object} model_inputs Processor outputs (must include `input_features`).
     * @param {Object} [decode_options]
     * @param {any} [decode_options.tokenizer] Tokenizer used for text reconstruction and word timestamps.
     * @param {boolean} [decode_options.return_token_timestamps=true]
     * @param {boolean} [decode_options.return_word_timestamps=true]
     * @param {boolean} [decode_options.return_utterance_timestamp=true]
     * @returns {Promise<{
     *  text: string,
     *  token_ids: number[],
     *  token_timestamps?: [number, number][],
     *  word_timestamps?: { text: string, timestamp: [number, number]}[],
     *  utterance_timestamp?: [number, number],
     * }>}
     */
    async transcribe(
        model_inputs,
        {
            tokenizer = null,
            return_token_timestamps = true,
            return_word_timestamps = true,
            return_utterance_timestamp = true,
        } = {},
    ) {
        const io = this.transducer.io;
        const vocabSize = this._resolveVocabSize(tokenizer);
        this._validateRuntimeConfig(vocabSize);

        const { feeds: encoderFeeds, disposables } = this._buildEncoderFeeds(model_inputs);
        let encoderOutputs;
        try {
            encoderOutputs = await this._runEncoder(encoderFeeds);
        } finally {
            for (const tensor of disposables) {
                tensor.dispose();
            }
        }

        const encoderOutput = this._getEncoderOutput(encoderOutputs);
        let frames;
        try {
            frames = this._encoderOutputToFrames(encoderOutput);
        } finally {
            const seen = new Set();
            for (const value of Object.values(encoderOutputs)) {
                if (value instanceof Tensor && !seen.has(value)) {
                    value.dispose();
                    seen.add(value);
                }
            }
        }
        const frameTime = this.transducer.subsampling_factor * this.transducer.frame_shift_s;

        const numLayers = this.transducer.decoder.num_layers;
        const hiddenSize = this.transducer.decoder.hidden_size;
        const blankId = this.transducer.blank_token_id;
        const maxSymbolsPerStep = this.transducer.max_symbols_per_step;

        /** @type {number[]} */
        const tokenIds = [];
        /** @type {[number, number][]} */
        const tokenTimestamps = [];

        let decoderState = {
            state1: new Tensor('float32', new Float32Array(numLayers * hiddenSize), [numLayers, 1, hiddenSize]),
            state2: new Tensor('float32', new Float32Array(numLayers * hiddenSize), [numLayers, 1, hiddenSize]),
        };

        const targetLengthTensor =
            this.transducer.decoder_token_length_dtype === 'int64'
                ? new Tensor('int64', BigInt64Array.from([1n]), [1])
                : new Tensor('int32', new Int32Array([1]), [1]);
        let emittedOnFrame = 0;

        try {
            for (let frameIndex = 0; frameIndex < frames.length; ) {
                const frameTensor = this._createFrameTensor(frames[frameIndex]);
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

                const logits = decoderOutput[io.decoder_output] ?? Object.values(decoderOutput)[0];
                const logitsData = logits.data;
                if (logitsData.length < vocabSize) {
                    throw new Error(
                        `Nemo Conformer TDT decoder output is too small (${logitsData.length}) for vocab_size=${vocabSize}.`,
                    );
                }
                const tokenId = argmax(logitsData, 0, vocabSize);

                const durationStart = this.transducer.duration_start_index ?? vocabSize;
                const hasDurationLogits = logitsData.length > durationStart;
                const step = hasDurationLogits
                    ? argmax(logitsData, durationStart, logitsData.length - durationStart) - durationStart
                    : 0;

                const newState = {
                    state1: decoderOutput[io.decoder_output_state_1] ?? decoderState.state1,
                    state2: decoderOutput[io.decoder_output_state_2] ?? decoderState.state2,
                };

                if (tokenId !== blankId) {
                    this._disposeDecoderState(decoderState, newState);
                    decoderState = newState;

                    tokenIds.push(tokenId);
                    const durationFrames = step > 0 ? step : 1;
                    tokenTimestamps.push([frameIndex * frameTime, (frameIndex + durationFrames) * frameTime]);
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
            targetLengthTensor.dispose();
            this._disposeDecoderState(decoderState);
        }

        const text = decodeTransducerText(tokenizer, tokenIds);

        const result = {
            text,
            token_ids: tokenIds,
        };

        if (return_token_timestamps) {
            result.token_timestamps = tokenTimestamps;
        }

        if (return_word_timestamps) {
            result.word_timestamps = buildTransducerWordTimestamps(tokenizer, tokenIds, tokenTimestamps);
        }

        if (return_utterance_timestamp) {
            if (tokenTimestamps.length > 0) {
                result.utterance_timestamp = [tokenTimestamps[0][0], tokenTimestamps[tokenTimestamps.length - 1][1]];
            } else {
                result.utterance_timestamp = [0, frames.length * frameTime];
            }
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
