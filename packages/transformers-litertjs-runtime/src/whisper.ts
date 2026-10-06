import { Tensor } from '@huggingface/transformers';
import { Tensor as LiteRtTensor, type CompiledModel } from '@litertjs/core';

const MAX_DECODE_LENGTH = 128;
const VOCAB_SIZE = 51_865;

export type GenerationConfig = {
    decoder_start_token_id?: number;
    eos_token_id?: number;
    pad_token_id?: number;
    no_timestamps_token_id?: number;
    suppress_tokens?: number[];
    begin_suppress_tokens?: number[];
    lang_to_id?: Record<string, number>;
    task_to_id?: Record<string, number>;
};

function createCausalMask(): Float32Array {
    const mask = new Float32Array(MAX_DECODE_LENGTH * MAX_DECODE_LENGTH);
    for (let row = 0; row < MAX_DECODE_LENGTH; ++row) {
        mask.fill(Number.NEGATIVE_INFINITY, row * MAX_DECODE_LENGTH + row + 1, (row + 1) * MAX_DECODE_LENGTH);
    }
    return mask;
}

function getPrompt(config: GenerationConfig, options: Record<string, unknown>): number[] {
    const language = String(options.language ?? 'en').toLowerCase();
    const languageToken = language.startsWith('<|') ? language : `<|${language}|>`;
    const languageId = config.lang_to_id?.[languageToken];
    if (languageId === undefined) {
        throw new Error(`LiteRT Whisper does not recognize language "${language}". Pass an ISO language code.`);
    }
    const task = String(options.task ?? 'transcribe');
    const taskId = config.task_to_id?.[task];
    if (taskId === undefined) throw new Error(`LiteRT Whisper does not support task "${task}".`);
    const prompt = [config.decoder_start_token_id ?? 50_258, languageId, taskId];
    if (options.return_timestamps !== true) prompt.push(config.no_timestamps_token_id ?? 50_363);
    return prompt;
}

async function deleteTensors(tensors: Record<string, LiteRtTensor> | LiteRtTensor[]): Promise<void> {
    for (const tensor of Object.values(tensors)) tensor.delete();
}

async function greedyGenerate(
    model: CompiledModel,
    inputFeatures: Tensor,
    generationConfig: GenerationConfig,
    options: Record<string, unknown>,
): Promise<Tensor> {
    if (options.return_timestamps) throw new Error('LiteRT Whisper does not currently support timestamp generation.');
    const encodeInput = new LiteRtTensor(inputFeatures.data as Float32Array, inputFeatures.dims);
    let encoded: Record<string, LiteRtTensor>;
    try {
        encoded = (await model.signatures.encode.run({ args_0: encodeInput })) as Record<string, LiteRtTensor>;
    } finally {
        encodeInput.delete();
    }
    const encoderOutput = encoded.output_0;
    const causalMask = new LiteRtTensor(createCausalMask(), [1, 1, MAX_DECODE_LENGTH, MAX_DECODE_LENGTH]);
    const tokens = new Int32Array(MAX_DECODE_LENGTH);
    const prompt = getPrompt(generationConfig, options);
    tokens.fill(generationConfig.pad_token_id ?? generationConfig.eos_token_id ?? 50_257);
    tokens.set(prompt);
    let length = prompt.length;
    const maxNewTokens = Math.min(
        Number(options.max_new_tokens ?? MAX_DECODE_LENGTH - length),
        MAX_DECODE_LENGTH - length,
    );
    const suppressed = new Set(generationConfig.suppress_tokens ?? []);
    const beginSuppressed = new Set(generationConfig.begin_suppress_tokens ?? []);
    const eosTokenId = generationConfig.eos_token_id ?? 50_257;
    try {
        for (let step = 0; step < maxNewTokens; ++step) {
            const tokenTensor = new LiteRtTensor(tokens, [1, MAX_DECODE_LENGTH]);
            let decoded: Record<string, LiteRtTensor> | undefined;
            try {
                decoded = (await model.signatures.decode.run({
                    args_0: encoderOutput,
                    args_1: tokenTensor,
                    args_2: causalMask,
                })) as Record<string, LiteRtTensor>;
                const logits = (await decoded.output_0.data()) as Float32Array;
                const offset = (length - 1) * VOCAB_SIZE;
                let nextToken = 0;
                let nextScore = Number.NEGATIVE_INFINITY;
                for (let token = 0; token < VOCAB_SIZE; ++token) {
                    if (suppressed.has(token) || (step === 0 && beginSuppressed.has(token))) continue;
                    const score = logits[offset + token];
                    if (score > nextScore) {
                        nextScore = score;
                        nextToken = token;
                    }
                }
                tokens[length++] = nextToken;
                if (nextToken === eosTokenId) break;
            } finally {
                tokenTensor.delete();
                if (decoded) await deleteTensors(decoded);
            }
        }
    } finally {
        causalMask.delete();
        await deleteTensors(encoded);
    }
    return new Tensor('int64', Array.from(tokens.subarray(0, length), BigInt), [1, length]);
}

export function createWhisperModel(model: CompiledModel, generationConfig: GenerationConfig): Function {
    const callable = () => {
        throw new Error('LiteRT Whisper supports `generate()` rather than a generic forward pass.');
    };
    return Object.assign(callable, {
        capabilities: { automaticSpeechRecognition: { version: 1, input: 'audio' } },
        generate: (options: Record<string, unknown>) =>
            greedyGenerate(model, options.inputs as Tensor, generationConfig, options),
        dispose: () => model.delete(),
    });
}
