import {
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    env,
    loadWasmBinary,
    loadWasmFactory,
    logger,
    type Tensor,
} from '@huggingface/transformers';
import { type LLGuidanceResponseFormat, type LoadBundledLLGuidanceOptions, loadBundledLLGuidance } from 'llguidance';

export type ResponseFormat = LLGuidanceResponseFormat;

export type LlguidanceLoadOptions = LoadBundledLLGuidanceOptions & {
    /** Whether to pre-load and cache llguidance WASM assets. Defaults to env.useWasmCache. */
    useWasmCache?: boolean;
};

const LLGUIDANCE_VERSION = '0.1.7';
const LLGUIDANCE_WASM_BASE = `https://cdn.jsdelivr.net/npm/llguidance@${LLGUIDANCE_VERSION}/wasm/`;
const DEFAULT_LLGUIDANCE_WASM_URL = `${LLGUIDANCE_WASM_BASE}llguidance_wasm_bg.wasm`;
const DEFAULT_LLGUIDANCE_WASM_FACTORY_URL = `${LLGUIDANCE_WASM_BASE}llguidance_wasm.js`;

type GuidanceMask = Uint32Array | Uint8Array | boolean[] | number[];

type GuidanceMaskResult =
    | { mask: GuidanceMask; vocabSize?: number; stop?: false }
    | { stop: true }
    | { backtrack?: number; ffTokens?: number[] };

type GuidanceCommitResult = {
    stop?: boolean;
    backtrack?: number;
    ffTokens?: number[];
};

type GuidanceInterpreter = {
    computeMask(): GuidanceMaskResult;
    commitToken(tokenId: number): GuidanceCommitResult | undefined;
};

type LlguidanceState = {
    completed: boolean;
    interpreter: GuidanceInterpreter;
    step: number;
};

export class LlguidanceConstraint {
    static async fromResponseFormat(
        tokenizer: unknown,
        response_format: ResponseFormat,
        loadOptions: LlguidanceLoadOptions = {},
    ) {
        logger.debug('[LlguidanceConstraint] loading llguidance', {
            response_format,
        });

        const runtime = await loadLLGuidance(loadOptions);
        const interpreter = runtime.createInterpreter({
            tokenizer,
            response_format,
        }) as GuidanceInterpreter;

        logger.debug('[LlguidanceConstraint] interpreter created');

        const state: LlguidanceState = {
            completed: false,
            interpreter,
            step: 0,
        };

        const logits_processor = new LogitsProcessorList();
        logits_processor.push(new LlguidanceLogitsProcessor(state));

        return {
            logits_processor,
            stopping_criteria: new LlguidanceStoppingCriteria(state),
        };
    }
}

async function loadLLGuidance(loadOptions: LlguidanceLoadOptions) {
    const { useWasmCache = env.useWasmCache, ...options } = loadOptions;
    if (!useWasmCache || isNodeLikeRuntime() || options.wasmFactory) {
        return loadBundledLLGuidance(options);
    }

    const wasmSource = options.wasm ?? options.wasmUrl ?? DEFAULT_LLGUIDANCE_WASM_URL;
    const wasmFactorySource = options.wasmFactoryUrl ?? DEFAULT_LLGUIDANCE_WASM_FACTORY_URL;
    const cachedOptions = { ...options };

    const [wasm, wasmFactoryUrl] = await Promise.all([
        loadCacheableWasm(wasmSource),
        loadCacheableWasmFactory(wasmFactorySource),
    ]);

    if (wasm) {
        cachedOptions.wasm = wasm;
        delete cachedOptions.wasmUrl;
    }

    if (wasm && wasmFactoryUrl) {
        cachedOptions.wasmFactoryUrl = wasmFactoryUrl;
    }

    return loadBundledLLGuidance(cachedOptions);
}

async function loadCacheableWasm(source: LoadBundledLLGuidanceOptions['wasm']) {
    const url = toCacheableURL(source);
    if (!url) return null;

    try {
        return await loadWasmBinary(url);
    } catch (error) {
        logger.warn('Failed to pre-load llguidance WASM binary:', error);
        return null;
    }
}

async function loadCacheableWasmFactory(source: LoadBundledLLGuidanceOptions['wasmFactoryUrl']) {
    const url = toCacheableURL(source);
    if (!url) return null;

    try {
        return await loadWasmFactory(url);
    } catch (error) {
        logger.warn('Failed to pre-load llguidance WASM factory:', error);
        return null;
    }
}

function toCacheableURL(source: unknown) {
    if (typeof source === 'string') {
        return isBlobURL(source) ? null : new URL(source, globalThis.location?.href).href;
    }
    if (source instanceof URL) {
        return isBlobURL(source.href) ? null : source.href;
    }
    return null;
}

function isBlobURL(url: string) {
    return url.startsWith('blob:');
}

function isNodeLikeRuntime() {
    return typeof process !== 'undefined' && Boolean(process.versions?.node);
}

class LlguidanceLogitsProcessor extends LogitsProcessor {
    private state: LlguidanceState;

    constructor(state: LlguidanceState) {
        super();
        this.state = state;
    }

    _call(_inputIds: bigint[][], logits: Tensor) {
        if (this.state.completed) {
            logger.debug('[LlguidanceLogitsProcessor] skip completed', {
                step: this.state.step,
            });
            return logits;
        }

        this.state.step++;
        const vocabSize = logits.dims.at(-1);
        logger.debug('[LlguidanceLogitsProcessor] compute mask', {
            step: this.state.step,
            logitsDims: logits.dims,
            vocabSize,
        });

        let result: GuidanceMaskResult;
        try {
            result = this.state.interpreter.computeMask();
        } catch (error) {
            if (!String((error as Error).message).includes('compute_mask() called after stop')) {
                throw error;
            }
            this.state.completed = true;
            logger.debug('[LlguidanceLogitsProcessor] compute after stop', {
                step: this.state.step,
            });
            return logits;
        }

        logger.debug('[LlguidanceLogitsProcessor] mask result', {
            step: this.state.step,
            result: summarizeMaskResult(result, vocabSize),
        });

        if ('stop' in result && result.stop) {
            this.state.completed = true;
            logger.debug('[LlguidanceLogitsProcessor] stopped by mask', {
                step: this.state.step,
            });
            return logits;
        }

        if ('mask' in result) {
            const applied = applyMask(logits, result.mask, result.vocabSize ?? vocabSize);
            logger.debug('[LlguidanceLogitsProcessor] mask applied', {
                step: this.state.step,
                ...applied,
            });
        }

        return logits;
    }

    onTokenSampled(tokenId: number, batchIdx: number, inputIds: bigint[][]) {
        logger.debug('[LlguidanceLogitsProcessor] token sampled', {
            step: this.state.step,
            tokenId,
            batchIdx,
            inputLength: inputIds[batchIdx]?.length,
            completed: this.state.completed,
        });

        if (this.state.completed) return;

        const result = this.state.interpreter.commitToken(tokenId);
        logger.debug('[LlguidanceLogitsProcessor] token committed', {
            step: this.state.step,
            tokenId,
            result,
        });

        if (result?.stop) {
            this.state.completed = true;
            logger.debug('[LlguidanceLogitsProcessor] stopped by commit', {
                step: this.state.step,
                tokenId,
            });
        }
    }

    onTokensSampled(tokenIds: number[], inputIds: bigint[][]) {
        logger.debug('[LlguidanceLogitsProcessor] tokens sampled', {
            step: this.state.step,
            tokenIds,
            inputLengths: inputIds.map((ids) => ids.length),
            completed: this.state.completed,
        });

        for (let batchIdx = 0; batchIdx < tokenIds.length; ++batchIdx) {
            this.onTokenSampled(tokenIds[batchIdx], batchIdx, inputIds);
        }
    }
}

class LlguidanceStoppingCriteria extends StoppingCriteria {
    private state: LlguidanceState;

    constructor(state: LlguidanceState) {
        super();
        this.state = state;
    }

    _call(inputIds: ArrayLike<unknown>[]) {
        const result = new Array(inputIds.length).fill(this.state.completed);
        logger.debug('[LlguidanceStoppingCriteria] call', {
            step: this.state.step,
            completed: this.state.completed,
            result,
            inputLengths: inputIds.map((ids) => ids.length),
        });
        return result;
    }
}

function summarizeMaskResult(result: GuidanceMaskResult, vocabSize?: number) {
    if ('stop' in result && result.stop) {
        return { stop: true };
    }

    if (!('mask' in result)) {
        return result;
    }

    return {
        maskLength: result.mask.length,
        vocabSize: result.vocabSize ?? vocabSize,
        allowed: countAllowed(result.mask, result.vocabSize ?? vocabSize),
        sampleAllowedTokenIds: sampleAllowedTokenIds(result.mask, result.vocabSize ?? vocabSize),
    };
}

function countAllowed(mask: GuidanceMask, vocabSize?: number) {
    if (!vocabSize) return undefined;

    let allowed = 0;
    for (let tokenId = 0; tokenId < vocabSize; ++tokenId) {
        if (isAllowed(mask, tokenId, vocabSize)) allowed++;
    }
    return allowed;
}

function sampleAllowedTokenIds(mask: GuidanceMask, vocabSize?: number) {
    if (!vocabSize) return [];

    const tokenIds: number[] = [];
    for (let tokenId = 0; tokenId < vocabSize && tokenIds.length < 25; ++tokenId) {
        if (isAllowed(mask, tokenId, vocabSize)) tokenIds.push(tokenId);
    }
    return tokenIds;
}

function isAllowed(mask: GuidanceMask, tokenId: number, vocabSize: number) {
    if (mask.length >= vocabSize) {
        return Boolean(mask[tokenId]);
    }
    return Boolean(Number(mask[tokenId >> 5]) & (1 << (tokenId & 31)));
}

function applyMask(logits: Tensor, mask: GuidanceMask, vocabSize?: number) {
    if (!vocabSize) {
        return { vocabSize, batchSize: 0, masked: 0, allowed: undefined };
    }

    const data = logits.data as Float32Array | Float64Array | number[];
    const batchSize = Math.max(1, data.length / vocabSize);
    let masked = 0;
    let allowed = 0;

    for (let batch = 0; batch < batchSize; ++batch) {
        const offset = batch * vocabSize;
        for (let tokenId = 0; tokenId < vocabSize; ++tokenId) {
            if (!isAllowed(mask, tokenId, vocabSize)) {
                data[offset + tokenId] = -Infinity;
                masked++;
            } else if (batch === 0) {
                allowed++;
            }
        }
    }

    return { vocabSize, batchSize, masked, allowed };
}
