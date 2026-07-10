import { LogitsProcessor, LogitsProcessorList, StoppingCriteria, type Tensor } from '@huggingface/transformers';
import { type LLGuidanceResponseFormat, loadBundledLLGuidance } from 'llguidance';

export type ResponseFormat = LLGuidanceResponseFormat;

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
    static async fromResponseFormat(tokenizer: unknown, response_format: ResponseFormat) {
        console.log('[LlguidanceConstraint] loading llguidance', {
            response_format,
        });

        const runtime = await loadBundledLLGuidance();
        const interpreter = runtime.createInterpreter({
            tokenizer,
            response_format,
        }) as GuidanceInterpreter;

        console.log('[LlguidanceConstraint] interpreter created');

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

class LlguidanceLogitsProcessor extends LogitsProcessor {
    private state: LlguidanceState;

    constructor(state: LlguidanceState) {
        super();
        this.state = state;
    }

    _call(_inputIds: bigint[][], logits: Tensor) {
        if (this.state.completed) {
            console.log('[LlguidanceLogitsProcessor] skip completed', {
                step: this.state.step,
            });
            return logits;
        }

        this.state.step++;
        const vocabSize = logits.dims.at(-1);
        console.log('[LlguidanceLogitsProcessor] compute mask', {
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
            console.log('[LlguidanceLogitsProcessor] compute after stop', {
                step: this.state.step,
            });
            return logits;
        }

        console.log('[LlguidanceLogitsProcessor] mask result', {
            step: this.state.step,
            result: summarizeMaskResult(result, vocabSize),
        });

        if ('stop' in result && result.stop) {
            this.state.completed = true;
            console.log('[LlguidanceLogitsProcessor] stopped by mask', {
                step: this.state.step,
            });
            return logits;
        }

        if ('mask' in result) {
            const applied = applyMask(logits, result.mask, result.vocabSize ?? vocabSize);
            console.log('[LlguidanceLogitsProcessor] mask applied', {
                step: this.state.step,
                ...applied,
            });
        }

        return logits;
    }

    onTokenSampled(tokenId: number, batchIdx: number, inputIds: bigint[][]) {
        console.log('[LlguidanceLogitsProcessor] token sampled', {
            step: this.state.step,
            tokenId,
            batchIdx,
            inputLength: inputIds[batchIdx]?.length,
            completed: this.state.completed,
        });

        if (this.state.completed) return;

        const result = this.state.interpreter.commitToken(tokenId);
        console.log('[LlguidanceLogitsProcessor] token committed', {
            step: this.state.step,
            tokenId,
            result,
        });

        if (result?.stop) {
            this.state.completed = true;
            console.log('[LlguidanceLogitsProcessor] stopped by commit', {
                step: this.state.step,
                tokenId,
            });
        }
    }

    onTokensSampled(tokenIds: number[], inputIds: bigint[][]) {
        console.log('[LlguidanceLogitsProcessor] tokens sampled', {
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
        console.log('[LlguidanceStoppingCriteria] call', {
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
