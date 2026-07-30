import {
    env,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    logger,
    type Tensor,
} from '@huggingface/transformers';
import { type LLGuidanceResponseFormat } from 'llguidance';

import { applyMask, forceToken, summarizeMaskResult } from './utils/mask';
import {
    type GuidanceInterpreter,
    type GuidanceMaskResult,
    type LlguidanceState,
    type LlguidanceStats,
} from './utils/types';
import { type LlguidanceLoadOptions, loadLLGuidance } from './utils/wasm';

export type ResponseFormat = LLGuidanceResponseFormat;
export type { LlguidanceLoadOptions, LlguidanceStats };

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
            stats: { steps: 0, computeMaskMs: 0, applyMaskMs: 0, commitTokenMs: 0 },
        };

        const logits_processor = new LogitsProcessorList();
        logits_processor.push(new LlguidanceLogitsProcessor(state));

        return {
            logits_processor,
            stopping_criteria: new LlguidanceStoppingCriteria(state),
            stats: state.stats,
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
            if (isDebugEnabled()) {
                logger.debug('[LlguidanceLogitsProcessor] skip completed', {
                    step: this.state.step,
                });
            }
            return logits;
        }

        this.state.step++;
        this.state.stats.steps = this.state.step;
        const vocabSize = logits.dims.at(-1);
        if (isDebugEnabled()) {
            logger.debug('[LlguidanceLogitsProcessor] compute mask', {
                step: this.state.step,
                logitsDims: logits.dims,
                vocabSize,
            });
        }

        let result: GuidanceMaskResult;
        const maskStart = performance.now();
        try {
            result = this.state.interpreter.computeMask();
            this.state.stats.computeMaskMs += performance.now() - maskStart;
        } catch (error) {
            this.state.stats.computeMaskMs += performance.now() - maskStart;
            if (!String((error as Error).message).includes('compute_mask() called after stop')) {
                throw error;
            }
            this.state.completed = true;
            if (isDebugEnabled()) {
                logger.debug('[LlguidanceLogitsProcessor] compute after stop', {
                    step: this.state.step,
                });
            }
            return logits;
        }

        if (isDebugEnabled()) {
            logger.debug('[LlguidanceLogitsProcessor] mask result', {
                step: this.state.step,
                result: summarizeMaskResult(result, vocabSize),
            });
        }

        if ('stop' in result && result.stop) {
            this.state.completed = true;
            if (isDebugEnabled()) {
                logger.debug('[LlguidanceLogitsProcessor] stopped by mask', {
                    step: this.state.step,
                });
            }
            return logits;
        }

        if ('ffTokens' in result && result.ffTokens?.length) {
            // Fast-forward splice: the grammar forces the next token, so ban
            // everything else instead of letting the model sample unconstrained.
            forceToken(logits, result.ffTokens[0], vocabSize);
            if (isDebugEnabled()) {
                logger.debug('[LlguidanceLogitsProcessor] forced splice token', {
                    step: this.state.step,
                    ffTokens: Array.from(result.ffTokens),
                    backtrack: 'backtrack' in result ? result.backtrack : undefined,
                });
            }
            return logits;
        }

        if ('mask' in result) {
            const applyStart = performance.now();
            if (isDebugEnabled()) {
                const applied = applyMask(logits, result.mask, result.vocabSize ?? vocabSize, true);
                logger.debug('[LlguidanceLogitsProcessor] mask applied', {
                    step: this.state.step,
                    ...applied,
                });
            } else {
                applyMask(logits, result.mask, result.vocabSize ?? vocabSize);
            }
            this.state.stats.applyMaskMs += performance.now() - applyStart;
        }

        return logits;
    }

    onTokenSampled(tokenId: number, batchIdx: number, inputIds: bigint[][]) {
        if (isDebugEnabled()) {
            logger.debug('[LlguidanceLogitsProcessor] token sampled', {
                step: this.state.step,
                tokenId,
                batchIdx,
                inputLength: inputIds[batchIdx]?.length,
                completed: this.state.completed,
            });
        }

        if (this.state.completed) return;

        const commitStart = performance.now();
        const result = this.state.interpreter.commitToken(tokenId);
        this.state.stats.commitTokenMs += performance.now() - commitStart;
        if (isDebugEnabled()) {
            logger.debug('[LlguidanceLogitsProcessor] token committed', {
                step: this.state.step,
                tokenId,
                result,
            });
        }

        if (result?.stop) {
            this.state.completed = true;
            if (isDebugEnabled()) {
                logger.debug('[LlguidanceLogitsProcessor] stopped by commit', {
                    step: this.state.step,
                    tokenId,
                });
            }
        }
    }

    onTokensSampled(tokenIds: number[], inputIds: bigint[][]) {
        if (isDebugEnabled()) {
            logger.debug('[LlguidanceLogitsProcessor] tokens sampled', {
                step: this.state.step,
                tokenIds,
                inputLengths: inputIds.map((ids) => ids.length),
                completed: this.state.completed,
            });
        }

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
        if (isDebugEnabled()) {
            logger.debug('[LlguidanceStoppingCriteria] call', {
                step: this.state.step,
                completed: this.state.completed,
                result,
                inputLengths: inputIds.map((ids) => ids.length),
            });
        }
        return result;
    }
}

function isDebugEnabled() {
    // Keep compatibility with Transformers.js releases that predate the LogLevel export.
    return env.logLevel <= 10;
}
