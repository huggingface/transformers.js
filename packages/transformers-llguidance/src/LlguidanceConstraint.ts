import { LogitsProcessor, LogitsProcessorList, StoppingCriteria, logger, type Tensor } from '@huggingface/transformers';
import { type LLGuidanceResponseFormat } from 'llguidance';

import { applyMask, summarizeMaskResult } from './utils/mask';
import { type GuidanceInterpreter, type GuidanceMaskResult, type LlguidanceState } from './utils/types';
import { type LlguidanceLoadOptions, loadLLGuidance } from './utils/wasm';

export type ResponseFormat = LLGuidanceResponseFormat;
export type { LlguidanceLoadOptions };

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
