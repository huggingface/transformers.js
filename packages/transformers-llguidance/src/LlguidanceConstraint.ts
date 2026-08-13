import {
    env,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    logger,
    type Tensor,
} from '@huggingface/transformers';
import {
    loadBundledLLGuidance,
    type JSONSchema,
    type LLGuidanceMaskResult,
    type LLGuidanceTokenizerSource,
} from 'llguidance';

import { applyMask, forceTokens, summarizeMaskResult } from './utils/mask';
import { type LlguidanceState, type LlguidanceStats } from './utils/types';

export type ResponseFormat =
    | { type: 'json_object' }
    | { type: 'json_schema'; json_schema: JSONSchema }
    | { type: 'regex'; regex: string };
export type { LlguidanceStats };

export class LlguidanceConstraint {
    static async fromResponseFormat(tokenizer: LLGuidanceTokenizerSource, response_format: ResponseFormat) {
        logger.debug('[LlguidanceConstraint] loading llguidance', {
            response_format,
        });

        const runtime = await loadBundledLLGuidance();
        const eosTokenIds = getEosTokenIds(tokenizer);
        const interpreter = runtime.createInterpreter({
            tokenizer,
            response_format,
        });

        logger.debug('[LlguidanceConstraint] interpreter created');

        const state: LlguidanceState = {
            completed: false,
            disposed: false,
            eosTokenIds,
            interpreter,
            step: 0,
            stats: {
                steps: 0,
                computeMaskMs: 0,
                applyMaskMs: 0,
                commitTokenMs: 0,
                trieNodesVisited: 0,
                sharedJsonMaskCacheHits: 0,
                sharedJsonMaskCacheMisses: 0,
            },
        };

        const logits_processor = new LogitsProcessorList();
        logits_processor.push(new LlguidanceLogitsProcessor(state));

        return {
            logits_processor,
            stopping_criteria: new LlguidanceStoppingCriteria(state),
            stats: state.stats,
            dispose: () => disposeState(state),
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
        assertSingleSequence(_inputIds.length, this.state);

        if (this.state.completed) {
            if (isDebugEnabled()) {
                logger.debug('[LlguidanceLogitsProcessor] skip completed', {
                    step: this.state.step,
                });
            }
            return logits;
        }
        if (this.state.disposed) {
            throw new Error('LlguidanceConstraint has been disposed.');
        }

        this.state.step++;
        this.state.stats.steps = this.state.step;
        const vocabSize = logits.dims.at(-1);
        if (vocabSize === undefined || !Number.isInteger(vocabSize) || vocabSize <= 0) {
            throw failState(this.state, 'LlguidanceConstraint requires logits with a vocabulary dimension.');
        }
        if (isDebugEnabled()) {
            logger.debug('[LlguidanceLogitsProcessor] compute mask', {
                step: this.state.step,
                logitsDims: logits.dims,
                vocabSize,
            });
        }

        let result: LLGuidanceMaskResult;
        const maskStart = performance.now();
        const maskWords = Math.ceil(vocabSize / 32);
        if (this.state.maskBuffer?.length !== maskWords) {
            this.state.maskBuffer = new Uint32Array(maskWords);
        }
        try {
            result = this.state.interpreter.computeMaskInto(this.state.maskBuffer);
        } catch (error) {
            disposeState(this.state);
            throw error;
        }
        this.state.stats.computeMaskMs += performance.now() - maskStart;
        const maskInstrumentation = this.state.interpreter.maskInstrumentation?.();
        if (maskInstrumentation) {
            this.state.stats.trieNodesVisited = maskInstrumentation.trieNodesVisited;
            this.state.stats.sharedJsonMaskCacheHits = maskInstrumentation.sharedJsonMaskCacheHits;
            this.state.stats.sharedJsonMaskCacheMisses = maskInstrumentation.sharedJsonMaskCacheMisses;
        }

        if (isDebugEnabled()) {
            logger.debug('[LlguidanceLogitsProcessor] mask result', {
                step: this.state.step,
                result: summarizeMaskResult(result),
            });
        }

        if ('stop' in result && result.stop) {
            this.state.stats.stopReason = result.reason;
            if (result.reason === 'dead_end') {
                throw failState(this.state, 'llguidance reached a dead end before satisfying the constraint.');
            }
            this.state.completed = true;
            try {
                forceTokens(logits, this.state.eosTokenIds);
            } catch (error) {
                disposeState(this.state);
                throw error;
            }
            disposeState(this.state);
            if (isDebugEnabled()) {
                logger.debug('[LlguidanceLogitsProcessor] stopped by mask', {
                    step: this.state.step,
                });
            }
            return logits;
        }

        if ('mask' in result) {
            const applyStart = performance.now();
            try {
                if (isDebugEnabled()) {
                    const applied = applyMask(logits, result.mask, result.vocabSize, true);
                    logger.debug('[LlguidanceLogitsProcessor] mask applied', {
                        step: this.state.step,
                        ...applied,
                    });
                } else {
                    applyMask(logits, result.mask, result.vocabSize);
                }
            } catch (error) {
                disposeState(this.state);
                throw error;
            }
            this.state.stats.applyMaskMs += performance.now() - applyStart;
        }

        return logits;
    }

    getRuntimeGenerationProcessor() {
        return {
            op: 'token-mask' as const,
            getMask: (vocabSize: number) => this.computeRuntimeMask(vocabSize),
        };
    }

    private computeRuntimeMask(vocabSize: number): Uint32Array {
        if (!Number.isInteger(vocabSize) || vocabSize <= 0) {
            throw failState(this.state, 'LlguidanceConstraint requires a positive vocabulary size.');
        }
        if (this.state.completed) return tokenMask(vocabSize, this.state.eosTokenIds);
        if (this.state.disposed) throw new Error('LlguidanceConstraint has been disposed.');

        this.state.step++;
        this.state.stats.steps = this.state.step;
        const maskWords = Math.ceil(vocabSize / 32);
        if (this.state.maskBuffer?.length !== maskWords) this.state.maskBuffer = new Uint32Array(maskWords);

        let result: LLGuidanceMaskResult;
        const maskStart = performance.now();
        try {
            result = this.state.interpreter.computeMaskInto(this.state.maskBuffer);
        } catch (error) {
            disposeState(this.state);
            throw error;
        }
        this.state.stats.computeMaskMs += performance.now() - maskStart;
        const instrumentation = this.state.interpreter.maskInstrumentation?.();
        if (instrumentation) {
            this.state.stats.trieNodesVisited = instrumentation.trieNodesVisited;
            this.state.stats.sharedJsonMaskCacheHits = instrumentation.sharedJsonMaskCacheHits;
            this.state.stats.sharedJsonMaskCacheMisses = instrumentation.sharedJsonMaskCacheMisses;
        }

        if ('stop' in result && result.stop) {
            this.state.stats.stopReason = result.reason;
            if (result.reason === 'dead_end') {
                throw failState(this.state, 'llguidance reached a dead end before satisfying the constraint.');
            }
            this.state.completed = true;
            disposeState(this.state);
            return tokenMask(vocabSize, this.state.eosTokenIds);
        }
        return this.state.maskBuffer;
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
        if (this.state.disposed) {
            throw new Error('LlguidanceConstraint has been disposed.');
        }

        const commitStart = performance.now();
        let result;
        try {
            result = this.state.interpreter.commitToken(tokenId);
        } catch (error) {
            disposeState(this.state);
            throw error;
        }
        this.state.stats.commitTokenMs += performance.now() - commitStart;
        if (isDebugEnabled()) {
            logger.debug('[LlguidanceLogitsProcessor] token committed', {
                step: this.state.step,
                tokenId,
                result,
            });
        }

        if (result.backtrack > 0) {
            throw failState(
                this.state,
                `llguidance requested backtracking by ${result.backtrack} token(s), which Transformers.js does not support.`,
            );
        }

        if (result.stop) {
            this.state.completed = true;
            this.state.stats.stopReason = 'accepted';
            disposeState(this.state);
            if (isDebugEnabled()) {
                logger.debug('[LlguidanceLogitsProcessor] stopped by commit', {
                    step: this.state.step,
                    tokenId,
                });
            }
        }
    }

    onTokensSampled(tokenIds: number[], inputIds: bigint[][]) {
        assertSingleSequence(tokenIds.length, this.state);
        assertSingleSequence(inputIds.length, this.state);
        if (isDebugEnabled()) {
            logger.debug('[LlguidanceLogitsProcessor] tokens sampled', {
                step: this.state.step,
                tokenIds,
                inputLengths: inputIds.map((ids) => ids.length),
                completed: this.state.completed,
            });
        }

        this.onTokenSampled(tokenIds[0], 0, inputIds);
    }
}

function tokenMask(vocabSize: number, tokenIds: readonly number[]): Uint32Array {
    const mask = new Uint32Array(Math.ceil(vocabSize / 32));
    for (const tokenId of tokenIds) {
        if (tokenId >= 0 && tokenId < vocabSize) mask[tokenId >> 5] |= 1 << (tokenId & 31);
    }
    return mask;
}

class LlguidanceStoppingCriteria extends StoppingCriteria {
    private state: LlguidanceState;

    constructor(state: LlguidanceState) {
        super();
        this.state = state;
    }

    _call(inputIds: ArrayLike<unknown>[]) {
        assertSingleSequence(inputIds.length, this.state);
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
    return env.logLevel <= DEBUG_LOG_LEVEL;
}

const DEBUG_LOG_LEVEL = 10;

function assertSingleSequence(batchSize: number, state: LlguidanceState) {
    if (batchSize !== 1) {
        throw failState(state, `LlguidanceConstraint currently supports batch size 1; received ${batchSize}.`);
    }
}

function failState(state: LlguidanceState, message: string) {
    disposeState(state);
    return new Error(message);
}

function disposeState(state: LlguidanceState) {
    if (state.disposed) return;
    state.disposed = true;
    state.maskBuffer = undefined;
    state.interpreter.dispose();
}

function getEosTokenIds(tokenizer: LLGuidanceTokenizerSource) {
    const source = tokenizer as Record<string, unknown>;
    const values = [
        source.eos_token_id,
        source.eosTokenId,
        source.eos_token_ids,
        source.eosTokenIds,
        source.eos_token,
        source.eos_tokens,
    ];

    if (typeof source.eosToken === 'function') {
        values.push(source.eosToken.call(tokenizer));
    }

    return [
        ...new Set(
            values
                .flat()
                .filter((value): value is number => typeof value === 'number' && Number.isInteger(value) && value >= 0),
        ),
    ];
}
