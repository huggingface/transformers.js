import { LogitsProcessor, LogitsProcessorList, type Tensor } from '@huggingface/transformers';

import {
    createTokenConstraint,
    prepareTokenizer,
    type JSONSchema,
    type TokenConstraint,
    type TokenizerSource,
} from './engine';
import { applyMask } from './utils/mask';

export type ResponseFormat =
    | { type: 'json_object' }
    | { type: 'json_schema'; json_schema: JSONSchema }
    | { type: 'regex'; regex: string };

type GenerationState = {
    constraint: TokenConstraint;
    enableThinking?: boolean;
    processedInputLength?: number;
    mask?: Uint32Array;
};

const WHITESPACE_REPETITION_PENALTY = 1.2;
const MAX_CONSECUTIVE_WHITESPACE_TOKENS = 4;

export class StructuredOutputProcessor extends LogitsProcessorList {
    private readonly state: GenerationState;

    /**
     * Precomputes the tokenizer-derived data structures used by every
     * constraint. The first processor per tokenizer otherwise pays this cost
     * (hundreds of milliseconds for large vocabularies) in its constructor;
     * call this once after loading the model to pay it early instead.
     */
    static warmup(tokenizer: TokenizerSource): void {
        prepareTokenizer(tokenizer);
    }

    constructor(tokenizer: TokenizerSource, responseFormat: ResponseFormat) {
        super();
        this.state = {
            constraint: createTokenConstraint(tokenizer, responseFormat),
        };
        this.push(new ConstraintLogitsProcessor(this.state));
    }
}

class ConstraintLogitsProcessor extends LogitsProcessor {
    constructor(private readonly state: GenerationState) {
        super();
    }

    setGenerationContext({ enable_thinking }: { enable_thinking?: boolean }): void {
        if (this.state.processedInputLength !== undefined)
            throw new Error('Generation context must be set before generation starts.');
        this.state.enableThinking = enable_thinking;
    }

    _call(inputIds: bigint[][], logits: Tensor) {
        assertSingleSequence(inputIds.length);
        const input = inputIds[0];
        this.syncInput(input);
        const logitsVocabSize = logits.dims.at(-1);
        const mask = this.fillMask(logitsVocabSize);
        applyMask(logits, mask, this.state.constraint.vocabSize);
        const repeatedWhitespace = this.state.constraint.repeatedWhitespace();
        if (repeatedWhitespace !== undefined) {
            discourageRepeatedWhitespace(logits, repeatedWhitespace.tokenIds, repeatedWhitespace.count);
        }
        return logits;
    }

    onTokensSampled(tokenIds: number[], inputIds: bigint[][]): void {
        assertSingleSequence(tokenIds.length);
        assertSingleSequence(inputIds.length);
        if (this.state.processedInputLength === undefined) return;
        const input = inputIds[0];
        this.syncInput(input.slice(0, -1));
        this.state.constraint.commit(tokenIds[0]);
        this.state.processedInputLength = input.length;
    }

    getRuntimeProcessor(inputIds: bigint[][]) {
        assertSingleSequence(inputIds.length);
        this.syncInput(inputIds[0]);
        return {
            op: 'token-mask' as const,
            getMask: (vocabSize: number) => this.fillMask(vocabSize, true),
        };
    }

    private syncInput(input: readonly bigint[]): void {
        if (this.state.processedInputLength === undefined) {
            this.state.constraint.initializePrompt(input, { enable_thinking: this.state.enableThinking });
        }
        const start = this.state.processedInputLength ?? input.length;
        for (let i = start; i < input.length; ++i) {
            if (this.state.constraint.commit(Number(input[i]))) {
                throw new Error(
                    'StructuredOutputProcessor observed the tokenizer EOS token after generation continued. Ensure the model generation config uses the same eos_token_id as the tokenizer.',
                );
            }
        }
        this.state.processedInputLength = input.length;
    }

    private fillMask(vocabSize: number | undefined, enforceWhitespaceLimit = false): Uint32Array {
        if (vocabSize === undefined || !Number.isInteger(vocabSize) || vocabSize <= 0) {
            throw new Error('StructuredOutputProcessor requires logits with a vocabulary dimension.');
        }
        const words = Math.ceil(vocabSize / 32);
        if (this.state.mask?.length !== words) this.state.mask = new Uint32Array(words);
        if (!this.state.constraint.fillMask(this.state.mask)) {
            throw new Error('The constraint reached a dead end before producing a valid output.');
        }
        const repeatedWhitespace = this.state.constraint.repeatedWhitespace();
        if (
            enforceWhitespaceLimit &&
            repeatedWhitespace !== undefined &&
            repeatedWhitespace.count >= MAX_CONSECUTIVE_WHITESPACE_TOKENS
        ) {
            for (const tokenId of repeatedWhitespace.tokenIds) {
                this.state.mask[tokenId >>> 5] &= ~(1 << (tokenId & 31));
            }
        }
        return this.state.mask;
    }
}

function assertSingleSequence(batchSize: number): void {
    if (batchSize !== 1) {
        throw new Error(`StructuredOutputProcessor currently supports batch size 1; received ${batchSize}.`);
    }
}

function discourageRepeatedWhitespace(logits: Tensor, tokenIds: readonly number[], count: number): void {
    const data = logits.data as Float32Array | Float64Array | number[];
    const stride = logits.dims.at(-1)!;
    const penalty = WHITESPACE_REPETITION_PENALTY ** count;
    for (let offset = 0; offset < data.length; offset += stride) {
        for (const tokenId of tokenIds) {
            const index = offset + tokenId;
            if (count >= MAX_CONSECUTIVE_WHITESPACE_TOKENS) {
                data[index] = -Infinity;
            } else if (data[index] < 0) {
                data[index] *= penalty;
            } else {
                data[index] /= penalty;
            }
        }
    }
}
