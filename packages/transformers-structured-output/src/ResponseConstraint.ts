import { LogitsProcessor, LogitsProcessorList, StoppingCriteria, type Tensor } from '@huggingface/transformers';

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
    completed: boolean;
    constraint: TokenConstraint;
    processedInputLength?: number;
    mask?: Uint32Array;
};

const WHITESPACE_REPETITION_PENALTY = 1.2;
const MAX_CONSECUTIVE_WHITESPACE_TOKENS = 4;

export class ResponseConstraint {
    /**
     * Precomputes the tokenizer-derived data structures used by every
     * constraint. The first constraint per tokenizer otherwise pays this cost
     * (hundreds of milliseconds for large vocabularies) inside
     * `fromResponseFormat`; call this once after loading the model to pay it
     * early instead.
     */
    static warmup(tokenizer: TokenizerSource): void {
        prepareTokenizer(tokenizer);
    }

    static fromResponseFormat(tokenizer: TokenizerSource, responseFormat: ResponseFormat) {
        const state: GenerationState = {
            completed: false,
            constraint: createTokenConstraint(tokenizer, responseFormat),
        };
        const logits_processor = new LogitsProcessorList();
        logits_processor.push(new ConstraintLogitsProcessor(state));
        return {
            logits_processor,
            stopping_criteria: new ConstraintStoppingCriteria(state),
        };
    }
}

class ConstraintLogitsProcessor extends LogitsProcessor {
    constructor(private readonly state: GenerationState) {
        super();
    }

    _call(inputIds: bigint[][], logits: Tensor) {
        assertSingleSequence(inputIds.length);
        this.state.processedInputLength ??= inputIds[0].length;
        if (this.state.completed) return logits;
        const logitsVocabSize = logits.dims.at(-1);
        if (logitsVocabSize === undefined || !Number.isInteger(logitsVocabSize) || logitsVocabSize <= 0) {
            throw new Error('ResponseConstraint requires logits with a vocabulary dimension.');
        }
        const words = Math.ceil(logitsVocabSize / 32);
        if (this.state.mask?.length !== words) this.state.mask = new Uint32Array(words);
        if (!this.state.constraint.fillMask(this.state.mask)) {
            throw new Error('The constraint reached a dead end before producing a valid output.');
        }
        applyMask(logits, this.state.mask, this.state.constraint.vocabSize);
        const repeatedWhitespace = this.state.constraint.repeatedWhitespace();
        if (repeatedWhitespace !== undefined) {
            discourageRepeatedWhitespace(logits, repeatedWhitespace.tokenIds, repeatedWhitespace.count);
        }
        return logits;
    }
}

class ConstraintStoppingCriteria extends StoppingCriteria {
    constructor(private readonly state: GenerationState) {
        super();
    }

    _call(inputIds: ArrayLike<number | bigint>[]) {
        assertSingleSequence(inputIds.length);
        const input = inputIds[0];
        const start = this.state.processedInputLength ?? input.length;
        for (let i = start; i < input.length && !this.state.completed; ++i) {
            this.state.completed = this.state.constraint.commit(Number(input[i]));
        }
        this.state.processedInputLength = input.length;
        return [this.state.completed];
    }
}

function assertSingleSequence(batchSize: number): void {
    if (batchSize !== 1) {
        throw new Error(`ResponseConstraint currently supports batch size 1; received ${batchSize}.`);
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
