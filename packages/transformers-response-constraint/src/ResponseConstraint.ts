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
    mask?: Uint32Array;
};

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
        return logits;
    }

    onTokensSampled(tokenIds: number[], inputIds: bigint[][]) {
        assertSingleSequence(tokenIds.length);
        assertSingleSequence(inputIds.length);
        if (!this.state.completed) this.state.completed = this.state.constraint.commit(tokenIds[0]);
    }
}

class ConstraintStoppingCriteria extends StoppingCriteria {
    constructor(private readonly state: GenerationState) {
        super();
    }

    _call(inputIds: ArrayLike<unknown>[]) {
        assertSingleSequence(inputIds.length);
        return [this.state.completed];
    }
}

function assertSingleSequence(batchSize: number): void {
    if (batchSize !== 1) {
        throw new Error(`ResponseConstraint currently supports batch size 1; received ${batchSize}.`);
    }
}
