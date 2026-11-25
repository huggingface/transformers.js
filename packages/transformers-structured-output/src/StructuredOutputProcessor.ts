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

export type StructuredOutputOptions = {
    /**
     * Whether the model may think before the constrained answer.
     * - `undefined` (default): thinking only continues if the prompt already started it (an open
     *   thinking prefill or a partially written opener). Otherwise behaves like `false`.
     * - `false`: no thinking region; constrain the first content token. An unfinished thinking
     *   prefill raises an error.
     * - `true`: the model may open the optional thinking region before the answer.
     * Thinking is opt-in because templates such as Gemma 4 omit thinking markers rather than write a
     * closed block, so the prompt alone cannot tell that thinking is off. See #1791.
     */
    enable_thinking?: boolean;
};

type GenerationState = {
    constraint: TokenConstraint;
    enableThinking?: boolean;
    processedInputLength?: number;
    mask?: Uint32Array;
};

const WHITESPACE_REPETITION_PENALTY = 1.2;
const MAX_CONSECUTIVE_WHITESPACE_TOKENS = 4;

export class StructuredOutputProcessor extends LogitsProcessorList {
    /**
     * Precomputes the tokenizer-derived data structures used by every
     * constraint. The first processor per tokenizer otherwise pays this cost
     * (hundreds of milliseconds for large vocabularies) in its constructor;
     * call this once after loading the model to pay it early instead.
     */
    static warmup(tokenizer: TokenizerSource): void {
        prepareTokenizer(tokenizer);
    }

    constructor(tokenizer: TokenizerSource, responseFormat: ResponseFormat, options: StructuredOutputOptions = {}) {
        super();
        const state: GenerationState = {
            constraint: createTokenConstraint(tokenizer, responseFormat),
            enableThinking: options.enable_thinking,
        };
        this.push(new ConstraintLogitsProcessor(state));
    }
}

class ConstraintLogitsProcessor extends LogitsProcessor {
    constructor(private readonly state: GenerationState) {
        super();
    }

    _call(inputIds: bigint[][], logits: Tensor) {
        assertSingleSequence(inputIds.length);
        const input = inputIds[0];
        if (this.state.processedInputLength === undefined) {
            this.state.constraint.initializePrompt(input, { enable_thinking: this.state.enableThinking });
        } else if (input.length < this.state.processedInputLength) {
            throw new Error(
                'StructuredOutputProcessor was reused for a new generation. Create a new processor for every generate() call.',
            );
        }
        const start = this.state.processedInputLength ?? input.length;
        for (let i = start; i < input.length; ++i) {
            try {
                if (this.state.constraint.commit(Number(input[i]))) {
                    throw new Error(
                        'StructuredOutputProcessor observed the tokenizer EOS token after generation continued. Ensure the model generation config uses the same eos_token_id as the tokenizer.',
                    );
                }
            } catch (error) {
                const message = error instanceof Error ? error.message : String(error);
                throw new Error(
                    `${message} The StructuredOutputProcessor may have been reused for another generation; create a new processor for every generate() call.`,
                    { cause: error },
                );
            }
        }
        this.state.processedInputLength = input.length;
        const logitsVocabSize = logits.dims.at(-1);
        if (logitsVocabSize === undefined || !Number.isInteger(logitsVocabSize) || logitsVocabSize <= 0) {
            throw new Error('StructuredOutputProcessor requires logits with a vocabulary dimension.');
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
