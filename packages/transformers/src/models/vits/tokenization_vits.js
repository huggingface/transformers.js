import { Decoder } from '@huggingface/tokenizers';

import { PreTrainedTokenizer } from '../../tokenization_utils.js';

class VitsDecoder extends Decoder {
    /** @type {Decoder['decode_chain']} */
    decode_chain(tokens) {
        let decoded = '';
        for (let i = 1; i < tokens.length; i += 2) {
            decoded += tokens[i];
        }
        return [decoded];
    }
}
export class VitsTokenizer extends PreTrainedTokenizer {
    constructor(tokenizerJSON, tokenizerConfig) {
        super(tokenizerJSON, tokenizerConfig);

        // Custom decoder function
        this._tokenizer.decoder = new VitsDecoder({ type: 'VitsDecoder' });
    }
}
