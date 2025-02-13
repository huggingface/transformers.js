// Although not strictly necessary, we delegate the tokenization to a worker thread to avoid
// any potential issues with the tokenizer blocking the main thread (especially for large inputs).

import { env, AutoTokenizer } from '@huggingface/transformers'

env.allowLocalModels = false;

// This is a map of all the tokenizer instances that we have loaded.
// model_id -> promise that resolves to tokenizer
const TOKENIZER_MAPPINGS = new Map();

// Listen for messages from the main thread
self.addEventListener('message', async (event) => {
    let tokenizerPromise = TOKENIZER_MAPPINGS.get(event.data.model_id);
    // Load the tokenizer if it hasn't been loaded yet
    if (!tokenizerPromise) {
        tokenizerPromise = AutoTokenizer.from_pretrained(event.data.model_id);

        TOKENIZER_MAPPINGS.set(event.data.model_id, new Promise((resolve) => {
            // Just for visualization purposes, we may need to modify the tokenizer slightly
            tokenizerPromise.then((tokenizer) => {
                // NOTE: We just remove the StripDecoder from the llama tokenizer
                switch (tokenizer.constructor.name) {
                    case 'LlamaTokenizer':
                    case 'Grok1Tokenizer':
                        // tokenizer.decoder.decoders.at(-1).constructor.name === 'StripDecoder'
                        tokenizer.decoder.decoders.pop();
                        break;
                    case 'T5Tokenizer':
                        tokenizer.decoder.addPrefixSpace = false;
                        break;
                }
                resolve(tokenizer);
            });
        }));
    }

    const tokenizer = await tokenizerPromise;

    const text = event.data.text;

    const start = performance.now();
    const token_ids = tokenizer.encode(text);
    const end = performance.now();
    console.log('[INFO]', `Tokenized ${text.length} characters in ${(end - start).toFixed(2)}ms`)

    let decoded = token_ids.map(x => tokenizer.decode([x]));

    let margins = [];

    // Minor post-processing for visualization purposes
    switch (tokenizer.constructor.name) {
        case 'BertTokenizer':
            margins = decoded.map((x, i) => i === 0 || x.startsWith('##') ? 0 : 8);
            decoded = decoded.map(x => x.replace('##', ''));
            break;
        case 'T5Tokenizer':
            if (decoded.length > 0 && decoded.length !== ' ') {
                decoded[0] = decoded[0].replace(/^ /, '');
            }
            break;
    }

    // Send the output back to the main thread
    self.postMessage({
        token_ids, decoded, margins
    });
});