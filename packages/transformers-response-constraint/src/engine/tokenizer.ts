import type { TokenizerSource } from './types';

type RecordLike = Record<string, unknown>;

export type TokenizerData = {
    tokens: Uint8Array[];
    eosTokenId: number;
    specialTokenIds: Set<number>;
};

const encoder = new TextEncoder();
let byteLevelMap: Map<string, number> | undefined;

export function extractTokenizer(tokenizer: TokenizerSource): TokenizerData {
    const source = asRecord(tokenizer, 'tokenizer');
    if (Array.isArray(source.tokens)) {
        return normalizeDirectTokenizer(source);
    }

    const tokenizerJson = getTokenizerJson(source);
    const vocabulary = getVocabulary(source, tokenizerJson);
    if (vocabulary === undefined) {
        throw new TypeError('Could not extract the tokenizer vocabulary.');
    }

    const vocabularyTokens = Object.keys(vocabulary);
    let size = 0;
    for (const token of vocabularyTokens) {
        const id = Number(vocabulary[token]);
        if (!Number.isInteger(id) || id < 0) throw new TypeError(`Tokenizer has an invalid token ID for ${token}.`);
        if (id + 1 > size) size = id + 1;
    }
    const tokenBytes = tokenBytesConverter(source, tokenizerJson);
    const tokens = new Array<Uint8Array | undefined>(size);
    for (const token of vocabularyTokens) {
        const id = Number(vocabulary[token]);
        tokens[id] = tokenBytes(token, id);
    }

    const addedTokens = field(tokenizerJson, 'added_tokens');
    if (Array.isArray(addedTokens)) {
        for (const added of addedTokens) {
            if (!isRecord(added) || !Number.isInteger(added.id)) continue;
            while (tokens.length <= Number(added.id)) tokens.push(undefined);
            const id = Number(added.id);
            tokens[id] = tokenBytes(typeof added.content === 'string' ? added.content : '', id);
        }
    }
    for (let id = 0; id < tokens.length; ++id) {
        if (tokens[id] === undefined) throw new Error(`Tokenizer vocabulary is missing token ID ${id}.`);
    }

    const eosTokenId = tokenId(source, tokenizerJson, ['eos_token_id', 'eosTokenId', 'eos_token', 'eosToken']);
    if (eosTokenId === undefined) throw new TypeError('Tokenizer does not expose an EOS token ID.');
    const specialTokenIds = new Set<number>([eosTokenId]);
    for (const value of [
        source.special_token_ids,
        source.specialTokenIds,
        source.all_special_ids,
        source.allSpecialIds,
    ]) {
        if (Array.isArray(value)) for (const id of value) if (Number.isInteger(id)) specialTokenIds.add(Number(id));
    }
    if (Array.isArray(addedTokens)) {
        for (const added of addedTokens) {
            if (isRecord(added) && added.special === true && Number.isInteger(added.id))
                specialTokenIds.add(Number(added.id));
        }
    }
    return { tokens: tokens as Uint8Array[], eosTokenId, specialTokenIds };
}

function normalizeDirectTokenizer(source: RecordLike): TokenizerData {
    const configuredTokens = source.tokens as unknown[];
    const tokens = configuredTokens.map((token, id) => {
        if (!(token instanceof Uint8Array) && !Array.isArray(token)) {
            throw new TypeError(`Tokenizer token ${id} must be a byte array.`);
        }
        const values = Array.from(token as ArrayLike<number>);
        if (values.some((byte) => !Number.isInteger(byte) || byte < 0 || byte > 255))
            throw new TypeError(`Tokenizer token ${id} is invalid.`);
        return Uint8Array.from(values);
    });
    const eosTokenId = Number(source.eosTokenId ?? source.eos_token_id);
    if (!Number.isInteger(eosTokenId) || eosTokenId < 0 || eosTokenId >= tokens.length) {
        throw new TypeError('A valid eos_token_id is required with tokenizer tokens.');
    }
    const configured = source.specialTokenIds ?? source.special_token_ids;
    const specialTokenIds = new Set<number>([eosTokenId]);
    if (Array.isArray(configured)) for (const id of configured) specialTokenIds.add(Number(id));
    return { tokens, eosTokenId, specialTokenIds };
}

function getTokenizerJson(source: RecordLike): unknown {
    const value = source._tokenizerJSON ?? source.tokenizerJSON ?? source.tokenizer_json;
    return typeof value === 'string' ? JSON.parse(value) : value;
}

function getVocabulary(source: RecordLike, tokenizerJson: unknown): RecordLike | undefined {
    const modelVocabulary = field(field(tokenizerJson, 'model'), 'vocab');
    if (isRecord(modelVocabulary)) return modelVocabulary;
    for (const name of ['get_vocab', 'getVocab']) {
        const method = source[name];
        if (typeof method !== 'function') continue;
        const value = method.call(source, true);
        if (value instanceof Map) return Object.fromEntries(value);
        if (isRecord(value)) return value;
    }
    return isRecord(source.vocab) ? source.vocab : undefined;
}

function tokenBytesConverter(source: RecordLike, tokenizerJson: unknown): (token: string, id: number) => Uint8Array {
    const decoder = field(tokenizerJson, 'decoder');
    if (field(decoder, 'type') === 'ByteLevel') return byteLevelTokenBytes;

    const decode = source.decode;
    if (typeof decode === 'function') {
        const byteFallback = hasComponent(decoder, 'ByteFallback');
        return (token, id) => {
            const fallback = byteFallback ? /^<0x([\da-f]{2})>$/i.exec(token) : null;
            if (fallback) return Uint8Array.of(Number.parseInt(fallback[1], 16));
            const decoded = decode.call(source, [id], {
                skip_special_tokens: false,
                clean_up_tokenization_spaces: false,
            });
            if (typeof decoded !== 'string') throw new TypeError(`Tokenizer.decode([${id}]) must return a string.`);
            return encoder.encode(decoded);
        };
    }

    if (decoder !== undefined && decoder !== null && field(decoder, 'type') !== 'ByteLevel') {
        throw new TypeError('Tokenizer decoder semantics require a tokenizer with a decode() method.');
    }
    if (hasComponent(decoder, 'ByteLevel') || hasComponent(field(tokenizerJson, 'pre_tokenizer'), 'ByteLevel')) {
        return byteLevelTokenBytes;
    }
    const modelType = field(field(tokenizerJson, 'model'), 'type');
    const sentencePiece = modelType === 'Unigram' || modelType === 'SentencePiece';
    const prefix = field(field(tokenizerJson, 'model'), 'continuing_subword_prefix');
    return (token) => {
        const fallback = /^<0x([\da-f]{2})>$/i.exec(token);
        if (fallback) return Uint8Array.of(Number.parseInt(fallback[1], 16));
        if (sentencePiece || token.includes('▁')) return encoder.encode(token.replaceAll('▁', ' '));
        return encoder.encode(
            typeof prefix === 'string' && token.startsWith(prefix) ? token.slice(prefix.length) : token,
        );
    };
}

function byteLevelTokenBytes(token: string): Uint8Array {
    const fallback = /^<0x([\da-f]{2})>$/i.exec(token);
    if (fallback) return Uint8Array.of(Number.parseInt(fallback[1], 16));
    const map = getByteLevelMap();
    const bytes: number[] = [];
    for (const character of token) {
        const byte = map.get(character);
        if (byte === undefined) bytes.push(...encoder.encode(character));
        else bytes.push(byte);
    }
    return Uint8Array.from(bytes);
}

function getByteLevelMap(): Map<string, number> {
    if (byteLevelMap !== undefined) return byteLevelMap;
    const visible = new Set<number>();
    for (let code = 33; code <= 126; ++code) visible.add(code);
    for (let code = 161; code <= 172; ++code) visible.add(code);
    for (let code = 174; code <= 255; ++code) visible.add(code);
    let extra = 0;
    byteLevelMap = new Map();
    for (let byte = 0; byte < 256; ++byte) {
        byteLevelMap.set(String.fromCharCode(visible.has(byte) ? byte : 256 + extra++), byte);
    }
    return byteLevelMap;
}

function tokenId(source: RecordLike, tokenizerJson: unknown, keys: string[]): number | undefined {
    const vocabulary = getVocabulary(source, tokenizerJson);
    for (const key of keys) {
        const value = source[key] ?? field(tokenizerJson, key);
        if (Number.isInteger(value)) return Number(value);
        if (typeof value === 'string' && Number.isInteger(vocabulary?.[value])) return Number(vocabulary![value]);
    }
    return undefined;
}

function hasComponent(value: unknown, type: string): boolean {
    if (Array.isArray(value)) return value.some((item) => hasComponent(item, type));
    return isRecord(value) && (value.type === type || Object.values(value).some((item) => hasComponent(item, type)));
}

function field(value: unknown, key: string): unknown {
    return isRecord(value) ? value[key] : undefined;
}

function asRecord(value: unknown, name: string): RecordLike {
    if ((typeof value !== 'object' && typeof value !== 'function') || value === null) {
        throw new TypeError(`${name} must be an object.`);
    }
    return value as RecordLike;
}

function isRecord(value: unknown): value is RecordLike {
    return value !== null && typeof value === 'object' && !Array.isArray(value);
}
