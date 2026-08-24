import type { ConstraintState, JSONSchema } from './types';

type Schema = boolean | Record<string, unknown>;
type Kind = 'null' | 'boolean' | 'number' | 'integer' | 'string' | 'array' | 'object';
type JsonNode =
    | { kind: 'null'; value: null }
    | { kind: 'boolean'; value: boolean }
    | { kind: 'number'; value: number; raw: string }
    | { kind: 'string'; value: string }
    | { kind: 'array'; value: unknown[]; items: JsonNode[] }
    | { kind: 'object'; value: Record<string, unknown>; entries: Array<{ key: string; node: JsonNode }> };
type Guidance = {
    itemSeparator: string;
    keySeparator: string;
    itemBytes: Uint8Array;
    keyBytes: Uint8Array;
    whitespaceFlexible: boolean;
};
type Mode =
    | 'value'
    | 'array-value'
    | 'object-key'
    | 'colon'
    | 'item-separator'
    | 'after-value'
    | 'string'
    | 'key-string'
    | 'escape'
    | 'key-escape'
    | 'unicode'
    | 'key-unicode'
    | 'number'
    | 'literal'
    | 'done'
    | 'dead';

type ObjectFrame = {
    kind: 'object';
    schema: Schema;
    seen: ReadonlySet<string>;
    entries: ReadonlyArray<{ key: string; node: JsonNode }>;
    key?: string;
    childSchema?: Schema;
};

type ArrayFrame = {
    kind: 'array';
    schema: Schema;
    length: number;
    items: readonly JsonNode[];
};

type Frame = ObjectFrame | ArrayFrame;

type JsonState = {
    mode: Mode;
    schema: Schema;
    stack: readonly Frame[];
    bytes?: readonly number[];
    // Code points started in the current string and UTF-8 continuation bytes
    // still expected (-1 once the byte sequence can no longer decode). Both are
    // pure functions of `bytes`, maintained incrementally so that length checks
    // do not have to re-decode the prefix on every byte.
    stringLength?: number;
    stringPending?: number;
    text?: string;
    literal?: string;
    literalValue?: null | boolean;
    index?: number;
    highSurrogate?: number;
    guidance: Guidance;
};

const DEFAULT_GUIDANCE: Guidance = {
    itemSeparator: ',',
    keySeparator: ':',
    itemBytes: Uint8Array.of(0x2c),
    keyBytes: Uint8Array.of(0x3a),
    whitespaceFlexible: true,
};
const DEAD: JsonState = { mode: 'dead', schema: false, stack: [], guidance: DEFAULT_GUIDANCE };
const encoder = new TextEncoder();
const decoder = new TextDecoder('utf-8', { fatal: true });
const schemaIds = new WeakMap<object, number>();
const propertyKeyBytes = new WeakMap<object, ReadonlyArray<{ key: string; bytes: Uint8Array }> | null>();
const finiteStringCache = new WeakMap<object, ReadonlyArray<{ value: string; bytes: Uint8Array }> | null>();
let nextSchemaId = 0;
const SIMPLE_ESCAPES: Record<number, string> = {
    0x22: '"',
    0x2f: '/',
    0x5c: '\\',
    0x62: '\b',
    0x66: '\f',
    0x6e: '\n',
    0x72: '\r',
    0x74: '\t',
};
const ASSERTED_FORMATS = new Set([
    'date',
    'time',
    'date-time',
    'duration',
    'email',
    'hostname',
    'ipv4',
    'ipv6',
    'uuid',
    'uri',
    'uri-reference',
    'regex',
    'json-pointer',
    'relative-json-pointer',
]);

export function compileJsonSchema(schema: JSONSchema, stringKeyClamp = Infinity): ConstraintState<JsonState> {
    registerSchemaContext(schema, schema);
    checkSchema(schema, '$');
    checkReferences(schema);
    const guidance = guidanceFrom(schema);
    return {
        initial: { mode: 'value', schema, stack: [], guidance },
        transition,
        viable: (state) => state !== DEAD,
        accepting: isAccepting,
        stringCapacity,
        maskKey: (state) => stateMaskKey(state, stringKeyClamp),
    };
}

function stringCapacity(state: JsonState): number | undefined {
    if (state.mode !== 'string' || state.highSurrogate !== undefined || finiteStringValues(state.schema) !== null)
        return undefined;
    if ((state.stringPending ?? 0) !== 0) return undefined;
    const maximum = directStringMaxLength(state.schema);
    if (maximum === undefined) return Infinity;
    return maximum - (state.stringLength ?? 0);
}

function stateMaskKey(state: JsonState, stringKeyClamp: number): string | undefined {
    if (state === DEAD) return undefined;
    const frames = state.stack.map((frame) => {
        if (frame.kind === 'object') {
            const entries = [...frame.entries]
                .sort((left, right) => left.key.localeCompare(right.key))
                .map((entry) => `${JSON.stringify(entry.key)}:${nodeKey(entry.node)}`)
                .join(',');
            return `o${schemaId(frame.schema)}[${entries}]${frame.key === undefined ? '' : `:${JSON.stringify(frame.key)}:${schemaId(frame.childSchema!)}`}`;
        }
        return `a${schemaId(frame.schema)}[${frame.items.map(nodeKey).join(',')}]`;
    });
    const key = [
        state.mode,
        schemaId(state.schema),
        state.bytes === undefined ? '' : (stringLengthKey(state, stringKeyClamp) ?? bytesKey(state.bytes)),
        state.text ?? '',
        state.literal ?? '',
        state.index ?? '',
        state.highSurrogate ?? '',
        ...frames,
    ].join('|');
    return key.length <= 2048 ? key : undefined;
}

// While inside a plain length-constrained string, the token mask depends only on
// the remaining capacity (clamped to the longest token) and whether minLength is
// already satisfied — not on the actual bytes typed so far. Keying on that makes
// every string-content step after the first share one cached mask. This is only
// sound when nothing can inspect the string's content: neither the string schema
// itself (pattern/format/const/enum or composition keywords) nor any enclosing
// frame (cross-value keywords such as uniqueItems, contains, or conditionals).
const CONTENT_DEPENDENT_STRING_KEYWORDS = [
    'pattern',
    'format',
    'const',
    'enum',
    '$ref',
    'allOf',
    'anyOf',
    'oneOf',
    'not',
    'if',
];
const CONTENT_DEPENDENT_FRAME_KEYWORDS = [
    '$ref',
    'allOf',
    'anyOf',
    'oneOf',
    'not',
    'if',
    'uniqueItems',
    'contains',
    'dependentSchemas',
    'dependencies',
];
const contentIndependentStrings = new WeakMap<object, boolean>();
const contentNeutralFrames = new WeakMap<object, boolean>();

function isContentIndependent(schema: Schema, keywords: readonly string[], cache: WeakMap<object, boolean>): boolean {
    if (schema === true) return true;
    if (schema === false) return false;
    let result = cache.get(schema);
    if (result === undefined) {
        result = keywords.every((keyword) => schema[keyword] === undefined);
        cache.set(schema, result);
    }
    return result;
}

function stringLengthKey(state: JsonState, clamp: number): string | undefined {
    if (state.mode !== 'string' || state.highSurrogate !== undefined) return undefined;
    if (!isContentIndependent(state.schema, CONTENT_DEPENDENT_STRING_KEYWORDS, contentIndependentStrings)) {
        return undefined;
    }
    for (const frame of state.stack) {
        if (!isContentIndependent(frame.schema, CONTENT_DEPENDENT_FRAME_KEYWORDS, contentNeutralFrames)) {
            return undefined;
        }
    }
    if ((state.stringPending ?? 0) !== 0) return undefined;
    const length = state.stringLength ?? 0;
    const maximum = directStringMaxLength(state.schema);
    const remaining = Math.min(maximum === undefined ? Infinity : maximum - length, clamp);
    const minimum =
        state.schema !== true && state.schema !== false && typeof state.schema.minLength === 'number'
            ? state.schema.minLength
            : 0;
    return `#${remaining === Infinity ? 'inf' : remaining}:${length >= minimum ? '' : length}`;
}

function nodeKey(node: JsonNode): string {
    if (node.kind === 'number') return `n${node.raw}`;
    if (node.kind === 'string') return `s${JSON.stringify(node.value)}`;
    if (node.kind === 'boolean') return node.value ? 't' : 'f';
    if (node.kind === 'null') return 'z';
    if (node.kind === 'array') return `[${node.items.map(nodeKey).join(',')}]`;
    return `{${[...node.entries]
        .sort((left, right) => left.key.localeCompare(right.key))
        .map((entry) => `${JSON.stringify(entry.key)}:${nodeKey(entry.node)}`)
        .join(',')}}`;
}

function bytesKey(bytes: readonly number[]): string {
    let result = '';
    for (const byte of bytes) result += byte.toString(16).padStart(2, '0');
    return result;
}

function schemaId(schema: Schema): string | number {
    if (schema === true) return 't';
    if (schema === false) return 'f';
    let id = schemaIds.get(schema);
    if (id === undefined) {
        id = nextSchemaId++;
        schemaIds.set(schema, id);
    }
    return id;
}

function transition(state: JsonState, byte: number): JsonState {
    if (state === DEAD) return DEAD;
    if (
        state.guidance.whitespaceFlexible &&
        isWhitespace(byte) &&
        allowsWhitespace(state.mode) &&
        !separatorExpects(state, byte)
    )
        return state;

    switch (state.mode) {
        case 'value':
        case 'array-value':
            if (state.mode === 'array-value' && byte === 0x5d) return closeArray(state);
            if (state.mode === 'array-value' && exceedsMaxItems(state.stack.at(-1) as ArrayFrame)) return DEAD;
            return startValue(state, byte);
        case 'object-key':
            if (byte === 0x7d) return closeObject(state);
            return byte === 0x22
                ? { ...state, mode: 'key-string', bytes: [], stringLength: 0, stringPending: 0 }
                : DEAD;
        case 'colon':
            return separatorByte(state, byte, state.guidance.keyBytes, 'value');
        case 'item-separator':
            return separatorByte(
                state,
                byte,
                state.guidance.itemBytes,
                state.stack.at(-1)?.kind === 'object' ? 'object-key' : 'array-value',
            );
        case 'after-value':
            return afterValue(state, byte);
        case 'string':
        case 'key-string':
            return stringByte(state, byte);
        case 'escape':
        case 'key-escape':
            return escapeByte(state, byte);
        case 'unicode':
        case 'key-unicode':
            return unicodeByte(state, byte);
        case 'number':
            return numberByte(state, byte);
        case 'literal':
            return literalByte(state, byte);
        case 'done':
        case 'dead':
            return DEAD;
    }
}

function startValue(state: JsonState, byte: number): JsonState {
    const kind = byteKind(byte);
    if (kind === undefined) return DEAD;
    const schema = selectSchema(state.schema, kind);
    if (schema === null) return DEAD;

    if (byte === 0x22) return { ...state, mode: 'string', schema, bytes: [], stringLength: 0, stringPending: 0 };
    if (byte === 0x7b) {
        return {
            mode: 'object-key',
            schema,
            stack: [...state.stack, { kind: 'object', schema, seen: new Set<string>(), entries: [] }],
            guidance: state.guidance,
        };
    }
    if (byte === 0x5b) {
        return {
            mode: 'array-value',
            schema: itemSchema(schema, 0),
            stack: [
                ...state.stack,
                {
                    kind: 'array',
                    schema,
                    length: 0,
                    items: [],
                },
            ],
            guidance: state.guidance,
        };
    }
    if (byte === 0x74) return { ...state, mode: 'literal', schema, literal: 'true', literalValue: true, index: 1 };
    if (byte === 0x66) return { ...state, mode: 'literal', schema, literal: 'false', literalValue: false, index: 1 };
    if (byte === 0x6e) return { ...state, mode: 'literal', schema, literal: 'null', literalValue: null, index: 1 };
    const text = String.fromCharCode(byte);
    if (!integerPrefixViable(schema, text)) return DEAD;
    return { ...state, mode: 'number', schema, text };
}

function isAccepting(state: JsonState): boolean {
    if (state.mode === 'done') return true;
    return (
        state.mode === 'number' &&
        state.stack.length === 0 &&
        numberComplete(state.text!) &&
        validateNode(state.schema, { kind: 'number', value: Number(state.text), raw: state.text! })
    );
}

function stringByte(state: JsonState, byte: number): JsonState {
    const isKey = state.mode === 'key-string';
    if (state.highSurrogate !== undefined && byte !== 0x5c) return DEAD;
    if (byte === 0x22) {
        const value = decodeString(state.bytes ?? []);
        if (value === null) return DEAD;
        if (isKey) return completeKey(state, value);
        return completeValue(state, { kind: 'string', value });
    }
    if (byte === 0x5c) {
        if (isKey ? !keyEscapeAllowed(state) : !stringEscapeAllowed(state)) return DEAD;
        return { ...state, mode: isKey ? 'key-escape' : 'escape' };
    }
    if (byte < 0x20) return DEAD;
    let length = state.stringLength ?? 0;
    let pending = state.stringPending ?? 0;
    if ((byte & 0xc0) === 0x80) {
        pending = pending > 0 ? pending - 1 : -1;
    } else {
        pending = pending === 0 ? utf8Continuations(byte) : -1;
        length++;
    }
    const bytes = [...(state.bytes ?? []), byte];
    if (isKey && !keyPrefixAllowed(state, bytes)) return DEAD;
    if (!isKey && !stringContentAllowed(state.schema, bytes, length)) return DEAD;
    return { ...state, bytes, stringLength: length, stringPending: pending };
}

function utf8Continuations(lead: number): number {
    if (lead < 0x80) return 0;
    if (lead < 0xc2) return -1;
    if (lead < 0xe0) return 1;
    if (lead < 0xf0) return 2;
    if (lead < 0xf5) return 3;
    return -1;
}

function escapeByte(state: JsonState, byte: number): JsonState {
    const isKey = state.mode === 'key-escape';
    if (state.highSurrogate !== undefined && byte !== 0x75) return DEAD;
    if (byte === 0x75) {
        return { ...state, mode: isKey ? 'key-unicode' : 'unicode', text: '' };
    }
    const escaped = SIMPLE_ESCAPES[byte];
    if (escaped === undefined) return DEAD;
    const bytes = [...(state.bytes ?? []), ...encoder.encode(escaped)];
    const length = (state.stringLength ?? 0) + 1;
    const pending = (state.stringPending ?? 0) === 0 ? 0 : -1;
    if (isKey && !keyPrefixAllowed(state, bytes)) return DEAD;
    if (!isKey && !stringContentAllowed(state.schema, bytes, length)) return DEAD;
    return { ...state, mode: isKey ? 'key-string' : 'string', bytes, stringLength: length, stringPending: pending };
}

function unicodeByte(state: JsonState, byte: number): JsonState {
    if (!isHex(byte)) return DEAD;
    const text = `${state.text ?? ''}${String.fromCharCode(byte)}`;
    if (text.length < 4) {
        if (state.mode === 'key-unicode' && !unicodeKeyPrefixAllowed(state, text)) return DEAD;
        return { ...state, text };
    }
    const codeUnit = Number.parseInt(text, 16);
    const isKey = state.mode === 'key-unicode';
    const mode = isKey ? 'key-string' : 'string';
    if (codeUnit >= 0xd800 && codeUnit <= 0xdbff) {
        if (state.highSurrogate !== undefined) return DEAD;
        return { ...state, mode, text: undefined, highSurrogate: codeUnit };
    }
    let value: string;
    if (codeUnit >= 0xdc00 && codeUnit <= 0xdfff) {
        if (state.highSurrogate === undefined) return DEAD;
        const codePoint = 0x10000 + ((state.highSurrogate - 0xd800) << 10) + (codeUnit - 0xdc00);
        value = String.fromCodePoint(codePoint);
    } else {
        if (state.highSurrogate !== undefined) return DEAD;
        value = String.fromCharCode(codeUnit);
    }
    const bytes = [...(state.bytes ?? []), ...encoder.encode(value)];
    const length = (state.stringLength ?? 0) + 1;
    const pending = (state.stringPending ?? 0) === 0 ? 0 : -1;
    if (isKey && !keyPrefixAllowed(state, bytes)) return DEAD;
    if (!isKey && !stringContentAllowed(state.schema, bytes, length)) return DEAD;
    return {
        ...state,
        mode,
        bytes,
        text: undefined,
        highSurrogate: undefined,
        stringLength: length,
        stringPending: pending,
    };
}

function completeKey(state: JsonState, key: string): JsonState {
    const frame = topObject(state);
    if (frame.seen.has(key)) return DEAD;
    const childSchema = propertySchema(frame.schema, key, frame.entries);
    if (childSchema === null) return DEAD;
    return {
        ...state,
        mode: 'colon',
        bytes: undefined,
        index: 0,
        stack: replaceTop(state.stack, { ...frame, key, childSchema }),
    };
}

function literalByte(state: JsonState, byte: number): JsonState {
    const index = state.index ?? 0;
    const literal = state.literal!;
    if (byte !== literal.charCodeAt(index)) return DEAD;
    if (index + 1 < literal.length) return { ...state, index: index + 1 };
    const node: JsonNode =
        state.literalValue === null ? { kind: 'null', value: null } : { kind: 'boolean', value: state.literalValue! };
    return completeValue(state, node);
}

function numberByte(state: JsonState, byte: number): JsonState {
    if (isNumberByte(byte)) {
        const text = `${state.text}${String.fromCharCode(byte)}`;
        if (numberPrefixValid(text) && integerPrefixViable(state.schema, text)) return { ...state, text };
    }
    const value = finishNumber(state);
    return value === DEAD ? DEAD : transition(value, byte);
}

// JSON Schema accepts any number encoding for integer fields ("0.9e1" is 9),
// but during generation that freedom creates traps a model cannot escape:
// after "0.9" under {type: "integer"} only digits and exponents stay viable,
// continuations such as "0.9e-" can never close the value, and zero-padding
// like "0.0e0000…" stays legal forever, so a stuck model streams digits until
// it hits the token limit. Integer-only fields therefore use the constrained
// shape -?(0|[1-9]\d*)(\.0+)?([eE]+?\d+)? — the value stays
// ±digits×10^exponent, so every prefix can still close — with fraction zeros
// and exponent digits capped (they carry no information beyond two digits), and
// bytes from which no in-range integer remains reachable are pruned.
const SCALE_DIGIT_LIMIT = 3;
const integerRestrictions = new WeakMap<object, { lo: number; hi: number } | null>();

function integerPrefixViable(schema: Schema, text: string): boolean {
    const restriction = integerRestriction(schema);
    if (restriction === null) return true;
    const { lo, hi } = restriction;
    if (text === '-') return lo <= Math.min(hi, 0);
    const match = /^(-?)(0|[1-9]\d*)(\.0{0,3})?(?:[eE]\+?(\d{0,3}))?$/.exec(text);
    if (match === null) return false;
    // "1.e" can never complete (the fraction needs a digit before the exponent)
    if (match[3] === '.' && match[4] !== undefined) return false;
    const negative = match[1] === '-';
    if (match[3] === undefined && match[4] === undefined) {
        return integerDigitsReachable(lo, hi, negative, match[2]);
    }
    return integerScaleReachable(lo, hi, negative, match[2], match[4] ?? '');
}

function integerRestriction(schema: Schema): { lo: number; hi: number } | null {
    if (schema === true || schema === false) return null;
    let restriction = integerRestrictions.get(schema);
    if (restriction === undefined) {
        restriction = computeIntegerRestriction(schema);
        integerRestrictions.set(schema, restriction);
    }
    return restriction;
}

function computeIntegerRestriction(schema: Record<string, unknown>): { lo: number; hi: number } | null {
    let integerOnly = false;
    if (schema.type !== undefined) {
        const types = Array.isArray(schema.type) ? schema.type.map(String) : [String(schema.type)];
        integerOnly = types.includes('integer') && !types.includes('number');
    } else if (typeof schema.const === 'number') {
        integerOnly = Number.isInteger(schema.const);
    } else if (Array.isArray(schema.enum)) {
        const numbers = schema.enum.filter((value): value is number => typeof value === 'number');
        integerOnly = numbers.length > 0 && numbers.every((value) => Number.isInteger(value));
    }
    if (!integerOnly) return null;
    let lo = -Infinity;
    let hi = Infinity;
    if (typeof schema.minimum === 'number') lo = Math.ceil(schema.minimum);
    if (typeof schema.exclusiveMinimum === 'number') lo = Math.max(lo, Math.floor(schema.exclusiveMinimum) + 1);
    if (typeof schema.maximum === 'number') hi = Math.floor(schema.maximum);
    if (typeof schema.exclusiveMaximum === 'number') hi = Math.min(hi, Math.ceil(schema.exclusiveMaximum) - 1);
    // Pruning must never reject a value the validator would accept; when a
    // bound is beyond exact integer arithmetic, drop it instead of guessing.
    if (!Number.isSafeInteger(lo)) lo = -Infinity;
    if (!Number.isSafeInteger(hi)) hi = Infinity;
    return { lo, hi };
}

// Whether some integer whose decimal representation starts with `digits`
// (extended by zero or more digits, or scaled by a later ".0…"/exponent) lies
// in [lo, hi]. The reachable values are ±[|P|·10^k, (|P|+1)·10^k - 1] for
// k ≥ 0, except that "0"/"-0" cannot take further digits.
function integerDigitsReachable(lo: number, hi: number, negative: boolean, digits: string): boolean {
    if (digits.length > 16) return negative ? lo === -Infinity : hi === Infinity;
    let low = Number(digits);
    let high = low + 1;
    const extensible = low !== 0;
    for (;;) {
        if (negative) {
            if (-(high - 1) <= hi && -low >= lo) return true;
            if (-low < lo) return false;
        } else {
            if (low <= hi && high - 1 >= lo) return true;
            if (low > hi) return false;
        }
        if (!extensible) return false;
        low *= 10;
        high *= 10;
    }
}

// Once a fraction or exponent starts, the digits are frozen and the value is
// exactly ±digits×10^E for a still-typeable exponent E ≥ 0. When exponent
// digits have been typed, reachable exponents are those obtainable by extending
// the typed prefix within the digit limit (leading zeros allowed).
function integerScaleReachable(
    lo: number,
    hi: number,
    negative: boolean,
    digits: string,
    exponentPrefix: string,
): boolean {
    if (digits.length > 16) return negative ? lo === -Infinity : hi === Infinity;
    const base = Number(digits);
    if (base === 0) return lo <= 0 && 0 <= hi;
    let magnitude = base;
    // A typeable exponent has at most SCALE_DIGIT_LIMIT digits, so 999 bounds
    // the search; finite bounds exit early via the overshoot check.
    for (let exponent = 0; exponent <= 999; ++exponent) {
        const value = negative ? -magnitude : magnitude;
        const inRange = (lo === -Infinity || value >= lo) && (hi === Infinity || value <= hi);
        if (inRange && exponentTypeable(exponent, exponentPrefix)) return true;
        if (negative ? lo !== -Infinity && value < lo : hi !== Infinity && value > hi) return false;
        magnitude *= 10;
    }
    return false;
}

// Whether exponent value `exponent` can be written by extending the typed
// digits `prefix` to a full exponent of at most SCALE_DIGIT_LIMIT digits
// (at least one digit total; leading zeros allowed).
function exponentTypeable(exponent: number, prefix: string): boolean {
    const typed = prefix === '' ? 0 : Number(prefix);
    for (let extra = Math.max(0, 1 - prefix.length); extra <= SCALE_DIGIT_LIMIT - prefix.length; ++extra) {
        const scale = 10 ** extra;
        if (exponent >= typed * scale && exponent < (typed + 1) * scale) return true;
    }
    return false;
}

function finishNumber(state: JsonState): JsonState {
    const text = state.text!;
    if (!numberComplete(text)) return DEAD;
    return completeValue(state, { kind: 'number', value: Number(text), raw: text });
}

function completeValue(state: JsonState, node: JsonNode): JsonState {
    if (!validateNode(state.schema, node)) return DEAD;
    if (state.stack.length === 0) return { mode: 'done', schema: state.schema, stack: [], guidance: state.guidance };
    const frame = state.stack.at(-1)!;
    if (frame.kind === 'object') {
        if (frame.key === undefined) return DEAD;
        const seen = new Set(frame.seen);
        seen.add(frame.key);
        const entries = [...frame.entries, { key: frame.key, node }];
        if (!partialObjectValid(frame.schema, entries, new Set())) return DEAD;
        return {
            mode: 'after-value',
            schema: frame.schema,
            stack: replaceTop(state.stack, { kind: 'object', schema: frame.schema, seen, entries }),
            guidance: state.guidance,
        };
    }
    return {
        mode: 'after-value',
        schema: frame.schema,
        stack: replaceTop(state.stack, { ...frame, length: frame.length + 1, items: [...frame.items, node] }),
        guidance: state.guidance,
    };
}

function afterValue(state: JsonState, byte: number): JsonState {
    const frame = state.stack.at(-1);
    if (frame === undefined) return DEAD;
    if (frame.kind === 'object') {
        if (byte === 0x7d) return closeObject(state);
        if (!objectCanAddProperty(frame)) return DEAD;
        return separatorByte(
            { ...state, mode: 'item-separator', index: 0 },
            byte,
            state.guidance.itemBytes,
            'object-key',
        );
    }
    if (byte === 0x5d) return closeArray(state);
    if (exceedsMaxItems(frame)) return DEAD;
    return separatorByte({ ...state, mode: 'item-separator', index: 0 }, byte, state.guidance.itemBytes, 'array-value');
}

function separatorByte(state: JsonState, byte: number, separator: Uint8Array, completedMode: Mode): JsonState {
    const index = state.index ?? 0;
    if (byte !== separator[index]) return DEAD;
    if (index + 1 < separator.length) return { ...state, index: index + 1 };
    const frame = state.stack.at(-1);
    return {
        ...state,
        mode: completedMode,
        schema:
            completedMode === 'value'
                ? topObject(state).childSchema!
                : completedMode === 'array-value' && frame?.kind === 'array'
                  ? itemSchema(frame.schema, frame.length)
                  : state.schema,
        index: undefined,
    };
}

function separatorExpects(state: JsonState, byte: number): boolean {
    if (state.mode === 'colon') return state.guidance.keyBytes[state.index ?? 0] === byte;
    if (state.mode === 'item-separator') return state.guidance.itemBytes[state.index ?? 0] === byte;
    return false;
}

function closeObject(state: JsonState): JsonState {
    const frame = state.stack.at(-1);
    if (frame?.kind !== 'object') return DEAD;
    const value: Record<string, unknown> = Object.create(null) as Record<string, unknown>;
    for (const entry of frame.entries) value[entry.key] = entry.node.value;
    return completeValue(
        { ...state, schema: frame.schema, stack: state.stack.slice(0, -1) },
        { kind: 'object', value, entries: [...frame.entries] },
    );
}

function closeArray(state: JsonState): JsonState {
    const frame = state.stack.at(-1);
    if (frame?.kind !== 'array') return DEAD;
    return completeValue(
        { ...state, schema: frame.schema, stack: state.stack.slice(0, -1) },
        { kind: 'array', value: frame.items.map((item) => item.value), items: [...frame.items] },
    );
}

function selectSchema(schema: Schema, kind: Kind): Schema | null {
    return schemaMayAcceptKind(schema, kind, new Set()) ? schema : null;
}

function propertySchema(
    schema: Schema,
    key: string,
    entries: ReadonlyArray<{ key: string; node: JsonNode }> = [],
): Schema | null {
    if (schema === true) return true;
    if (schema === false) return null;
    const schemas: Schema[] = [];
    const direct = directPropertySchema(schema, key);
    if (direct === null) return null;
    if (direct !== true) schemas.push(direct);
    for (const candidate of [schema, ...(Array.isArray(schema.allOf) ? schema.allOf.filter(isSchema) : [])]) {
        if (candidate === true || candidate === false) continue;
        const branch = selectedConditionalBranch(candidate, entries);
        if (branch === undefined || branch === true) continue;
        if (branch === false) return null;
        const branchSchema = directPropertySchema(branch, key);
        if (branchSchema === null) return null;
        if (branchSchema !== true) schemas.push(branchSchema);
    }
    if (schemas.length === 0) return true;
    return schemas.length === 1 ? schemas[0] : { allOf: schemas };
}

function directPropertySchema(schema: Record<string, unknown>, key: string): Schema | null {
    const properties = isRecord(schema.properties) ? schema.properties : {};
    const schemas: Schema[] = [];
    if (isSchema(properties[key])) schemas.push(properties[key]);
    if (isRecord(schema.patternProperties)) {
        for (const [pattern, candidate] of Object.entries(schema.patternProperties)) {
            if (new RegExp(pattern, 'u').test(key) && isSchema(candidate)) schemas.push(candidate);
        }
    }
    if (schemas.length === 0) {
        if (schema.additionalProperties === false) return null;
        return isSchema(schema.additionalProperties) ? schema.additionalProperties : true;
    }
    return schemas.length === 1 ? schemas[0] : { allOf: schemas };
}

function selectedConditionalBranch(
    schema: Record<string, unknown>,
    entries: ReadonlyArray<{ key: string; node: JsonNode }>,
): Schema | undefined {
    if (!isSchema(schema.if) || schema.if === true || schema.if === false) return undefined;
    const required = Array.isArray(schema.if.required) ? schema.if.required : [];
    if (required.length === 0 || !required.every((key) => entries.some((entry) => entry.key === key))) return undefined;
    const node = objectNode(entries);
    const branch = validateNode(schema.if, node) ? schema.then : schema.else;
    return isSchema(branch) ? branch : undefined;
}

function partialObjectValid(
    schema: Schema,
    entries: ReadonlyArray<{ key: string; node: JsonNode }>,
    seen: Set<object>,
): boolean {
    if (schema === true) return true;
    if (schema === false) return false;
    if (seen.has(schema)) return true;
    seen.add(schema);
    try {
        for (const entry of entries) {
            const child = directPropertySchema(schema, entry.key);
            if (child === null || (child !== true && !validateNode(child, entry.node))) return false;
        }
        if (
            typeof schema.$ref === 'string' &&
            !partialObjectValid(resolveReference(schema, schema.$ref), entries, seen)
        )
            return false;
        if (
            Array.isArray(schema.allOf) &&
            !schema.allOf.every((child) => !isSchema(child) || partialObjectValid(child, entries, seen))
        )
            return false;
        const branch = selectedConditionalBranch(schema, entries);
        return branch === undefined || partialObjectValid(branch, entries, seen);
    } finally {
        seen.delete(schema);
    }
}

function objectNode(entries: ReadonlyArray<{ key: string; node: JsonNode }>): Extract<JsonNode, { kind: 'object' }> {
    const value: Record<string, unknown> = Object.create(null) as Record<string, unknown>;
    for (const entry of entries) value[entry.key] = entry.node.value;
    return { kind: 'object', value, entries: [...entries] };
}

function itemSchema(schema: Schema, index: number): Schema {
    if (schema === true || schema === false || hasDeferredStructure(schema)) return schema === false ? false : true;
    if (Array.isArray(schema.prefixItems) && isSchema(schema.prefixItems[index])) return schema.prefixItems[index];
    if (Array.isArray(schema.items)) {
        if (isSchema(schema.items[index])) return schema.items[index];
        return isSchema(schema.additionalItems) ? schema.additionalItems : true;
    }
    return isSchema(schema.items) ? schema.items : true;
}

function keyPrefixAllowed(state: JsonState, bytes: readonly number[]): boolean {
    const frame = topObject(state);
    if (frame.schema === true || frame.schema === false) return true;
    const properties = encodedPropertyKeys(frame.schema);
    if (properties === null) return true;
    return properties.some(({ key, bytes: propertyBytes }) => {
        if (frame.seen.has(key) || bytes.length > propertyBytes.length) return false;
        for (let index = 0; index < bytes.length; ++index) {
            if (bytes[index] !== propertyBytes[index]) return false;
        }
        return true;
    });
}

function keyEscapeAllowed(state: JsonState): boolean {
    const frame = topObject(state);
    if (frame.schema === true || frame.schema === false) return true;
    const properties = encodedPropertyKeys(frame.schema);
    if (properties === null) return true;
    if (state.highSurrogate !== undefined) return true;
    const bytes = state.bytes ?? [];
    return properties.some(({ key, bytes: propertyBytes }) => {
        if (frame.seen.has(key) || propertyBytes.length <= bytes.length) return false;
        for (let index = 0; index < bytes.length; ++index) {
            if (bytes[index] !== propertyBytes[index]) return false;
        }
        const next = propertyBytes[bytes.length];
        return next < 0x20 || next === 0x22 || next === 0x5c || next >= 0x80;
    });
}

function objectCanAddProperty(frame: ObjectFrame): boolean {
    if (frame.schema === true || frame.schema === false) return true;
    const properties = encodedPropertyKeys(frame.schema);
    return properties === null || properties.some(({ key }) => !frame.seen.has(key));
}

function stringContentAllowed(schema: Schema, bytes: readonly number[], length: number): boolean {
    const values = finiteStringValues(schema);
    if (
        values !== null &&
        !values.some(({ bytes: valueBytes }) => {
            if (bytes.length > valueBytes.length) return false;
            for (let index = 0; index < bytes.length; ++index) {
                if (bytes[index] !== valueBytes[index]) return false;
            }
            return true;
        })
    )
        return false;
    const maximum = directStringMaxLength(schema);
    return maximum === undefined || length <= maximum;
}

function stringEscapeAllowed(state: JsonState): boolean {
    const values = finiteStringValues(state.schema);
    const maximum = directStringMaxLength(state.schema);
    if (maximum !== undefined && (state.stringLength ?? 0) >= maximum) return false;
    if (values === null) return true;
    if (state.highSurrogate !== undefined) return true;
    const bytes = state.bytes ?? [];
    return values.some(({ bytes: valueBytes }) => {
        if (valueBytes.length <= bytes.length) return false;
        for (let index = 0; index < bytes.length; ++index) {
            if (bytes[index] !== valueBytes[index]) return false;
        }
        const next = valueBytes[bytes.length];
        return next < 0x20 || next === 0x22 || next === 0x5c || next >= 0x80;
    });
}

function directStringMaxLength(schema: Schema): number | undefined {
    return schema !== true && schema !== false && typeof schema.maxLength === 'number' ? schema.maxLength : undefined;
}

function finiteStringValues(schema: Schema): ReadonlyArray<{ value: string; bytes: Uint8Array }> | null {
    if (schema === true || schema === false) return null;
    let values = finiteStringCache.get(schema);
    if (values !== undefined) return values;
    let candidates: string[] | null = null;
    if (typeof schema.const === 'string') candidates = [schema.const];
    else if (Array.isArray(schema.enum))
        candidates = schema.enum.filter((value): value is string => typeof value === 'string');
    values =
        candidates === null
            ? null
            : [...new Set(candidates)]
                  .filter((value) => validateNode(schema, { kind: 'string', value }))
                  .map((value) => ({ value, bytes: encoder.encode(value) }));
    finiteStringCache.set(schema, values);
    return values;
}

function unicodeKeyPrefixAllowed(state: JsonState, hexPrefix: string): boolean {
    const frame = topObject(state);
    if (frame.schema === true || frame.schema === false) return true;
    const properties = encodedPropertyKeys(frame.schema);
    if (properties === null) return true;
    const prefix = decodeString(state.bytes ?? []);
    if (prefix === null) return true;
    return properties.some(({ key }) => {
        if (frame.seen.has(key) || !key.startsWith(prefix)) return false;
        const offset = prefix.length;
        let codeUnit = key.charCodeAt(offset);
        if (state.highSurrogate !== undefined) {
            if (codeUnit !== state.highSurrogate) return false;
            codeUnit = key.charCodeAt(offset + 1);
        }
        return Number.isInteger(codeUnit) && codeUnit.toString(16).padStart(4, '0').startsWith(hexPrefix.toLowerCase());
    });
}

function encodedPropertyKeys(
    schema: Record<string, unknown>,
): ReadonlyArray<{ key: string; bytes: Uint8Array }> | null {
    let properties = propertyKeyBytes.get(schema);
    if (properties === undefined) {
        const keys = constrainedPropertyKeys(schema, new Set());
        properties = keys === null ? null : [...keys].map((key) => ({ key, bytes: encoder.encode(key) }));
        propertyKeyBytes.set(schema, properties);
    }
    return properties;
}

function constrainedPropertyKeys(schema: Schema, seen: Set<object>): Set<string> | null {
    if (schema === true || schema === false || seen.has(schema)) return null;
    seen.add(schema);
    try {
        const properties = isRecord(schema.properties) ? schema.properties : {};
        let result =
            schema.additionalProperties === false && !isRecord(schema.patternProperties)
                ? new Set(Object.keys(properties))
                : null;
        if (typeof schema.$ref === 'string') {
            result = intersectKeySets(result, constrainedPropertyKeys(resolveReference(schema, schema.$ref), seen));
        }
        if (Array.isArray(schema.allOf)) {
            for (const child of schema.allOf) {
                if (isSchema(child)) result = intersectKeySets(result, constrainedPropertyKeys(child, seen));
            }
        }
        for (const keyword of ['anyOf', 'oneOf'] as const) {
            if (!Array.isArray(schema[keyword])) continue;
            let union: Set<string> | null = new Set();
            for (const child of schema[keyword]) {
                if (!isSchema(child) || !schemaMayAcceptKind(child, 'object', new Set())) continue;
                const childKeys = constrainedPropertyKeys(child, seen);
                if (childKeys === null) {
                    union = null;
                    break;
                }
                for (const key of childKeys) union.add(key);
            }
            result = intersectKeySets(result, union);
        }
        return result;
    } finally {
        seen.delete(schema);
    }
}

function intersectKeySets(left: Set<string> | null, right: Set<string> | null): Set<string> | null {
    if (left === null) return right;
    if (right === null) return left;
    return new Set([...left].filter((key) => right.has(key)));
}

function exceedsMaxItems(frame: ArrayFrame): boolean {
    return frame.schema !== true &&
        frame.schema !== false &&
        !hasDeferredStructure(frame.schema) &&
        typeof frame.schema.maxItems === 'number'
        ? frame.length >= frame.schema.maxItems
        : false;
}

const schemaContexts = new WeakMap<object, Schema>();

function validateNode(schema: Schema, node: JsonNode, active = new Map<object, Set<JsonNode>>()): boolean {
    if (schema === true) return true;
    if (schema === false) return false;
    let nodes = active.get(schema);
    if (nodes?.has(node)) return true;
    if (nodes === undefined) {
        nodes = new Set();
        active.set(schema, nodes);
    }
    nodes.add(node);
    try {
        if (typeof schema.$ref === 'string' && !validateNode(resolveReference(schema, schema.$ref), node, active))
            return false;
        if (schema.type !== undefined) {
            const types = Array.isArray(schema.type) ? schema.type.map(String) : [String(schema.type)];
            if (!types.some((type) => nodeHasType(node, type))) return false;
        }
        if (schema.const !== undefined && !nodeEqualsValue(node, schema.const)) return false;
        if (Array.isArray(schema.enum) && !schema.enum.some((value) => nodeEqualsValue(node, value))) return false;
        if (
            Array.isArray(schema.allOf) &&
            !schema.allOf.every((child) => isSchema(child) && validateNode(child, node, active))
        )
            return false;
        if (
            Array.isArray(schema.anyOf) &&
            !schema.anyOf.some((child) => isSchema(child) && validateNode(child, node, active))
        )
            return false;
        if (
            Array.isArray(schema.oneOf) &&
            schema.oneOf.filter((child) => isSchema(child) && validateNode(child, node, active)).length !== 1
        )
            return false;
        if (isSchema(schema.not) && validateNode(schema.not, node, active)) return false;
        if (isSchema(schema.if)) {
            const branch = validateNode(schema.if, node, active) ? schema.then : schema.else;
            if (isSchema(branch) && !validateNode(branch, node, active)) return false;
        }
        if (node.kind === 'number' && !validateNumericNode(schema, node)) return false;
        if (node.kind === 'string' && !validateStringNode(schema, node.value)) return false;
        if (node.kind === 'array' && !validateArrayNode(schema, node, active)) return false;
        if (node.kind === 'object' && !validateObjectNode(schema, node, active)) return false;
        return true;
    } finally {
        nodes.delete(node);
    }
}

function validateNumericNode(schema: Record<string, unknown>, node: Extract<JsonNode, { kind: 'number' }>): boolean {
    const value = decimal(node.raw);
    if (schema.minimum !== undefined && compareDecimal(value, decimal(String(schema.minimum))) < 0) return false;
    if (schema.maximum !== undefined && compareDecimal(value, decimal(String(schema.maximum))) > 0) return false;
    if (schema.exclusiveMinimum !== undefined && compareDecimal(value, decimal(String(schema.exclusiveMinimum))) <= 0)
        return false;
    if (schema.exclusiveMaximum !== undefined && compareDecimal(value, decimal(String(schema.exclusiveMaximum))) >= 0)
        return false;
    return schema.multipleOf === undefined || decimalMultiple(value, decimal(String(schema.multipleOf)));
}

function validateStringNode(schema: Record<string, unknown>, value: string): boolean {
    const length = [...value].length;
    if (typeof schema.minLength === 'number' && length < schema.minLength) return false;
    if (typeof schema.maxLength === 'number' && length > schema.maxLength) return false;
    if (typeof schema.pattern === 'string' && !new RegExp(schema.pattern, 'u').test(value)) return false;
    return typeof schema.format !== 'string' || formatMatches(schema.format, value);
}

function validateArrayNode(
    schema: Record<string, unknown>,
    node: Extract<JsonNode, { kind: 'array' }>,
    active: Map<object, Set<JsonNode>>,
): boolean {
    if (typeof schema.minItems === 'number' && node.items.length < schema.minItems) return false;
    if (typeof schema.maxItems === 'number' && node.items.length > schema.maxItems) return false;
    if (schema.uniqueItems === true) {
        for (let index = 0; index < node.items.length; ++index) {
            if (node.items.slice(0, index).some((other) => nodesEqual(other, node.items[index]))) return false;
        }
    }
    const legacyTuple = Array.isArray(schema.items) ? schema.items : undefined;
    const prefix = (Array.isArray(schema.prefixItems) ? schema.prefixItems : legacyTuple) ?? [];
    for (let index = 0; index < Math.min(prefix.length, node.items.length); ++index) {
        if (!isSchema(prefix[index]) || !validateNode(prefix[index], node.items[index], active)) return false;
    }
    const remaining = legacyTuple ? schema.additionalItems : schema.items;
    if (isSchema(remaining)) {
        for (let index = prefix.length; index < node.items.length; ++index) {
            if (!validateNode(remaining, node.items[index], active)) return false;
        }
    }
    if (isSchema(schema.contains)) {
        const matches = node.items.filter((item) => validateNode(schema.contains as Schema, item, active)).length;
        const minimum = typeof schema.minContains === 'number' ? schema.minContains : 1;
        if (matches < minimum || (typeof schema.maxContains === 'number' && matches > schema.maxContains)) return false;
    }
    return true;
}

function validateObjectNode(
    schema: Record<string, unknown>,
    node: Extract<JsonNode, { kind: 'object' }>,
    active: Map<object, Set<JsonNode>>,
): boolean {
    const entries = new Map(node.entries.map((entry) => [entry.key, entry.node]));
    if (entries.size !== node.entries.length) return false;
    if (typeof schema.minProperties === 'number' && entries.size < schema.minProperties) return false;
    if (typeof schema.maxProperties === 'number' && entries.size > schema.maxProperties) return false;
    const required = Array.isArray(schema.required) ? schema.required : [];
    if (!required.every((key) => typeof key === 'string' && entries.has(key))) return false;
    const properties = isRecord(schema.properties) ? schema.properties : {};
    const patterns = isRecord(schema.patternProperties) ? schema.patternProperties : {};
    for (const [key, child] of entries) {
        if (
            isSchema(schema.propertyNames) &&
            !validateNode(schema.propertyNames, { kind: 'string', value: key }, active)
        )
            return false;
        let matched = false;
        if (isSchema(properties[key])) {
            matched = true;
            if (!validateNode(properties[key], child, active)) return false;
        }
        for (const [pattern, patternSchema] of Object.entries(patterns)) {
            if (new RegExp(pattern, 'u').test(key)) {
                matched = true;
                if (!isSchema(patternSchema) || !validateNode(patternSchema, child, active)) return false;
            }
        }
        if (!matched) {
            if (schema.additionalProperties === false) return false;
            if (isSchema(schema.additionalProperties) && !validateNode(schema.additionalProperties, child, active))
                return false;
        }
    }
    const dependentRequired = isRecord(schema.dependentRequired) ? schema.dependentRequired : {};
    for (const [key, dependencies] of Object.entries(dependentRequired)) {
        if (
            entries.has(key) &&
            Array.isArray(dependencies) &&
            !dependencies.every((dependency) => typeof dependency === 'string' && entries.has(dependency))
        )
            return false;
    }
    const dependentSchemas = isRecord(schema.dependentSchemas) ? schema.dependentSchemas : {};
    for (const [key, dependency] of Object.entries(dependentSchemas)) {
        if (entries.has(key) && (!isSchema(dependency) || !validateNode(dependency, node, active))) return false;
    }
    const dependencies = isRecord(schema.dependencies) ? schema.dependencies : {};
    for (const [key, dependency] of Object.entries(dependencies)) {
        if (!entries.has(key)) continue;
        if (Array.isArray(dependency)) {
            if (!dependency.every((requiredKey) => typeof requiredKey === 'string' && entries.has(requiredKey)))
                return false;
        } else if (!isSchema(dependency) || !validateNode(dependency, node, active)) return false;
    }
    return true;
}

function nodeHasType(node: JsonNode, type: string): boolean {
    if (type === 'integer') return node.kind === 'number' && decimalInteger(decimal(node.raw));
    if (type === 'number') return node.kind === 'number';
    return node.kind === type;
}

function nodeEqualsValue(node: JsonNode, value: unknown): boolean {
    if (node.kind === 'number')
        return typeof value === 'number' && compareDecimal(decimal(node.raw), decimal(String(value))) === 0;
    if (node.kind === 'array')
        return (
            Array.isArray(value) &&
            node.items.length === value.length &&
            node.items.every((item, index) => nodeEqualsValue(item, value[index]))
        );
    if (node.kind === 'object') {
        if (!isRecord(value)) return false;
        const entries = new Map(node.entries.map((entry) => [entry.key, entry.node]));
        return (
            entries.size === Object.keys(value).length &&
            Object.entries(value).every(([key, child]) => entries.has(key) && nodeEqualsValue(entries.get(key)!, child))
        );
    }
    return Object.is(node.value, value);
}

function nodesEqual(left: JsonNode, right: JsonNode): boolean {
    if (left.kind !== right.kind) return false;
    if (left.kind === 'number' && right.kind === 'number')
        return compareDecimal(decimal(left.raw), decimal(right.raw)) === 0;
    if (left.kind === 'array' && right.kind === 'array')
        return (
            left.items.length === right.items.length &&
            left.items.every((item, index) => nodesEqual(item, right.items[index]))
        );
    if (left.kind === 'object' && right.kind === 'object') {
        const rightEntries = new Map(right.entries.map((entry) => [entry.key, entry.node]));
        return (
            left.entries.length === rightEntries.size &&
            left.entries.every(
                (entry) => rightEntries.has(entry.key) && nodesEqual(entry.node, rightEntries.get(entry.key)!),
            )
        );
    }
    return Object.is(left.value, right.value);
}

type Decimal = { coefficient: bigint; exponent: bigint };

function decimal(raw: string): Decimal {
    const match = /^(-?)(\d+)(?:\.(\d+))?(?:[eE]([+-]?\d+))?$/.exec(raw)!;
    const fraction = match[3] ?? '';
    let digits = `${match[2]}${fraction}`.replace(/^0+/, '');
    if (digits === '') return { coefficient: 0n, exponent: 0n };
    const trailing = /0+$/.exec(digits)?.[0].length ?? 0;
    if (trailing > 0) digits = digits.slice(0, -trailing);
    return {
        coefficient: BigInt(`${match[1]}${digits}`),
        exponent: BigInt(match[4] ?? '0') - BigInt(fraction.length) + BigInt(trailing),
    };
}

function compareDecimal(left: Decimal, right: Decimal): number {
    if (left.coefficient === right.coefficient && left.exponent === right.exponent) return 0;
    if (left.coefficient === 0n) return right.coefficient < 0n ? 1 : -1;
    if (right.coefficient === 0n) return left.coefficient < 0n ? -1 : 1;
    if (left.coefficient < 0n && right.coefficient >= 0n) return -1;
    if (left.coefficient >= 0n && right.coefficient < 0n) return 1;
    const negative = left.coefficient < 0n;
    const leftAbsolute = left.coefficient < 0n ? -left.coefficient : left.coefficient;
    const rightAbsolute = right.coefficient < 0n ? -right.coefficient : right.coefficient;
    const leftMagnitude = BigInt(leftAbsolute.toString().length) + left.exponent;
    const rightMagnitude = BigInt(rightAbsolute.toString().length) + right.exponent;
    if (leftMagnitude !== rightMagnitude) {
        const comparison = leftMagnitude < rightMagnitude ? -1 : 1;
        return negative ? -comparison : comparison;
    }
    const exponent = left.exponent < right.exponent ? left.exponent : right.exponent;
    const scaledLeft = left.coefficient * 10n ** (left.exponent - exponent);
    const scaledRight = right.coefficient * 10n ** (right.exponent - exponent);
    return scaledLeft < scaledRight ? -1 : scaledLeft > scaledRight ? 1 : 0;
}

function decimalInteger(value: Decimal): boolean {
    return value.coefficient === 0n || value.exponent >= 0n;
}

function decimalMultiple(value: Decimal, divisor: Decimal): boolean {
    if (divisor.coefficient === 0n) return false;
    if (value.coefficient === 0n) return true;
    const numerator = value.coefficient < 0n ? -value.coefficient : value.coefficient;
    const denominator = divisor.coefficient < 0n ? -divisor.coefficient : divisor.coefficient;
    const exponent = value.exponent - divisor.exponent;
    if (exponent >= 0n) {
        return ((numerator % denominator) * modularPower(10n, exponent, denominator)) % denominator === 0n;
    }
    const places = -exponent;
    if (places >= BigInt(numerator.toString().length)) return false;
    return numerator % (denominator * 10n ** places) === 0n;
}

function modularPower(base: bigint, exponent: bigint, modulus: bigint): bigint {
    if (modulus === 1n) return 0n;
    let result = 1n;
    base %= modulus;
    while (exponent > 0n) {
        if (exponent & 1n) result = (result * base) % modulus;
        base = (base * base) % modulus;
        exponent >>= 1n;
    }
    return result;
}

function hasDeferredStructure(schema: Schema): boolean {
    return (
        schema !== true &&
        schema !== false &&
        ['$ref', 'allOf', 'anyOf', 'oneOf', 'not', 'if'].some((key) => schema[key] !== undefined)
    );
}

function schemaMayAcceptKind(schema: Schema, kind: Kind, seen: Set<object>): boolean {
    if (schema === true) return true;
    if (schema === false) return false;
    if (seen.has(schema)) return true;
    seen.add(schema);
    if (typeof schema.$ref === 'string' && !schemaMayAcceptKind(resolveReference(schema, schema.$ref), kind, seen))
        return false;
    if (!allowsKind(schema, kind)) return false;
    if (
        Array.isArray(schema.allOf) &&
        !schema.allOf.every((child) => isSchema(child) && schemaMayAcceptKind(child, kind, seen))
    )
        return false;
    if (
        Array.isArray(schema.anyOf) &&
        !schema.anyOf.some((child) => isSchema(child) && schemaMayAcceptKind(child, kind, seen))
    )
        return false;
    if (
        Array.isArray(schema.oneOf) &&
        !schema.oneOf.some((child) => isSchema(child) && schemaMayAcceptKind(child, kind, seen))
    )
        return false;
    return true;
}

function resolveReference(owner: object, reference: string): Schema {
    if (!reference.startsWith('#'))
        throw new TypeError(`External JSON Schema reference ${JSON.stringify(reference)} is unsupported.`);
    const root = schemaContexts.get(owner);
    if (root === undefined) throw new Error('Missing JSON Schema context.');
    if (reference === '#') return root;
    if (!reference.startsWith('#/')) {
        throw new TypeError(`JSON Schema reference ${JSON.stringify(reference)} must contain a JSON Pointer.`);
    }
    let current: unknown = root;
    for (const encoded of reference.slice(2).split('/')) {
        const key = decodeURIComponent(encoded).replace(/~1/g, '/').replace(/~0/g, '~');
        if (!isRecord(current) || !(key in current))
            throw new TypeError(`JSON Schema reference ${JSON.stringify(reference)} does not resolve.`);
        current = current[key];
    }
    if (!isSchema(current))
        throw new TypeError(`JSON Schema reference ${JSON.stringify(reference)} does not resolve to a schema.`);
    return current;
}

function registerSchemaContext(schema: Schema, root: Schema, seen = new Set<object>()): void {
    if (!isRecord(schema) || seen.has(schema)) return;
    seen.add(schema);
    schemaContexts.set(schema, root);
    for (const child of schemaChildren(schema)) registerSchemaContext(child, root, seen);
}

function schemaChildren(schema: Record<string, unknown>): Schema[] {
    const result: Schema[] = [];
    for (const key of [
        'not',
        'if',
        'then',
        'else',
        'contains',
        'propertyNames',
        'additionalProperties',
        'additionalItems',
    ]) {
        if (isSchema(schema[key])) result.push(schema[key] as Schema);
    }
    if (isSchema(schema.items)) result.push(schema.items);
    if (Array.isArray(schema.items)) result.push(...schema.items.filter(isSchema));
    for (const key of ['prefixItems', 'allOf', 'anyOf', 'oneOf']) {
        if (Array.isArray(schema[key])) result.push(...(schema[key] as unknown[]).filter(isSchema));
    }
    for (const key of ['properties', 'patternProperties', 'dependentSchemas', '$defs', 'definitions']) {
        if (isRecord(schema[key]))
            result.push(...Object.values(schema[key] as Record<string, unknown>).filter(isSchema));
    }
    if (isRecord(schema.dependencies)) {
        result.push(...Object.values(schema.dependencies).filter(isSchema));
    }
    return result;
}

function checkReferences(schema: Schema, seen = new Set<object>()): void {
    if (!isRecord(schema) || seen.has(schema)) return;
    seen.add(schema);
    if (typeof schema.$ref === 'string') resolveReference(schema, schema.$ref);
    for (const child of schemaChildren(schema)) checkReferences(child, seen);
}

function assertPattern(value: unknown, path: string): void {
    if (typeof value !== 'string') throw new TypeError(`${path} must be a string.`);
    try {
        new RegExp(value, 'u');
    } catch {
        throw new TypeError(`${path} must be a valid Unicode RegExp.`);
    }
}

function guidanceFrom(schema: Schema): Guidance {
    if (!isRecord(schema) || schema['x-guidance'] === undefined) return DEFAULT_GUIDANCE;
    if (!isRecord(schema['x-guidance'])) throw new TypeError('x-guidance must be an object.');
    const source = schema['x-guidance'];
    for (const key of Object.keys(source)) {
        if (!['item_separator', 'key_separator', 'whitespace_flexible'].includes(key)) {
            throw new TypeError(`Unsupported x-guidance option ${JSON.stringify(key)}.`);
        }
    }
    const itemSeparator = source.item_separator ?? ',';
    const keySeparator = source.key_separator ?? ':';
    const whitespaceFlexible = source.whitespace_flexible ?? true;
    if (typeof itemSeparator !== 'string' || itemSeparator.length === 0) {
        throw new TypeError('x-guidance.item_separator must be a non-empty string.');
    }
    if (typeof keySeparator !== 'string' || keySeparator.length === 0) {
        throw new TypeError('x-guidance.key_separator must be a non-empty string.');
    }
    if (typeof whitespaceFlexible !== 'boolean')
        throw new TypeError('x-guidance.whitespace_flexible must be a boolean.');
    return {
        itemSeparator,
        keySeparator,
        itemBytes: encoder.encode(itemSeparator),
        keyBytes: encoder.encode(keySeparator),
        whitespaceFlexible,
    };
}

function formatMatches(format: string, value: string): boolean {
    switch (format) {
        case 'date':
            return validDate(value);
        case 'time':
            return validTime(value);
        case 'date-time': {
            const separator = value.search(/[Tt]/);
            return separator > 0 && validDate(value.slice(0, separator)) && validTime(value.slice(separator + 1));
        }
        case 'duration':
            return /^(?:P\d+W|P(?=\d|T\d)(?:\d+Y)?(?:\d+M)?(?:\d+D)?(?:T(?=\d)(?:\d+H)?(?:\d+M)?(?:\d+S)?)?)$/.test(
                value,
            );
        case 'email': {
            const at = value.lastIndexOf('@');
            if (at <= 0 || at === value.length - 1) return false;
            const local = value.slice(0, at);
            const domain = value.slice(at + 1);
            if (
                local.startsWith('.') ||
                local.endsWith('.') ||
                local.includes('..') ||
                !/^[A-Za-z0-9!#$%&'*+\-/=?^_`{|}~.]+$/.test(local)
            )
                return false;
            return /^\[(.+)\]$/.test(domain) ? validIpv4(domain.slice(1, -1)) : validHostname(domain);
        }
        case 'hostname':
            return validHostname(value);
        case 'ipv4':
            return validIpv4(value);
        case 'ipv6':
            return validIpv6(value);
        case 'uuid':
            return /^[\da-f]{8}(?:-[\da-f]{4}){3}-[\da-f]{12}$/i.test(value);
        case 'uri':
            return validUriText(value, true);
        case 'uri-reference':
            return validUriText(value, false);
        case 'regex':
            try {
                new RegExp(value, 'u');
                return true;
            } catch {
                return false;
            }
        case 'json-pointer':
            return /^(?:\/(?:[^~/]|~[01])*)*$/u.test(value);
        case 'relative-json-pointer':
            return /^(?:0|[1-9]\d*)(?:#|(?:\/(?:[^~/]|~[01])*)*)$/u.test(value);
        default:
            return true;
    }
}

function validDate(value: string): boolean {
    const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
    if (!match) return false;
    const year = Number(match[1]);
    const month = Number(match[2]);
    const day = Number(match[3]);
    const days = [
        31,
        year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0) ? 29 : 28,
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    return month >= 1 && month <= 12 && day >= 1 && day <= days[month - 1];
}

function validTime(value: string): boolean {
    const match = /^(\d{2}):(\d{2}):(\d{2})(?:\.\d+)?(?:[Zz]|[+-](\d{2}):(\d{2}))$/.exec(value);
    if (!match) return false;
    const hour = Number(match[1]);
    const minute = Number(match[2]);
    const second = Number(match[3]);
    if (hour > 23 || minute > 59 || second > 60 || (second === 60 && (hour !== 23 || minute !== 59))) return false;
    return match[4] === undefined || (Number(match[4]) <= 23 && Number(match[5]) <= 59);
}

function validHostname(value: string): boolean {
    return (
        value.length > 0 &&
        value.length <= 253 &&
        !value.endsWith('.') &&
        value.split('.').every((label) => /^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$/.test(label))
    );
}

function validIpv4(value: string): boolean {
    const parts = value.split('.');
    return parts.length === 4 && parts.every((part) => /^(?:0|[1-9]\d{0,2})$/.test(part) && Number(part) <= 255);
}

function validIpv6(value: string): boolean {
    if (value.includes(':::') || value.split('::').length > 2) return false;
    const compressed = value.includes('::');
    const groups = value.split(':').filter(Boolean);
    if (
        !groups.every((group, index) =>
            group.includes('.') ? index === groups.length - 1 && validIpv4(group) : /^[\da-f]{1,4}$/i.test(group),
        )
    )
        return false;
    const count = groups.reduce((total, group) => total + (group.includes('.') ? 2 : 1), 0);
    return compressed ? count < 8 : count === 8;
}

function validUriText(value: string, absolute: boolean): boolean {
    if (/[\s\\<>^`{|}\0]/.test(value) || /%(?![\da-f]{2})/i.test(value)) return false;
    let rest = value;
    const scheme = /^[A-Za-z][A-Za-z0-9+.-]*:/.exec(rest);
    if (absolute && !scheme) return false;
    if (scheme) rest = rest.slice(scheme[0].length);
    const fragment = rest.indexOf('#');
    if (fragment !== -1) {
        if (!uriQuery.test(rest.slice(fragment + 1))) return false;
        rest = rest.slice(0, fragment);
    }
    const query = rest.indexOf('?');
    if (query !== -1) {
        if (!uriQuery.test(rest.slice(query + 1))) return false;
        rest = rest.slice(0, query);
    }
    if (rest.startsWith('//')) {
        const slash = rest.indexOf('/', 2);
        const authority = rest.slice(2, slash === -1 ? undefined : slash);
        const path = slash === -1 ? '' : rest.slice(slash);
        return validAuthority(authority) && uriPath.test(path);
    }
    if (!uriPath.test(rest)) return false;
    if (scheme || rest === '' || rest.startsWith('/')) return true;
    return !rest.slice(0, rest.indexOf('/') === -1 ? undefined : rest.indexOf('/')).includes(':');
}

const uriEncoded = '%[\\da-fA-F]{2}';
const uriPchar = `(?:[A-Za-z0-9._~!$&'()*+,;=:@-]|${uriEncoded})`;
const uriPath = new RegExp(`^(?:${uriPchar}|/)*$`);
const uriQuery = new RegExp(`^(?:${uriPchar}|[/?])*$`);
const uriRegName = new RegExp(`^(?:[A-Za-z0-9._~!$&'()*+,;=-]|${uriEncoded})*$`);
const uriUserInfo = new RegExp(`^(?:[A-Za-z0-9._~!$&'()*+,;=:-]|${uriEncoded})*$`);

function validAuthority(authority: string): boolean {
    const at = authority.lastIndexOf('@');
    if (at !== -1) {
        if (authority.indexOf('@') !== at || !uriUserInfo.test(authority.slice(0, at))) return false;
        authority = authority.slice(at + 1);
    }
    if (authority.startsWith('[')) {
        const close = authority.indexOf(']');
        if (close === -1) return false;
        const host = authority.slice(1, close);
        const suffix = authority.slice(close + 1);
        return (
            (validIpv6(host) || /^v[\da-f]+\.[A-Za-z0-9._~!$&'()*+,;=:-]+$/i.test(host)) &&
            (suffix === '' || /^:\d*$/.test(suffix))
        );
    }
    const colon = authority.lastIndexOf(':');
    let host = authority;
    if (colon !== -1) {
        if (authority.indexOf(':') !== colon || !/^\d*$/.test(authority.slice(colon + 1))) return false;
        host = authority.slice(0, colon);
    }
    return uriRegName.test(host);
}

function checkSchema(schema: Schema, path: string): void {
    if (typeof schema === 'boolean') return;
    if (!isRecord(schema)) throw new TypeError(`${path} must be a boolean or object schema.`);
    const supported = new Set([
        'type',
        'const',
        'enum',
        'properties',
        'required',
        'additionalProperties',
        'minProperties',
        'maxProperties',
        'items',
        'prefixItems',
        'minItems',
        'maxItems',
        'uniqueItems',
        'minLength',
        'maxLength',
        'pattern',
        'format',
        'minimum',
        'maximum',
        'exclusiveMinimum',
        'exclusiveMaximum',
        'multipleOf',
        'contains',
        'minContains',
        'maxContains',
        'patternProperties',
        'propertyNames',
        'dependentRequired',
        'dependentSchemas',
        'dependencies',
        'anyOf',
        'allOf',
        'oneOf',
        'not',
        'if',
        'then',
        'else',
        '$ref',
        '$defs',
        'definitions',
        'additionalItems',
        'x-guidance',
        'title',
        'description',
        '$schema',
        '$id',
        '$comment',
        'default',
        'examples',
        'deprecated',
        'readOnly',
        'writeOnly',
        'contentEncoding',
        'contentMediaType',
        'contentSchema',
    ]);
    for (const key of Object.keys(schema)) {
        if (!supported.has(key))
            throw new TypeError(`${path}: unsupported JSON Schema keyword ${JSON.stringify(key)}.`);
    }
    const configuredTypes = schema.type === undefined ? [] : Array.isArray(schema.type) ? schema.type : [schema.type];
    if (
        configuredTypes.some(
            (type) => !['null', 'boolean', 'number', 'integer', 'string', 'array', 'object'].includes(String(type)),
        )
    ) {
        throw new TypeError(`${path}.type contains an unsupported JSON type.`);
    }
    for (const keyword of [
        'not',
        'if',
        'then',
        'else',
        'contains',
        'propertyNames',
        'additionalProperties',
        'additionalItems',
    ]) {
        if (schema[keyword] !== undefined && !isSchema(schema[keyword])) {
            throw new TypeError(`${path}.${keyword} must be a boolean or object schema.`);
        }
    }
    for (const keyword of ['allOf', 'anyOf', 'oneOf', 'prefixItems']) {
        const value = schema[keyword];
        if (
            value !== undefined &&
            (!Array.isArray(value) ||
                (keyword !== 'prefixItems' && value.length === 0) ||
                value.some((child) => !isSchema(child)))
        ) {
            throw new TypeError(`${path}.${keyword} must be an array of schemas.`);
        }
    }
    for (const keyword of ['properties', 'patternProperties', 'dependentSchemas', '$defs', 'definitions']) {
        const value = schema[keyword];
        if (value !== undefined && (!isRecord(value) || Object.values(value).some((child) => !isSchema(child)))) {
            throw new TypeError(`${path}.${keyword} must be an object containing schemas.`);
        }
    }
    if (
        schema.items !== undefined &&
        !isSchema(schema.items) &&
        !(Array.isArray(schema.items) && schema.items.every(isSchema))
    ) {
        throw new TypeError(`${path}.items must be a schema or array of schemas.`);
    }
    if (
        schema.required !== undefined &&
        (!Array.isArray(schema.required) ||
            schema.required.some((key) => typeof key !== 'string') ||
            new Set(schema.required).size !== schema.required.length)
    ) {
        throw new TypeError(`${path}.required must be an array of unique strings.`);
    }
    if (schema.dependentRequired !== undefined) {
        if (!isRecord(schema.dependentRequired)) throw new TypeError(`${path}.dependentRequired must be an object.`);
        for (const dependency of Object.values(schema.dependentRequired)) {
            if (
                !Array.isArray(dependency) ||
                dependency.some((key) => typeof key !== 'string') ||
                new Set(dependency).size !== dependency.length
            ) {
                throw new TypeError(`${path}.dependentRequired values must be arrays of unique strings.`);
            }
        }
    }
    if (schema.dependencies !== undefined && !isRecord(schema.dependencies)) {
        throw new TypeError(`${path}.dependencies must be an object.`);
    }
    if (schema.$ref !== undefined && typeof schema.$ref !== 'string')
        throw new TypeError(`${path}.$ref must be a string.`);
    if (schema.enum !== undefined && (!Array.isArray(schema.enum) || schema.enum.length === 0)) {
        throw new TypeError(`${path}.enum must be a non-empty array.`);
    }
    for (const keyword of ['minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum', 'multipleOf']) {
        const value = schema[keyword];
        if (
            value !== undefined &&
            (typeof value !== 'number' || !Number.isFinite(value) || (keyword === 'multipleOf' && value <= 0))
        ) {
            throw new TypeError(
                `${path}.${keyword} must be ${keyword === 'multipleOf' ? 'a positive' : 'a finite'} number.`,
            );
        }
    }
    if (schema.pattern !== undefined) {
        assertPattern(schema.pattern, `${path}.pattern`);
        throw new TypeError(`${path}.pattern cannot be enforced incrementally and is unsupported.`);
    }
    if (schema.format !== undefined && typeof schema.format !== 'string') {
        throw new TypeError(`${path}.format must be a string.`);
    }
    if (typeof schema.format === 'string' && ASSERTED_FORMATS.has(schema.format)) {
        throw new TypeError(
            `${path}.format ${JSON.stringify(schema.format)} cannot be enforced incrementally and is unsupported.`,
        );
    }
    if (isRecord(schema.patternProperties)) {
        for (const pattern of Object.keys(schema.patternProperties))
            assertPattern(pattern, `${path}.patternProperties`);
    }
    if (schema.uniqueItems !== undefined && typeof schema.uniqueItems !== 'boolean') {
        throw new TypeError(`${path}.uniqueItems must be a boolean.`);
    }
    for (const keyword of [
        'minItems',
        'maxItems',
        'minContains',
        'maxContains',
        'minProperties',
        'maxProperties',
        'minLength',
        'maxLength',
    ]) {
        const value = schema[keyword];
        if (value !== undefined && (!Number.isInteger(value) || (value as number) < 0)) {
            throw new TypeError(`${path}.${keyword} must be a non-negative integer.`);
        }
    }
    if (schema['x-guidance'] !== undefined && path !== '$') {
        throw new TypeError('x-guidance is only supported on the root schema.');
    }
    schemaChildren(schema).forEach((child, index) => checkSchema(child, `${path}.schema[${index}]`));
}

function allowsKind(schema: Record<string, unknown>, kind: Kind): boolean {
    if (
        schema.const !== undefined &&
        valueKind(schema.const) !== kind &&
        !(kind === 'number' && valueKind(schema.const) === 'integer')
    )
        return false;
    if (
        Array.isArray(schema.enum) &&
        !schema.enum.some((value) => valueKind(value) === kind || (kind === 'number' && valueKind(value) === 'integer'))
    )
        return false;
    if (schema.type === undefined) return true;
    const types = Array.isArray(schema.type) ? schema.type : [schema.type];
    return (
        types.includes(kind) ||
        (kind === 'number' && types.includes('integer')) ||
        (kind === 'integer' && types.includes('number'))
    );
}

function valueKind(value: unknown): Kind | undefined {
    if (value === null) return 'null';
    if (Array.isArray(value)) return 'array';
    if (isRecord(value)) return 'object';
    if (typeof value === 'number') return Number.isInteger(value) ? 'integer' : 'number';
    if (['boolean', 'string'].includes(typeof value)) return typeof value as Kind;
    return undefined;
}

function byteKind(byte: number): Kind | undefined {
    if (byte === 0x22) return 'string';
    if (byte === 0x7b) return 'object';
    if (byte === 0x5b) return 'array';
    if (byte === 0x74 || byte === 0x66) return 'boolean';
    if (byte === 0x6e) return 'null';
    if (byte === 0x2d || isDigit(byte)) return 'number';
    return undefined;
}

function numberPrefixValid(value: string): boolean {
    // The exponent may only start after at least one fraction digit — "1.e"
    // would be a syntactically extendable prefix that can never complete.
    return (
        /^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d*)?$/.test(value) ||
        /^-?(?:0|[1-9]\d*)\.$/.test(value) ||
        value === '-'
    );
}

function numberComplete(value: string): boolean {
    return /^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$/.test(value);
}

function isNumberByte(byte: number): boolean {
    return isDigit(byte) || byte === 0x2e || byte === 0x45 || byte === 0x65 || byte === 0x2b || byte === 0x2d;
}

function decodeString(bytes: readonly number[]): string | null {
    try {
        return decoder.decode(Uint8Array.from(bytes));
    } catch {
        return null;
    }
}

function topObject(state: JsonState): ObjectFrame {
    return state.stack.at(-1) as ObjectFrame;
}

function replaceTop(stack: readonly Frame[], frame: Frame): Frame[] {
    return [...stack.slice(0, -1), frame];
}

function allowsWhitespace(mode: Mode): boolean {
    return ['value', 'array-value', 'object-key', 'colon', 'item-separator', 'after-value', 'done'].includes(mode);
}

function isWhitespace(byte: number): boolean {
    return byte === 0x09 || byte === 0x0a || byte === 0x0d || byte === 0x20;
}

function isDigit(byte: number): boolean {
    return byte >= 0x30 && byte <= 0x39;
}

function isHex(byte: number): boolean {
    return isDigit(byte) || (byte >= 0x41 && byte <= 0x46) || (byte >= 0x61 && byte <= 0x66);
}

function isSchema(value: unknown): value is Schema {
    return typeof value === 'boolean' || isRecord(value);
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return value !== null && typeof value === 'object' && !Array.isArray(value);
}
