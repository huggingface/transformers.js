import { compileRegex, hasContentParser, validateTransformStrings } from './content_parsers.js';

// Start anchors locate the assistant response within a prompt prefix upstream. This parser only
// receives generated response text, but accepts the metadata so upstream tokenizer configs load.
const TOP_LEVEL_KEYS = new Set(['version', 'defaults', 'fields', 'start_anchor', 'start_anchor_pattern']);
const FIELD_KEYS = new Set([
    'open',
    'open_pattern',
    'close',
    'close_pattern',
    'content',
    'content_args',
    'repeats',
    'join',
    'optional',
    'transform',
    'transform_each',
]);

export function loadResponseTemplate(spec) {
    if (spec?.__responseTemplate === true) return spec;
    assertObject(spec, 'response_template');
    if ((spec.version ?? 1) !== 1) throw new Error(`Unsupported response_template version: ${spec.version}`);
    assertKnownKeys(spec, TOP_LEVEL_KEYS, 'response_template');
    if (spec.defaults !== undefined) assertObject(spec.defaults, 'response_template.defaults');
    assertObject(spec.fields, 'response_template.fields');
    if (Object.keys(spec.fields).length === 0) throw new Error('response_template.fields must not be empty.');

    const fields = {};
    let implicit = null;
    for (const [name, raw] of Object.entries(spec.fields)) {
        const field = buildField(name, raw);
        fields[name] = field;
        if (field.open === null) {
            if (implicit !== null) throw new Error('response_template may define at most one field without an opener.');
            implicit = name;
        }
    }
    return { __responseTemplate: true, defaults: { ...(spec.defaults ?? {}) }, fields, implicit };
}

function buildField(name, raw) {
    assertObject(raw, `Field '${name}'`);
    assertKnownKeys(raw, FIELD_KEYS, `Field '${name}'`);
    const content = raw.content ?? 'text';
    if (!hasContentParser(content)) throw new Error(`Unknown response_template content parser: ${content}`);
    const open = compileAnchor(raw, `Field '${name}'`, 'open', 'open_pattern');
    const close = compileAnchor(raw, `Field '${name}'`, 'close', 'close_pattern');
    if (raw.join !== undefined && typeof raw.join !== 'string')
        throw new Error(`Field '${name}': join must be a string.`);
    if (raw.join !== undefined && !raw.repeats) throw new Error(`Field '${name}': join requires repeats.`);
    if (raw.transform_each && raw.transform === undefined)
        throw new Error(`Field '${name}': transform_each requires transform.`);
    const transform = raw.transform ?? null;
    validateTransformStrings(`Field '${name}'`, transform);
    if ([...(open?.namedGroups ?? []), ...(close?.namedGroups ?? [])].length && transform === null) {
        throw new Error(`Field '${name}' has named regex groups but no transform.`);
    }
    return {
        name,
        open,
        close,
        content,
        contentArgs: raw.content_args ?? {},
        repeats: raw.repeats ?? false,
        join: raw.join ?? null,
        transform,
        transformEach: raw.transform_each ?? false,
    };
}

function compileAnchor(source, scope, literalKey, patternKey) {
    if (source[literalKey] !== undefined && source[patternKey] !== undefined) {
        throw new Error(`${scope}: cannot specify both '${literalKey}' and '${patternKey}'.`);
    }
    if (source[literalKey] !== undefined) {
        const raw = source[literalKey];
        const values = typeof raw === 'string' ? [raw] : raw;
        if (
            !Array.isArray(values) ||
            values.length === 0 ||
            values.some((value) => typeof value !== 'string' || !value)
        ) {
            throw new Error(`${scope}: '${literalKey}' must be a non-empty string or array of non-empty strings.`);
        }
        const literals = [...new Set(values)];
        return { literals, pattern: null, partialPrefixes: literals, namedGroups: [] };
    }
    if (source[patternKey] !== undefined) {
        if (typeof source[patternKey] !== 'string') throw new Error(`${scope}: '${patternKey}' must be a string.`);
        const pattern = compileRegex(source[patternKey]);
        return {
            literals: null,
            pattern,
            partialPrefixes: literalRegexPrefix(source[patternKey]),
            namedGroups: [...pattern.source.matchAll(/\(\?<([A-Za-z_]\w*)>/g)].map((match) => match[1]),
        };
    }
    return null;
}

export function findMatch(anchor, text, position) {
    let best = null;
    if (anchor.literals) {
        for (const literal of anchor.literals) {
            const start = text.indexOf(literal, position);
            if (start < 0) continue;
            const match = { start, end: start + literal.length, groups: {} };
            if (!best || match.start < best.start || (match.start === best.start && match.end > best.end)) best = match;
        }
    } else {
        anchor.pattern.lastIndex = position;
        const match = anchor.pattern.exec(text);
        if (match?.[0].length) {
            best = { start: match.index, end: match.index + match[0].length, groups: match.groups ?? {} };
        }
    }
    return best;
}

export function partialLiteralStart(anchors, text, position) {
    let start = text.length;
    for (const anchor of anchors) {
        if (!anchor?.partialPrefixes?.length) continue;
        for (let index = position; index < text.length; ++index) {
            const suffix = text.slice(index);
            if (anchor.partialPrefixes.some((literal) => literal.startsWith(suffix) && literal !== suffix)) {
                start = Math.min(start, index);
                break;
            }
        }
    }
    return start;
}

function literalRegexPrefix(pattern) {
    let prefix = '';
    for (let i = pattern.startsWith('^') ? 1 : 0; i < pattern.length; ++i) {
        const character = pattern[i];
        if (character === '\\') {
            const escaped = pattern[++i];
            if (escaped === undefined || /[dDsSwWbBAZ]/.test(escaped)) break;
            prefix += escaped === 'n' ? '\n' : escaped === 't' ? '\t' : escaped;
        } else if (/[.*+?()[\]{}$^|]/.test(character)) {
            break;
        } else {
            prefix += character;
        }
    }
    return prefix ? [prefix] : [];
}

function assertObject(value, name) {
    if (value === null || typeof value !== 'object' || Array.isArray(value))
        throw new Error(`${name} must be an object.`);
}

function assertKnownKeys(value, allowed, name) {
    const unknown = Object.keys(value).filter((key) => !allowed.has(key));
    if (unknown.length) throw new Error(`${name} has unknown keys: ${unknown.sort()}`);
}
