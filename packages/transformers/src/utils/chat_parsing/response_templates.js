import { compileRegex, hasContentParser, validateTransformStrings } from './content_parsers.js';

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

    const startAnchor = compileAnchor(spec, 'response_template', 'start_anchor', 'start_anchor_pattern');
    if (startAnchor === null)
        throw new Error("response_template must define 'start_anchor' or 'start_anchor_pattern'.");
    return {
        __responseTemplate: true,
        defaults: { ...(spec.defaults ?? {}) },
        fields,
        implicit,
        startAnchor,
        truncatePastLastAnchor(text) {
            let end = null;
            for (const match of findMatches(startAnchor, text, 0)) end = match.end;
            return end === null ? text : text.slice(end);
        },
    };
}

function buildField(name, raw) {
    assertObject(raw, `Field '${name}'`);
    assertKnownKeys(raw, FIELD_KEYS, `Field '${name}'`);
    const content = raw.content ?? 'text';
    if (!hasContentParser(content)) throw new Error(`Unknown response_template content parser: ${content}`);
    const open = compileAnchor(raw, `Field '${name}'`, 'open', 'open_pattern');
    const close = compileAnchor(raw, `Field '${name}'`, 'close', 'close_pattern');
    if (raw.join !== undefined && typeof raw.join !== 'string')
        throw new Error(`Field '${name}': 'join' must be a string.`);
    if (raw.join !== undefined && !raw.repeats) throw new Error(`Field '${name}': 'join' requires 'repeats': true.`);
    if (raw.transform_each !== undefined && typeof raw.transform_each !== 'boolean') {
        throw new Error(`Field '${name}': transform_each must be a boolean.`);
    }
    if (raw.transform_each && raw.transform === undefined) {
        throw new Error(`Field '${name}': transform_each requires transform.`);
    }
    const transform = raw.transform ?? null;
    validateTransformStrings(`Field '${name}'`, transform);
    const namedGroups = [...(open?.namedGroups ?? []), ...(close?.namedGroups ?? [])];
    if (namedGroups.length && transform === null) {
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
        optional: raw.optional ?? true,
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
        return {
            literals,
            pattern: null,
            canExtend: literals.some((a) => literals.some((b) => a !== b && a.startsWith(b))),
            namedGroups: [],
        };
    }
    if (source[patternKey] !== undefined) {
        if (typeof source[patternKey] !== 'string') throw new Error(`${scope}: '${patternKey}' must be a string.`);
        const pattern = compileRegex(source[patternKey]);
        return {
            literals: null,
            pattern,
            canExtend: true,
            namedGroups: [...pattern.source.matchAll(/\(\?<([A-Za-z_]\w*)>/g)].map((match) => match[1]),
        };
    }
    return null;
}

export function findMatches(anchor, text, position) {
    const matches = [];
    if (anchor.literals) {
        for (const literal of anchor.literals) {
            let index = text.indexOf(literal, position);
            while (index >= 0) {
                matches.push({ start: index, end: index + literal.length, groups: {} });
                index = text.indexOf(literal, index + 1);
            }
        }
    } else {
        anchor.pattern.lastIndex = position;
        for (const match of text.matchAll(anchor.pattern)) {
            matches.push({ start: match.index, end: match.index + match[0].length, groups: match.groups ?? {} });
        }
    }
    matches.sort((a, b) => a.start - b.start || b.end - b.start - (a.end - a.start));
    return matches.filter((match, index) => index === 0 || match.start !== matches[index - 1].start);
}

function assertObject(value, name) {
    if (value === null || typeof value !== 'object' || Array.isArray(value))
        throw new Error(`${name} must be an object.`);
}

function assertKnownKeys(value, allowed, name) {
    const unknown = Object.keys(value).filter((key) => !allowed.has(key));
    if (unknown.length) throw new Error(`${name} has unknown keys: ${unknown.sort()}`);
}
