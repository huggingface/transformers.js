const CONTENT_PARSERS = {
    text: parseText,
    int: parseInteger,
    float: parseFloatValue,
    bool: parseBoolean,
    json: parseJSON,
    'xml-inline': parseXMLInline,
    'kv-lines': parseKeyValueLines,
};

export const STREAMABLE_PARSERS = new Set(['text', 'int', 'float', 'bool']);

function parseText(text, args) {
    return (args.strip ?? true) ? text.trim() : text;
}

function parseInteger(text, args) {
    const value = parseText(text, args);
    if (!/^[+-]?\d+$/.test(value)) throw new Error(`Could not parse ${JSON.stringify(value)} as an integer.`);
    return Number.parseInt(value, 10);
}

function parseFloatValue(text, args) {
    const value = parseText(text, args);
    if (!value || !Number.isFinite(Number(value)))
        throw new Error(`Could not parse ${JSON.stringify(value)} as a float.`);
    return Number(value);
}

function parseBoolean(text, args) {
    return ['true', '1'].includes(parseText(text, args).toLowerCase());
}

const LAX_OPEN = '\x01';
const LAX_CLOSE = '\x02';

function parseJSON(text, args) {
    const stringDelimiters = args.string_delims ?? [];
    if (stringDelimiters.length && (text.includes(LAX_OPEN) || text.includes(LAX_CLOSE))) {
        throw new Error('json: input contains reserved sentinel characters (\\x01/\\x02); cannot parse safely.');
    }

    let working = text;
    const captured = [];
    for (const [open, close] of stringDelimiters) {
        const pattern = new RegExp(`${escapeRegex(open)}(.*?)${escapeRegex(close)}`, 'gs');
        working = working.replace(pattern, (_, value) => {
            const index = captured.push(value) - 1;
            return `${LAX_OPEN}${index}${LAX_CLOSE}`;
        });
    }
    if (args.unquoted_keys) working = working.replace(/(?<=[{,])(\w+):/gu, '"$1":');
    for (let i = 0; i < captured.length; ++i) {
        working = working.replaceAll(`${LAX_OPEN}${i}${LAX_CLOSE}`, JSON.stringify(captured[i]));
    }

    try {
        return JSON.parse(working);
    } catch (error) {
        if (args.allow_non_json) return parseText(text, args);
        const detail =
            working === text
                ? `Content: ${JSON.stringify(text)}`
                : `Original: ${JSON.stringify(text)}\nTransformed: ${JSON.stringify(working)}`;
        throw new Error(`json parser could not parse region as JSON.\n${detail}\nError: ${error.message}`);
    }
}

function parseXMLInline(text, args) {
    if (typeof args.tag_pattern !== 'string') throw new Error("xml-inline: 'tag_pattern' content_arg is required");
    const pattern = compileRegex(args.tag_pattern);
    const output = {};
    for (const match of text.matchAll(pattern)) {
        const key = match.groups?.key;
        if (key === undefined) {
            throw new Error(`xml-inline: tag_pattern must have a named group 'key'. Pattern: ${args.tag_pattern}`);
        }
        const value = parseNested(match.groups?.value ?? '', args.value_parser);
        if (Object.hasOwn(output, key) && args.merge_duplicates) {
            output[key] = Array.isArray(output[key]) ? [...output[key], value] : [output[key], value];
        } else {
            output[key] = value;
        }
    }
    return output;
}

function parseKeyValueLines(text, args) {
    const lineSeparator = args.line_sep ?? '\n';
    const keyValueSeparator = args.kv_sep ?? ':';
    const output = {};
    for (let line of text.split(lineSeparator)) {
        line = parseText(line, args);
        const index = line.indexOf(keyValueSeparator);
        if (!line || index < 0) continue;
        const key = parseText(line.slice(0, index), args);
        const value = parseText(line.slice(index + keyValueSeparator.length), args);
        output[key] = parseNested(value, args.value_parser);
    }
    return output;
}

function parseNested(value, parser) {
    return parser ? parseContent(value, parser.name ?? 'text', parser.args ?? {}) : value;
}

export function parseContent(text, name, args = {}) {
    const parser = CONTENT_PARSERS[name];
    if (!parser) throw new Error(`Unknown response_template content parser: ${name}`);
    return parser(text, args);
}

export function hasContentParser(name) {
    return Object.hasOwn(CONTENT_PARSERS, name);
}

const PLACEHOLDER = /^\{(\w+(?:\.\w+)*)\}$/;
const ANY_PLACEHOLDER = /\{\w+(?:\.\w+)*\}/;

function applyTransform(transform, scope) {
    if (Array.isArray(transform)) return transform.map((value) => applyTransform(value, scope));
    if (isObject(transform))
        return Object.fromEntries(Object.entries(transform).map(([key, value]) => [key, applyTransform(value, scope)]));
    if (typeof transform !== 'string') return transform;
    const match = PLACEHOLDER.exec(transform);
    if (!match) return transform;

    const [root, ...keys] = match[1].split('.');
    if (!Object.hasOwn(scope, root)) {
        throw new Error(
            `Transform placeholder '{${match[1]}}' is not defined. Available: ${Object.keys(scope).sort()}`,
        );
    }
    let value = scope[root];
    for (const key of keys) {
        if (!isObject(value) || !Object.hasOwn(value, key)) {
            throw new Error(`Transform placeholder '{${match[1]}}' cannot resolve key '${key}'.`);
        }
        value = value[key];
    }
    return value;
}

export function validateTransformStrings(scope, transform) {
    if (Array.isArray(transform)) {
        for (const value of transform) validateTransformStrings(scope, value);
    } else if (isObject(transform)) {
        for (const value of Object.values(transform)) validateTransformStrings(scope, value);
    } else if (typeof transform === 'string' && ANY_PLACEHOLDER.test(transform) && !PLACEHOLDER.test(transform)) {
        throw new Error(
            `${scope}: transform string ${JSON.stringify(transform)} mixes a placeholder with literal text.`,
        );
    }
}

export function processField(body, field, captures) {
    const value = parseContent(body, field.content, field.contentArgs);
    if (field.transform === null) return value;
    if (!field.transformEach) return applyTransform(field.transform, { ...captures, content: value });
    if (!Array.isArray(value)) throw new Error(`Field '${field.name}': transform_each requires an array.`);
    return value.map((item) => {
        if (!isObject(item)) throw new Error(`Field '${field.name}': transform_each requires object elements.`);
        return applyTransform(field.transform, { ...captures, ...item });
    });
}

export function compileRegex(pattern) {
    return new RegExp(pattern.replaceAll('(?P<', '(?<').replaceAll('\\Z', '$(?![\\s\\S])'), 'gs');
}

function escapeRegex(value) {
    return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function isObject(value) {
    return value !== null && typeof value === 'object' && !Array.isArray(value);
}
