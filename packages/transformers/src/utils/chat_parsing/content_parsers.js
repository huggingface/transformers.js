/**
 * The parsers used by chat response parsing. Each parser takes a chunk of captured text and parses it
 * into a single key in the output message.
 */

import { compile_pattern, escape_pattern, search } from './regex.js';

export const is_object = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);
export const type_name = (value) => (value === null ? 'null' : Array.isArray(value) ? 'array' : typeof value);

/**
 * Python's `int(text)`: optional sign, digits with optional single underscores, surrounding whitespace.
 * @param {string} text
 * @returns {number}
 */
export function to_int(text) {
    const value = text.trim();
    if (!/^[+-]?\d+(?:_\d+)*$/.test(value)) throw new Error(`invalid literal for int(): ${JSON.stringify(text)}`);
    return Number(value.replaceAll('_', ''));
}

const DIGITS = String.raw`\d(?:_?\d)*`;
const FLOAT = new RegExp(
    String.raw`^[+-]?(?:(?:${DIGITS}(?:\.(?:${DIGITS})?)?|\.${DIGITS})(?:e[+-]?${DIGITS})?|inf(?:inity)?|nan)$`,
);

/**
 * Python's `float(text)`, including `inf`, `infinity` and `nan`.
 * @param {string} text
 * @returns {number}
 */
export function to_float(text) {
    const value = text.trim().toLowerCase();
    if (!FLOAT.test(value)) throw new Error(`could not convert string to float: ${JSON.stringify(text)}`);
    return value.endsWith('nan') ? NaN : Number(value.replaceAll('_', '').replace(/inf(?:inity)?$/, 'Infinity'));
}

function _text(text, args) {
    let value = (args.strip ?? true) ? text.trim() : text;
    if (args.strip_prefix && value.startsWith(args.strip_prefix)) {
        value = value.slice(args.strip_prefix.length);
    }
    if (args.strip_suffix && value.endsWith(args.strip_suffix)) {
        value = value.slice(0, -args.strip_suffix.length);
    }
    return (args.strip ?? true) ? value.trim() : value;
}

function _int(text, args) {
    return to_int(_text(text, args));
}

function _float(text, args) {
    return to_float(_text(text, args));
}

function _bool(text, args) {
    return ['true', '1'].includes(_text(text, args).toLowerCase());
}

// Sentinel characters for lax-JSON string pre-extraction — ASCII control chars
// that should never appear in real LLM output.
const LAX_OPEN = '\x01';
const LAX_CLOSE = '\x02';

/**
 * JSON parser with optional dialect knobs for LLM-emitted quirks.
 *
 * `args`:
 *   - `unquoted_keys` (bool): quote bare-identifier keys before parsing.
 *   - `string_delims` ([[open, close], ...]): strings delimited by these
 *     custom markers are pre-extracted, then restored as standard JSON strings.
 *   - `prefix` / `suffix` (string): restore syntax consumed by region delimiters.
 *   - `allow_non_json` (bool): return stripped text if parsing fails.
 */
function _json(text, args) {
    const string_delims = args.string_delims ?? [];
    const unquoted_keys = args.unquoted_keys ?? false;

    if (string_delims.length && (text.includes(LAX_OPEN) || text.includes(LAX_CLOSE))) {
        throw new Error('json: input contains reserved sentinel characters (\\x01/\\x02); cannot parse safely.');
    }

    let working = `${args.prefix ?? ''}${text}${args.suffix ?? ''}`;
    const captured = [];
    for (const [open_d, close_d] of string_delims) {
        const pattern = compile_pattern(`${escape_pattern(open_d)}(.*?)${escape_pattern(close_d)}`);
        working = working.replace(pattern.regex, (_, value) => `${LAX_OPEN}${captured.push(value) - 1}${LAX_CLOSE}`);
    }

    if (unquoted_keys) {
        working = working.replace(compile_pattern(String.raw`(?<=[{,])(\w+):`).regex, '"$1":');
    }

    for (let i = 0; i < captured.length; ++i) {
        working = working.replace(`${LAX_OPEN}${i}${LAX_CLOSE}`, () => JSON.stringify(captured[i]));
    }

    try {
        return JSON.parse(working);
    } catch (error) {
        if (args.allow_non_json) return _text(text, args);
        const transformed = working === text ? '' : `\nTransformed: ${JSON.stringify(working)}`;
        throw new Error(
            `json parser could not parse region as JSON.\nContent: ${JSON.stringify(text)}${transformed}\nError: ${error.message}`,
        );
    }
}

function _sub_parse(raw, value_parser) {
    if (value_parser == null) return raw;
    return parse_content(raw, value_parser.name ?? 'text', value_parser.args ?? {});
}

/**
 * Parse shallow XML-ish tags into an object. `tag_pattern` regex must have named
 * groups `key` and `value`. Optional `value_parser` recurses; `merge_duplicates`
 * collects duplicate keys into a list.
 */
function _xml_inline(text, args) {
    const tag_pattern = args.tag_pattern;
    if (tag_pattern == null) throw new Error("xml-inline: 'tag_pattern' content_arg is required");
    const value_parser = args.value_parser;
    const merge = args.merge_duplicates ?? false;

    // A Map keeps model-generated keys (e.g. `__proto__`) from touching object prototypes
    const out = new Map();
    for (const m of text.matchAll(compile_pattern(tag_pattern).regex)) {
        const key = m.groups?.key;
        if (key === undefined) {
            throw new Error(`xml-inline: tag_pattern must have a named group 'key'. Pattern: ${tag_pattern}`);
        }
        const value = _sub_parse(m.groups.value ?? '', value_parser);
        if (out.has(key) && merge) {
            if (!Array.isArray(out.get(key))) out.set(key, [out.get(key)]);
            out.get(key).push(value);
        } else {
            out.set(key, value);
        }
    }
    return Object.fromEntries(out);
}

/** Parse line-delimited `key<sep>value` pairs into an object. */
function _kv_lines(text, args) {
    const line_sep = args.line_sep ?? '\n';
    const kv_sep = args.kv_sep ?? ':';
    const value_parser = args.value_parser;

    const out = new Map();
    for (let line of text.split(line_sep)) {
        line = _text(line, args);
        const index = line.indexOf(kv_sep);
        if (!line || index < 0) continue;
        const k = _text(line.slice(0, index), args);
        const v = _text(line.slice(index + kv_sep.length), args);
        out.set(k, _sub_parse(v, value_parser));
    }
    return Object.fromEntries(out);
}

/** Parse a Python-style list of function calls without evaluating model output. */
function _python_call_list(text) {
    let pos = 0;
    const fail = (message) => {
        throw new Error(`python-call-list: ${message} at position ${pos}. Content: ${JSON.stringify(text)}`);
    };
    const skip = () => {
        while (/\s/.test(text[pos] ?? '')) ++pos;
    };
    const take = (value) => {
        skip();
        if (!text.startsWith(value, pos)) fail(`expected ${JSON.stringify(value)}`);
        pos += value.length;
    };
    const identifier = () => {
        skip();
        const match = /^[A-Za-z_]\w*/.exec(text.slice(pos));
        if (match === null) fail('expected identifier');
        pos += match[0].length;
        return match[0];
    };
    const string = () => {
        skip();
        const quote = text[pos++];
        let out = '';
        while (pos < text.length) {
            const char = text[pos++];
            if (char === quote) return out;
            if (char !== '\\') {
                out += char;
                continue;
            }
            if (pos >= text.length) fail('unterminated escape sequence');
            const escaped = text[pos++];
            const simple = { a: '\x07', b: '\b', f: '\f', n: '\n', r: '\r', t: '\t', v: '\v' };
            if (Object.hasOwn(simple, escaped)) {
                out += simple[escaped];
            } else if (/[0-7]/.test(escaped)) {
                let digits = escaped;
                while (digits.length < 3 && /[0-7]/.test(text[pos] ?? '')) digits += text[pos++];
                out += String.fromCodePoint(Number.parseInt(digits, 8));
            } else if (escaped === '\n') {
                // Python ignores an escaped physical newline.
            } else if (escaped === '\r') {
                if (text[pos] === '\n') ++pos;
            } else if (escaped === '\\' || escaped === '"' || escaped === "'") {
                out += escaped;
            } else if (escaped === 'N') {
                fail('named unicode escapes are not supported');
            } else if (escaped === 'x' || escaped === 'u' || escaped === 'U') {
                const size = escaped === 'x' ? 2 : escaped === 'u' ? 4 : 8;
                const digits = text.slice(pos, pos + size);
                if (!new RegExp(`^[0-9a-fA-F]{${size}}$`).test(digits)) fail(`invalid \\${escaped} escape`);
                const codepoint = Number.parseInt(digits, 16);
                if (codepoint > 0x10ffff) fail('unicode escape is outside the valid range');
                out += String.fromCodePoint(codepoint);
                pos += size;
            } else {
                out += `\\${escaped}`;
            }
        }
        fail('unterminated string');
    };
    const sequence = (open, close) => {
        take(open);
        const values = [];
        skip();
        if (text[pos] === close) {
            ++pos;
            return values;
        }
        while (true) {
            values.push(value());
            skip();
            if (text[pos] === close) {
                ++pos;
                return values;
            }
            take(',');
            skip();
            if (text[pos] === close) {
                ++pos;
                return values;
            }
        }
    };
    const mapping = () => {
        take('{');
        const entries = [];
        skip();
        if (text[pos] === '}') {
            ++pos;
            return {};
        }
        while (true) {
            skip();
            const key = text[pos] === '"' || text[pos] === "'" ? string() : identifier();
            take(':');
            entries.push([key, value()]);
            skip();
            if (text[pos] === '}') {
                ++pos;
                return Object.fromEntries(entries);
            }
            take(',');
            skip();
            if (text[pos] === '}') {
                ++pos;
                return Object.fromEntries(entries);
            }
        }
    };
    const value = () => {
        skip();
        const char = text[pos];
        if (char === '"' || char === "'") return string();
        if (char === '[') return sequence('[', ']');
        if (char === '(') return sequence('(', ')');
        if (char === '{') return mapping();
        const number = /^[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)/.exec(text.slice(pos));
        if (number !== null) {
            pos += number[0].length;
            return Number(number[0]);
        }
        const name = identifier();
        if (name === 'True' || name === 'true') return true;
        if (name === 'False' || name === 'false') return false;
        if (name === 'None' || name === 'null') return null;
        fail(`unsupported value ${JSON.stringify(name)}`);
    };
    const call = () => {
        const name = identifier();
        take('(');
        const entries = [];
        skip();
        if (text[pos] === ')') {
            ++pos;
            return { name, arguments: {} };
        }
        while (true) {
            const key = identifier();
            take('=');
            entries.push([key, value()]);
            skip();
            if (text[pos] === ')') {
                ++pos;
                return { name, arguments: Object.fromEntries(entries) };
            }
            take(',');
            skip();
            if (text[pos] === ')') {
                ++pos;
                return { name, arguments: Object.fromEntries(entries) };
            }
        }
    };

    take('[');
    const calls = [];
    skip();
    if (text[pos] !== ']') {
        while (true) {
            calls.push(call());
            skip();
            if (text[pos] === ']') break;
            take(',');
            skip();
            if (text[pos] === ']') break;
        }
    }
    take(']');
    skip();
    if (pos !== text.length) fail('unexpected trailing content');
    return calls;
}

export const CONTENT_PARSERS = {
    text: _text,
    int: _int,
    float: _float,
    bool: _bool,
    json: _json,
    'xml-inline': _xml_inline,
    'kv-lines': _kv_lines,
    'python-call-list': _python_call_list,
};

// Parsers whose output is the verbatim body text (modulo whitespace) — chunks
// from these fields stream with `dirty=false` because each chunk is part of
// the final value. Structured parsers (`json`, `xml-inline`, `kv-lines`) only
// produce a meaningful value on close, so their chunks stream raw text
// flagged `dirty=true` while the parsed value is delivered in `region_close`.
export const STREAMABLE_PARSERS = new Set(['text', 'int', 'float', 'bool']);

/**
 * Parse `text` with the named content parser.
 * @param {string} text The text to parse.
 * @param {string} name The content parser name.
 * @param {Object} args The parser's `content_args`.
 * @returns {any}
 */
export function parse_content(text, name, args) {
    if (!Object.hasOwn(CONTENT_PARSERS, name)) throw new Error(`Unknown content parser '${name}'`);
    return CONTENT_PARSERS[name](text, args);
}

const PLACEHOLDER = compile_pattern(String.raw`\{(\w+(?:\.\w+)*)\}`);

/** The placeholder path if `text` is entirely a `{name}` placeholder, otherwise `null`. */
function full_placeholder(text) {
    PLACEHOLDER.sticky.lastIndex = 0;
    const m = PLACEHOLDER.sticky.exec(text);
    return m !== null && m[0].length === text.length ? m[1] : null;
}

/**
 * Recursively walk a transform template, which is used to restructure
 * parsed output into the actual shape we want. A dotted placeholder like
 * `{content.args}` descends into keys of the looked-up value.
 */
function _apply_transform(transform, scope) {
    if (Array.isArray(transform)) return transform.map((v) => _apply_transform(v, scope));
    if (is_object(transform)) {
        return Object.fromEntries(Object.entries(transform).map(([k, v]) => [k, _apply_transform(v, scope)]));
    }
    if (typeof transform !== 'string') return transform;
    const path = full_placeholder(transform);
    if (path === null) return transform;
    const [root, ...keys] = path.split('.');
    if (!Object.hasOwn(scope, root)) {
        throw new Error(
            `transform placeholder '{${path}}' is not defined. Available: ${JSON.stringify(Object.keys(scope).sort())}`,
        );
    }
    let value = scope[root];
    for (const key of keys) {
        if (!is_object(value)) {
            throw new Error(`transform placeholder '{${path}}' cannot index into ${type_name(value)} at '${key}'`);
        }
        if (!Object.hasOwn(value, key)) {
            throw new Error(
                `transform placeholder '{${path}}' is missing key '${key}'. Available: ${JSON.stringify(Object.keys(value).sort())}`,
            );
        }
        value = value[key];
    }
    return value;
}

/**
 * Walk a transform template and reject any string that mixes a `{name}`
 * placeholder with literal text. Only whole-string placeholders (e.g.
 * `"{content}"`) and plain literals are supported. Called from the template
 * loader so authors get a clear error at load time, not at parse time.
 * @param {string} scope Description of the field, for error messages.
 * @param {any} transform The transform template.
 */
export function validate_transform_strings(scope, transform) {
    if (Array.isArray(transform) || is_object(transform)) {
        for (const v of Object.values(transform)) validate_transform_strings(scope, v);
        return;
    }
    if (typeof transform !== 'string') return;
    if (search(PLACEHOLDER, transform, 0) !== null && full_placeholder(transform) === null) {
        throw new Error(
            `${scope}: transform string ${JSON.stringify(transform)} mixes a {placeholder} with literal text. ` +
                'Use either a whole-string placeholder (e.g. "{content}") or a plain literal; ' +
                'string interpolation is not supported.',
        );
    }
}

/**
 * Run `body` through the field's content parser, then optionally apply the
 * transform template. When `transform_each` is set, the parsed content must
 * be a list and the template is applied to each element (with the element's
 * keys unpacked into the template scope, alongside any regex captures).
 * @param {string} body The region's text.
 * @param {import('./response_templates.js').ResponseTemplateField} field The field being closed.
 * @param {Record<string, string>} captures Named groups captured by the field's open pattern.
 * @returns {any}
 */
export function process_field(body, field, captures) {
    const value = parse_content(body, field.content, field.content_args);
    if (field.transform === null) return value;
    if (field.transform_each) {
        if (!Array.isArray(value)) {
            throw new Error(
                `Field '${field.name}': transform_each requires the parsed content to be a list, got ${type_name(value)}.`,
            );
        }
        return value.map((item) => {
            if (!is_object(item)) {
                throw new Error(
                    `Field '${field.name}': transform_each requires each list element to be a dict, got ${type_name(item)}.`,
                );
            }
            return _apply_transform(field.transform, { ...captures, ...item });
        });
    }
    return _apply_transform(field.transform, { ...captures, content: value });
}
