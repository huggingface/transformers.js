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
    return (args.strip ?? true) ? text.trim() : text;
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

const NON_FINITE = { NaN: NaN, Infinity: Infinity, '-Infinity': -Infinity };
const NON_FINITE_TOKEN = /-Infinity|Infinity|NaN/y;

/**
 * Python's `json.loads` accepts `NaN`, `Infinity` and `-Infinity`, but `JSON.parse` doesn't. Replace them (outside
 * strings) in value positions. All original string values are escaped into a separate namespace,
 * so the reviver never mistakes user strings for a non-finite placeholder.
 * @param {string} text
 * @returns {string|null} The rewritten text, or `null` if there was nothing to replace.
 */
function _replace_non_finite(text) {
    let out = '';
    let replaced = false;
    let previous;
    for (let i = 0; i < text.length; ++i) {
        const char = text[i];
        if (char === '"') {
            const start = i++;
            while (i < text.length && text[i] !== '"') {
                if (text[i] === '\\') ++i;
                ++i;
            }
            if (i >= text.length) return null;
            const quoted = text.slice(start, i + 1);
            let next = i + 1;
            while (/[ \t\r\n]/.test(text[next] ?? '')) ++next;
            try {
                out += text[next] === ':' ? quoted : JSON.stringify('s:' + JSON.parse(quoted));
            } catch {
                return null;
            }
            previous = '"';
            continue;
        }
        NON_FINITE_TOKEN.lastIndex = i;
        const match = NON_FINITE_TOKEN.exec(text);
        if (match !== null) {
            let next = i + match[0].length;
            while (/[ \t\r\n]/.test(text[next] ?? '')) ++next;
            const after = text[next];
            if (
                (previous === undefined || ':,['.includes(previous)) &&
                (after === undefined || ',]}'.includes(after))
            ) {
                out += JSON.stringify('n:' + match[0]);
                i += match[0].length - 1;
                previous = 'v';
                replaced = true;
                continue;
            }
        }
        out += char;
        if (!/[ \t\r\n]/.test(char)) previous = char;
    }
    return replaced ? out : null;
}

function _revive_non_finite(_key, value) {
    if (typeof value === 'string') {
        if (value.startsWith('s:')) return value.slice(2);
        const token = value.startsWith('n:') ? value.slice(2) : '';
        if (Object.hasOwn(NON_FINITE, token)) return NON_FINITE[token];
    }
    return value;
}

/**
 * JSON parser with optional dialect knobs for LLM-emitted quirks. Like Python's `json.loads`, it accepts
 * `NaN`, `Infinity` and `-Infinity`.
 *
 * `args`:
 *   - `unquoted_keys` (bool): quote bare-identifier keys before parsing.
 *   - `string_delims` ([[open, close], ...]): strings delimited by these
 *     custom markers are pre-extracted, then restored as standard JSON strings.
 *   - `allow_non_json` (bool): return stripped text if parsing fails.
 */
function _json(text, args) {
    const string_delims = args.string_delims ?? [];
    const unquoted_keys = args.unquoted_keys ?? false;

    if (string_delims.length && (text.includes(LAX_OPEN) || text.includes(LAX_CLOSE))) {
        throw new Error('json: input contains reserved sentinel characters (\\x01/\\x02); cannot parse safely.');
    }

    let working = text;
    const captured = [];
    for (const [open_d, close_d] of string_delims) {
        const pattern = compile_pattern(`${escape_pattern(open_d)}(.*?)${escape_pattern(close_d)}`);
        working = working.replace(pattern.regex, (_, value) => `${LAX_OPEN}${captured.push(value) - 1}${LAX_CLOSE}`);
    }

    if (unquoted_keys) {
        // Consume quoted strings first so key-like text and escaped quotes inside them stay untouched.
        working = working.replace(
            compile_pattern(String.raw`"(?:\\.|[^"\\])*"|(?<=[{,])(\s*)(\w+)(\s*):`).regex,
            (match, before, key, after) => (key === undefined ? match : `${before}"${key}"${after}:`),
        );
    }

    for (let i = 0; i < captured.length; ++i) {
        working = working.replace(`${LAX_OPEN}${i}${LAX_CLOSE}`, () => JSON.stringify(captured[i]));
    }

    try {
        return JSON.parse(working);
    } catch (error) {
        const non_finite = _replace_non_finite(working);
        if (non_finite !== null) {
            try {
                return JSON.parse(non_finite, _revive_non_finite);
            } catch {}
        }
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
    const duplicates = new Set();
    for (const m of text.matchAll(compile_pattern(tag_pattern).regex)) {
        const key = m.groups?.key;
        if (key === undefined) {
            throw new Error(`xml-inline: tag_pattern must have a named group 'key'. Pattern: ${tag_pattern}`);
        }
        const value = _sub_parse(m.groups.value ?? '', value_parser);
        if (out.has(key) && merge) {
            if (!duplicates.has(key)) {
                out.set(key, [out.get(key)]);
                duplicates.add(key);
            }
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

const CALL_NAME = /[A-Za-z_][A-Za-z0-9_]*(?:[-.][A-Za-z0-9_]+)*/y;
const CALL_NUMBER = /-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?/y;
const CALL_ESCAPES = { '\\': '\\', "'": "'", '"': '"', '/': '/', b: '\b', f: '\f', n: '\n', r: '\r', t: '\t' };
const CALL_MAX_DEPTH = 100;

/**
 * Parse the strict `python-call-list` grammar without evaluating model output. Calls take keyword arguments
 * whose values are quoted strings, JSON numbers, lists, string-keyed dicts, or Python/JSON booleans and null.
 * Names are ASCII identifiers with optional `-`/`.` segments and no reserved words. Strings use JSON escapes
 * plus `\'`; lone surrogates and raw LF/CR are errors. Only space, tab, LF and CR count as whitespace.
 *
 * Trailing commas, duplicates, positional arguments and other Python syntax are rejected. Integers must be
 * safe integers, floats must be finite, and value containers may nest at most 100 levels (excluding calls and
 * the outer call list). The only language difference is numeric representation: JS has no separate float type.
 */
function _python_call_list(text) {
    let pos = 0;
    let depth = 0;
    const fail = (message) => {
        throw new Error(`python-call-list: ${message} at position ${pos}. Content: ${JSON.stringify(text)}`);
    };
    /** Count only nested value lists and dicts, not the surrounding call list or argument parentheses. */
    const nested = (parse) => {
        if (++depth > CALL_MAX_DEPTH) fail('lists and dicts may nest at most 100 levels');
        const result = parse();
        --depth;
        return result;
    };
    const at = (regex) => {
        regex.lastIndex = pos;
        return regex.exec(text);
    };
    const skip = () => {
        while (/[ \t\n\r]/.test(text[pos] ?? '')) ++pos;
    };
    const take = (value) => {
        skip();
        if (!text.startsWith(value, pos)) fail(`expected ${JSON.stringify(value)}`);
        pos += value.length;
    };
    const name = () => {
        skip();
        const match = at(CALL_NAME);
        if (match === null) fail('expected ASCII name');
        pos += match[0].length;
        return match[0];
    };
    const string = () => {
        skip();
        const quote = text[pos];
        if (quote !== "'" && quote !== '"') fail('dict keys must be quoted strings');
        ++pos;
        let out = '';
        while (pos < text.length) {
            const char = text[pos++];
            if (char === quote) {
                for (let i = 0; i < out.length; ++i) {
                    const code = out.charCodeAt(i);
                    if (code >= 0xd800 && code <= 0xdbff) {
                        const low = out.charCodeAt(++i);
                        if (!(low >= 0xdc00 && low <= 0xdfff)) fail('lone surrogate in string');
                    } else if (code >= 0xdc00 && code <= 0xdfff) {
                        fail('lone surrogate in string');
                    }
                }
                return out;
            }
            if (char === '\n' || char === '\r') fail('raw LF and CR are not allowed in strings');
            if (char !== '\\') {
                out += char;
                continue;
            }
            if (pos >= text.length) break;
            const escaped = text[pos++];
            if (Object.hasOwn(CALL_ESCAPES, escaped)) {
                out += CALL_ESCAPES[escaped];
            } else if (escaped === 'u') {
                const digits = text.slice(pos, pos + 4);
                if (!/^[0-9a-fA-F]{4}$/.test(digits)) fail('invalid \\u escape');
                out += String.fromCharCode(Number.parseInt(digits, 16));
                pos += 4;
            } else {
                fail(`unknown escape \\${escaped}`);
            }
        }
        fail('unterminated string');
    };
    /** Comma-separated items up to `close` (the opener is already consumed), with no trailing comma. */
    const items = (close, item) => {
        const values = [];
        skip();
        if (text[pos] === close) {
            ++pos;
            return values;
        }
        while (true) {
            values.push(item());
            skip();
            if (text[pos] === close) {
                ++pos;
                return values;
            }
            take(',');
            skip();
            if (text[pos] === close) fail('trailing commas are not supported');
        }
    };
    const mapping = () =>
        nested(() => {
            take('{');
            const seen = new Set();
            const entries = items('}', () => {
                const key = string();
                if (seen.has(key)) fail(`dict key repeated: ${JSON.stringify(key)}`);
                seen.add(key);
                take(':');
                return [key, value()];
            });
            return Object.fromEntries(entries);
        });
    const value = () => {
        skip();
        const char = text[pos];
        if (char === "'" || char === '"') return string();
        if (char === '[') {
            return nested(() => {
                ++pos;
                return items(']', value);
            });
        }
        if (char === '{') return mapping();
        const number = at(CALL_NUMBER);
        if (number !== null) {
            pos += number[0].length;
            const parsed = Number(number[0]);
            if (!/[.eE]/.test(number[0]) && !Number.isSafeInteger(parsed)) fail('integer literal is out of range');
            if (!Number.isFinite(parsed)) fail('number literal is out of range');
            return parsed;
        }
        const id = name();
        if (id === 'True' || id === 'true') return true;
        if (id === 'False' || id === 'false') return false;
        if (id === 'None' || id === 'null') return null;
        fail(`unsupported value ${JSON.stringify(id)}`);
    };
    const call = () => {
        const fn = name();
        take('(');
        const seen = new Set();
        const entries = items(')', () => {
            skip();
            const start = pos;
            const match = at(CALL_NAME);
            if (match !== null) {
                pos += match[0].length;
                skip();
            }
            if (match === null || text[pos] !== '=' || text[pos + 1] === '=') {
                pos = start;
                fail('positional arguments are not supported; only keyword arguments are supported');
            }
            pos = start;
            const key = name();
            if (seen.has(key)) fail(`keyword argument repeated: ${JSON.stringify(key)}`);
            seen.add(key);
            take('=');
            return [key, value()];
        });
        return { name: fn, arguments: Object.fromEntries(entries) };
    };
    take('[');
    const calls = items(']', call);
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
// the final value. Structured parsers (`json`, `xml-inline`, `kv-lines`, `python-call-list`) only
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
 * @param {Record<string, string>} captures Named groups from the field's open and close patterns; closing groups take precedence.
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
