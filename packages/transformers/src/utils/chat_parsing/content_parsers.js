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
 * strings) with placeholder strings that `_revive_non_finite` turns back into numbers.
 * @param {string} text
 * @returns {string|null} The rewritten text, or `null` if there was nothing to replace.
 */
function _replace_non_finite(text) {
    if (text.includes(LAX_OPEN) || text.includes(LAX_CLOSE)) return null;
    let out = '';
    let replaced = false;
    let in_string = false;
    for (let i = 0; i < text.length; ++i) {
        const char = text[i];
        if (in_string) {
            out += char;
            if (char === '\\') out += text[++i] ?? '';
            else if (char === '"') in_string = false;
            continue;
        }
        if (char === '"') {
            in_string = true;
            out += char;
            continue;
        }
        NON_FINITE_TOKEN.lastIndex = i;
        const match = NON_FINITE_TOKEN.exec(text);
        if (match !== null) {
            // Escaped, because raw control characters aren't allowed inside JSON strings.
            out += `"\\u0001${match[0]}\\u0002"`;
            i += match[0].length - 1;
            replaced = true;
            continue;
        }
        out += char;
    }
    return replaced ? out : null;
}

function _revive_non_finite(_key, value) {
    if (typeof value === 'string' && value.startsWith(LAX_OPEN) && value.endsWith(LAX_CLOSE)) {
        const token = value.slice(1, -1);
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
        working = working.replace(compile_pattern(String.raw`(?<=[{,])(\w+):`).regex, '"$1":');
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

// Python keywords can't be used as names (soft keywords such as `match` or `type` can).
const PY_KEYWORDS = new Set([
    'False',
    'None',
    'True',
    'and',
    'as',
    'assert',
    'async',
    'await',
    'break',
    'class',
    'continue',
    'def',
    'del',
    'elif',
    'else',
    'except',
    'finally',
    'for',
    'from',
    'global',
    'if',
    'import',
    'in',
    'is',
    'lambda',
    'nonlocal',
    'not',
    'or',
    'pass',
    'raise',
    'return',
    'try',
    'while',
    'with',
    'yield',
]);
const PY_IDENTIFIER = /[\p{L}\p{Nl}_][\p{L}\p{Nl}\p{Mn}\p{Mc}\p{Nd}\p{Pc}]*/uy;
const PY_DIGITS = String.raw`\d(?:_?\d)*`;
const PY_NUMBER = new RegExp(
    String.raw`0[xX](?:_?[0-9a-fA-F])+|0[oO](?:_?[0-7])+|0[bB](?:_?[01])+` +
        `|(?:(?:${PY_DIGITS})?\\.${PY_DIGITS}|${PY_DIGITS}\\.)(?:[eE][+-]?${PY_DIGITS})?|${PY_DIGITS}[eE][+-]?${PY_DIGITS}` +
        String.raw`|[1-9](?:_?\d)*|0(?:_?0)*`,
    'y',
);
const PY_STRING_START = /([A-Za-z]{0,2})('''|"""|'|")/y;
const PY_ESCAPES = { a: '\x07', b: '\b', f: '\f', n: '\n', r: '\r', t: '\t', v: '\v' };

/**
 * Parse a Python-style list of keyword-argument calls, e.g. `[get_weather(city="Paris")]`, without evaluating
 * model output. The syntax is Python's own, matching the `ast`-based parser in Python transformers: any `str`
 * literal (quotes, triple quotes, `r`/`u` prefixes, implicit concatenation), Python number literals, comments and
 * redundant parentheses. Positional arguments, bytes, f-strings and `\N{...}` escapes are rejected.
 */
function _python_call_list(text) {
    let pos = 0;
    const fail = (message) => {
        throw new Error(`python-call-list: ${message} at position ${pos}. Content: ${JSON.stringify(text)}`);
    };
    const at = (regex) => {
        regex.lastIndex = pos;
        return regex.exec(text);
    };
    const skip = () => {
        while (pos < text.length) {
            const char = text[pos];
            if (/\s/.test(char)) {
                ++pos;
            } else if (char === '#') {
                while (pos < text.length && text[pos] !== '\n' && text[pos] !== '\r') ++pos;
            } else if (char === '\\' && (text[pos + 1] === '\n' || text[pos + 1] === '\r')) {
                pos += 2; // explicit line continuation
            } else {
                break;
            }
        }
    };
    const take = (value) => {
        skip();
        if (!text.startsWith(value, pos)) fail(`expected ${JSON.stringify(value)}`);
        pos += value.length;
    };
    const identifier = () => {
        skip();
        const match = at(PY_IDENTIFIER);
        if (match === null) fail('expected identifier');
        pos += match[0].length;
        return match[0].normalize('NFKC'); // like Python's identifier normalization
    };
    const name = () => {
        const id = identifier();
        if (PY_KEYWORDS.has(id)) fail(`${JSON.stringify(id)} is a reserved word`);
        return id;
    };
    const string_literal = () => {
        const match = at(PY_STRING_START);
        const prefix = match[1].toLowerCase();
        if (prefix !== '' && prefix !== 'r' && prefix !== 'u')
            fail(`unsupported string prefix ${JSON.stringify(match[1])}`);
        const raw = prefix === 'r';
        const quote = match[2];
        const triple = quote.length === 3;
        pos += match[0].length;
        let out = '';
        while (pos < text.length) {
            if (text.startsWith(quote, pos)) {
                pos += quote.length;
                return out;
            }
            const char = text[pos++];
            if (char === '\n' || char === '\r') {
                if (!triple) fail('unterminated string');
                // Python reads source with universal newlines.
                if (char === '\r' && text[pos] === '\n') ++pos;
                out += '\n';
                continue;
            }
            if (char !== '\\') {
                out += char;
                continue;
            }
            if (pos >= text.length) break;
            const escaped = text[pos++];
            if (raw) {
                // Raw strings keep the backslash, but it still stops the next character from closing the string.
                out += char + escaped;
            } else if (Object.hasOwn(PY_ESCAPES, escaped)) {
                out += PY_ESCAPES[escaped];
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
    const string = () => {
        // Adjacent literals are concatenated: `"a" 'b'` is `"ab"`.
        let out = string_literal();
        while (true) {
            skip();
            if (at(PY_STRING_START) === null) return out;
            out += string_literal();
        }
    };
    /** Comma-separated items up to `close` (the opener is already consumed); trailing commas are allowed. */
    const items = (close, item) => {
        const values = [];
        while (true) {
            skip();
            if (text[pos] === close) {
                ++pos;
                return values;
            }
            values.push(item());
            skip();
            if (text[pos] === close) {
                ++pos;
                return values;
            }
            take(',');
        }
    };
    const group_or_tuple = () => {
        take('(');
        skip();
        if (text[pos] === ')') {
            ++pos;
            return [];
        }
        const first = value();
        skip();
        // `(x)` is just `x`; only `(x,)` is a tuple.
        if (text[pos] === ')') {
            ++pos;
            return first;
        }
        take(',');
        return [first, ...items(')', value)];
    };
    const mapping = () => {
        take('{');
        const entries = items('}', () => {
            skip();
            let key;
            if (at(PY_STRING_START) !== null) {
                key = string();
            } else {
                // Bare identifiers are accepted as keys, as in `{city: "Paris"}`.
                key = identifier();
                if (PY_KEYWORDS.has(key)) fail('dict keys must be strings');
            }
            take(':');
            return [key, value()];
        });
        return Object.fromEntries(entries);
    };
    const value = () => {
        skip();
        const char = text[pos];
        if (at(PY_STRING_START) !== null) return string();
        if (char === '[') {
            ++pos;
            return items(']', value);
        }
        if (char === '(') return group_or_tuple();
        if (char === '{') return mapping();
        if (char === '-' || char === '+') {
            ++pos;
            const operand = value();
            if (typeof operand !== 'number') fail('unary + and - only apply to numbers');
            return char === '-' ? -operand : operand;
        }
        const number = at(PY_NUMBER);
        if (number !== null) {
            pos += number[0].length;
            if (/[\p{L}\p{N}_.]/u.test(text[pos] ?? '')) fail('invalid number literal');
            // Integers beyond 2^53 lose precision here (Python keeps them exact).
            return Number(number[0].replaceAll('_', ''));
        }
        const id = identifier();
        if (id === 'True' || id === 'true') return true;
        if (id === 'False' || id === 'false') return false;
        if (id === 'None' || id === 'null') return null;
        fail(`unsupported value ${JSON.stringify(id)}`);
    };
    const call = () => {
        skip();
        if (text[pos] === '(') {
            ++pos;
            const result = call();
            take(')');
            return result;
        }
        const fn = name();
        take('(');
        const seen = new Set();
        const entries = items(')', () => {
            skip();
            const start = pos;
            const match = at(PY_IDENTIFIER);
            if (match !== null) {
                pos += match[0].length;
                skip();
            }
            if (match === null || text[pos] !== '=' || text[pos + 1] === '=') {
                pos = start;
                fail('only keyword arguments are supported');
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
    const top = () => {
        skip();
        if (text[pos] === '(') {
            ++pos;
            const calls = top();
            take(')');
            return calls;
        }
        take('[');
        return items(']', call);
    };

    const calls = top();
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
