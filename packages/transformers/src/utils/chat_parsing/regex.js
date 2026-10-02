/**
 * Response templates are written for Python's `regex` module, so their patterns are translated to
 * JavaScript once, at load time. Besides syntax (`(?P<name>`, `\Z`, ...), this keeps Python's semantics:
 * `\w`, `\d` and `\b` are Unicode-aware, and `$` also matches before a trailing newline. Patterns always
 * use `DOTALL`, like in Python.
 *
 * Each pattern also gets a "partial" variant, which emulates `regex`'s `partial=True` search: every
 * atom `X` becomes `(?:X|$)`, so the pattern also matches any prefix of a match that runs into the
 * end of the input. Streaming parsers use it to hold back text that may still become a delimiter.
 */

/** Python's Unicode `\w` (UTS #18). */
const WORD = String.raw`\p{Alphabetic}\p{M}\p{Nd}\p{Pc}\p{Join_Control}`;
const W = `[${WORD}]`;

/** Python escapes whose JavaScript equivalent differs, outside and inside character classes. */
const ESCAPES = {
    w: W,
    W: `[^${WORD}]`,
    d: String.raw`\p{Nd}`,
    D: String.raw`\P{Nd}`,
    b: `(?:(?<=${W})(?!${W})|(?<!${W})(?=${W}))`,
    B: `(?:(?<=${W})(?=${W})|(?<!${W})(?!${W}))`,
    A: String.raw`(?<![\s\S])`,
    Z: String.raw`(?![\s\S])`,
};
const CLASS_ESCAPES = { w: WORD, d: String.raw`\p{Nd}`, D: String.raw`\P{Nd}`, b: String.raw`\x08` };
/** Escapes that mean the same in both languages. */
const NATIVE_ESCAPES = new Set('sSntrfv0');
/** Characters that must be escaped to be literal in a `u`-mode regular expression. */
const SYNTAX_CHARACTERS = new Set('^$\\.*+?()[]{}|/');
const HEX_LENGTHS = { x: 2, u: 4, U: 8 };

const QUANTIFIER = /\{(\d*)(,?)(\d*)\}/y;
const GROUP = /\((?:\?(?:(:|=|!|<=|<!)|P?<(\w+)>|P=(\w+)\)))?/y;

/**
 * Translate a Python `regex` pattern to JavaScript.
 * @param {string} pattern The Python pattern.
 * @returns {{ source: string, partial: string, flags: string, names: string[] }} The JavaScript source,
 * its partial-matching variant, the flags to compile both with, and the names of all named groups.
 */
function translate(pattern) {
    // Global inline flags must lead the pattern, as in Python.
    const leading = /^(?:\(\?[a-zA-Z]+\))*/.exec(pattern)[0];
    let flags = 'su';
    for (const flag of new Set(leading.replace(/[(?)]/g, ''))) {
        if (flag === 'i') flags += flag;
        else if (flag !== 's' && flag !== 'u') throw new Error(`unsupported inline flag '${flag}'`);
    }
    const names = [];
    let i = leading.length;

    /** Return a literal character, escaped where JavaScript requires it. */
    const literal = (character, in_class) =>
        SYNTAX_CHARACTERS.has(character) || (in_class && character === '-') ? `\\${character}` : character;

    /** Read the character at `i`, which may be a surrogate pair. */
    const next_character = () => {
        const character = String.fromCodePoint(/** @type {number} */ (pattern.codePointAt(i)));
        i += character.length;
        return character;
    };

    /** Read the escape sequence at `i` and return its JavaScript equivalent. */
    function escape(in_class) {
        ++i;
        if (i >= pattern.length) throw new Error('bad escape (end of pattern)');
        const e = next_character();
        if (in_class && e === 'W') throw new Error(String.raw`\W inside a character class is not supported`);
        const replacement = (in_class ? CLASS_ESCAPES : ESCAPES)[e];
        if (replacement !== undefined) return replacement;
        if (NATIVE_ESCAPES.has(e)) return `\\${e}`;
        if (e in HEX_LENGTHS) {
            const hex = pattern.slice(i, (i += HEX_LENGTHS[e]));
            if (!/^[\da-fA-F]+$/.test(hex) || hex.length !== HEX_LENGTHS[e]) throw new Error(`bad escape \\${e}${hex}`);
            return `\\u{${hex}}`;
        }
        if (/\d/.test(e)) {
            // Backreference: Python reads at most two digits.
            return /\d/.test(pattern[i]) ? `\\${e}${pattern[i++]}` : `\\${e}`;
        }
        if (/[A-Za-z]/.test(e)) throw new Error(`bad escape \\${e}`);
        return literal(e, in_class);
    }

    /** Read the character class at `i`. */
    function character_class() {
        let out = pattern[i++];
        if (pattern[i] === '^') out += pattern[i++];
        if (pattern[i] === ']') {
            // A leading `]` is literal in Python
            out += '\\]';
            ++i;
        }
        while (pattern[i] !== ']') {
            if (i >= pattern.length) throw new Error('unterminated character set');
            if (pattern[i] === '\\') {
                out += escape(true);
            } else {
                const character = next_character();
                out += character === '[' ? '\\[' : character;
            }
        }
        ++i;
        return `${out}]`;
    }

    /** Translate up to the end of the current group, returning `[source, partial]`. */
    function walk(depth) {
        let source = '';
        let partial = '';
        /** Append to both outputs; atoms are optional at the end of the input in the partial variant. */
        const append = (text, partial_text = `(?:${text}|$)`) => {
            source += text;
            partial += partial_text;
        };
        while (i < pattern.length) {
            const c = pattern[i];
            if (c === '\\') {
                append(escape(false));
            } else if (c === '[') {
                append(character_class());
            } else if (c === '(') {
                GROUP.lastIndex = i;
                const [head, kind, name, reference] = /** @type {RegExpExecArray} */ (GROUP.exec(pattern));
                if (head === '(' && pattern[i + 1] === '?') {
                    throw new Error(`unsupported group syntax at position ${i}`);
                }
                i += head.length;
                if (reference !== undefined) {
                    append(`\\k<${reference}>`);
                    continue;
                }
                if (name !== undefined) names.push(name);
                const open = name !== undefined ? `(?<${name}>` : head;
                const [inner, inner_partial] = walk(depth + 1);
                if (kind === '<=' || kind === '<!' || kind === '!') {
                    // Lookbehinds and negative lookaheads stay exact
                    append(`${open}${inner})`, `${open}${inner})`);
                } else if (kind === '=') {
                    append(`${open}${inner})`, `${open}${inner_partial})`);
                } else {
                    append(`${open}${inner})`, `${open}${inner_partial}|$)`);
                }
            } else if (c === ')') {
                if (depth === 0) throw new Error(`unbalanced parenthesis at position ${i}`);
                ++i;
                return [source, partial];
            } else if (c === '*' || c === '+' || c === '?' || c === '{') {
                let quantifier = c;
                if (c === '{') {
                    QUANTIFIER.lastIndex = i;
                    const m = QUANTIFIER.exec(pattern);
                    if (!m || (!m[1] && !m[2])) {
                        // Not a valid repetition, so a literal `{`, as in Python
                        append('\\{');
                        ++i;
                        continue;
                    }
                    quantifier = `{${m[1] || '0'}${m[2]}${m[3]}}`;
                    i += m[0].length;
                } else {
                    ++i;
                }
                if (pattern[i] === '+') throw new Error('possessive quantifiers are not supported');
                if (pattern[i] === '?') {
                    quantifier += '?';
                    ++i;
                }
                source += quantifier;
                partial += quantifier;
            } else if (c === '|' || c === '^' || c === '$') {
                ++i;
                // Python's `$` also matches before a trailing newline
                const text = c === '$' ? String.raw`(?=\n?(?![\s\S]))` : c;
                source += text;
                partial += text;
            } else if (c === '.') {
                ++i;
                append('.');
            } else {
                append(literal(next_character(), false));
            }
        }
        if (depth > 0) throw new Error('missing ), unterminated subpattern');
        return [source, partial];
    }

    const [source, partial] = walk(0);
    return { source, partial, flags, names };
}

/**
 * @typedef {Object} CompiledPattern
 * @property {RegExp} regex Global regex, for searching and iterating over matches.
 * @property {RegExp} sticky Sticky regex, for matching at a given position.
 * @property {RegExp} partial Global regex that also matches prefixes of matches that run into the end of the input.
 * @property {string[]} names Names of the pattern's named groups.
 */

/** @type {Map<string, CompiledPattern>} */
const cache = new Map();

/**
 * Compile (and cache) a Python `regex` pattern.
 * @param {string} pattern The Python pattern.
 * @returns {CompiledPattern}
 */
export function compile_pattern(pattern) {
    let compiled = cache.get(pattern);
    if (!compiled) {
        const { source, partial, flags, names } = translate(pattern);
        compiled = {
            regex: new RegExp(source, `${flags}g`),
            sticky: new RegExp(source, `${flags}y`),
            partial: new RegExp(partial, `${flags}g`),
            names,
        };
        cache.set(pattern, compiled);
    }
    return compiled;
}

/**
 * @typedef {Object} Match
 * @property {number} start Start index of the match.
 * @property {number} end End index of the match (the end of the input for partial matches).
 * @property {Record<string, string>} groups Named groups that participated in the match.
 * @property {boolean} partial Whether the match is incomplete and ran into the end of the input.
 */

/**
 * Find the first match at or after `position`, like Python's `pattern.search(text, position, partial=partial)`.
 * @param {CompiledPattern} pattern The compiled pattern.
 * @param {string} text The text to search.
 * @param {number} position Where to start searching.
 * @param {boolean} [partial=false] Whether to also report incomplete matches that run into the end of `text`.
 * @returns {Match|null}
 */
export function search(pattern, text, position, partial = false) {
    let match;
    if (partial) {
        pattern.partial.lastIndex = position;
        const candidate = pattern.partial.exec(text);
        if (candidate === null) return null;
        // A complete match at the same position takes precedence over the partial one.
        pattern.sticky.lastIndex = candidate.index;
        match = pattern.sticky.exec(text);
        if (match === null) return { start: candidate.index, end: text.length, groups: {}, partial: true };
    } else {
        pattern.regex.lastIndex = position;
        match = pattern.regex.exec(text);
        if (match === null) return null;
    }
    /** @type {Record<string, string>} */
    const groups = {};
    for (const [key, value] of Object.entries(match.groups ?? {})) {
        if (value !== undefined) groups[key] = value;
    }
    return { start: match.index, end: match.index + match[0].length, groups, partial: false };
}

/**
 * Escape literal text for use in a Python `regex` pattern.
 * @param {string} text The literal text.
 * @returns {string}
 */
export function escape_pattern(text) {
    return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
