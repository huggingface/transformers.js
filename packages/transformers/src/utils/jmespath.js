/**
 * @file Minimal JMESPath implementation for chat response schema transforms.
 *
 * Single-pass recursive-descent evaluator — no tokens, no AST, no visitor.
 * Context is threaded through all functions; the only mutable state is the cursor.
 *
 * Supports: field access, current node (`@`), wildcard projections (`[*]`),
 * multi-select hash (`{key: expr}`), multi-select list (`[expr, ...]`),
 * literal strings (`'value'`), and subexpressions (`a.b`).
 *
 * @module utils/jmespath
 */

const WS = /\s/;
const ID_CHAR = /[a-zA-Z0-9_]/;

/**
 * Access a field on an object, returning null for non-objects or missing keys.
 * @param {any} obj
 * @param {string} key
 * @returns {any}
 */
const fieldAccess = (obj, key) =>
    (obj !== null && typeof obj === "object" && !Array.isArray(obj)) ? obj[key] ?? null : null;

/**
 * Evaluate a JMESPath expression against data.
 * @param {string} expression The JMESPath expression string.
 * @param {any} data The data to search.
 * @returns {any}
 */
export function jmespath_search(expression, data) {
    let cursor = 0;

    /** Advance past whitespace. */
    const skipWhitespace = () => {
        while (cursor < expression.length && WS.test(expression[cursor])) cursor++;
    };

    /**
     * Read an identifier (e.g. `foo`, `tool_name`) from the expression.
     * @returns {string}
     */
    const readIdentifier = () => {
        skipWhitespace();
        const start = cursor;
        while (cursor < expression.length && ID_CHAR.test(expression[cursor])) cursor++;
        if (cursor === start) {
            throw new Error(`Expected identifier at position ${cursor} in: ${expression}`);
        }
        return expression.slice(start, cursor);
    };

    /**
     * Parse and evaluate an expression with optional `.rhs` subexpression chaining.
     * @param {any} context The current data context.
     * @returns {any}
     */
    const parseExpression = (context) => {
        let result = parsePrimary(context);
        skipWhitespace();
        while (cursor < expression.length && expression[cursor] === ".") {
            cursor++; // skip '.'
            result = parsePrimary(result);
        }
        return result;
    };

    /**
     * Parse and evaluate a single primary construct (identifier, literal, `@`, `{}`, `[]`).
     * @param {any} context The current data context.
     * @returns {any}
     */
    const parsePrimary = (context) => {
        skipWhitespace();
        const ch = expression[cursor];

        if (ch === "@") {
            cursor++;
            return context;
        }

        if (ch === "'" || ch === '"') {
            const start = ++cursor;
            const end = expression.indexOf(ch, start);
            if (end === -1) {
                throw new Error(`Unterminated string literal in: ${expression}`);
            }
            cursor = end + 1;
            return expression.slice(start, end);
        }

        if (ch === "{") return parseMultiSelectHash(context);
        if (ch === "[") return parseBracket(context);

        // Bare identifier — field access
        return fieldAccess(context, readIdentifier());
    };

    /**
     * Parse `{key: expr, key: expr, ...}` and evaluate each value against context.
     * @param {any} context
     * @returns {Record<string, any> | null}
     */
    const parseMultiSelectHash = (context) => {
        if (context === null) return null;
        cursor++; // skip '{'
        /** @type {Record<string, any>} */
        const result = Object.create(null);
        let first = true;
        skipWhitespace();
        while (expression[cursor] !== "}") {
            if (!first) cursor++; // skip ','
            first = false;
            const key = readIdentifier();
            skipWhitespace();
            cursor++; // skip ':'
            result[key] = parseExpression(context);
            skipWhitespace();
        }
        cursor++; // skip '}'
        return result;
    };

    /**
     * Parse `[*]` (wildcard projection) or `[expr, ...]` (multi-select list).
     * @param {any} context
     * @returns {any}
     */
    const parseBracket = (context) => {
        cursor++; // skip '['
        skipWhitespace();

        if (expression[cursor] === "*") {
            // Wildcard projection: [*] applies RHS to each array element
            cursor++; // skip '*'
            skipWhitespace();
            cursor++; // skip ']'

            if (!Array.isArray(context)) return null;

            // Check for a .rhs after the projection
            skipWhitespace();
            if (cursor >= expression.length || expression[cursor] !== ".") return context;
            cursor++; // skip '.'

            // Snapshot cursor position — replay the RHS for each element
            const rhsStart = cursor;
            /** @type {any[]} */
            const results = [];
            for (const item of context) {
                cursor = rhsStart;
                const value = parsePrimary(item);
                if (value !== null) results.push(value);
            }
            return results;
        }

        // Multi-select list: [expr, expr, ...]
        if (context === null) return null;
        /** @type {any[]} */
        const elements = [];
        let first = true;
        skipWhitespace();
        while (expression[cursor] !== "]") {
            if (!first) cursor++; // skip ','
            first = false;
            elements.push(parseExpression(context));
            skipWhitespace();
        }
        cursor++; // skip ']'
        return elements;
    };

    return parseExpression(data);
}
