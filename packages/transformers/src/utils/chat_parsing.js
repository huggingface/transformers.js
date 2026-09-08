/**
 * @file Chat response parsing utilities for schema-driven extraction of structured
 * data from model output strings.
 *
 * @module utils/chat_parsing
 */

import { jmespath_search } from './jmespath.js';

/**
 * Convert Python-style named groups `(?P<name>...)` to JS-style `(?<name>...)`.
 * @param {string} pattern
 * @returns {string}
 * @private
 */
function _convertNamedGroups(pattern) {
    return pattern.replaceAll('(?P<', '(?<');
}

/**
 * Extract data from a regex match object.
 * - Named groups → dict of non-undefined values.
 * - Single unnamed group → the captured string.
 * @param {RegExpExecArray} match
 * @returns {Record<string, string> | string}
 * @private
 */
function _parseReMatch(match) {
    if (match.groups) {
        const filtered = Object.create(null);
        let hasValue = false;
        for (const [key, val] of Object.entries(match.groups)) {
            if (val !== undefined) {
                filtered[key] = val;
                hasValue = true;
            }
        }
        if (hasValue) return filtered;
    }
    // Unnamed groups: must have exactly one
    const groupCount = match.length - 1;
    if (groupCount === 1) return match[1];
    throw new Error(
        groupCount === 0
            ? `Regex has no capture groups: ${match[0]}`
            : `Regex has multiple unnamed groups: ${Array.from(match).slice(1)}`,
    );
}

/**
 * Convert Gemma4 tool call format (unquoted keys, `<|"|>` string delimiters) to valid JSON.
 * Handles all JSON types: strings, numbers, booleans, null, arrays, and nested objects.
 * @param {string} text
 * @returns {string} Valid JSON string.
 * @private
 */
function _gemma4JsonToJson(text) {
    const strings = [];
    // Extract gemma-quoted strings into placeholders
    text = text.replace(/<\|"\|>(.*?)<\|"\|>/gs, (_, s) => {
        strings.push(s);
        return `\x00${strings.length - 1}\x00`;
    });
    // Quote bare keys
    text = text.replace(/(?<=[{,])(\w+):/g, '"$1":');
    // Restore captured strings with proper JSON escaping
    text = text.replace(/\x00(\d+)\x00/g, (_, idx) => JSON.stringify(strings[idx]));
    return text;
}

/**
 * Recursively parse content against a JSON schema with regex extraction directives.
 *
 * @param {string | Record<string, any> | any[] | null} nodeContent The content to parse.
 * @param {Record<string, any>} nodeSchema The schema controlling the parsing.
 * @returns {any} The parsed data structure.
 */
export function recursive_parse(nodeContent, nodeSchema) {
    // 1. Const: return immediately
    if ('const' in nodeSchema) {
        return nodeSchema['const'];
    }

    // 2. Null content
    if (nodeContent == null) {
        return null;
    }

    // 3. Setup and validation
    const nodeType = nodeSchema.type ?? null;
    const hasRegex =
        'x-regex' in nodeSchema ||
        'x-regex-iterator' in nodeSchema ||
        'x-regex-key-value' in nodeSchema ||
        'x-regex-substitutions' in nodeSchema;
    if (hasRegex && typeof nodeContent !== 'string') {
        throw new TypeError(
            `Schema node got a non-string input, but has a regex for parsing.\n` +
                `Input: ${nodeContent}\nSchema: ${JSON.stringify(nodeSchema)}`,
        );
    }

    // 4. Apply x-regex-substitutions (regex-based string replacements)
    if ('x-regex-substitutions' in nodeSchema) {
        for (const [pattern, replacement] of nodeSchema['x-regex-substitutions']) {
            nodeContent = /** @type {string} */ (nodeContent).replace(new RegExp(pattern, 'gs'), replacement);
        }
    }

    // 5. Apply x-regex (single match extraction)
    if ('x-regex' in nodeSchema) {
        const match = new RegExp(_convertNamedGroups(nodeSchema['x-regex']), 's').exec(/** @type {string} */ (nodeContent));
        if (!match) return null;
        nodeContent = _parseReMatch(match);
    }

    // 6. Apply x-regex-iterator (multiple match extraction for arrays)
    if ('x-regex-iterator' in nodeSchema) {
        if (nodeType !== 'array') {
            throw new TypeError(`Schema node with type ${nodeType} cannot use x-regex-iterator.`);
        }
        const regex = new RegExp(_convertNamedGroups(nodeSchema['x-regex-iterator']), 'gs');
        const results = [];
        let match;
        while ((match = regex.exec(/** @type {string} */ (nodeContent))) !== null) {
            results.push(_parseReMatch(match));
        }
        if (results.length === 0) return null;
        nodeContent = results;
    }

    // 7. Apply x-regex-key-value (key-value pair extraction for objects)
    if ('x-regex-key-value' in nodeSchema) {
        if (nodeType !== 'object') {
            throw new TypeError(`Schema node with type ${nodeType} cannot use x-regex-key-value.`);
        }
        const regex = new RegExp(_convertNamedGroups(nodeSchema['x-regex-key-value']), 'gs');
        const output = Object.create(null);
        let match;
        let hasEntries = false;
        while ((match = regex.exec(/** @type {string} */ (nodeContent))) !== null) {
            const groups = _parseReMatch(match);
            if (typeof groups !== 'object' || !('key' in groups) || !('value' in groups)) {
                throw new Error(
                    `Regex for x-regex-key-value must have named groups 'key' and 'value'.\n` +
                        `Match groups: ${JSON.stringify(groups)}`,
                );
            }
            output[groups.key] = groups.value;
            hasEntries = true;
        }
        if (!hasEntries) return null;
        nodeContent = output;
    }

    // 8. Apply x-parser
    if ('x-parser' in nodeSchema) {
        let parser = nodeSchema['x-parser'];
        if (parser === 'gemma4-tool-call') {
            if (typeof nodeContent !== 'string') {
                throw new TypeError(`Node has gemma4-tool-call parser but got non-string input: ${nodeContent}`);
            }
            nodeContent = _gemma4JsonToJson(/** @type {string} */ (nodeContent));
            parser = 'json'; // fall through to JSON parser
        }
        if (parser === 'json') {
            if (typeof nodeContent !== 'string') {
                throw new TypeError(`Node has JSON parser but got non-string input: ${nodeContent}`);
            }
            const parserArgs = nodeSchema['x-parser-args'] ?? {};
            try {
                nodeContent = JSON.parse(/** @type {string} */ (nodeContent));
            } catch (e) {
                if (parserArgs.allow_non_json) {
                    // keep nodeContent as-is
                } else {
                    throw new Error(
                        `Node has JSON parser but could not parse its contents as JSON.\n` +
                            `Content: ${nodeContent}\nError: ${e.message}`,
                    );
                }
            }
            if (parserArgs.transform != null) {
                nodeContent = jmespath_search(parserArgs.transform, nodeContent);
            }
        } else {
            throw new Error(`Unknown parser "${parser}" for schema node: ${JSON.stringify(nodeSchema)}`);
        }
    }

    // 9. Type-specific handling
    if (nodeType === 'object') {
        const properties = nodeSchema.properties ?? {};
        const parsed = Object.create(null);

        if (typeof nodeContent === 'string') {
            if (!nodeSchema.properties) {
                throw new Error(
                    `Object node received string content but has no properties to parse.\nContent: ${nodeContent}`,
                );
            }
            for (const [key, childSchema] of Object.entries(properties)) {
                const childResult = recursive_parse(nodeContent, childSchema);
                if (childResult != null) {
                    parsed[key] = childResult;
                }
            }
        } else if (typeof nodeContent === 'object' && !Array.isArray(nodeContent)) {
            for (const [key, childSchema] of Object.entries(properties)) {
                if ('const' in childSchema) {
                    parsed[key] = childSchema['const'];
                } else if (key in nodeContent) {
                    parsed[key] = recursive_parse(nodeContent[key], childSchema);
                } else if ('default' in childSchema) {
                    parsed[key] = childSchema['default'];
                }
            }
            const additionalSchema = nodeSchema.additionalProperties;
            if (additionalSchema !== false) {
                const childSchema = typeof additionalSchema === 'object' && additionalSchema !== null ? additionalSchema : {};
                for (const [key, value] of Object.entries(nodeContent)) {
                    if (!(key in properties)) {
                        parsed[key] = recursive_parse(value, childSchema);
                    }
                }
            }
        } else {
            throw new TypeError(`Expected a dict or str for schema node with type object, got ${typeof nodeContent}`);
        }

        const required = nodeSchema.required ?? [];
        const missing = required.filter((key) => !(key in parsed));
        if (missing.length > 0) {
            const preview = typeof nodeContent === 'string' ? nodeContent.slice(0, 500) : JSON.stringify(nodeContent);
            throw new Error(
                `Required fields [${missing}] are missing from parsed output.\n` +
                    `Parsed: ${JSON.stringify(parsed)}\nInput: ${preview}`,
            );
        }
        return parsed;
    }

    if (nodeType === 'array') {
        if (!nodeContent) return [];
        if ('items' in nodeSchema) {
            if (!Array.isArray(nodeContent)) {
                throw new TypeError(`Expected a list for schema node with type array, got ${typeof nodeContent}`);
            }
            return nodeContent.map((item) => recursive_parse(item, nodeSchema.items));
        }
        if ('prefixItems' in nodeSchema) {
            if (!Array.isArray(nodeContent)) {
                if (nodeSchema.prefixItems.length === 1) {
                    nodeContent = [nodeContent];
                } else {
                    throw new TypeError(`Expected a list for schema node with type array, got ${typeof nodeContent}`);
                }
            }
            if (nodeContent.length !== nodeSchema.prefixItems.length) {
                throw new Error(
                    `Array node has ${nodeContent.length} items, but schema has ${nodeSchema.prefixItems.length} prefixItems.`,
                );
            }
            return nodeContent.map((item, i) => recursive_parse(item, nodeSchema.prefixItems[i]));
        }
        throw new Error(`Array node has no items or prefixItems schema defined.`);
    }

    if (nodeType === 'integer') {
        if (Number.isInteger(nodeContent)) return nodeContent;
        if (typeof nodeContent !== 'string') {
            throw new TypeError(`Expected a string or int for type integer, got ${typeof nodeContent}: ${nodeContent}`);
        }
        const val = parseInt(nodeContent, 10);
        if (isNaN(val)) throw new Error(`Type 'integer', but content is not a valid integer: "${nodeContent}"`);
        return val;
    }

    if (nodeType === 'number') {
        if (typeof nodeContent === 'number') return nodeContent;
        if (typeof nodeContent !== 'string') {
            throw new TypeError(`Expected a string or number for type number, got ${typeof nodeContent}: ${nodeContent}`);
        }
        const val = Number(nodeContent);
        if (isNaN(val)) throw new Error(`Type 'number', but content is not a valid number: "${nodeContent}"`);
        return val;
    }

    if (nodeType === 'boolean') {
        if (typeof nodeContent === 'boolean') return nodeContent;
        if (typeof nodeContent !== 'string') {
            throw new TypeError(`Expected a string or bool for type boolean, got ${typeof nodeContent}: ${nodeContent}`);
        }
        const lower = nodeContent.toLowerCase();
        if (lower === 'true' || lower === '1') return true;
        if (lower === 'false' || lower === '0') return false;
        throw new Error(`Invalid boolean value: ${nodeContent}`);
    }

    if (nodeType === 'string') {
        if (typeof nodeContent !== 'string') {
            throw new TypeError(`Expected a string for type string, got ${typeof nodeContent}: ${nodeContent}`);
        }
        return nodeContent;
    }

    if (nodeType == null || nodeType === 'any') {
        return nodeContent;
    }

    throw new TypeError(`Unsupported schema type "${nodeType}" for node: ${nodeContent}`);
}
