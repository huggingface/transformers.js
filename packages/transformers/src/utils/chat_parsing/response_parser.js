import { processField, STREAMABLE_PARSERS } from './content_parsers.js';
import { findMatch, loadResponseTemplate, partialLiteralStart, truncatePastLastAnchor } from './response_templates.js';

/**
 * @typedef {Object} ParseResponseOptions
 * @property {string} prefix Full chat prompt sent to the model. Parsing starts after its last matching template start anchor; pass an empty string when `text` contains the complete assistant message.
 * @property {boolean} [partial=false] Treat `text` as an accumulated in-progress generation. Incomplete structured fields and missing required fields are omitted instead of failing final validation.
 */

/**
 * Parse generated assistant text according to a response template.
 *
 * @param {string} text Generated assistant text, excluding the prompt prefix.
 * @param {Object} responseTemplate Model-specific response format.
 * @param {ParseResponseOptions} options Parsing context and finalization mode.
 * @returns {Object.<string, *>} Parsed assistant message.
 */
export function parseResponse(text, responseTemplate, options) {
    const { prefix, partial = false } = options ?? {};
    if (typeof text !== 'string') throw new TypeError('parse_response expects a string.');
    if (typeof prefix !== 'string') {
        throw new TypeError('parse_response requires a string prefix. Pass an empty string to opt out.');
    }
    const parser = new ResponseParser(responseTemplate);
    const responsePrefix = truncatePastLastAnchor(parser.spec.startAnchor, prefix);
    parser.consume(responsePrefix + text, !partial);
    return parser.snapshot(partial);
}

class ResponseParser {
    constructor(responseTemplate) {
        this.spec = loadResponseTemplate(responseTemplate);
        this.output = clone(this.spec.defaults);
        this.current = this.spec.implicit;
        this.captures = {};
        this.body = null;
        this.position = 0;
    }

    consume(text, final) {
        while (true) {
            const watch = this.watchlist();
            let best = null;
            for (const item of watch) {
                const anchor = item.kind === 'open' ? item.field.open : item.field.close;
                const match = findMatch(anchor, text, this.position, final && item.kind === 'close');
                if (!match) continue;
                const candidate = { ...item, match };
                if (!best || compareCandidates(candidate, best) < 0) best = candidate;
            }

            if (!best) {
                const holdStart = final
                    ? text.length
                    : partialLiteralStart(
                          watch.map((item) => (item.kind === 'open' ? item.field.open : item.field.close)),
                          text,
                          this.position,
                      );
                this.accumulate(text.slice(this.position, holdStart));
                this.position = holdStart;
                if (final) this.closeCurrent();
                break;
            }

            this.accumulate(text.slice(this.position, best.match.start));
            this.position = best.match.end;
            if (best.kind === 'open') {
                this.closeCurrent();
                this.current = best.field.name;
                this.captures = best.match.groups;
                this.body = '';
            } else {
                this.closeCurrent();
                if (best.match.start === best.match.end) break;
            }
        }
    }

    snapshot(partial) {
        const output = clone(this.output);
        if (partial && this.current !== null && this.body !== null) {
            const field = this.spec.fields[this.current];
            if (STREAMABLE_PARSERS.has(field.content) || field.close === null) {
                try {
                    this.store(output, field, processField(this.body, field, this.captures));
                } catch {
                    // A partial numeric value may not be parseable yet.
                }
            }
        }
        if (!partial) {
            const missing = Object.values(this.spec.fields)
                .filter((field) => !field.optional && !Object.hasOwn(output, field.name))
                .map((field) => field.name);
            if (missing.length)
                throw new Error(`Required response_template fields missing from parsed output: ${missing}`);
        }
        for (const [key, value] of Object.entries(output)) {
            if (!Object.hasOwn(this.spec.defaults, key) && isEmpty(value)) delete output[key];
        }
        return output;
    }

    watchlist() {
        if (this.current !== null && this.current !== this.spec.implicit) {
            const field = this.spec.fields[this.current];
            return field.close ? [{ kind: 'close', field }] : [];
        }
        const watch = Object.values(this.spec.fields)
            .filter((field) => field.open)
            .map((field) => ({ kind: 'open', field }));
        const implicit = this.spec.implicit === null ? null : this.spec.fields[this.spec.implicit];
        if (implicit?.close) watch.push({ kind: 'close', field: implicit });
        return watch;
    }

    accumulate(text) {
        if (text && this.current !== null) {
            this.body = (this.body ?? '') + text;
        }
    }

    closeCurrent() {
        if (this.current !== null && this.body !== null) {
            const field = this.spec.fields[this.current];
            this.store(this.output, field, processField(this.body, field, this.captures));
        }
        this.current = this.spec.implicit;
        this.captures = {};
        this.body = null;
    }

    store(output, field, value) {
        if (field.join !== null) {
            if (typeof value !== 'string') {
                throw new Error(`Field '${field.name}': join requires each match to parse to a string.`);
            }
            setOwn(
                output,
                field.name,
                Object.hasOwn(output, field.name) ? output[field.name] + field.join + value : value,
            );
        } else if (field.repeats) {
            if (!Object.hasOwn(output, field.name)) setOwn(output, field.name, []);
            output[field.name].push(value);
        } else {
            setOwn(output, field.name, value);
        }
    }
}

function compareCandidates(a, b) {
    const kindRank = { open: 0, close: 1 };
    return (
        a.match.start - b.match.start ||
        b.match.end - b.match.start - (a.match.end - a.match.start) ||
        kindRank[a.kind] - kindRank[b.kind] ||
        a.field.name.localeCompare(b.field.name)
    );
}

function isEmpty(value) {
    return (
        value === null ||
        value === '' ||
        (Array.isArray(value) && value.length === 0) ||
        (value && typeof value === 'object' && Object.keys(value).length === 0)
    );
}

function clone(value) {
    if (Array.isArray(value)) return value.map(clone);
    if (value !== null && typeof value === 'object') {
        return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, clone(item)]));
    }
    return value;
}

function setOwn(object, key, value) {
    Object.defineProperty(object, key, { value, writable: true, enumerable: true, configurable: true });
}
