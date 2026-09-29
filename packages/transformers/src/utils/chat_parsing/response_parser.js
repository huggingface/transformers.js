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
    const parser = new ResponseParser(responseTemplate, { prefix });
    const snapshot = parser.feed(text);
    return partial ? snapshot : parser.finalize();
}

export class ResponseParser {
    /**
     * @param {Object} responseTemplate Model-specific response format.
     * @param {Object} options
     * @param {string} options.prefix Full chat prompt sent to the model. Pass an empty string to opt out.
     */
    constructor(responseTemplate, options) {
        const { prefix } = options ?? {};
        if (typeof prefix !== 'string') {
            throw new TypeError('ResponseParser requires a string prefix. Pass an empty string to opt out.');
        }
        this.spec = loadResponseTemplate(responseTemplate);
        this.output = clone(this.spec.defaults);
        for (const value of Object.values(this.output)) deepFreeze(value);
        this.current = this.spec.implicit;
        this.captures = {};
        this.body = null;
        this.position = 0;
        this.buffer = truncatePastLastAnchor(this.spec.startAnchor, prefix);
        this.finalized = false;
        this.process(false);
    }

    /**
     * Parse the next decoded text chunk and return the current partial message.
     * @param {string} text Newly decoded text. Do not pass previously consumed text again.
     * @returns {Object.<string, *>}
     */
    feed(text) {
        if (this.finalized) throw new Error('ResponseParser is already finalized.');
        if (typeof text !== 'string') throw new TypeError('ResponseParser.feed expects a string.');
        this.buffer += text;
        this.process(false);
        return this.snapshot(true);
    }

    /**
     * Finish parsing, resolve end-of-input anchors, and validate required fields.
     * @returns {Object.<string, *>}
     */
    finalize() {
        if (this.finalized) throw new Error('ResponseParser is already finalized.');
        this.process(true);
        const output = this.snapshot(false);
        this.finalized = true;
        return output;
    }

    process(final) {
        while (true) {
            const watch = this.watchlist();
            let best = null;
            let pendingStart = this.buffer.length;
            for (const item of watch) {
                const anchor = item.kind === 'open' ? item.field.open : item.field.close;
                const match = findMatch(anchor, this.buffer, this.position, final && item.kind === 'close');
                if (!final && anchor.pattern) {
                    const prefix = anchor.partialPrefixes[0];
                    const prefixStart = prefix ? this.buffer.indexOf(prefix, this.position) : -1;
                    if (prefixStart >= 0 && (!match || prefixStart < match.start || match.end === this.buffer.length)) {
                        pendingStart = Math.min(pendingStart, prefixStart);
                    }
                    if (match?.end === this.buffer.length) {
                        pendingStart = Math.min(pendingStart, match.start);
                        continue;
                    }
                }
                if (!match) continue;
                const candidate = { ...item, match };
                if (!best || compareCandidates(candidate, best) < 0) best = candidate;
            }

            if (best && best.match.start >= pendingStart) best = null;

            if (!best) {
                const holdStart = final
                    ? this.buffer.length
                    : Math.min(
                          pendingStart,
                          partialLiteralStart(
                              watch.map((item) => (item.kind === 'open' ? item.field.open : item.field.close)),
                              this.buffer,
                              this.position,
                          ),
                      );
                this.accumulate(this.buffer.slice(this.position, holdStart));
                this.position = holdStart;
                if (final) this.closeCurrent();
                break;
            }

            this.accumulate(this.buffer.slice(this.position, best.match.start));
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
        const output = { ...this.output };
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
        return Object.freeze(output);
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
            const previous = Object.hasOwn(output, field.name) ? output[field.name] : [];
            setOwn(output, field.name, Object.freeze([...previous, deepFreeze(value)]));
        } else {
            setOwn(output, field.name, deepFreeze(value));
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

function deepFreeze(value) {
    if (value !== null && typeof value === 'object' && !Object.isFrozen(value)) {
        for (const item of Object.values(value)) deepFreeze(item);
        Object.freeze(value);
    }
    return value;
}

function setOwn(object, key, value) {
    Object.defineProperty(object, key, { value, writable: true, enumerable: true, configurable: true });
}
