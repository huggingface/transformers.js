import { processField, STREAMABLE_PARSERS } from './content_parsers.js';
import { findMatch, loadResponseTemplate, partialLiteralStart } from './response_templates.js';

export function parseResponse(text, responseTemplate) {
    if (typeof text !== 'string') throw new TypeError('parse_response expects a string.');
    const parser = new BestEffortResponseParser(responseTemplate);
    parser.consume(text);
    return parser.snapshot();
}

class BestEffortResponseParser {
    constructor(responseTemplate) {
        this.spec = loadResponseTemplate(responseTemplate);
        this.output = clone(this.spec.defaults);
        this.current = this.spec.implicit;
        this.captures = {};
        this.body = '';
        this.position = 0;
    }

    consume(text) {
        while (this.position < text.length) {
            const watch = this.watchlist();
            let best = null;
            for (const item of watch) {
                const anchor = item.kind === 'open' ? item.field.open : item.field.close;
                const match = findMatch(anchor, text, this.position);
                if (!match) continue;
                const candidate = { ...item, match };
                if (!best || compareCandidates(candidate, best) < 0) best = candidate;
            }

            if (!best) {
                const holdStart = partialLiteralStart(
                    watch.map((item) => (item.kind === 'open' ? item.field.open : item.field.close)),
                    text,
                    this.position,
                );
                this.accumulate(text.slice(this.position, holdStart));
                this.position = holdStart;
                break;
            }

            if (best.kind === 'orphan_close') {
                this.storeBestEffort(best.field, text.slice(this.position, best.match.start), best.match.groups);
                this.position = best.match.end;
                this.current = this.spec.implicit;
                this.captures = {};
                this.body = '';
                continue;
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
            }
        }
    }

    snapshot() {
        const output = clone(this.output);
        if (this.current !== null && this.body) {
            const field = this.spec.fields[this.current];
            if (STREAMABLE_PARSERS.has(field.content) || field.close === null) {
                try {
                    this.store(output, field, processField(this.body, field, this.captures));
                } catch {
                    // A partial numeric value may not be parseable yet.
                }
            }
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
        if (this.position === 0 && !this.body) {
            for (const field of Object.values(this.spec.fields)) {
                if (field.open && field.close) watch.push({ kind: 'orphan_close', field });
            }
        }
        return watch;
    }

    accumulate(text) {
        if (text && this.current !== null) this.body += text;
    }

    closeCurrent() {
        if (this.current !== null && this.body) {
            const field = this.spec.fields[this.current];
            this.storeBestEffort(field, this.body, this.captures);
        }
        this.current = this.spec.implicit;
        this.captures = {};
        this.body = '';
    }

    storeBestEffort(field, body, captures) {
        if (!body) return;
        try {
            this.store(this.output, field, processField(body, field, captures));
        } catch {
            // Best-effort parsing omits malformed or incomplete structured regions.
        }
    }

    store(output, field, value) {
        if (field.join !== null) {
            if (typeof value !== 'string') return;
            output[field.name] = Object.hasOwn(output, field.name) ? output[field.name] + field.join + value : value;
        } else if (field.repeats) {
            (output[field.name] ??= []).push(value);
        } else {
            output[field.name] = value;
        }
    }
}

function compareCandidates(a, b) {
    const kindRank = { open: 0, close: 1, orphan_close: 2 };
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
