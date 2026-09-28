import { processField, STREAMABLE_PARSERS } from './content_parsers.js';
import { findMatches, loadResponseTemplate } from './response_templates.js';

export function parseResponse(text, responseTemplate, { prefix = null, tools = null } = {}) {
    const parser = new ResponseParser(responseTemplate, { prefix, tools });
    parser.feed(text);
    return parser.finalize()[0];
}

export class ResponseParser {
    constructor(responseTemplate, { prefix = null, tools = null } = {}) {
        this.spec = loadResponseTemplate(responseTemplate);
        if (prefix === null) {
            throw new Error('ResponseParser requires prefix. Pass prefix: "" when no prompt context is needed.');
        }
        this.toolParameters = collectToolParameters(tools);
        this.buffer = '';
        this.position = 0;
        this.output = { ...this.spec.defaults };
        this.implicitName = this.spec.implicit;
        this.current = this.implicitName;
        this.captures = {};
        this.body = '';
        this.opened = false;
        this.finalized = false;
        this.initial_events = [];
        if (prefix) {
            this.buffer = this.spec.truncatePastLastAnchor(prefix);
            this.process(this.initial_events, false);
        }
    }

    feed(text) {
        if (this.finalized) throw new Error('ResponseParser already finalized.');
        if (text) this.buffer += text;
        const events = [];
        this.process(events, false);
        return events;
    }

    finalize() {
        if (this.finalized) throw new Error('ResponseParser already finalized.');
        const events = [];
        this.process(events, true);
        const missing = Object.values(this.spec.fields).filter(
            (field) => !field.optional && !Object.hasOwn(this.output, field.name),
        );
        if (missing.length)
            throw new Error(
                `Required response_template fields missing from parsed output: ${missing.map((field) => field.name)}`,
            );
        for (const [key, value] of Object.entries(this.output)) {
            if (!Object.hasOwn(this.spec.defaults, key) && isEmpty(value)) delete this.output[key];
        }
        this.finalized = true;
        return [this.output, events];
    }

    process(events, eos) {
        while (true) {
            const { best, holdStart } = this.scan(this.watchlist(), eos);
            if (best) {
                if (best.match.start > this.position)
                    this.accumulate(events, this.buffer.slice(this.position, best.match.start));
                this.position = best.match.end;
                if (best.kind === 'open') {
                    this.closeCurrent(events);
                    this.current = best.field.name;
                    this.captures = best.match.groups;
                    this.body = '';
                    this.opened = true;
                    events.push({ type: 'region_open', field: this.current });
                } else {
                    const hadContent = this.opened;
                    this.closeCurrent(events);
                    if (!hadContent && best.match.start === best.match.end) break;
                }
                continue;
            }
            if (eos) {
                if (this.position < this.buffer.length) this.accumulate(events, this.buffer.slice(this.position));
                this.position = this.buffer.length;
                this.closeCurrent(events);
            } else if (holdStart > this.position) {
                this.accumulate(events, this.buffer.slice(this.position, holdStart));
                this.position = holdStart;
            }
            break;
        }
    }

    watchlist() {
        if (this.current !== null && this.current !== this.implicitName) {
            const field = this.spec.fields[this.current];
            return field.close ? [{ kind: 'close', field }] : [];
        }
        const watch = Object.values(this.spec.fields)
            .filter((field) => field.open)
            .map((field) => ({ kind: 'open', field }));
        const implicit = this.implicitName === null ? null : this.spec.fields[this.implicitName];
        if (implicit?.close) watch.push({ kind: 'close', field: implicit });
        return watch;
    }

    scan(watch, eos) {
        let best = null;
        let holdStart = this.buffer.length;
        for (const item of watch) {
            const anchor = item.kind === 'open' ? item.field.open : item.field.close;
            const matches = findMatches(anchor, this.buffer, this.position);
            const match = matches[0];
            if (!eos && anchor.pattern && !match) {
                // Native RegExp cannot report partial matches, so retain the unconsumed text until the pattern resolves.
                holdStart = Math.min(holdStart, this.position);
                continue;
            }
            if (
                !eos &&
                match &&
                match.end === this.buffer.length &&
                (anchor.pattern || anchor.canExtend || match.start === match.end)
            ) {
                holdStart = Math.min(holdStart, match.start);
                continue;
            }
            if (!eos && anchor.literals)
                holdStart = Math.min(holdStart, partialLiteralStart(anchor.literals, this.buffer, this.position));
            if (!match) continue;
            const candidate = { ...item, match };
            if (!best || compareCandidates(candidate, best) < 0) best = candidate;
        }
        if (best && best.match.start >= holdStart) best = null;
        return { best, holdStart };
    }

    accumulate(events, text) {
        if (!text || this.current === null) return;
        const field = this.spec.fields[this.current];
        if (!this.opened) {
            events.push({ type: 'region_open', field: this.current });
            this.opened = true;
        }
        this.body += text;
        events.push({ type: 'region_chunk', field: this.current, text, dirty: !STREAMABLE_PARSERS.has(field.content) });
    }

    closeCurrent(events) {
        if (this.current === null || !this.opened) {
            this.resetToImplicit();
            return;
        }
        const field = this.spec.fields[this.current];
        let value = processField(this.body, field, this.captures);
        if (this.toolParameters.size) value = coerceToolCalls(value, this.toolParameters);
        if (field.join !== null) {
            if (typeof value !== 'string') throw new Error(`Field '${field.name}': join requires string values.`);
            this.output[this.current] = Object.hasOwn(this.output, this.current)
                ? this.output[this.current] + field.join + value
                : value;
        } else if (field.repeats) {
            (this.output[this.current] ??= []).push(value);
        } else {
            this.output[this.current] = value;
        }
        events.push({ type: 'region_close', field: this.current, value });
        this.resetToImplicit();
    }

    resetToImplicit() {
        this.current = this.implicitName;
        this.captures = {};
        this.body = '';
        this.opened = false;
    }
}

function compareCandidates(a, b) {
    return (
        a.match.start - b.match.start ||
        b.match.end - b.match.start - (a.match.end - a.match.start) ||
        (a.kind === b.kind ? 0 : a.kind === 'open' ? -1 : 1) ||
        a.field.name.localeCompare(b.field.name)
    );
}

function partialLiteralStart(literals, text, position) {
    for (let index = position; index < text.length; ++index) {
        const suffix = text.slice(index);
        if (literals.some((literal) => literal.startsWith(suffix) && literal !== suffix)) return index;
    }
    return text.length;
}

function collectToolParameters(tools) {
    const output = new Map();
    for (const tool of tools ?? []) {
        const fn = tool?.function ?? tool;
        if (typeof fn?.name !== 'string') continue;
        output.set(fn.name, fn.parameters?.properties ?? {});
    }
    return output;
}

function schemaTypes(schema) {
    if (schema === null || typeof schema !== 'object' || Array.isArray(schema)) return [];
    const output =
        typeof schema.type === 'string'
            ? [schema.type]
            : Array.isArray(schema.type)
              ? schema.type.filter((x) => typeof x === 'string')
              : [];
    for (const key of ['anyOf', 'oneOf']) for (const choice of schema[key] ?? []) output.push(...schemaTypes(choice));
    if (schema.nullable && !output.includes('null')) output.push('null');
    return output;
}

function coerce(raw, types) {
    for (const type of types) {
        const value = raw.trim();
        if (type === 'integer' && /^[+-]?\d+$/.test(value)) return Number.parseInt(value, 10);
        if (type === 'number' && value && Number.isFinite(Number(value))) return Number(value);
        if (type === 'boolean' && ['true', '1', 'false', '0'].includes(value.toLowerCase()))
            return ['true', '1'].includes(value.toLowerCase());
        if (type === 'null' && ['null', 'None'].includes(value)) return null;
        if (type === 'object' || type === 'array') {
            try {
                const decoded = JSON.parse(raw);
                if (
                    (type === 'object' && decoded && typeof decoded === 'object' && !Array.isArray(decoded)) ||
                    (type === 'array' && Array.isArray(decoded))
                )
                    return decoded;
            } catch {}
        }
    }
    return raw;
}

function coerceToolCalls(value, toolParameters) {
    if (Array.isArray(value)) return value.map((item) => coerceToolCalls(item, toolParameters));
    const fn = value?.function;
    if (
        typeof fn?.name !== 'string' ||
        fn.arguments === null ||
        typeof fn.arguments !== 'object' ||
        Array.isArray(fn.arguments)
    )
        return value;
    const parameters = toolParameters.get(fn.name);
    if (!parameters) return value;
    for (const [key, argument] of Object.entries(fn.arguments)) {
        const types = schemaTypes(parameters[key]);
        if (!types.length) continue;
        if (typeof argument === 'string') fn.arguments[key] = coerce(argument, types);
        else if (Array.isArray(argument) && !types.includes('array'))
            fn.arguments[key] = argument.map((item) => (typeof item === 'string' ? coerce(item, types) : item));
    }
    return value;
}

function isEmpty(value) {
    return (
        value === null ||
        value === '' ||
        (Array.isArray(value) && value.length === 0) ||
        (value && typeof value === 'object' && Object.keys(value).length === 0)
    );
}
