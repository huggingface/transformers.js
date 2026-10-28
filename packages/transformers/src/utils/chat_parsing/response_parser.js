import { STREAMABLE_PARSERS, is_object, process_field, to_float, to_int } from './content_parsers.js';
import { search } from './regex.js';
import { load_response_template, truncate_past_last_anchor } from './response_templates.js';

/**
 * @typedef {{ type: 'region_open', field: string }
 *   | { type: 'region_chunk', field: string, text: string, dirty: boolean }
 *   | { type: 'region_close', field: string, value: any }} ResponseEvent
 * An event emitted by {@link ResponseParser}. `dirty` flags chunks from structured parsers (json,
 * xml-inline, kv-lines), whose raw text is only parsed into the final value on close.
 */

/**
 * @typedef {Object} ResponseParserOptions
 * @property {string} prefix The chat prompt sent to the model before generation. Pass `""` to opt out.
 * @property {Object[]|null} [tools=null] Tools available to the model, in the same format as
 * `apply_chat_template` accepts. When set, tool-call arguments are cast using the calling tool's JSON schema.
 */

/**
 * @typedef {import('./response_templates.js').ResponseTemplateField} ResponseTemplateField
 * @typedef {['open'|'close', ResponseTemplateField]} WatchItem
 * @typedef {['open'|'close', ResponseTemplateField, import('./regex.js').Match]} Candidate
 */

/**
 * The JSON schema types a tool parameter accepts.
 * @param {any} schema The parameter's JSON schema.
 * @returns {string[]}
 */
function schema_types(schema) {
    if (!is_object(schema)) return [];
    const declared = schema.type;
    const types = typeof declared === 'string' ? [declared] : [];
    if (Array.isArray(declared)) types.push(...declared.filter((t) => typeof t === 'string'));
    for (const union_name of ['anyOf', 'oneOf']) {
        for (const choice of schema[union_name] ?? []) types.push(...schema_types(choice));
    }
    if (schema.nullable && !types.includes('null')) types.push('null');
    return types;
}

/**
 * Cast raw text to the first JSON schema type it is valid for.
 * @param {string} raw The raw argument text.
 * @param {string[]} types Candidate types, in order.
 * @returns {any}
 */
function coerce(raw, types) {
    for (const type_name of types) {
        try {
            if (type_name === 'integer') return to_int(raw);
            if (type_name === 'number') {
                const number = to_float(raw);
                if (!Number.isFinite(number)) continue; // NaN / inf are not valid JSON numbers
                return number;
            }
            const value = raw.trim().toLowerCase();
            if (type_name === 'boolean' && ['true', '1', 'false', '0'].includes(value)) {
                return value === 'true' || value === '1';
            }
            if (type_name === 'null' && ['null', 'None'].includes(raw.trim())) return null;
            if (type_name === 'object' || type_name === 'array') {
                const decoded = JSON.parse(raw);
                if (type_name === 'object' ? is_object(decoded) : Array.isArray(decoded)) return decoded;
            }
        } catch {
            continue;
        }
    }
    return raw; // `string` params, unknown types and failed casts all keep the original text
}

/**
 * The main function for response parsing when you don't want streaming. Takes generated output
 * and the prompt prefix and parses them without streaming any events, then returns the parsed message.
 *
 * @param {string} text The generated text, excluding the prompt.
 * @param {Object} response_template The response template.
 * @param {ResponseParserOptions} options Parsing options.
 * @returns {Record<string, any>} The parsed message.
 */
export function parse_response(text, response_template, options) {
    const parser = new ResponseParser(response_template, options);
    parser.feed(text);
    return parser.finalize()[0];
}

/**
 * This class implements a streaming parser with a `response_template`. If you don't need streaming and
 * just want to parse a complete message, use `parse_response` instead. Streaming parsing emits events
 * indicating when regions (message fields) are opened and closed, with the model writing to the region
 * that is currently open.
 *
 * ```javascript
 * const parser = new ResponseParser(response_template, { prefix: chat_prompt });
 * for (const event of parser.initial_events) handle(event);
 * for (const chunk of model_text_stream) {
 *     for (const event of parser.feed(chunk)) handle(event);
 * }
 * const [message, final_events] = parser.finalize();
 * for (const event of final_events) handle(event);
 * ```
 *
 * Events are either `region_open`, `region_chunk`, or `region_close`.
 *
 * The parser requires the chat `prefix` (i.e. the chat history, the prefill before the current generation).
 * This is because chat templates or assistant prefills can sometimes write part of the message, and if we
 * only see the model output, and not the template, then we can't reliably parse the message in those cases.
 * Any events produced while consuming the prefix are exposed as `initial_events`, so renderers can show
 * prefill regions before the model writes anything; closed prefill regions also land in the output message.
 */
export class ResponseParser {
    /**
     * @param {Object} response_template The response template.
     * @param {ResponseParserOptions} options Parsing options.
     */
    constructor(response_template, { prefix, tools = null } = /** @type {any} */ ({})) {
        this._spec = load_response_template(response_template);
        if (typeof prefix !== 'string') {
            throw new Error(
                '`ResponseParser`/`parse_response` requires `prefix` (the chat prompt sent to the model before ' +
                    'generation), because chat templates often pre-write part of the assistant message (e.g. an ' +
                    'opening `<think>` tag) that the parser must see to parse the output correctly. If the generation ' +
                    'already contains the complete message, pass `prefix: ""` to opt out explicitly.',
            );
        }
        /**
         * Maps tool name -> schema `properties`, used to cast parsed tool-call arguments
         * @type {Map<string, Record<string, any>>}
         */
        this._tool_params = new Map();
        for (const tool of tools ?? []) {
            const fn = is_object(tool) ? (Object.hasOwn(tool, 'function') ? tool.function : tool) : null;
            if (is_object(fn) && typeof fn.name === 'string') {
                const properties = is_object(fn.parameters) ? fn.parameters.properties : null;
                this._tool_params.set(fn.name, is_object(properties) ? properties : {});
            }
        }
        this._buffer = '';
        this._pos = 0;
        /** @type {Record<string, any>} */
        this._output = Object.assign(Object.create(null), structuredClone(this._spec.defaults));
        this._implicit_name = this._spec.implicit;
        // Unified current-region state: starts in the implicit region (or a
        // null sink if none was declared), and returns there after every close.
        // For explicit regions `_opened` flips to true eagerly on the open
        // match; for the implicit region it flips lazily on the first character.
        /** @type {string|null} */
        this._current = this._implicit_name;
        /** @type {Record<string, string>} */
        this._captures = {};
        this._body = '';
        this._opened = false;
        this._finalized = false;
        /** @type {ResponseEvent[]} */
        this.initial_events = [];
        if (prefix) this._consume_prefix(prefix);
    }

    /**
     * Loads the prefix (the chat prefill sent to the model), right-truncates it to the start of the
     * assistant message (as determined by start_anchor) and then runs the remainder through the parser.
     * Events produced while processing the prefix are stashed on `initial_events` so callers can replay
     * them into a renderer before feeding model output.
     * @param {string} prefix
     * @private
     */
    _consume_prefix(prefix) {
        const truncated = truncate_past_last_anchor(this._spec, prefix);
        if (!truncated) return;
        this._buffer = truncated;
        this._process(this.initial_events, false);
    }

    /**
     * Feeds more generated text into the parser, and returns any events that result
     * (regions entered or left). Call this after each generation step.
     * @param {string} text Newly generated text.
     * @returns {ResponseEvent[]}
     */
    feed(text) {
        if (this._finalized) throw new Error('ResponseParser already finalized');
        if (text) this._buffer += text;
        /** @type {ResponseEvent[]} */
        const events = [];
        this._process(events, false);
        return events;
    }

    /**
     * Close the stream and return the final message together with any finalization events.
     * This is necessary because some regions may only end at the end of the sequence, so you
     * won't see the event telling you they're ready until the sequence is finalized.
     * @returns {[Record<string, any>, ResponseEvent[]]}
     */
    finalize() {
        if (this._finalized) throw new Error('ResponseParser already finalized');
        /** @type {ResponseEvent[]} */
        const events = [];
        this._process(events, true);
        const missing = Object.values(this._spec.fields)
            .filter((f) => !f.optional && !(f.name in this._output))
            .map((f) => f.name);
        if (missing.length) {
            throw new Error(`Required response_template fields missing from parsed output: ${JSON.stringify(missing)}`);
        }
        const defaults = this._spec.defaults;
        this._output = Object.fromEntries(
            Object.entries(this._output).filter(([k, v]) => Object.hasOwn(defaults, k) || !is_empty(v)),
        );
        this._finalized = true;
        return [this._output, events];
    }

    /**
     * @param {ResponseEvent[]} events
     * @param {boolean} eos
     * @private
     */
    _process(events, eos) {
        while (true) {
            const watch = this._watchlist();
            const [best, hold_start] = this._scan(watch, eos);

            if (best !== null) {
                const [kind, field, m] = best;
                if (m.start > this._pos) this._accumulate(events, this._buffer.slice(this._pos, m.start));
                this._pos = m.end;
                if (kind === 'open') {
                    this._close_current(events);
                    this._open_explicit(events, field, m);
                } else {
                    // "close" (always the implicit region's close here,
                    // since explicit regions only expose their own close)
                    const had_content = this._opened;
                    this._close_current(events);
                    // Zero-width close on an already-empty region would just
                    // re-fire next iteration -- bail out to make progress.
                    if (!had_content && m.start === m.end) break;
                }
                continue;
            }

            // No committable match in the current buffer.
            if (eos) {
                if (this._pos < this._buffer.length) {
                    this._accumulate(events, this._buffer.slice(this._pos));
                    this._pos = this._buffer.length;
                }
                this._close_current(events);
                break;
            }
            // Stream everything up to the earliest still-pending delimiter. When
            // nothing is pending `hold_start === this._buffer.length`, so this flushes
            // the whole buffer; otherwise we hold the (possibly partial) delimiter
            // back until more input resolves it.
            if (hold_start > this._pos) {
                this._accumulate(events, this._buffer.slice(this._pos, hold_start));
                this._pos = hold_start;
            }
            break;
        }
    }

    /**
     * Patterns we care about right now: the close of the currently-open
     * explicit region, or -- if we're in the implicit/null region -- every
     * explicit open plus the implicit's own close (if any).
     * @returns {WatchItem[]}
     * @private
     */
    _watchlist() {
        if (this._current !== null && this._current !== this._implicit_name) {
            const field = this._spec.fields[this._current];
            return field.close !== null ? [['close', field]] : [];
        }
        /** @type {WatchItem[]} */
        const watch = [];
        for (const field of Object.values(this._spec.fields)) {
            if (field.open !== null) watch.push(['open', field]);
        }
        if (this._implicit_name !== null) {
            const impl = this._spec.fields[this._implicit_name];
            if (impl.close !== null) watch.push(['close', impl]);
        }
        return watch;
    }

    /**
     * Single pass over the watched delimiters, using partial matching to decide -- per delimiter --
     * whether it can be committed now or must be held. Returns `[best, hold_start]`:
     *
     * - `best` is the earliest-starting delimiter we can safely commit *now*
     *   (longest on ties, opens before closes), or `null`.
     * - `hold_start` is the leftmost buffer position occupied by a still-pending
     *   match: a partial (incomplete) delimiter, or a complete one ending at the
     *   buffer edge that more input could still grow. Text before it is safe to
     *   emit; text from it onward must be held. It stays `this._buffer.length` when
     *   nothing is pending, letting the caller flush the whole buffer.
     *
     * A complete match is committable only if it starts strictly before
     * `hold_start` -- otherwise an earlier (or co-located) pending delimiter could
     * turn out to be the real one. At EOS nothing can grow, so partial matching is
     * skipped and every complete match is committable.
     *
     * (A partial search with no real match returns a zero-width partial match at the
     * buffer end; that lands in the pending branch with `start === this._buffer.length`,
     * a no-op for `hold_start`.)
     * @param {WatchItem[]} watch
     * @param {boolean} eos
     * @returns {[Candidate|null, number]}
     * @private
     */
    _scan(watch, eos) {
        /** @type {Candidate|null} */
        let best = null;
        let hold_start = this._buffer.length;
        for (const [kind, field] of watch) {
            // The watchlist only includes fields whose delimiter is set.
            const anchor = /** @type {import('./response_templates.js').Anchor} */ (
                kind === 'open' ? field.open : field.close
            );
            const m = search(anchor, this._buffer, this._pos, !eos);
            if (m === null) continue;
            if (!eos && (m.partial || this._can_grow(anchor, m))) {
                // Pending: can't commit, and blocks emitting from its start onward.
                hold_start = Math.min(hold_start, m.start);
                continue;
            }
            /** @type {Candidate} */
            const candidate = [kind, field, m];
            if (best === null || compare_candidates(candidate, best) < 0) best = candidate;
        }
        // A committable match co-located with or after a pending one must wait too:
        // the pending delimiter starts no later and might be the one that fires.
        if (best !== null && best[2].start >= hold_start) best = null;
        return [best, hold_start];
    }

    /**
     * Whether a *complete* match ending at the current buffer edge could still
     * change as more input arrives -- in which case we defer rather than commit. A
     * match ending before the edge has already seen its terminating character and is
     * final. At the edge: zero-width matches (`$` / `\Z`) are only real at true
     * EOS; a fully-present literal that no other literal in its set extends cannot
     * grow (the fast path that keeps literal delimiters zero-latency); anything
     * else (regex delimiters, prefix-overlapping literal lists) might.
     * @param {import('./response_templates.js').Anchor} anchor
     * @param {import('./regex.js').Match} m
     * @private
     */
    _can_grow(anchor, m) {
        if (m.end !== this._buffer.length) return false;
        if (m.start === m.end) return true;
        return anchor.literals === null || anchor.literal_can_extend;
    }

    /**
     * Route `text` into the currently active region. When the current
     * region is the null sink (no implicit declared, no explicit open), we
     * silently discard. Every routed chunk emits a `region_chunk` event so
     * consumers can render live; `dirty: true` flags chunks from structured
     * parsers (json, xml-inline, kv-lines) and transformed text whose raw
     * text will only be parsed into the final value on close.
     * @param {ResponseEvent[]} events
     * @param {string} text
     * @private
     */
    _accumulate(events, text) {
        if (!text || this._current === null) return;
        const field = this._spec.fields[this._current];
        if (!this._opened) {
            events.push({ type: 'region_open', field: this._current });
            this._opened = true;
        }
        this._body += text;
        const dirty =
            !STREAMABLE_PARSERS.has(field.content) ||
            Boolean(field.content_args.strip_prefix) ||
            Boolean(field.content_args.strip_suffix);
        events.push({ type: 'region_chunk', field: this._current, text, dirty });
    }

    /**
     * @param {ResponseEvent[]} events
     * @param {ResponseTemplateField} field
     * @param {import('./regex.js').Match} m
     * @private
     */
    _open_explicit(events, field, m) {
        this._current = field.name;
        this._captures = m.groups;
        this._body = '';
        this._opened = true;
        events.push({ type: 'region_open', field: field.name });
    }

    /**
     * Close the current region and reset to the implicit/null region.
     * Skipped (aside from the reset) when the current region never opened --
     * avoids vacuous open/close pairs at every explicit boundary.
     * @param {ResponseEvent[]} events
     * @private
     */
    _close_current(events) {
        if (this._current === null || !this._opened) {
            this._reset_to_implicit();
            return;
        }
        const name = this._current;
        const field = this._spec.fields[name];
        let value = process_field(this._body, field, this._captures);
        if (this._tool_params.size) value = this._coerce_tool_calls(value);
        if (field.join !== null) {
            if (typeof value !== 'string') {
                throw new Error(
                    `Field '${field.name}': 'join' requires each match to parse to a string, got ${typeof value}.`,
                );
            }
            const previous = this._output[name];
            this._output[name] = previous == null ? value : previous + field.join + value;
        } else if (field.repeats) {
            (this._output[name] ??= []).push(value);
        } else {
            this._output[name] = value;
        }
        events.push({ type: 'region_close', field: name, value });
        this._reset_to_implicit();
    }

    /**
     * Cast string tool-call arguments using the calling tool's JSON schema.
     * @param {any} value A parsed region value.
     * @returns {any}
     * @private
     */
    _coerce_tool_calls(value) {
        if (Array.isArray(value)) return value.map((item) => this._coerce_tool_calls(item));
        const fn = is_object(value) ? value.function : null;
        if (!is_object(fn)) return value;
        const { name, arguments: args } = fn;
        if (typeof name !== 'string' || !is_object(args)) return value;
        const properties = this._tool_params.get(name);
        if (properties) {
            for (const [key, argument] of Object.entries(args)) {
                const types = Object.hasOwn(properties, key) ? schema_types(properties[key]) : [];
                if (types.length === 0) continue;
                if (typeof argument === 'string') {
                    args[key] = coerce(argument, types);
                } else if (Array.isArray(argument) && !types.includes('array')) {
                    // duplicate keys collected by `merge_duplicates`
                    args[key] = argument.map((item) => (typeof item === 'string' ? coerce(item, types) : item));
                }
            }
        }
        return value;
    }

    /** @private */
    _reset_to_implicit() {
        this._current = this._implicit_name;
        this._captures = {};
        this._body = '';
        this._opened = false;
    }
}

/**
 * Order candidate matches: earliest start, then longest, then opens before closes, then field name.
 * @param {Candidate} a
 * @param {Candidate} b
 * @private
 */
function compare_candidates([a_kind, a_field, a], [b_kind, b_field, b]) {
    return (
        a.start - b.start ||
        b.end - b.start - (a.end - a.start) ||
        (a_kind === 'open' ? 0 : 1) - (b_kind === 'open' ? 0 : 1) ||
        (a_field.name < b_field.name ? -1 : a_field.name > b_field.name ? 1 : 0)
    );
}

/** @private */
function is_empty(v) {
    return (
        v == null ||
        ((typeof v === 'string' || Array.isArray(v)) && v.length === 0) ||
        (is_object(v) && Object.keys(v).length === 0)
    );
}
