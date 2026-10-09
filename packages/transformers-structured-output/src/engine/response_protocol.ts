import { compileRegex } from './regex';
import type { ConstraintState, TokenizerSource } from './types';

type Matcher = { machine: ConstraintState<unknown>; eager: boolean };
type MatchProgress = { state: unknown; accepted: boolean };
export type ProtocolConfig = {
    thinkingOpen?: Matcher;
    thinkingClose?: Uint8Array[];
    toolOpen?: Matcher;
    toolClose?: Uint8Array[];
    toolStringDelimiters?: Uint8Array[];
    contentOpen?: Matcher;
    contentClose?: Matcher;
    startAnchors?: Uint8Array[];
};
type ProtocolState =
    | {
          phase: 'start';
          structured?: unknown;
          thinking?: MatchProgress;
          content?: MatchProgress;
          tool?: MatchProgress;
          framingWhitespace: boolean;
      }
    | { phase: 'thinking'; closeBuffer: Uint8Array }
    | { phase: 'tool'; closeBuffer: Uint8Array }
    | { phase: 'afterTool'; tool: MatchProgress; canEnd: boolean }
    | { phase: 'content' | 'ending'; progress: MatchProgress }
    | { phase: 'structured'; state: unknown }
    | { phase: 'dead' | 'finished' };

export function withResponseProtocol(
    structured: ConstraintState<unknown>,
    protocol: ProtocolConfig,
): ConstraintState<ProtocolState> {
    const payload = (state: unknown): ProtocolState =>
        structured.viable(state) ? { phase: 'structured', state } : { phase: 'dead' };
    const start = (allowThinking: boolean): ProtocolState => ({
        phase: 'start',
        structured: protocol.contentOpen === undefined ? structured.initial : undefined,
        thinking: allowThinking && protocol.thinkingOpen ? initialMatch(protocol.thinkingOpen) : undefined,
        content: protocol.contentOpen === undefined ? undefined : initialMatch(protocol.contentOpen),
        tool: protocol.toolOpen === undefined ? undefined : initialMatch(protocol.toolOpen),
        framingWhitespace: true,
    });
    const initial = start(true);
    const framed = (
        phase: 'content' | 'ending',
        progress: MatchProgress,
        matcher: Matcher,
        byte: number,
    ): ProtocolState => {
        const next = advanceMatch(matcher, progress, byte);
        if (next.complete === 'after') return phase === 'ending' ? { phase: 'finished' } : payload(structured.initial);
        if (next.complete === 'before' && phase === 'content')
            return payload(structured.transition(structured.initial, byte));
        return next.progress === undefined ? { phase: 'dead' } : { phase, progress: next.progress };
    };
    const transition = (state: ProtocolState, byte: number): ProtocolState => {
        switch (state.phase) {
            case 'dead':
                return state;
            case 'finished':
                return { phase: 'dead' };
            case 'structured': {
                const next = structured.transition(state.state, byte);
                if (structured.viable(next)) return next === state.state ? state : payload(next);
                return structured.accepting(state.state) && protocol.contentClose !== undefined
                    ? framed('ending', initialMatch(protocol.contentClose), protocol.contentClose, byte)
                    : { phase: 'dead' };
            }
            case 'content':
                return framed('content', state.progress, protocol.contentOpen!, byte);
            case 'ending':
                return framed('ending', state.progress, protocol.contentClose!, byte);
            case 'thinking': {
                const close = advanceLiteralSearch(protocol.thinkingClose!, state.closeBuffer, byte);
                if (close.complete === 'after') return start(false);
                if (close.complete === 'before') return transition(start(false), byte);
                return { phase: 'thinking', closeBuffer: close.buffer };
            }
            case 'tool': {
                const close = advanceLiteralSearch(protocol.toolClose!, state.closeBuffer, byte);
                if (close.complete === 'after')
                    return protocol.toolOpen === undefined
                        ? { phase: 'finished' }
                        : { phase: 'afterTool', tool: initialMatch(protocol.toolOpen), canEnd: true };
                if (close.complete === 'before') {
                    const after: ProtocolState =
                        protocol.toolOpen === undefined
                            ? { phase: 'finished' }
                            : { phase: 'afterTool', tool: initialMatch(protocol.toolOpen), canEnd: true };
                    return transition(after, byte);
                }
                return { phase: 'tool', closeBuffer: close.buffer };
            }
            case 'afterTool': {
                if (isFramingWhitespace(byte)) return state;
                const tool = advanceMatch(protocol.toolOpen!, state.tool, byte);
                if (tool.complete === 'after') return { phase: 'tool', closeBuffer: new Uint8Array() };
                if (tool.complete === 'before')
                    return transition({ phase: 'tool', closeBuffer: new Uint8Array() }, byte);
                return tool.progress === undefined
                    ? { phase: 'dead' }
                    : { phase: 'afterTool', tool: tool.progress, canEnd: false };
            }
            case 'start': {
                const thinking = state.thinking && advanceMatch(protocol.thinkingOpen!, state.thinking, byte);
                if (thinking?.complete === 'after') return { phase: 'thinking', closeBuffer: new Uint8Array() };
                if (thinking?.complete === 'before')
                    return transition({ phase: 'thinking', closeBuffer: new Uint8Array() }, byte);
                const content = state.content && advanceMatch(protocol.contentOpen!, state.content, byte);
                if (content?.complete === 'after') return payload(structured.initial);
                if (content?.complete === 'before') return payload(structured.transition(structured.initial, byte));
                const tool = state.tool && advanceMatch(protocol.toolOpen!, state.tool, byte);
                if (tool?.complete === 'after') return { phase: 'tool', closeBuffer: new Uint8Array() };
                if (tool?.complete === 'before')
                    return transition({ phase: 'tool', closeBuffer: new Uint8Array() }, byte);
                const next = state.structured === undefined ? undefined : structured.transition(state.structured, byte);
                const direct = next !== undefined && structured.viable(next) ? next : undefined;
                const framingWhitespace = state.framingWhitespace && isFramingWhitespace(byte);
                const thinkingProgress = thinking?.progress ?? (framingWhitespace ? state.thinking : undefined);
                const contentProgress = content?.progress ?? (framingWhitespace ? state.content : undefined);
                const toolProgress = tool?.progress ?? (framingWhitespace ? state.tool : undefined);
                if (thinkingProgress === undefined && contentProgress === undefined && toolProgress === undefined)
                    return direct === undefined ? { phase: 'dead' } : payload(direct);
                return {
                    phase: 'start',
                    structured: direct,
                    thinking: thinkingProgress,
                    content: contentProgress,
                    tool: toolProgress,
                    framingWhitespace,
                };
            }
        }
    };
    return {
        initial,
        initialize: (bytes, context) => {
            let start = -1;
            for (const anchor of protocol.startAnchors ?? []) {
                const index = lastIndexOfBytes(bytes, anchor);
                if (index >= 0) start = Math.max(start, index + anchor.length);
            }
            let state: ProtocolState = initial;
            let atContentBoundary = false;
            if (start >= 0) {
                for (let i = start; i < bytes.length; ++i) {
                    // Chat templates can prefill framing newlines after a closed thinking
                    // block or content opener. They precede, rather than belong to, the
                    // constrained payload. Only skip them in the prompt, never generation.
                    if (atContentBoundary && isFramingWhitespace(bytes[i])) continue;
                    const next = transition(state, bytes[i]);
                    atContentBoundary =
                        state.phase !== 'structured' &&
                        ((next.phase === 'structured' && next.state === structured.initial) ||
                            (state.phase === 'thinking' &&
                                next.phase === 'start' &&
                                next.structured === structured.initial));
                    state = next;
                }
            }
            if (state.phase === 'dead')
                throw new Error('The prompt prefill does not satisfy the response protocol or constraint.');
            if (context?.enable_thinking !== false) return state;
            if (state.phase === 'thinking') {
                throw new Error('Thinking is disabled but the prompt prefills an unfinished thinking region.');
            }
            if (state.phase !== 'start') return state;
            const withoutThinking = { ...state, thinking: undefined };
            if (
                withoutThinking.structured !== undefined ||
                withoutThinking.content !== undefined ||
                withoutThinking.tool !== undefined
            )
                return withoutThinking;
            throw new Error('The prompt is incompatible with disabled thinking.');
        },
        transition,
        viable: (state) => state.phase !== 'dead',
        accepting: (state) =>
            state.phase === 'finished' ||
            (state.phase === 'afterTool' && state.canEnd) ||
            (state.phase === 'structured' && structured.accepting(state.state)),
        allowsSpecial: (state, next, bytes) => {
            if (bytes.length === 0) return false;
            if (state.phase === 'thinking') {
                return next.phase !== 'thinking' || next.closeBuffer.length > state.closeBuffer.length;
            }
            if (state.phase === 'tool') {
                return (
                    protocol.toolStringDelimiters?.some((delimiter) => bytesEqual(bytes, delimiter)) ||
                    next.phase !== 'tool' ||
                    next.closeBuffer.length > state.closeBuffer.length
                );
            }
            if (state.phase === 'structured') return next.phase === 'ending' || next.phase === 'finished';
            if (state.phase === 'start' || state.phase === 'afterTool') {
                // Only protocol branches justify special tokens, never the payload branch.
                let candidate: ProtocolState = state.phase === 'start' ? { ...state, structured: undefined } : state;
                for (const byte of bytes) candidate = transition(candidate, byte);
                return candidate.phase !== 'dead';
            }
            return state.phase === 'content' || state.phase === 'ending';
        },
        stringCapacity: (state) =>
            state.phase === 'structured' ? structured.stringCapacity?.(state.state) : undefined,
        maskKey: (state) => {
            if (state.phase === 'thinking') return `thinking:${Array.from(state.closeBuffer).join(',')}`;
            if (state.phase === 'tool') return `tool:${Array.from(state.closeBuffer).join(',')}`;
            if (state.phase !== 'structured') return undefined;
            const key = structured.maskKey?.(state.state);
            return key === undefined ? undefined : `structured:${key}`;
        },
    };
}

function isFramingWhitespace(byte: number): boolean {
    return byte === 0x09 || byte === 0x0a || byte === 0x0d || byte === 0x20;
}

function initialMatch(matcher: Matcher): MatchProgress {
    return { state: matcher.machine.initial, accepted: matcher.machine.accepting(matcher.machine.initial) };
}
function advanceMatch(
    matcher: Matcher,
    progress: MatchProgress,
    byte: number,
): { progress?: MatchProgress; complete?: 'before' | 'after' } {
    const next = matcher.machine.transition(progress.state, byte);
    if (!matcher.machine.viable(next)) return progress.accepted ? { complete: 'before' } : {};
    const accepted = matcher.machine.accepting(next);
    return accepted && matcher.eager ? { complete: 'after' } : { progress: { state: next, accepted } };
}
function advanceLiteralSearch(
    literals: Uint8Array[],
    buffer: Uint8Array,
    byte: number,
): { buffer: Uint8Array; complete?: 'before' | 'after' } {
    if (
        literals.some((literal) => bytesEqual(buffer, literal)) &&
        !literals.some((literal) => startsWithBytes(literal, appendByte(buffer, byte)))
    ) {
        return { buffer: new Uint8Array(), complete: 'before' };
    }
    const candidate = appendByte(buffer, byte);
    let next = new Uint8Array();
    for (let start = 0; start < candidate.length; ++start) {
        const suffix = candidate.subarray(start);
        if (literals.some((literal) => startsWithBytes(literal, suffix))) {
            next = suffix.slice();
            break;
        }
    }
    const complete = literals.some((literal) => bytesEqual(next, literal));
    const canExtend =
        complete && literals.some((literal) => literal.length > next.length && startsWithBytes(literal, next));
    return { buffer: next, complete: complete && !canExtend ? 'after' : undefined };
}

export function responseProtocol(source: TokenizerSource): ProtocolConfig | undefined {
    const template = asRecord(asRecord(source)?.response_template);
    const fields = asRecord(template?.fields);
    const thinking = asRecord(fields?.thinking);
    const toolCalls = asRecord(fields?.tool_calls);
    const content = asRecord(fields?.content);
    if (content === undefined || (thinking === undefined && toolCalls === undefined)) return undefined;
    if (template?.start_anchor_pattern !== undefined) {
        throw new Error(
            'Response-area-aware constraints do not support start_anchor_pattern; use start_anchor with a literal string or list of strings.',
        );
    }
    if (thinking?.close_pattern !== undefined)
        throw new Error(
            'Response-area-aware constraints do not support a thinking close_pattern; use close with a literal string or list of strings.',
        );
    if (toolCalls?.close_pattern !== undefined)
        throw new Error(
            'Response-area-aware constraints do not support a tool_calls close_pattern; use close with a literal string or list of strings.',
        );
    const thinkingOpen = thinking && anchorMatcher(thinking, 'open', 'open_pattern');
    const thinkingClose = thinking && literalAnchor(thinking.close, 'response_template.fields.thinking.close');
    if (thinking !== undefined && (thinkingOpen === undefined || thinkingClose === undefined))
        throw new Error(
            'Response-area-aware constraints require supported thinking open and literal close delimiters.',
        );
    const toolOpen = toolCalls && anchorMatcher(toolCalls, 'open', 'open_pattern');
    const toolClose = toolCalls && literalAnchor(toolCalls.close, 'response_template.fields.tool_calls.close');
    const stringDelimiters = asRecord(toolCalls?.content_args)?.string_delims;
    const toolStringDelimiters = Array.isArray(stringDelimiters)
        ? literalAnchor(stringDelimiters.flat(), 'response_template.fields.tool_calls.content_args.string_delims')
        : undefined;
    if (toolCalls !== undefined && (toolOpen === undefined || toolClose === undefined))
        throw new Error(
            'Response-area-aware constraints require supported tool_calls open and literal close delimiters.',
        );
    const contentOpen = anchorMatcher(content, 'open', 'open_pattern');
    if ((content.open !== undefined || content.open_pattern !== undefined) && contentOpen === undefined)
        throw new Error('Response-area-aware constraints require a supported content opener.');
    return {
        thinkingOpen,
        thinkingClose,
        toolOpen,
        toolClose,
        toolStringDelimiters,
        contentOpen,
        contentClose: anchorMatcher(content, 'close', 'close_pattern'),
        startAnchors: literalAnchor(template?.start_anchor, 'response_template.start_anchor'),
    };
}
function anchorMatcher(field: Record<string, unknown>, literalKey: string, patternKey: string): Matcher | undefined {
    if (field[literalKey] !== undefined) {
        const literals = literalAnchor(field[literalKey], `response_template field ${literalKey}`);
        if (literals === undefined) return undefined;
        const source = literals
            .map((literal) => new TextDecoder().decode(literal).replace(/[\\^$.*+?()[\]{}|]/g, '\\$&'))
            .join('|');
        const canExtend = literals.some((a) => literals.some((b) => a !== b && startsWithBytes(a, b)));
        return { machine: compileRegex(source) as ConstraintState<unknown>, eager: !canExtend };
    }
    if (typeof field[patternKey] === 'string') {
        try {
            return { machine: compileRegex(field[patternKey] as string) as ConstraintState<unknown>, eager: false };
        } catch (error) {
            throw new Error(`Unsupported response template ${patternKey}: ${String(error)}`);
        }
    }
    return undefined;
}
function literalAnchor(value: unknown, name: string): Uint8Array[] | undefined {
    const values = typeof value === 'string' ? [value] : Array.isArray(value) ? value : undefined;
    if (values === undefined || values.length === 0 || !values.every((item) => typeof item === 'string'))
        return undefined;
    return [...new Set(values as string[])].map((item) => {
        if (item.length === 0) throw new TypeError(`${name} cannot contain an empty string.`);
        return new TextEncoder().encode(item);
    });
}
function appendByte(bytes: Uint8Array, byte: number): Uint8Array {
    const result = new Uint8Array(bytes.length + 1);
    result.set(bytes);
    result[bytes.length] = byte;
    return result;
}
function startsWithBytes(value: Uint8Array, prefix: Uint8Array): boolean {
    if (prefix.length > value.length) return false;
    for (let i = 0; i < prefix.length; ++i) if (value[i] !== prefix[i]) return false;
    return true;
}
function bytesEqual(a: Uint8Array, b: Uint8Array): boolean {
    return a.length === b.length && startsWithBytes(a, b);
}
function lastIndexOfBytes(value: Uint8Array, search: Uint8Array): number {
    for (let start = value.length - search.length; start >= 0; --start)
        if (startsWithBytes(value.subarray(start), search)) return start;
    return -1;
}
function asRecord(value: unknown): Record<string, unknown> | undefined {
    return value !== null && (typeof value === 'object' || typeof value === 'function')
        ? (value as Record<string, unknown>)
        : undefined;
}
