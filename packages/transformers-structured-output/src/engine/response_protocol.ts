import { compileRegex } from './regex';
import type { ConstraintState, TokenizerSource } from './types';

type Matcher = { machine: ConstraintState<unknown>; eager: boolean; source: string };
type MatchProgress = { state: unknown; accepted: boolean };
export type ProtocolConfig = {
    thinkingOpen: Matcher;
    thinkingClose: Uint8Array[];
    contentOpen?: Matcher;
    /** Content opener after a thinking region, which may first repeat the start anchor. */
    contentOpenAfterThinking?: Matcher;
    contentClose?: Matcher;
    startAnchor?: Matcher;
};
// Templates frame the answer with whitespace after a closed thinking block or content opener
// (Qwen3 emits `</think>\n\n`). It precedes, rather than belongs to, the constrained payload.
const MAX_FRAMING_WHITESPACE = 4;

type ProtocolState =
    | { phase: 'start'; structured?: unknown; thinking?: MatchProgress; content?: MatchProgress }
    | { phase: 'thinking'; closeBuffer: Uint8Array }
    | { phase: 'content'; progress: MatchProgress; opener: Matcher }
    | { phase: 'ending'; progress: MatchProgress }
    | { phase: 'structured'; state: unknown; framing?: number }
    | { phase: 'dead' | 'finished' };

export function withResponseProtocol(
    structured: ConstraintState<unknown>,
    protocol: ProtocolConfig,
): ConstraintState<ProtocolState> {
    const payload = (state: unknown): ProtocolState =>
        structured.viable(state) ? { phase: 'structured', state } : { phase: 'dead' };
    const boundary: ProtocolState = { phase: 'structured', state: structured.initial, framing: 0 };
    const afterThinking = (): ProtocolState =>
        protocol.contentOpenAfterThinking === undefined
            ? boundary
            : {
                  phase: 'content',
                  progress: initialMatch(protocol.contentOpenAfterThinking),
                  opener: protocol.contentOpenAfterThinking,
              };
    const initial: ProtocolState = {
        phase: 'start',
        structured: protocol.contentOpen === undefined ? structured.initial : undefined,
        thinking: initialMatch(protocol.thinkingOpen),
        content: protocol.contentOpen === undefined ? undefined : initialMatch(protocol.contentOpen),
    };
    const framed = (
        phase: 'content' | 'ending',
        progress: MatchProgress,
        matcher: Matcher,
        byte: number,
    ): ProtocolState => {
        const next = advanceMatch(matcher, progress, byte);
        if (next.complete === 'after') return phase === 'ending' ? { phase: 'finished' } : boundary;
        if (next.complete === 'before' && phase === 'content') return transition(boundary, byte);
        if (next.progress === undefined) return { phase: 'dead' };
        return phase === 'content'
            ? { phase, progress: next.progress, opener: matcher }
            : { phase, progress: next.progress };
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
                if (state.framing !== undefined && state.framing < MAX_FRAMING_WHITESPACE && isFramingWhitespace(byte))
                    return { phase: 'structured', state: state.state, framing: state.framing + 1 };
                return structured.accepting(state.state) && protocol.contentClose !== undefined
                    ? framed('ending', initialMatch(protocol.contentClose), protocol.contentClose, byte)
                    : { phase: 'dead' };
            }
            case 'content':
                return framed('content', state.progress, state.opener, byte);
            case 'ending':
                return framed('ending', state.progress, protocol.contentClose!, byte);
            case 'thinking': {
                const close = advanceLiteralSearch(protocol.thinkingClose, state.closeBuffer, byte);
                if (close.complete === 'after') return afterThinking();
                if (close.replay !== undefined) return close.replay.reduce(transition, afterThinking());
                return { phase: 'thinking', closeBuffer: close.buffer };
            }
            case 'start': {
                const thinking = state.thinking && advanceMatch(protocol.thinkingOpen, state.thinking, byte);
                if (thinking?.complete === 'after') return { phase: 'thinking', closeBuffer: new Uint8Array() };
                if (thinking?.complete === 'before')
                    return transition({ phase: 'thinking', closeBuffer: new Uint8Array() }, byte);
                const content = state.content && advanceMatch(protocol.contentOpen!, state.content, byte);
                if (content?.complete === 'after') return boundary;
                if (content?.complete === 'before') return transition(boundary, byte);
                const next = state.structured === undefined ? undefined : structured.transition(state.structured, byte);
                const direct = next !== undefined && structured.viable(next) ? next : undefined;
                if (thinking?.progress === undefined && content?.progress === undefined)
                    return direct === undefined ? { phase: 'dead' } : payload(direct);
                return { phase: 'start', structured: direct, thinking: thinking?.progress, content: content?.progress };
            }
        }
    };
    // A special token must start by advancing a delimiter; any bytes after a completed delimiter
    // may continue into the payload (`</think>{`), but its first byte may not.
    const advancesDelimiter = (state: ProtocolState, byte: number): boolean => {
        const next = transition(state, byte);
        return next.phase !== 'dead' && (next.phase !== 'structured' || next.state === structured.initial);
    };
    const accepting = (state: ProtocolState): boolean =>
        state.phase === 'finished' || (state.phase === 'structured' && structured.accepting(state.state));
    return {
        initial,
        initialize: (bytes, context) => {
            const start = protocol.startAnchor === undefined ? -1 : lastMatchEnd(protocol.startAnchor.machine, bytes);
            let state: ProtocolState = initial;
            if (start >= 0) for (let i = start; i < bytes.length; ++i) state = transition(state, bytes[i]);
            // Without a generation prompt, the last anchor opens an earlier turn that has since closed.
            if (
                state.phase === 'dead' &&
                protocol.contentClose !== undefined &&
                lastMatchEnd(protocol.contentClose.machine, bytes.subarray(start)) >= 0
            )
                state = initial;
            if (state.phase === 'dead')
                throw new Error('The prompt prefill does not satisfy the response protocol or constraint.');
            if (context?.enable_thinking !== false) return state;
            if (
                state.phase === 'thinking' ||
                (state.phase === 'start' && state.structured === undefined && state.content === undefined)
            ) {
                throw new Error('Thinking is disabled but the prompt prefills an unfinished thinking region.');
            }
            if (state.phase !== 'start') return state;
            if (state.structured !== undefined) return payload(state.structured);
            if (state.content !== undefined)
                return { phase: 'content', progress: state.content, opener: protocol.contentOpen! };
            throw new Error('The prompt is incompatible with disabled thinking.');
        },
        transition,
        viable: (state) => state.phase !== 'dead',
        accepting,
        // The end token may itself complete the content closer, e.g. Qwen3's `\s*<|im_end|>` once a
        // newline has started it.
        acceptsEnd: (state, bytes) => {
            if (accepting(state)) return true;
            let next = state;
            for (const byte of bytes) next = transition(next, byte);
            return next.phase === 'finished' || (next.phase === 'ending' && next.progress.accepted);
        },
        allowsSpecial: (state, next, bytes) => {
            if (bytes.length === 0) return false;
            if (state.phase === 'thinking') {
                return next.phase !== 'thinking' || next.closeBuffer.length > state.closeBuffer.length;
            }
            if (state.phase === 'structured') return next.phase === 'ending' || next.phase === 'finished';
            if (state.phase === 'start') {
                // Only protocol branches justify special tokens, never the payload branch.
                let candidate: ProtocolState = { ...state, structured: undefined };
                if (!advancesDelimiter(candidate, bytes[0])) return false;
                for (const byte of bytes) candidate = transition(candidate, byte);
                return candidate.phase !== 'dead';
            }
            return state.phase === 'ending' || advancesDelimiter(state, bytes[0]);
        },
        stringCapacity: (state) =>
            state.phase === 'structured' ? structured.stringCapacity?.(state.state) : undefined,
        maskKey: (state) => {
            if (state.phase === 'thinking') return `thinking:${Array.from(state.closeBuffer).join(',')}`;
            if (state.phase !== 'structured') return undefined;
            const key = structured.maskKey?.(state.state);
            if (key === undefined) return undefined;
            return state.framing === undefined ? `structured:${key}` : `framing:${state.framing}:${key}`;
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
): { buffer: Uint8Array; complete?: 'after'; replay?: Uint8Array } {
    const candidate = appendByte(buffer, byte);
    if (!literals.some((literal) => startsWithBytes(literal, candidate))) {
        // A longer alternative stopped matching: keep the longest delimiter it already passed
        // (`</think>` of `</think>\n\n`) and replay the bytes after it.
        const completed = literals.reduce<Uint8Array | undefined>(
            (best, literal) =>
                startsWithBytes(buffer, literal) && literal.length > (best?.length ?? 0) ? literal : best,
            undefined,
        );
        if (completed !== undefined) return { buffer: new Uint8Array(), replay: candidate.subarray(completed.length) };
    }
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

const warnedTemplates = new Set<string>();

/**
 * Builds the thinking-aware protocol from the tokenizer's response template. Templates the
 * protocol cannot represent fall back to the plain constraint with a one-time warning, so an
 * unsupported template never turns a working constraint into an error.
 */
export function responseProtocol(source: TokenizerSource): ProtocolConfig | undefined {
    const template = asRecord(asRecord(source)?.response_template);
    const fields = asRecord(template?.fields);
    const thinking = asRecord(fields?.reasoning_content);
    const content = asRecord(fields?.content);
    if (template === undefined || thinking === undefined || content === undefined) return undefined;
    try {
        return buildProtocol(template, thinking, content);
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        if (!warnedTemplates.has(message)) {
            warnedTemplates.add(message);
            console.warn(
                `StructuredOutputProcessor: ${message} Falling back to constraining the response from its first token.`,
            );
        }
        return undefined;
    }
}
function buildProtocol(
    template: Record<string, unknown>,
    thinking: Record<string, unknown>,
    content: Record<string, unknown>,
): ProtocolConfig {
    if (thinking.close_pattern !== undefined)
        throw new Error('Thinking-aware constraints do not support a reasoning_content close_pattern.');
    const thinkingOpen = anchorMatcher(thinking, 'open', 'open_pattern');
    const thinkingClose = literalAnchor(thinking.close, 'response_template.fields.reasoning_content.close');
    if (thinkingOpen === undefined || thinkingClose === undefined)
        throw new Error('Thinking-aware constraints require reasoning_content open and literal close delimiters.');
    const contentOpen = anchorMatcher(content, 'open', 'open_pattern');
    if ((content.open !== undefined || content.open_pattern !== undefined) && contentOpen === undefined)
        throw new Error('Thinking-aware constraints require a supported content opener.');
    const startAnchor = anchorMatcher(template, 'start_anchor', 'start_anchor_pattern');
    // Multi-message formats (gpt-oss) start a new assistant message between thinking and the
    // final answer, e.g. `<|end|><|start|>assistant<|channel|>final<|message|>`.
    const contentOpenAfterThinking =
        contentOpen === undefined || startAnchor === undefined
            ? contentOpen
            : compileMatcher(`(?:${startAnchor.source})?(?:${contentOpen.source})`, contentOpen.eager);
    return {
        thinkingOpen,
        thinkingClose,
        contentOpen,
        contentOpenAfterThinking,
        contentClose: anchorMatcher(content, 'close', 'close_pattern'),
        startAnchor,
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
        return compileMatcher(source, !canExtend);
    }
    if (typeof field[patternKey] === 'string') {
        try {
            return compileMatcher(field[patternKey] as string, false);
        } catch (error) {
            throw new Error(`Unsupported response template ${patternKey}: ${String(error)}`);
        }
    }
    return undefined;
}
function compileMatcher(source: string, eager: boolean): Matcher {
    return { machine: compileRegex(source) as ConstraintState<unknown>, eager, source };
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
// End of the last non-overlapping match, scanning left to right like the parser's `matchAll`.
function lastMatchEnd(machine: ConstraintState<unknown>, bytes: Uint8Array): number {
    let last = -1;
    for (let start = 0; start < bytes.length; ) {
        let state = machine.initial;
        let end = -1;
        for (let i = start; i < bytes.length; ++i) {
            state = machine.transition(state, bytes[i]);
            if (!machine.viable(state)) break;
            if (machine.accepting(state)) end = i + 1;
        }
        if (end > start) {
            last = end;
            start = end;
        } else {
            ++start;
        }
    }
    return last;
}
function asRecord(value: unknown): Record<string, unknown> | undefined {
    return value !== null && (typeof value === 'object' || typeof value === 'function')
        ? (value as Record<string, unknown>)
        : undefined;
}
