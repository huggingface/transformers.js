import type { ToolCall } from '../types.ts';
import { ParserStrategyBase } from './ParserStrategyBase';
import type { ParseResult, ParserContext } from './types.ts';
import { splitTopLevel } from './utils';

export class ParserStrategyGemma4 extends ParserStrategyBase {
    readonly id = 'gemma4';

    supports(context: ParserContext): boolean {
        return context.modelType === 'gemma4' || /gemma-4/i.test(context.modelId);
    }

    parseAssistantContent(content: string, nextId: (prefix: string) => string): ParseResult {
        const base = super.parseAssistantContent(content, nextId);
        const thoughtBlocks = extractGemmaThoughtBlocks(content);
        const thoughtText = thoughtBlocks.join('\n\n').trim();

        const gemmaCalls = this.parseGemmaToolCalls(content, nextId);
        const visibleText = stripGemmaToolTokens(stripGemmaThoughtTokens(base.visibleText))
            .replace(/(^|\n)\s*call:[^\n{}]+\{[^\n]*\}\s*(?=\n|$)/g, '$1')
            .replace(/<\|tool_response\|>/g, '')
            .replace(/<\|tool_response>/g, '')
            .replace(/<tool_response\|>/g, '')
            .replace(/<\|turn\|>/g, '')
            .replace(/<\|turn>/g, '')
            .replace(/<turn\|>/g, '')
            .replace(/<\|channel\|>/g, '')
            .replace(/<\|channel>/g, '')
            .replace(/<channel\|>/g, '')
            .replace(/<\|tool_call\|>/g, '')
            .replace(/<\|tool_call>/g, '')
            .replace(/<eos>/g, '')
            .trim();

        const sanitizedVisibleText = sanitizeVisibleText(visibleText);

        if (gemmaCalls.length === 0) {
            return {
                thinkingText: joinNonEmpty(base.thinkingText, thoughtText),
                visibleText: sanitizedVisibleText,
                toolCalls: base.toolCalls,
            };
        }

        return {
            thinkingText: joinNonEmpty(base.thinkingText, thoughtText),
            visibleText: sanitizedVisibleText,
            toolCalls: [...base.toolCalls, ...gemmaCalls],
        };
    }

    private parseGemmaToolCalls(content: string, nextId: (prefix: string) => string): ToolCall[] {
        const toolCalls: ToolCall[] = [];
        const regex = /<\|tool_call>([\s\S]*?)<tool_call\|>/g;
        let match: RegExpExecArray | null;

        while ((match = regex.exec(content)) !== null) {
            const parsed = parseSingleGemmaCall(match[1].trim(), nextId);
            if (parsed) {
                toolCalls.push(parsed);
            }
        }

        // Fallback: some tokenizer paths strip special tokens, leaving bare
        // `call:toolName{...}` text with no <|tool_call> wrappers.
        const bareRegex = /(^|\n)\s*(call:[^\n{}]+\{[^\n]*\})\s*(?=\n|$)/g;
        while ((match = bareRegex.exec(content)) !== null) {
            const parsed = parseSingleGemmaCall(match[2].trim(), nextId);
            if (parsed) {
                toolCalls.push(parsed);
            }
        }

        return toolCalls;
    }
}

function parseSingleGemmaCall(raw: string, nextId: (prefix: string) => string): ToolCall | null {
    if (!raw.startsWith('call:')) {
        return null;
    }

    const openBrace = raw.indexOf('{');
    const closeBrace = raw.lastIndexOf('}');
    const header = openBrace >= 0 ? raw.slice(0, openBrace) : raw;
    const name = header.slice('call:'.length).trim();
    if (!name) {
        return null;
    }

    const argsText = openBrace >= 0 && closeBrace > openBrace ? raw.slice(openBrace + 1, closeBrace) : '';
    return {
        id: nextId('toolcall'),
        name,
        args: parseGemmaArgs(argsText),
    };
}

function parseGemmaArgs(input: string): Record<string, unknown> {
    const out: Record<string, unknown> = {};
    if (!input.trim()) {
        return out;
    }

    const parts = splitTopLevel(input, ',');
    for (const part of parts) {
        const [keyRaw, ...rest] = part.split(':');
        const key = keyRaw?.trim();
        if (!key) {
            continue;
        }
        const rawValue = rest.join(':').trim();
        out[key] = parseGemmaValue(rawValue);
    }

    return out;
}

function parseGemmaValue(raw: string): unknown {
    const normalized = raw.replace(/<\|"\|>/g, '"').trim();
    if (normalized === 'true') return true;
    if (normalized === 'false') return false;
    if (normalized === 'null') return null;
    if (/^-?\d+(\.\d+)?$/.test(normalized)) return Number(normalized);
    if (
        (normalized.startsWith('"') && normalized.endsWith('"')) ||
        (normalized.startsWith("'") && normalized.endsWith("'"))
    ) {
        return normalized.slice(1, -1);
    }
    return normalized;
}

function stripGemmaToolTokens(text: string): string {
    return text.replace(/<\|tool_call>[\s\S]*?<tool_call\|>/g, '').trim();
}

function stripGemmaThoughtTokens(text: string): string {
    const strippedClosed = text.replace(/<\|channel>thought[\s\S]*?<channel\|>/g, '');
    const openIdx = strippedClosed.lastIndexOf('<|channel>thought');
    if (openIdx >= 0) {
        return strippedClosed.slice(0, openIdx).trim();
    }
    return strippedClosed.trim();
}

function extractGemmaThoughtBlocks(content: string): string[] {
    const blocks: string[] = [];
    const matches = [...content.matchAll(/<\|channel>thought\s*([\s\S]*?)<channel\|>/g)];
    for (const match of matches) {
        const block = match[1].trim();
        if (block) {
            blocks.push(block);
        }
    }

    // Streaming partial support: if the thought channel is opened but not closed
    // yet, still surface it as thinking text for progressive UI updates.
    const lastOpenIdx = content.lastIndexOf('<|channel>thought');
    const lastCloseIdx = content.lastIndexOf('<channel|>');
    if (lastOpenIdx >= 0 && lastOpenIdx > lastCloseIdx) {
        const partial = content.slice(lastOpenIdx + '<|channel>thought'.length).trim();
        if (partial) {
            blocks.push(partial);
        }
    }

    return blocks;
}

function joinNonEmpty(a: string, b: string): string {
    if (a && b) return `${a}\n\n${b}`;
    return a || b || '';
}

function sanitizeVisibleText(text: string): string {
    const lines = text.split('\n');
    const kept = lines.filter((line) => {
        const trimmed = line.trim();
        if (!trimmed) return false;
        if (trimmed.startsWith('<|') || trimmed.endsWith('|>')) return false;
        if (trimmed.startsWith('call:')) return false;
        return true;
    });

    return kept.join('\n').trim();
}
