import type { ToolCall } from '../types.ts';
import { ParserStrategyBase } from './ParserStrategyBase';
import type { ParseResult, ParserContext } from './types.ts';
import { asRecord } from './utils';

export class ParserStrategyGranite extends ParserStrategyBase {
    readonly id = 'granite';

    supports(context: ParserContext): boolean {
        return context.modelType === 'granite' || /granite/i.test(context.modelId);
    }

    parseAssistantContent(content: string, nextId: (prefix: string) => string): ParseResult {
        const normalized = content.replace(/<\|end_of_text\|>/g, '');
        const toolCalls = parseGraniteToolCalls(normalized, nextId);
        const base = super.parseAssistantContent(stripGraniteToolCalls(normalized), nextId);

        if (toolCalls.length === 0) {
            return base;
        }

        return {
            ...base,
            toolCalls,
        };
    }
}

function stripGraniteToolCalls(content: string): string {
    const withoutClosedCalls = content.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, '');
    const openCallIndex = withoutClosedCalls.lastIndexOf('<tool_call>');
    if (openCallIndex >= 0) {
        return withoutClosedCalls.slice(0, openCallIndex);
    }

    const toolCallOpen = '<tool_call>';
    for (let i = toolCallOpen.length - 1; i > 0; i--) {
        const partialOpen = toolCallOpen.slice(0, i);
        if (withoutClosedCalls.endsWith(partialOpen)) {
            return withoutClosedCalls.slice(0, -partialOpen.length);
        }
    }

    return withoutClosedCalls;
}

function parseGraniteToolCalls(content: string, nextId: (prefix: string) => string): ToolCall[] {
    const toolCalls: ToolCall[] = [];
    const regex = /<tool_call>([\s\S]*?)<\/tool_call>/g;
    let match: RegExpExecArray | null;

    while ((match = regex.exec(content)) !== null) {
        const raw = match[1].trim();
        if (!raw) {
            continue;
        }

        try {
            const parsed = JSON.parse(raw) as { name?: unknown; arguments?: unknown; id?: unknown };
            if (typeof parsed.name !== 'string') {
                continue;
            }

            toolCalls.push({
                id: typeof parsed.id === 'string' ? parsed.id : nextId('toolcall'),
                name: parsed.name,
                args: asRecord(parsed.arguments),
            });
        } catch {
            continue;
        }
    }

    return toolCalls;
}
