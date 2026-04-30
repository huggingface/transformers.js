import type { ToolCall, ToolMap } from '../interfaces.d.ts';
import type { ParseResult, ParserContext, ParserStrategy } from './types.ts';
import { asRecord } from './utils';

export class BaseParserStrategy implements ParserStrategy {
    readonly id: string = 'base';

    supports(_context: ParserContext): boolean {
        return true;
    }

    formatTools(tools: ToolMap): Array<Record<string, unknown>> {
        return Object.entries(tools).map(([name, tool]) => ({
            type: 'function',
            function: {
                name,
                description: tool.description,
                parameters: tool.inputSchema,
            },
        }));
    }

    parseAssistantContent(content: string, nextId: (prefix: string) => string): ParseResult {
        const thinkingMatches = [...content.matchAll(/<think>([\s\S]*?)<\/think>/g)];
        const thinkingText = thinkingMatches
            .map((match) => match[1].trim())
            .filter(Boolean)
            .join('\n\n');

        const withoutThinking = content.replace(/<think>[\s\S]*?<\/think>/g, '');
        const toolCalls = this.parseToolCallsFromTaggedJson(withoutThinking, nextId);
        const visibleText = withoutThinking.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, '').trim();

        return { thinkingText, visibleText, toolCalls };
    }

    private parseToolCallsFromTaggedJson(text: string, nextId: (prefix: string) => string): ToolCall[] {
        const toolCalls: ToolCall[] = [];
        const regex = /<tool_call>([\s\S]*?)<\/tool_call>/g;
        let match: RegExpExecArray | null;

        while ((match = regex.exec(text)) !== null) {
            const raw = match[1].trim();
            if (!raw) {
                continue;
            }
            try {
                const parsed = JSON.parse(raw) as { name?: unknown; args?: unknown; id?: unknown };
                if (typeof parsed.name !== 'string') {
                    continue;
                }
                toolCalls.push({
                    id: typeof parsed.id === 'string' ? parsed.id : nextId('toolcall'),
                    name: parsed.name,
                    args: asRecord(parsed.args),
                });
            } catch {
                continue;
            }
        }

        return toolCalls;
    }
}
