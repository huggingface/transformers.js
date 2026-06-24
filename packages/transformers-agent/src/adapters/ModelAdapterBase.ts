import type { ToolList } from '../Tool';
import type { Message, ToolCall, ToolCallResult } from '../types';
import type { ModelAdapter, ModelAdapterContext, ParseResult } from './types';
import { asRecord } from './utils';

export class ModelAdapterBase implements ModelAdapter {
    readonly id: string = 'base';

    supports(_context: ModelAdapterContext): boolean {
        return true;
    }

    formatTools(tools: ToolList): Array<Record<string, unknown>> {
        return tools.map((tool) => ({
            type: 'function',
            function: {
                name: tool.name,
                description: tool.description,
                parameters: tool.inputSchema,
            },
        }));
    }

    formatMessages(messages: ReadonlyArray<Message>): Array<Record<string, unknown>> {
        return messages.map((message) => this.formatMessage(message));
    }

    preparePromptForGeneration(prompt: string): string {
        return prompt;
    }

    normalizeAssistantContent(content: string): string {
        return content.replace(/<eos>/g, '').trim();
    }

    parseAssistantContent(content: string, nextId: (prefix: string) => string): ParseResult {
        const normalized = this.normalizeAssistantContent(content);
        const thinkingMatches = [...normalized.matchAll(/<think>([\s\S]*?)<\/think>/g)];
        const thinkingText = thinkingMatches
            .map((match) => match[1].trim())
            .filter(Boolean)
            .join('\n\n');

        const withoutThinking = normalized.replace(/<think>[\s\S]*?<\/think>/g, '');
        const toolCalls = this.parseToolCallsFromTaggedJson(withoutThinking, nextId);
        const visibleText = withoutThinking.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, '').trim();

        return { thinkingText, visibleText, toolCalls };
    }

    useKvCache(enableThinking: boolean): boolean {
        return !enableThinking;
    }

    protected formatMessage(message: Message): Record<string, unknown> {
        if (message.role === 'assistant') {
            const toolCalls = message.tool_calls?.map((call) => ({
                id: call.id,
                type: 'function',
                function: {
                    name: call.function.name,
                    arguments: JSON.stringify(call.function.arguments),
                },
            }));

            return {
                role: 'assistant',
                content: message.content,
                ...(toolCalls ? { tool_calls: toolCalls } : {}),
            };
        }

        if (message.role === 'tool') {
            return {
                role: 'tool',
                content: message.content,
                tool_call_id: message.tool_call_id,
                name: message.name,
            };
        }

        return {
            role: message.role,
            content: message.content,
        };
    }

    protected toToolResultResponse(result: ToolCallResult): unknown {
        const content = result.output.content;
        if (content.length === 1 && content[0].type === 'structured') {
            return content[0].data;
        }
        if (content.length === 1 && content[0].type === 'text') {
            return content[0].text;
        }
        return content;
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
