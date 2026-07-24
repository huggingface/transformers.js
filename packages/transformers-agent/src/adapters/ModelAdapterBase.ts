import type { ToolList } from '../Tool';
import type { Message, MessageContent, ToolResponse } from '../types';
import type { ModelAdapter, ModelAdapterContext, ParseResult, ParsedToolCall } from './types';
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
        return messages.flatMap((message) => this.formatMessage(message));
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
        let withoutThinking = normalized.replace(/<think>[\s\S]*?<\/think>/g, '');
        let partialThinking = '';
        const openThinkingIndex = withoutThinking.lastIndexOf('<think>');
        if (openThinkingIndex >= 0) {
            partialThinking = withoutThinking.slice(openThinkingIndex + '<think>'.length).trim();
            withoutThinking = withoutThinking.slice(0, openThinkingIndex);
        }
        const toolCalls = this.parseToolCallsFromTaggedJson(withoutThinking, nextId);
        const visibleText = stripToolCalls(withoutThinking).trim();
        return {
            thinkingText: [thinkingText, partialThinking].filter(Boolean).join('\n\n'),
            visibleText,
            toolCalls,
        };
    }

    protected formatMessage(message: Message): Array<Record<string, unknown>> {
        const toolCalls = this.getToolCalls(message);
        const toolResponses = this.getToolResponses(message);
        const text = this.stringifyTextContent(message.content);
        const formatted: Array<Record<string, unknown>> = [];

        if (message.role === 'assistant' && (text !== undefined || toolCalls.length > 0)) {
            formatted.push({
                role: 'assistant',
                content: text,
                ...(toolCalls.length > 0
                    ? {
                          tool_calls: toolCalls.map((call) => ({
                              id: call.callID,
                              type: 'function',
                              function: { name: call.name, arguments: JSON.stringify(call.arguments) },
                          })),
                      }
                    : {}),
            });
        } else if (text !== undefined) {
            formatted.push({ role: message.role, content: text });
        }

        for (const response of toolResponses) {
            formatted.push({
                role: 'tool',
                content: this.stringifyToolResponse(response),
                tool_call_id: response.callID,
                name: response.name,
            });
        }
        return formatted;
    }

    protected getToolCalls(message: Message) {
        return typeof message.content === 'string'
            ? []
            : message.content.filter((part) => part.type === 'tool-call').map((part) => part.value);
    }

    protected getToolResponses(message: Message): ToolResponse[] {
        return typeof message.content === 'string'
            ? []
            : message.content.filter((part) => part.type === 'tool-response').map((part) => part.value);
    }

    protected stringifyTextContent(content: Message['content']): string | undefined {
        if (typeof content === 'string') return content;
        const text: string[] = [];
        for (const part of content) {
            if (part.type === 'text') {
                text.push(part.value);
            } else if (part.type === 'image' || part.type === 'audio') {
                throw new Error('Multimodal message content is not supported by the current model adapter yet.');
            }
        }
        return text.length > 0 ? text.join('') : undefined;
    }

    protected stringifyToolResponse(response: ToolResponse): string {
        if ('errorMessage' in response) return response.errorMessage;
        this.assertSupportedToolResult(response);
        if (response.result.length === 1) {
            const item = response.result[0];
            if (item.type === 'text') return item.value;
            if (item.type === 'object') return JSON.stringify(item.value);
        }
        return JSON.stringify(response.result);
    }

    protected toolResponseValue(response: ToolResponse): unknown {
        if ('errorMessage' in response) return { error: response.errorMessage };
        this.assertSupportedToolResult(response);
        if (response.result.length === 1) return response.result[0].value;
        return response.result.map((item) => item.value);
    }

    private assertSupportedToolResult(response: Extract<ToolResponse, { result: unknown }>): void {
        for (const item of response.result) {
            if (item.type === 'image' || item.type === 'audio') {
                throw new Error('Multimodal tool responses are not supported by the current model adapter yet.');
            }
            if (item.type === 'object' && JSON.stringify(item.value) === undefined) {
                throw new Error('Object tool responses must be JSON-serializable.');
            }
        }
    }

    private parseToolCallsFromTaggedJson(text: string, nextId: (prefix: string) => string): ParsedToolCall[] {
        const toolCalls: ParsedToolCall[] = [];
        const regex = /<tool_call>([\s\S]*?)<\/tool_call>/g;
        let match: RegExpExecArray | null;
        while ((match = regex.exec(text)) !== null) {
            const raw = match[1].trim();
            if (!raw) continue;
            try {
                const parsed = JSON.parse(raw) as { name?: unknown; args?: unknown; id?: unknown };
                if (typeof parsed.name !== 'string') continue;
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

function stripToolCalls(content: string): string {
    const withoutClosedCalls = content.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, '');
    const openCallIndex = withoutClosedCalls.lastIndexOf('<tool_call>');
    if (openCallIndex >= 0) return withoutClosedCalls.slice(0, openCallIndex);
    const marker = '<tool_call>';
    for (let length = marker.length - 1; length > 0; length--) {
        const partial = marker.slice(0, length);
        if (withoutClosedCalls.endsWith(partial)) return withoutClosedCalls.slice(0, -partial.length);
    }
    return withoutClosedCalls;
}
