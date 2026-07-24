import type { Message } from '../types';
import { ModelAdapterBase } from './ModelAdapterBase';
import type { ModelAdapterContext, ParseResult, ParsedToolCall } from './types';

const IM_END_TOKEN = '<|im_end|>';

export class ModelAdapterQwen3 extends ModelAdapterBase {
    readonly id = 'qwen3';

    private enableThinking = false;

    supports(context: ModelAdapterContext): boolean {
        this.enableThinking = context.enableThinking === true;
        return context.modelType?.startsWith('qwen3_') === true || /qwen3/i.test(context.modelId);
    }

    formatMessages(messages: ReadonlyArray<Message>): Array<Record<string, unknown>> {
        return messages
            .map((message) => {
                const toolCalls = this.getToolCalls(message);
                if (message.role !== 'assistant' || toolCalls.length === 0) {
                    return this.formatMessage(message);
                }

                return [
                    {
                        role: 'assistant',
                        content: this.stringifyTextContent(message.content),
                        tool_calls: toolCalls.map((call) => ({
                            id: call.callID,
                            type: 'function',
                            function: {
                                name: call.name,
                                arguments: call.arguments,
                            },
                        })),
                    },
                ];
            })
            .flat();
    }

    normalizeAssistantContent(content: string): string {
        return content.replaceAll(IM_END_TOKEN, '').trim();
    }

    parseAssistantContent(content: string, nextId: (prefix: string) => string): ParseResult {
        const normalized = this.normalizeAssistantContent(content);
        const { thinkingText, remainingText } = extractThinking(normalized, this.enableThinking);
        const toolCalls = parseQwenToolCalls(remainingText, nextId);
        const visibleText = stripQwenToolCalls(remainingText).trim();

        return {
            thinkingText,
            visibleText,
            toolCalls,
        };
    }
}

function extractThinking(content: string, enableThinking = false): { thinkingText: string; remainingText: string } {
    const closeIndex = content.indexOf('</think>');
    if (closeIndex < 0) {
        if (enableThinking && !content.startsWith('<tool_call>')) {
            return { thinkingText: content.replace(/^<think>\s*/, '').trim(), remainingText: '' };
        }
        return { thinkingText: '', remainingText: content };
    }

    const rawThinking = content
        .slice(0, closeIndex)
        .replace(/^<think>\s*/, '')
        .trim();
    const remainingText = content.slice(closeIndex + '</think>'.length);
    return { thinkingText: rawThinking, remainingText };
}

function parseQwenToolCalls(content: string, nextId: (prefix: string) => string): ParsedToolCall[] {
    const toolCalls: ParsedToolCall[] = [];
    const toolCallRegex = /<tool_call>([\s\S]*?)<\/tool_call>/g;
    let toolCallMatch: RegExpExecArray | null;

    while ((toolCallMatch = toolCallRegex.exec(content)) !== null) {
        const functionMatch = /<function=([^>\s]+)>([\s\S]*?)<\/function>/.exec(toolCallMatch[1]);
        if (!functionMatch) {
            continue;
        }

        const [, name, body] = functionMatch;
        const args: Record<string, unknown> = {};
        const parameterRegex = /<parameter=([^>\s]+)>([\s\S]*?)<\/parameter>/g;
        let parameterMatch: RegExpExecArray | null;

        while ((parameterMatch = parameterRegex.exec(body)) !== null) {
            args[parameterMatch[1]] = parameterMatch[2].trim();
        }

        toolCalls.push({
            id: nextId('toolcall'),
            name,
            args,
        });
    }

    return toolCalls;
}

function stripQwenToolCalls(content: string): string {
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
