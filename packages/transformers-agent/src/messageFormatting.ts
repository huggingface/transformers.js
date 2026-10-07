import type { ToolList } from './Tool';
import type { Message, ToolResponse } from './types';

type ModelFamily = 'default' | 'gemma4' | 'qwen3';

export function getModelFamily(modelId: string, modelType?: string): ModelFamily {
    if (modelType === 'gemma4' || /gemma-4/i.test(modelId)) return 'gemma4';
    if (modelType?.startsWith('qwen3_') || /qwen3/i.test(modelId)) return 'qwen3';
    return 'default';
}

export function formatTools(tools: ToolList): Array<Record<string, unknown>> {
    return tools.map((tool) => ({
        type: 'function',
        function: {
            name: tool.name,
            description: tool.description,
            parameters: tool.inputSchema,
        },
    }));
}

export function formatMessages(messages: ReadonlyArray<Message>, family: ModelFamily): Array<Record<string, unknown>> {
    if (family === 'gemma4') return formatGemmaMessages(messages);
    return messages.flatMap((message) => formatMessage(message, family === 'qwen3'));
}

function formatMessage(message: Message, qwen: boolean): Array<Record<string, unknown>> {
    const toolCalls = getToolCalls(message);
    const text = stringifyTextContent(message.content);
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
                          function: {
                              name: call.name,
                              arguments: qwen ? call.arguments : JSON.stringify(call.arguments),
                          },
                      })),
                  }
                : {}),
        });
    } else if (text !== undefined) {
        formatted.push({ role: message.role, content: text });
    }

    for (const response of getToolResponses(message)) {
        formatted.push({
            role: 'tool',
            content: stringifyToolResponse(response),
            tool_call_id: response.callID,
            name: response.name,
        });
    }
    return formatted;
}

function formatGemmaMessages(messages: ReadonlyArray<Message>): Array<Record<string, unknown>> {
    const formatted: Array<Record<string, unknown>> = [];

    for (let i = 0; i < messages.length; i++) {
        const message = messages[i];
        const thinking = (message as Message & { thinking?: string }).thinking;
        const toolCalls = getToolCalls(message);
        if (message.role !== 'assistant' || toolCalls.length === 0) {
            formatted.push(...formatMessage(message, false));
            continue;
        }

        const { responses, consumedMessages } = collectFollowingToolResponses(messages, i, toolCalls);
        const text = stringifyTextContent(message.content);
        if (thinking) {
            formatted.push({
                role: 'assistant',
                content: [
                    `<|channel>thought\n${thinking}<channel|>`,
                    ...toolCalls.map(
                        (call) => `<|tool_call>call:${call.name}{${formatGemmaObject(call.arguments)}}<tool_call|>`,
                    ),
                    ...responses.map(
                        (response) =>
                            `<|tool_response>response:${response.name}{${formatGemmaObject(
                                toolResponseValue(response),
                            )}}<tool_response|>`,
                    ),
                    text ?? '',
                ].join(''),
            });
        } else {
            formatted.push({
                role: 'assistant',
                ...(text !== undefined ? { content: text } : {}),
                tool_calls: toolCalls.map((call) => ({
                    function: { name: call.name, arguments: call.arguments },
                })),
                ...(responses.length > 0
                    ? {
                          tool_responses: responses.map((response) => ({
                              name: response.name,
                              response: toolResponseValue(response),
                          })),
                      }
                    : {}),
            });
        }
        i += consumedMessages;
    }
    return formatted;
}

function collectFollowingToolResponses(
    messages: ReadonlyArray<Message>,
    assistantIndex: number,
    toolCalls: ReturnType<typeof getToolCalls>,
): { responses: ToolResponse[]; consumedMessages: number } {
    const responses: ToolResponse[] = [];
    let consumedMessages = 0;
    for (let i = assistantIndex + 1; i < messages.length; i++) {
        const message = messages[i];
        const next = getToolResponses(message);
        if (next.length === 0 || (Array.isArray(message.content) && next.length !== message.content.length)) break;
        if (!next.every((response) => toolCalls.some((call) => call.callID === response.callID))) break;
        responses.push(...next);
        consumedMessages++;
    }
    return { responses, consumedMessages };
}

function getToolCalls(message: Message) {
    return typeof message.content === 'string'
        ? []
        : message.content.filter((part) => part.type === 'tool-call').map((part) => part.value);
}

function getToolResponses(message: Message): ToolResponse[] {
    return typeof message.content === 'string'
        ? []
        : message.content.filter((part) => part.type === 'tool-response').map((part) => part.value);
}

function stringifyTextContent(content: Message['content']): string | undefined {
    if (typeof content === 'string') return content;
    const text: string[] = [];
    for (const part of content) {
        if (part.type === 'text') text.push(part.value);
        if (part.type === 'image' || part.type === 'audio') {
            throw new Error('Multimodal message content is not supported yet.');
        }
    }
    return text.length > 0 ? text.join('') : undefined;
}

function stringifyToolResponse(response: ToolResponse): string {
    if ('errorMessage' in response) return response.errorMessage;
    assertSupportedToolResult(response);
    if (response.result.length === 1) {
        const item = response.result[0];
        if (item.type === 'text') return item.value;
        if (item.type === 'object') return JSON.stringify(item.value);
    }
    return JSON.stringify(response.result);
}

function toolResponseValue(response: ToolResponse): unknown {
    if ('errorMessage' in response) return { error: response.errorMessage };
    assertSupportedToolResult(response);
    if (response.result.length === 1) return response.result[0].value;
    return response.result.map((item) => item.value);
}

function assertSupportedToolResult(response: Extract<ToolResponse, { result: unknown }>): void {
    for (const item of response.result) {
        if (item.type === 'image' || item.type === 'audio') {
            throw new Error('Multimodal tool responses are not supported yet.');
        }
        if (item.type === 'object' && JSON.stringify(item.value) === undefined) {
            throw new Error('Object tool responses must be JSON-serializable.');
        }
    }
}

function formatGemmaObject(value: unknown): string {
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
        return `value:${formatGemmaValue(value)}`;
    }
    return Object.entries(value as Record<string, unknown>)
        .map(([key, entry]) => `${key}:${formatGemmaValue(entry)}`)
        .join(',');
}

function formatGemmaValue(value: unknown): string {
    if (typeof value === 'string') return `<|"|>${value}<|"|>`;
    if (typeof value === 'number' || typeof value === 'boolean') return String(value);
    if (value === null) return 'null';
    if (Array.isArray(value)) return `[${value.map(formatGemmaValue).join(',')}]`;
    if (typeof value === 'object') return `{${formatGemmaObject(value)}}`;
    return `<|"|>${String(value)}<|"|>`;
}
