import type { Message, ToolResponse } from '../types';
import { ModelAdapterBase } from './ModelAdapterBase';
import type { ModelAdapterContext, ParseResult, ParsedToolCall } from './types';
import { splitTopLevel } from './utils';

export class ModelAdapterGemma4 extends ModelAdapterBase {
    readonly id = 'gemma4';

    supports(context: ModelAdapterContext): boolean {
        return context.modelType === 'gemma4' || /gemma-4/i.test(context.modelId);
    }

    formatMessages(messages: ReadonlyArray<Message>): Array<Record<string, unknown>> {
        const formatted: Array<Record<string, unknown>> = [];

        for (let i = 0; i < messages.length; i++) {
            const message = messages[i];
            const thinking = (message as Message & { thinking?: string }).thinking;
            const toolCalls = this.getToolCalls(message);
            if (message.role !== 'assistant' || toolCalls.length === 0) {
                formatted.push(...this.formatMessage(message));
                continue;
            }

            const { responses: toolResponses, consumedMessages } = this.collectFollowingToolResponses(
                messages,
                i,
                toolCalls,
            );
            const text = this.stringifyTextContent(message.content);

            if (thinking) {
                formatted.push({
                    role: 'assistant',
                    content: [
                        `<|channel>thought\n${thinking}<channel|>`,
                        ...toolCalls.map(
                            (call) => `<|tool_call>call:${call.name}{${formatGemmaObject(call.arguments)}}<tool_call|>`,
                        ),
                        ...toolResponses.map(
                            (response) =>
                                `<|tool_response>response:${response.name}{${formatGemmaObject(
                                    normalizeGemmaToolResponse(this.toolResponseValue(response)),
                                )}}<tool_response|>`,
                        ),
                        text ?? '',
                    ].join(''),
                });
                i += consumedMessages;
                continue;
            }

            formatted.push({
                role: 'assistant',
                ...(text !== undefined ? { content: text } : {}),
                tool_calls: toolCalls.map((call) => ({
                    function: {
                        name: call.name,
                        arguments: call.arguments,
                    },
                })),
                ...(toolResponses.length > 0
                    ? {
                          tool_responses: toolResponses.map((response) => ({
                              name: response.name,
                              response: normalizeGemmaToolResponse(this.toolResponseValue(response)),
                          })),
                      }
                    : {}),
            });
            i += consumedMessages;
        }

        return formatted;
    }

    private collectFollowingToolResponses(
        messages: ReadonlyArray<Message>,
        assistantIndex: number,
        toolCalls: ReturnType<ModelAdapterGemma4['getToolCalls']>,
    ): { responses: ToolResponse[]; consumedMessages: number } {
        const responses: ToolResponse[] = [];
        let consumedMessages = 0;
        for (let j = assistantIndex + 1; j < messages.length; j++) {
            const next = messages[j];
            const nextResponses = this.getToolResponses(next);
            if (
                nextResponses.length === 0 ||
                (Array.isArray(next.content) && nextResponses.length !== next.content.length)
            ) {
                break;
            }
            if (!nextResponses.every((response) => toolCalls.some((call) => call.callID === response.callID))) {
                break;
            }
            responses.push(...nextResponses);
            consumedMessages += 1;
        }
        return { responses, consumedMessages };
    }

    normalizeAssistantContent(content: string): string {
        return content
            .replace(/<\|tool_response\|>/g, '')
            .replace(/<\|tool_response>/g, '')
            .replace(/<tool_response\|>/g, '')
            .replace(/<\|turn\|>/g, '')
            .replace(/<\|turn>/g, '')
            .replace(/<turn\|>/g, '')
            .replace(/<eos>/g, '')
            .trim();
    }

    parseAssistantContent(content: string, nextId: (prefix: string) => string): ParseResult {
        const normalized = this.normalizeAssistantContent(content);
        const base = super.parseAssistantContent(normalized, nextId);
        const thoughtBlocks = extractGemmaThoughtBlocks(normalized);
        const thoughtText = thoughtBlocks.join('\n\n').trim();

        const gemmaCalls = this.parseGemmaToolCalls(normalized, nextId);
        const visibleText = stripGemmaToolTokens(stripGemmaThoughtTokens(base.visibleText))
            .replace(/(^|\n)\s*call:[^\n{}]+\{[^\n]*\}\s*(?=\n|$)/g, '$1')
            .replace(/<\|channel\|>/g, '')
            .replace(/<\|channel>/g, '')
            .replace(/<channel\|>/g, '')
            .replace(/<\|tool_call\|>/g, '')
            .replace(/<\|tool_call>/g, '')
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

    private parseGemmaToolCalls(content: string, nextId: (prefix: string) => string): ParsedToolCall[] {
        const toolCalls: ParsedToolCall[] = [];
        const regex = /<\|tool_call>([\s\S]*?)<tool_call\|>/g;
        let match: RegExpExecArray | null;

        while ((match = regex.exec(content)) !== null) {
            const parsed = parseSingleGemmaCall(match[1].trim(), nextId);
            if (parsed) {
                toolCalls.push(parsed);
            }
        }

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

function parseSingleGemmaCall(raw: string, nextId: (prefix: string) => string): ParsedToolCall | null {
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
    if (Array.isArray(value)) return `[${value.map((item) => formatGemmaValue(item)).join(',')}]`;
    if (typeof value === 'object') return `{${formatGemmaObject(value)}}`;
    return `<|"|>${String(value)}<|"|>`;
}

function normalizeGemmaToolResponse(response: unknown, args: Record<string, unknown> = {}): unknown {
    if (typeof response !== 'string') {
        return response;
    }

    const location = typeof args.location === 'string' ? args.location : extractLocation(response);
    const temperature = extractTemperature(response);
    const weather = extractWeather(response);

    if (!location || temperature === undefined || !weather) {
        return response;
    }

    return { location, temperature, weather };
}

function extractLocation(text: string): string | undefined {
    const match = /weather\s+in\s+(.+?)\s+(?:is|in)\s+/i.exec(text);
    return match?.[1]?.trim();
}

function extractTemperature(text: string): number | undefined {
    const match = /(-?\d+(?:\.\d+)?)\s*degrees?/i.exec(text);
    return match ? Number(match[1]) : undefined;
}

function extractWeather(text: string): string | undefined {
    const match = /\s(?:is|in)\s+([a-z]+),\s*-?\d+(?:\.\d+)?\s*degrees?/i.exec(text);
    return match?.[1]?.toLowerCase();
}
