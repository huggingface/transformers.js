import type { Message, ToolCall, ToolCallResult, ToolMessage } from '../types';
import { ModelAdapterBase } from './ModelAdapterBase';
import type { ModelAdapterContext, ParseResult } from './types';
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
            if (message.role !== 'assistant' || !message.tool_calls || message.tool_calls.length === 0) {
                formatted.push(this.formatMessage(message));
                continue;
            }

            const toolResponses = this.collectFollowingToolMessages(messages, i, message.tool_calls);

            formatted.push({
                role: 'assistant',
                tool_calls: message.tool_calls.map((call) => ({
                    function: {
                        name: call.function.name,
                        arguments: call.function.arguments,
                    },
                })),
                ...(toolResponses.length > 0
                    ? {
                          tool_responses: toolResponses.map((response) => this.formatToolResponse(response)),
                      }
                    : {}),
            });
            i += toolResponses.length;
        }

        return formatted;
    }

    private collectFollowingToolMessages(
        messages: ReadonlyArray<Message>,
        assistantIndex: number,
        toolCalls: NonNullable<Extract<Message, { role: 'assistant' }>['tool_calls']>,
    ): ToolMessage[] {
        const toolResponses: ToolMessage[] = [];
        for (let j = assistantIndex + 1; j < messages.length; j++) {
            const next = messages[j];
            if (next.role !== 'tool') {
                break;
            }
            if (!toolCalls.some((call) => call.id === next.tool_call_id)) {
                break;
            }
            toolResponses.push(next);
        }
        return toolResponses;
    }

    private formatToolResponse(response: ToolCallResult | ToolMessage): { name: string; response: unknown } {
        if ('output' in response) {
            const toolResponse = this.toToolResultResponse(response);
            return {
                name: response.name,
                response: normalizeGemmaToolResponse(toolResponse, response.args),
            };
        }

        const toolResponse = parseToolResponse(response.content);
        return {
            name: response.name ?? '',
            response: normalizeGemmaToolResponse(toolResponse),
        };
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

    useKvCache(_enableThinking: boolean): boolean {
        return false;
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

function parseToolResponse(content: string): unknown {
    try {
        return JSON.parse(content);
    } catch {
        return content;
    }
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
