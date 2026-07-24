import type { ToolList } from '../Tool';
import type { Message } from '../types';

export interface ModelAdapterContext {
    modelId: string;
    modelType?: string;
    chatTemplate?: string;
    enableThinking?: boolean;
}

export interface ParseResult {
    thinkingText: string;
    visibleText: string;
    toolCalls: ParsedToolCall[];
}

export interface ParsedToolCall {
    id: string;
    name: string;
    args: Record<string, unknown>;
}

export interface ModelAdapter {
    readonly id: string;
    supports(context: ModelAdapterContext): boolean;
    formatTools(tools: ToolList): Array<Record<string, unknown>>;
    formatMessages(messages: ReadonlyArray<Message>): Array<Record<string, unknown>>;
    preparePromptForGeneration(prompt: string): string;
    normalizeAssistantContent(content: string): string;
    parseAssistantContent(content: string, nextId: (prefix: string) => string): ParseResult;
}
