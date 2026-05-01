import type { ToolCall, ToolMap } from '../types.ts';

export interface ParserContext {
    modelId: string;
    modelType?: string;
    chatTemplate?: string;
}

export interface ParseResult {
    thinkingText: string;
    visibleText: string;
    toolCalls: ToolCall[];
}

export interface ParserStrategy {
    readonly id: string;
    supports(context: ParserContext): boolean;
    formatTools(tools: ToolMap): Array<Record<string, unknown>>;
    parseAssistantContent(content: string, nextId: (prefix: string) => string): ParseResult;
}
