import type { ToolList } from '../Tool.ts';
import type { ToolCall } from '../types.ts';

export interface ParserContext {
    modelId: string;
    modelType?: string;
    chatTemplate?: string;
    enableThinking?: boolean;
}

export interface ParseResult {
    thinkingText: string;
    visibleText: string;
    toolCalls: ToolCall[];
}

export interface ParserStrategy {
    readonly id: string;
    supports(context: ParserContext): boolean;
    formatTools(tools: ToolList): Array<Record<string, unknown>>;
    parseAssistantContent(content: string, nextId: (prefix: string) => string): ParseResult;
}
