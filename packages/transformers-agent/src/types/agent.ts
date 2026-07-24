import type { Model } from '../Model';
import type { ModelAdapter } from '../adapters';
import type { ToolList } from '../Tool';
import type { ToolCall, ToolResponse } from './tools';

export type { ModelAdapter, ModelAdapterContext, ParseResult, ParsedToolCall } from '../adapters/types';

export interface Usage {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
}

export interface PromptResult {
    response: string;
    thinking: string;
    toolCalls: ToolCall[];
    usage: Usage;
}

export interface StreamChunk extends PromptResult {
    done: boolean;
}

export type Prompt = string | Message[];

export type MessageContent = TextContent | ImageContent | AudioContent | ToolCallContent | ToolResponseContent;

export interface TextContent {
    type: 'text';
    value: string;
}

export interface ImageContent {
    type: 'image';
    value: string | Blob | ArrayBuffer | Uint8Array;
}

export interface AudioContent {
    type: 'audio';
    value: string | ArrayBuffer | Uint8Array;
}

export interface ToolCallContent {
    type: 'tool-call';
    value: ToolCall;
}

export interface ToolResponseContent {
    type: 'tool-response';
    value: ToolResponse;
}

export interface Message {
    role: 'system' | 'user' | 'assistant';
    content: string | MessageContent[];
}

export interface AgentConfig {
    model: Model;
    tools?: ToolList;
    maxNewTokens?: number;
    temperature?: number;
    enableThinking?: boolean;
    initialPrompts?: Array<Message>;
    adapter?: ModelAdapter;
}
