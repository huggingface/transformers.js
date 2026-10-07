import type { Model } from '../Model';
import type { ToolList } from '../Tool';
import type { ToolCall, ToolResponse } from './tools';

export type Prompt = string | Message[];

export type MessageContent = TextContent | ImageContent | AudioContent | ToolCallContent | ToolResponseContent;

export type LanguageModelMessageContent = MessageContent | ThinkingContent;

export interface Usage {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
    tokensPerSecond: number;
    timeToFirstTokenMs: number;
    generationTimeMs: number;
    totalTimeMs: number;
}

export interface TextContent {
    type: 'text';
    value: string;
}

export interface ThinkingContent {
    type: 'thinking';
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
}
