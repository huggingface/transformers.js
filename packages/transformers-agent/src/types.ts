import type { DataType, DeviceType } from '@huggingface/transformers';
import type { Model } from './Model.ts';
import type { ToolList } from './Tool.ts';

export interface JSONSchemaObject {
    type: 'object';
    properties?: Record<string, JSONSchemaProperty>;
    required?: string[];
    additionalProperties?: boolean;
    description?: string;
}

export type JSONSchemaProperty =
    | { type: 'string'; description?: string; enum?: string[]; pattern?: string; default?: string }
    | { type: 'number'; description?: string; minimum?: number; maximum?: number; default?: number }
    | { type: 'integer'; description?: string; minimum?: number; maximum?: number; default?: number }
    | { type: 'boolean'; description?: string; default?: boolean }
    | { type: 'array'; description?: string; items?: JSONSchemaProperty }
    | { type: 'object'; description?: string; properties?: Record<string, JSONSchemaProperty>; required?: string[] };

export interface TextContent {
    type: 'text';
    text: string;
}

export interface ImageContent {
    type: 'image';
    data: string;
    mimeType: 'image/png' | 'image/jpeg' | 'image/gif' | 'image/webp';
}

export interface StructuredContent {
    type: 'structured';
    data: Record<string, unknown>;
}

export type ContentBlock = TextContent | ImageContent | StructuredContent;

export interface ToolCallOutput {
    content: ContentBlock[];
    isError?: boolean;
}

export interface ParserContext {
    modelId: string;
    modelType?: string;
    chatTemplate?: string;
}

export interface ToolCall {
    name: string;
    args: Record<string, unknown>;
    id: string;
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

export interface ToolCallResult extends ToolCall {
    output: ToolCallOutput;
    durationMs: number;
}

export interface Usage {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
}

export interface RunResult {
    thinkingText: string;
    text: string;
    tools: ToolCallResult[];
    usage: Usage;
}

export interface RequestResult {
    done: boolean;
    runs: RunResult[];
    usage: Usage;
}

export type StreamChunk = RequestResult;

export interface SystemMessage {
    role: 'system';
    content: string;
}

export interface UserMessage {
    role: 'user';
    content: string;
}

export interface AssistantMessage {
    role: 'assistant';
    content: string;
    toolCalls?: ToolCall[];
}

export interface ToolMessage {
    role: 'tool';
    toolCallId: string;
    name: string;
    content: string;
}

export type Message = SystemMessage | UserMessage | AssistantMessage | ToolMessage;

export type BeforeToolCallHook = (call: ToolCall) => void | Promise<void>;
export type AfterToolCallHook = (call: ToolCall, output: ToolCallOutput, durationMs: number) => void | Promise<void>;
export type OnStepHook = (step: RunResult) => void | Promise<void>;
export type Unsubscribe = () => void;

export interface ModelConfig {
    modelId: string;
    device?: DeviceType;
    dtype?: DataType | Record<string, DataType>;
}

export interface AgentConfig {
    model: Model;
    system: string;
    tools?: ToolList;
    maxSteps?: number;
    maxNewTokens?: number;
    temperature?: number;
    parser?: ParserStrategy;
    enableThinking?: boolean;
}
