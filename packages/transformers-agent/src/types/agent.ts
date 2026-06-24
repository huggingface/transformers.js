import type { Model } from '../Model';
import type { ModelAdapter } from '../adapters/types';
import type { ToolList } from '../Tool';
import type { ToolCall, ToolCallOutput, ToolCallResult } from './tools';

export type { ModelAdapter, ModelAdapterContext, ParseResult } from '../adapters/types';

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
    content?: string;
    tool_calls?: Array<{
        id: string;
        type?: 'function';
        function: {
            name: string;
            arguments: Record<string, unknown>;
        };
    }>;
}

export interface ToolMessage {
    role: 'tool';
    tool_call_id: string;
    name?: string;
    content: string;
}

export type Message = SystemMessage | UserMessage | AssistantMessage | ToolMessage;

export type BeforeToolCallHook = (call: ToolCall) => void | Promise<void>;
export type AfterToolCallHook = (call: ToolCall, output: ToolCallOutput, durationMs: number) => void | Promise<void>;
export type OnStepHook = (step: RunResult) => void | Promise<void>;
export type Unsubscribe = () => void;

export interface AgentConfig {
    model: Model;
    tools?: ToolList;
    maxSteps?: number;
    maxNewTokens?: number;
    temperature?: number;
    enableThinking?: boolean;
    initialPrompts?: Array<Message>;
    adapter?: ModelAdapter;
}
