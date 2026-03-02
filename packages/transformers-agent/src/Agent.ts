import type {
    AgentConfig,
    BeforeToolCallHook,
    AfterToolCallHook,
    OnStepHook,
    ToolMap,
    Message,
    RunResult,
    StreamChunk,
} from './interfaces.d.ts';
import type { Model } from './Model.ts';

export class Agent {
    readonly system: string;
    readonly tools: ToolMap;
    readonly history: ReadonlyArray<Message> = [];

    constructor(config: AgentConfig) {
        this.system = config.system;
        this.tools = config.tools ?? {};
    }

    async run(input: string): Promise<RunResult> {
        throw new Error('Not implemented');
    }

    async *stream(input: string): AsyncIterable<StreamChunk> {
        throw new Error('Not implemented');
    }

    clearHistory(): void {
        throw new Error('Not implemented');
    }

    onBeforeToolCall(hook: BeforeToolCallHook): this {
        throw new Error('Not implemented');
    }

    onAfterToolCall(hook: AfterToolCallHook): this {
        throw new Error('Not implemented');
    }

    onStep(hook: OnStepHook): this {
        throw new Error('Not implemented');
    }
}
