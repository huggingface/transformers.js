import type {
    AfterToolCallHook,
    BeforeToolCallHook,
    Message,
    ToolCall,
    ToolCallOutput,
    ToolCallResult,
} from '../types';
import type { ToolList } from '../Tool';

export class ToolExecutor {
    private readonly beforeHooks: BeforeToolCallHook[] = [];
    private readonly afterHooks: AfterToolCallHook[] = [];
    private readonly toolsByName: Map<string, ToolList[number]>;

    constructor(tools: ToolList) {
        this.toolsByName = new Map(tools.map((tool) => [tool.name, tool]));
    }

    onBeforeToolCall(hook: BeforeToolCallHook): void {
        this.beforeHooks.push(hook);
    }

    onAfterToolCall(hook: AfterToolCallHook): void {
        this.afterHooks.push(hook);
    }

    async execute(call: ToolCall): Promise<ToolCallResult> {
        const tool = this.toolsByName.get(call.name);
        const start = performance.now();

        await this.emitBeforeHooks(call);

        let output: ToolCallOutput;
        if (!tool) {
            output = {
                isError: true,
                content: [{ type: 'text', text: `Unknown tool: ${call.name}` }],
            };
        } else {
            try {
                output = await tool.execute(call.args, undefined);
            } catch (error) {
                output = {
                    isError: true,
                    content: [{ type: 'text', text: this.errorToString(error) }],
                };
            }
        }

        const durationMs = performance.now() - start;
        await this.emitAfterHooks(call, output, durationMs);

        return {
            ...call,
            output,
            durationMs,
        };
    }

    createToolMessage(result: ToolCallResult): Message {
        const content = this.toToolMessageContent(result.output.content);
        return {
            role: 'tool',
            toolCallId: result.id,
            name: result.name,
            content,
        };
    }

    private toToolMessageContent(content: ToolCallOutput['content']): string {
        if (content.length === 1 && content[0].type === 'text') {
            return content[0].text;
        }
        return JSON.stringify(content);
    }

    private async emitBeforeHooks(call: ToolCall): Promise<void> {
        for (const hook of this.beforeHooks) {
            await hook(call);
        }
    }

    private async emitAfterHooks(call: ToolCall, output: ToolCallOutput, durationMs: number): Promise<void> {
        for (const hook of this.afterHooks) {
            await hook(call, output, durationMs);
        }
    }

    private errorToString(error: unknown): string {
        if (error instanceof Error) {
            return error.message;
        }
        return String(error);
    }
}
