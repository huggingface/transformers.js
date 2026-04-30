import type {
    AgentConfig,
    BeforeToolCallHook,
    AfterToolCallHook,
    OnStepHook,
    ToolMap,
    Message,
    RunItem,
    RunResult,
    StepResult,
    ToolCall,
    ToolCallOutput,
    ToolCallResult,
    StepUsage,
    TotalUsage,
    StreamChunk,
} from './interfaces.d.ts';
import { DynamicCache, TextStreamer } from '@huggingface/transformers';
import { BaseParserStrategy, ParserRegistry } from './parsers';

export class Agent {
    readonly model: AgentConfig['model'];
    readonly system: string;
    readonly tools: ToolMap;
    readonly maxSteps: number;
    readonly maxNewTokens: number;
    readonly temperature: number | undefined;
    readonly enableThinking: boolean;
    readonly parser: NonNullable<AgentConfig['parser']>;

    private readonly beforeToolCallHooks: BeforeToolCallHook[] = [];
    private readonly afterToolCallHooks: AfterToolCallHook[] = [];
    private readonly stepHooks: OnStepHook[] = [];
    private _history: Message[] = [];
    private itemIdCounter = 0;
    private readonly parserRegistry = new ParserRegistry();

    get history(): ReadonlyArray<Message> {
        return this._history;
    }

    constructor(config: AgentConfig) {
        this.model = config.model;
        this.system = config.system;
        this.tools = config.tools ?? {};
        this.maxSteps = config.maxSteps ?? 10;
        this.maxNewTokens = config.maxNewTokens ?? 1024;
        this.temperature = config.temperature;
        this.enableThinking = config.enableThinking ?? false;
        this.parser = config.parser ?? this.resolveParser();
    }

    async run(input: string): Promise<RunResult> {
        let last: RunResult | undefined;
        for await (const chunk of this.stream(input)) {
            last = chunk;
        }

        if (last) {
            return last;
        }

        return this.createRunSnapshot({
            done: true,
            stepIndex: -1,
            items: [],
            thinkingText: '',
            text: '',
            steps: [],
            usage: this.emptyUsage(0),
        });
    }

    async *stream(input: string): AsyncIterable<StreamChunk> {
        if (!this.model.isInitialized) {
            throw new Error('Model is not initialized. Call model.init() before creating an Agent run.');
        }

        this._history.push({ role: 'user', content: input });
        const conversation = this.toModelMessages(this._history);
        const pastKeyValues = new DynamicCache();

        const items: RunItem[] = [];
        const steps: StepResult[] = [];
        let fullText = '';
        let fullThinkingText = '';
        let done = false;
        let totalPromptTokens = 0;
        let totalCompletionTokens = 0;

        for (let stepIndex = 0; stepIndex < this.maxSteps; stepIndex++) {
            const stepItems: RunItem[] = [];
            const stepToolCalls: ToolCallResult[] = [];
            const stepToolMessages: Message[] = [];
            let stepText = '';
            let stepThinkingText = '';
            const promptTokens = this.estimatePromptTokens(conversation);

            let previewText = '';
            const previewItemId = this.nextItemId('text_preview');
            const deltaQueue: string[] = [];

            let generationDone = false;
            let generationError: unknown;
            const assistantPromise = this.generateAssistantMessage(conversation, pastKeyValues, {
                streaming: true,
                onDelta: (delta) => {
                    deltaQueue.push(delta);
                },
            })
                .then((assistant) => {
                    generationDone = true;
                    return assistant;
                })
                .catch((error) => {
                    generationDone = true;
                    generationError = error;
                    return { role: 'assistant' as const, content: '', completionTokens: 0 };
                });

            while (!generationDone || deltaQueue.length > 0) {
                while (deltaQueue.length > 0) {
                    const delta = deltaQueue.shift();
                    if (!delta) {
                        continue;
                    }
                    previewText += delta;

                    const previewParsed = this.parser.parseAssistantContent(previewText, (prefix) => this.nextItemId(prefix));
                    const previewItems: RunItem[] = [];

                    if (previewParsed.thinkingText.length > 0) {
                        previewItems.push({
                            type: 'thinking',
                            id: `${previewItemId}_thinking`,
                            stepIndex,
                            text: previewParsed.thinkingText,
                        });
                    }

                    if (previewParsed.visibleText.length > 0) {
                        previewItems.push({
                            type: 'text',
                            id: previewItemId,
                            stepIndex,
                            text: previewParsed.visibleText,
                        });
                    }

                    if (previewItems.length === 0) {
                        continue;
                    }

                    const previewStep: StepResult = {
                        stepIndex,
                        items: [...stepItems, ...previewItems],
                        thinkingText: stepThinkingText + previewParsed.thinkingText,
                        text: stepText + previewParsed.visibleText,
                        toolCalls: [...stepToolCalls],
                        usage: this.emptyUsage(),
                    };

                    yield this.createRunSnapshot({
                        done: false,
                        stepIndex,
                        items: [...items, ...previewItems],
                        thinkingText: fullThinkingText + previewParsed.thinkingText,
                        text: fullText + previewParsed.visibleText,
                        steps: [...steps, previewStep],
                        usage: this.emptyUsage(stepIndex + 1),
                    });
                }

                if (!generationDone) {
                    await this.delay(0);
                }
            }

            if (generationError !== undefined) {
                throw generationError;
            }

            const assistant = await assistantPromise;
            const completionTokens = assistant.completionTokens;
            const stepUsage = this.makeStepUsage(promptTokens, completionTokens);
            totalPromptTokens += promptTokens;
            totalCompletionTokens += completionTokens;
            const parsed = this.parser.parseAssistantContent(assistant.content, (prefix) => this.nextItemId(prefix));

            if (parsed.thinkingText.length > 0) {
                const thinkingItem: RunItem = {
                    type: 'thinking',
                    id: this.nextItemId('thinking'),
                    stepIndex,
                    text: parsed.thinkingText,
                };
                items.push(thinkingItem);
                stepItems.push(thinkingItem);
                stepThinkingText += parsed.thinkingText;
                fullThinkingText += parsed.thinkingText;
            }

            const toolCalls = parsed.toolCalls;

            for (const call of toolCalls) {
                const toolCallItem: RunItem = {
                    type: 'tool.call',
                    id: call.id,
                    stepIndex,
                    name: call.name,
                    args: call.args,
                };
                items.push(toolCallItem);
                stepItems.push(toolCallItem);

                const result = await this.executeToolCall(call);
                const toolResultItem: RunItem = {
                    type: 'tool.result',
                    id: result.id,
                    stepIndex,
                    name: result.name,
                    output: result.output,
                    durationMs: result.durationMs,
                };
                items.push(toolResultItem);
                stepItems.push(toolResultItem);
                stepToolCalls.push(result);

                const toolMessage = this.createToolMessage(result);
                stepToolMessages.push(toolMessage);
            }

            if (parsed.visibleText.length > 0) {
                const textItem: RunItem = {
                    type: 'text',
                    id: this.nextItemId('text'),
                    stepIndex,
                    text: parsed.visibleText,
                };
                items.push(textItem);
                stepItems.push(textItem);
                stepText += parsed.visibleText;
                fullText += parsed.visibleText;
            }

            const assistantMessage = {
                role: 'assistant' as const,
                content: parsed.visibleText,
                toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
            };
            this._history.push(assistantMessage);
            conversation.push(this.toModelMessage(assistantMessage));

            for (const toolMessage of stepToolMessages) {
                this._history.push(toolMessage);
                conversation.push(this.toModelMessage(toolMessage));
            }

            const stepResult: StepResult = {
                stepIndex,
                items: stepItems,
                thinkingText: stepThinkingText,
                text: stepText,
                toolCalls: stepToolCalls,
                usage: stepUsage,
            };
            steps.push(stepResult);

            const shouldContinue = stepToolCalls.length > 0;
            done = !shouldContinue;

            const snapshot = this.createRunSnapshot({
                done,
                stepIndex,
                items,
                thinkingText: fullThinkingText,
                text: fullText,
                steps,
                usage: this.makeTotalUsage(totalPromptTokens, totalCompletionTokens, stepIndex + 1),
            });

            await this.emitStepHooks(stepResult);
            yield snapshot;

            if (!shouldContinue) {
                return;
            }
        }

        const finalStepIndex = steps.length > 0 ? steps[steps.length - 1].stepIndex : -1;
        yield this.createRunSnapshot({
            done: true,
            stepIndex: finalStepIndex,
            items,
            thinkingText: fullThinkingText,
            text: fullText,
            steps,
            usage: this.makeTotalUsage(totalPromptTokens, totalCompletionTokens, steps.length),
        });
    }

    clearHistory(): void {
        this._history = [];
    }

    onBeforeToolCall(hook: BeforeToolCallHook): this {
        this.beforeToolCallHooks.push(hook);
        return this;
    }

    onAfterToolCall(hook: AfterToolCallHook): this {
        this.afterToolCallHooks.push(hook);
        return this;
    }

    onStep(hook: OnStepHook): this {
        this.stepHooks.push(hook);
        return this;
    }

    private async emitStepHooks(step: StepResult): Promise<void> {
        for (const hook of this.stepHooks) {
            await hook(step);
        }
    }

    private emptyUsage(steps = 0): TotalUsage {
        return {
            promptTokens: 0,
            completionTokens: 0,
            totalTokens: 0,
            steps,
        };
    }

    private createRunSnapshot(snapshot: RunResult): RunResult {
        return {
            done: snapshot.done,
            stepIndex: snapshot.stepIndex,
            items: [...snapshot.items],
            thinkingText: snapshot.thinkingText,
            text: snapshot.text,
            steps: snapshot.steps.map((step) => ({
                ...step,
                items: [...step.items],
                toolCalls: [...step.toolCalls],
            })),
            usage: { ...snapshot.usage },
        };
    }

    private toModelMessages(history: ReadonlyArray<Message>): Array<Record<string, unknown>> {
        return [{ role: 'system', content: this.system }, ...history.map((message) => this.toModelMessage(message))];
    }

    private toModelMessage(message: Message): Record<string, unknown> {
        if (message.role === 'assistant') {
            const toolCalls = message.toolCalls?.map((call) => ({
                id: call.id,
                type: 'function',
                function: {
                    name: call.name,
                    arguments: JSON.stringify(call.args),
                },
            }));
            return {
                role: 'assistant',
                content: message.content,
                ...(toolCalls ? { tool_calls: toolCalls } : {}),
            };
        }

        if (message.role === 'tool') {
            return {
                role: 'tool',
                content: message.content,
                tool_call_id: message.toolCallId,
                name: message.name,
            };
        }

        return {
            role: message.role,
            content: message.content,
        };
    }

    private async generateAssistantMessage(
        conversation: Array<Record<string, unknown>>,
        pastKeyValues: DynamicCache,
        options?: { streaming?: boolean; onDelta?: (text: string) => void },
    ): Promise<{ role: 'assistant'; content: string; completionTokens: number }> {
        let completionTokens = 0;
        let streamedRawText = '';
        const tokenizer = this.model.tokenizer;
        const model = this.model.model;

        const streamer = options?.streaming
            ? new TextStreamer(tokenizer, {
                  skip_prompt: true,
                  skip_special_tokens: false,
                  callback_function: (text: string) => {
                      streamedRawText += text;
                      options.onDelta?.(text);
                  },
                  token_callback_function: (tokens: bigint[]) => {
                      completionTokens += tokens.length;
                  },
              })
            : undefined;

        const input = tokenizer.apply_chat_template(conversation as never, {
            tools: this.parser.formatTools(this.tools),
            add_generation_prompt: true,
            return_dict: true,
            enable_thinking: this.enableThinking,
        }) as {
            input_ids: { dims: number[]; size: number };
            attention_mask: unknown;
        };

        const output = (await model.generate({
            ...input,
            max_new_tokens: this.maxNewTokens,
            do_sample: this.temperature !== undefined,
            ...(this.temperature !== undefined ? { temperature: this.temperature } : {}),
            //past_key_values: pastKeyValues,
            return_dict_in_generate: true,
            ...(streamer ? { streamer } : {}),
        })) as { sequences?: { slice: (a: null, b: [number, number]) => unknown }; past_key_values?: unknown };

        const promptLength = input.input_ids?.dims?.[1] ?? 0;
        let contentFromOutput = '';
        if (output.sequences && promptLength > 0) {
            const completionTokensTensor = output.sequences.slice(null, [promptLength, Number.MAX_SAFE_INTEGER]);
            const decoded = tokenizer.batch_decode(completionTokensTensor as never, {
                skip_special_tokens: false,
            });
            contentFromOutput = typeof decoded?.[0] === 'string' ? decoded[0] : '';
        } else if (output.sequences) {
            const decoded = tokenizer.batch_decode(output.sequences as never, {
                skip_special_tokens: false,
            });
            contentFromOutput = typeof decoded?.[0] === 'string' ? decoded[0] : '';
        }

        if (!completionTokens && contentFromOutput.length > 0) {
            const encodedCompletion = tokenizer(contentFromOutput, {
                add_special_tokens: false,
                return_tensor: false,
            });
            const ids = (encodedCompletion as { input_ids?: number[] | number[][] }).input_ids;
            if (Array.isArray(ids) && ids.length > 0) {
                completionTokens = Array.isArray(ids[0]) ? (ids[0] as number[]).length : (ids as number[]).length;
            }
        }

        return {
            role: 'assistant',
            content: options?.streaming && streamedRawText.length > 0 ? streamedRawText : contentFromOutput,
            completionTokens,
        };
    }

    private estimatePromptTokens(conversation: Array<Record<string, unknown>>): number {
        try {
            const rendered = this.model.tokenizer.apply_chat_template(conversation as never, {
                return_dict: true,
                add_generation_prompt: true,
                tools: this.parser.formatTools(this.tools),
                enable_thinking: this.enableThinking,
            });
            const ids = (rendered as { input_ids?: { size?: number; dims?: number[] } }).input_ids;
            if (!ids) {
                return 0;
            }
            if (typeof ids.size === 'number') {
                return ids.size;
            }
            if (Array.isArray(ids.dims) && ids.dims.length > 1) {
                return ids.dims[0] * ids.dims[1];
            }
            return 0;
        } catch {
            return 0;
        }
    }

    private makeStepUsage(promptTokens: number, completionTokens: number): StepUsage {
        return {
            promptTokens,
            completionTokens,
            totalTokens: promptTokens + completionTokens,
        };
    }

    private makeTotalUsage(promptTokens: number, completionTokens: number, steps: number): TotalUsage {
        return {
            promptTokens,
            completionTokens,
            totalTokens: promptTokens + completionTokens,
            steps,
        };
    }

    private async executeToolCall(call: ToolCall): Promise<ToolCallResult> {
        const tool = this.tools[call.name];
        const start = performance.now();

        await this.emitBeforeToolCallHooks(call);

        let output: ToolCallOutput;
        if (!tool) {
            output = {
                isError: true,
                content: [{ type: 'text', text: `Unknown tool: ${call.name}` }],
            };
        } else {
            try {
                output = await tool.execute(call.args);
            } catch (error) {
                output = {
                    isError: true,
                    content: [{ type: 'text', text: this.errorToString(error) }],
                };
            }
        }

        const durationMs = performance.now() - start;
        await this.emitAfterToolCallHooks(call, output, durationMs);

        return {
            ...call,
            output,
            durationMs,
        };
    }

    private async emitBeforeToolCallHooks(call: ToolCall): Promise<void> {
        for (const hook of this.beforeToolCallHooks) {
            await hook(call);
        }
    }

    private async emitAfterToolCallHooks(call: ToolCall, output: ToolCallOutput, durationMs: number): Promise<void> {
        for (const hook of this.afterToolCallHooks) {
            await hook(call, output, durationMs);
        }
    }

    private createToolMessage(result: ToolCallResult): Message {
        return {
            role: 'tool',
            toolCallId: result.id,
            name: result.name,
            content: JSON.stringify(result.output.content),
        };
    }

    private nextItemId(prefix: string): string {
        this.itemIdCounter += 1;
        return `${prefix}_${this.itemIdCounter}`;
    }

    private resolveParser(): NonNullable<AgentConfig['parser']> {
        const modelConfig = this.readModelConfig();
        return (
            this.parserRegistry.resolve({
                modelId: this.model.modelId,
                modelType: this.tryReadString(modelConfig, 'model_type'),
                chatTemplate: this.tryReadString(this.model.tokenizer, 'chat_template'),
            }) ?? new BaseParserStrategy()
        );
    }

    private readModelConfig(): Record<string, unknown> {
        const modelUnknown = this.model.model as unknown as Record<string, unknown>;
        const cfg = modelUnknown?.config;
        if (!cfg || typeof cfg !== 'object' || Array.isArray(cfg)) {
            return {};
        }
        return cfg as Record<string, unknown>;
    }

    private tryReadString(obj: unknown, key: string): string | undefined {
        if (!obj || typeof obj !== 'object') {
            return undefined;
        }
        const value = (obj as Record<string, unknown>)[key];
        return typeof value === 'string' ? value : undefined;
    }

    private errorToString(error: unknown): string {
        if (error instanceof Error) {
            return error.message;
        }
        return String(error);
    }

    private async delay(ms: number): Promise<void> {
        await new Promise((resolve) => {
            setTimeout(resolve, ms);
        });
    }
}
