import type {
    AgentConfig,
    BeforeToolCallHook,
    AfterToolCallHook,
    OnStepHook,
    Message,
    RunResult,
    RequestResult,
    ToolCallResult,
    Usage,
    StreamChunk,
    ModelAdapter,
} from './types';
import type { ToolList } from './Tool';
import { DynamicCache, TextStreamer } from '@huggingface/transformers';
import { ModelAdapterBase, ModelAdapterRegistry } from './adapters';
import type { Model } from './Model';
import { ToolExecutor } from './tools/ToolExecutor';

export class Agent {
    readonly model: Model;
    readonly initialPrompts: ReadonlyArray<Message>;
    readonly tools: ToolList;
    readonly maxSteps: number;
    readonly maxNewTokens: number;
    readonly temperature: number | undefined;
    readonly enableThinking: boolean;
    readonly adapter: ModelAdapter;

    private readonly stepHooks: OnStepHook[] = [];
    private readonly toolExecutor: ToolExecutor;
    private _history: Message[] = [];
    private _modelHistory: Message[] = [];
    private itemIdCounter = 0;
    private readonly adapterRegistry = new ModelAdapterRegistry();
    private _kvCache: DynamicCache | null = null;

    get history(): ReadonlyArray<Message> {
        return this._history;
    }

    constructor(config: AgentConfig) {
        this.model = config.model;
        this.initialPrompts = this.cloneMessages(config.initialPrompts ?? []);
        this.tools = config.tools ?? [];
        this.maxSteps = config.maxSteps ?? 10;
        this.maxNewTokens = config.maxNewTokens ?? 1024;
        this.temperature = config.temperature;
        this.enableThinking = config.enableThinking ?? false;
        this.adapter = config.adapter ?? this.resolveAdapter();
        this.toolExecutor = new ToolExecutor(this.tools);
        this.clearHistory();
    }

    async run(input: string): Promise<RequestResult> {
        let last: RequestResult = { done: true, runs: [], usage: this.makeUsage(0, 0) };
        await this.runAgentLoop(input, (snapshot) => {
            last = snapshot;
        });
        return last;
    }

    async *stream(input: string): AsyncIterable<StreamChunk> {
        const queue: StreamChunk[] = [];
        let done = false;
        let loopError: unknown;

        const loopPromise = this.runAgentLoop(input, (snapshot) => {
            queue.push(snapshot);
        })
            .catch((error) => {
                loopError = error;
            })
            .finally(() => {
                done = true;
            });

        while (!done || queue.length > 0) {
            const chunk = queue.shift();
            if (chunk) {
                yield chunk;
                continue;
            }
            await this.delay(0);
        }

        await loopPromise;
        if (loopError !== undefined) {
            throw loopError;
        }
    }

    private async runAgentLoop(input: string, onUpdate: (snapshot: RequestResult) => void): Promise<void> {
        if (!this.model.isInitialized) {
            throw new Error('Model is not initialized. Call model.init() before creating an Agent run.');
        }

        const userMessage: Message = { role: 'user', content: input };
        this._history.push(userMessage);
        this._modelHistory.push(userMessage);

        this.assertStringMessageContent(this._modelHistory);
        let conversation = this.toModelMessages(this._modelHistory);

        let totalPromptTokens = 0;
        let totalCompletionTokens = 0;
        const results: RunResult[] = [];

        for (let stepIndex = 0; stepIndex < this.maxSteps; stepIndex++) {
            const promptTokens = this.estimatePromptTokens(conversation);
            let assistantRaw = '';
            let completionTokens = 0;
            let generationDone = false;
            let generationError: unknown;

            let previewRaw = '';
            const previewQueue: string[] = [];

            // restart generation with a streaming callback bound to preview queue
            generationDone = false;
            generationError = undefined;
            assistantRaw = '';
            completionTokens = 0;
            void (async () => {
                try {
                    const assistant = await this.generateAssistantMessage(conversation, (delta) => {
                        previewQueue.push(delta);
                    });
                    assistantRaw = assistant.modelContent;
                    completionTokens = assistant.completionTokens;
                } catch (error) {
                    generationError = error;
                } finally {
                    generationDone = true;
                }
            })();

            while (true) {
                const delta = previewQueue.shift();
                if (delta) {
                    previewRaw += delta;
                    const previewParsed = this.adapter.parseAssistantContent(previewRaw, (prefix) =>
                        this.nextItemId(prefix),
                    );

                    const current = this.createRunResult({
                        thinkingText: previewParsed.thinkingText,
                        text: previewParsed.visibleText,
                        tools: [],
                        usage: this.makeUsage(0, 0),
                    });
                    onUpdate({
                        done: false,
                        runs: [...results, current],
                        usage: this.makeUsage(totalPromptTokens, totalCompletionTokens),
                    });
                    continue;
                }

                if (generationDone) break;
                await this.delay(0);
            }

            if (generationError !== undefined) {
                throw generationError;
            }

            const parsed = this.adapter.parseAssistantContent(assistantRaw, (prefix) => this.nextItemId(prefix));
            totalPromptTokens += promptTokens;
            totalCompletionTokens += completionTokens;

            const executedTools: ToolCallResult[] = [];
            for (const call of parsed.toolCalls) {
                const result = await this.toolExecutor.execute(call);
                executedTools.push(result);
            }

            const assistantMessage: Message = {
                role: 'assistant',
                content: parsed.visibleText,
                thinking: parsed.thinkingText || undefined,
                toolCalls:
                    parsed.toolCalls.length > 0
                        ? parsed.toolCalls.map((call) => this.toOpenAIToolCall(call))
                        : undefined,
            };

            const modelAssistantMessage: Message = {
                role: 'assistant',
                content: parsed.visibleText,
                thinking: parsed.thinkingText || undefined,
                toolCalls:
                    parsed.toolCalls.length > 0
                        ? parsed.toolCalls.map((call) => this.toOpenAIToolCall(call))
                        : undefined,
            };

            if (parsed.visibleText.length > 0 || parsed.toolCalls.length > 0) {
                this._history.push(assistantMessage);
                this._modelHistory.push(modelAssistantMessage);
            }

            for (const result of executedTools) {
                const toolMessage = this.toolExecutor.createToolMessage(result);
                this._history.push(toolMessage);
                this._modelHistory.push(toolMessage);
            }

            const done = executedTools.length === 0;
            const runResult = this.createRunResult({
                thinkingText: parsed.thinkingText,
                text: parsed.visibleText,
                tools: executedTools,
                usage: this.makeUsage(promptTokens, completionTokens),
            });
            results.push(runResult);

            await this.emitStepHooks(runResult);
            onUpdate({ done, runs: [...results], usage: this.makeUsage(totalPromptTokens, totalCompletionTokens) });

            if (done) {
                return;
            }

            conversation = this.toModelMessages(this._modelHistory);
        }

        onUpdate({ done: true, runs: [...results], usage: this.makeUsage(totalPromptTokens, totalCompletionTokens) });
    }

    clearHistory(): void {
        this._history = this.cloneMessages(this.initialPrompts);
        this._modelHistory = this.cloneMessages(this.initialPrompts);
        this.assertStringMessageContent(this._modelHistory);
        this._kvCache = null;
    }

    onBeforeToolCall(hook: BeforeToolCallHook): this {
        this.toolExecutor.onBeforeToolCall(hook);
        return this;
    }

    onAfterToolCall(hook: AfterToolCallHook): this {
        this.toolExecutor.onAfterToolCall(hook);
        return this;
    }

    onStep(hook: OnStepHook): this {
        this.stepHooks.push(hook);
        return this;
    }

    private async emitStepHooks(step: RunResult): Promise<void> {
        for (const hook of this.stepHooks) {
            await hook(step);
        }
    }

    private createRunResult(snapshot: RunResult): RunResult {
        return {
            thinkingText: snapshot.thinkingText,
            text: snapshot.text,
            tools: [...snapshot.tools],
            usage: { ...snapshot.usage },
        };
    }

    private toModelMessages(history: ReadonlyArray<Message>): Array<Record<string, unknown>> {
        return this.adapter.formatMessages(history);
    }

    private cloneMessages(messages: ReadonlyArray<Message>): Message[] {
        return messages.map((message) => ({
            ...message,
            ...(message.role === 'assistant' && message.toolCalls ? { toolCalls: [...message.toolCalls] } : {}),
        }));
    }

    private assertStringMessageContent(messages: ReadonlyArray<Message>): void {
        for (const message of messages) {
            if ('content' in message && message.content !== undefined && typeof message.content !== 'string') {
                throw new Error(
                    `Multimodal message content is not supported yet. ${message.role} messages currently require string content.`,
                );
            }
        }
    }

    private toOpenAIToolCall(call: ToolCallResult | { id: string; name: string; args: Record<string, unknown> }) {
        return {
            id: call.id,
            type: 'function' as const,
            function: {
                name: call.name,
                arguments: call.args,
            },
        };
    }

    private async generateAssistantMessage(
        conversation: Array<Record<string, unknown>>,
        onDelta?: (text: string) => void,
    ): Promise<{
        role: 'assistant';
        content: string;
        modelContent: string;
        completionTokens: number;
        promptTokenCount: number;
    }> {
        let completionTokens = 0;
        let streamedRawText = '';
        const tokenizer = this.model.tokenizer;
        const model = this.model.model;

        const streamer = new TextStreamer(tokenizer, {
            skip_prompt: true,
            skip_special_tokens: false,
            callback_function: (text: string) => {
                streamedRawText += text;
                onDelta?.(text);
            },
            token_callback_function: (tokens: bigint[]) => {
                completionTokens += tokens.length;
            },
        });

        const rawInput = tokenizer.apply_chat_template(
            conversation as never,
            {
                tools: this.adapter.formatTools(this.tools),
                add_generation_prompt: true,
                tokenize: false,
                enable_thinking: this.enableThinking,
            } as never,
        );
        const prompt = this.adapter.preparePromptForGeneration(String(rawInput));
        const input = (
            tokenizer as unknown as (
                text: string[],
                options: Record<string, unknown>,
            ) => {
                input_ids?: { dims?: number[]; size?: number };
            }
        )([prompt], {
            add_special_tokens: false,
            padding: true,
            truncation: true,
            return_tensor: true,
            return_dict: true,
        });

        const fullPromptLength = input.input_ids?.dims?.[1] ?? 0;

        let generationInput = input;
        let generationPromptLength = fullPromptLength;
        const useKvCache = this.adapter.useKvCache(this.enableThinking);

        const generationOptions = {
            ...generationInput,
            max_new_tokens: this.maxNewTokens,
            ...(this.temperature !== undefined
                ? { temperature: this.temperature, do_sample: true }
                : { do_sample: false }),
            ...(useKvCache ? { past_key_values: this._kvCache } : {}),
            streamer,
            ...(useKvCache ? { return_dict_in_generate: true } : {}),
        };

        const output = (await model.generate(generationOptions)) as
            | { past_key_values?: DynamicCache; sequences?: unknown }
            | unknown;

        this._kvCache =
            useKvCache && typeof output === 'object' && output !== null
                ? ((output as { past_key_values?: DynamicCache }).past_key_values ?? null)
                : null;

        const sequences =
            typeof output === 'object' && output !== null && 'sequences' in output
                ? (output as { sequences?: unknown }).sequences
                : output;
        const modelRawText = this.decodeGeneratedContinuation(sequences, generationPromptLength) ?? streamedRawText;
        const modelContent = this.adapter.normalizeAssistantContent(modelRawText);

        console.log({ rawInput, prompt, modelRawText, modelContent });

        return {
            role: 'assistant',
            content: modelContent,
            modelContent,
            completionTokens,
            promptTokenCount: fullPromptLength,
        };
    }

    private decodeGeneratedContinuation(sequences: unknown, promptLength: number): string | null {
        if (!sequences || typeof sequences !== 'object') {
            return null;
        }

        const tensor = sequences as {
            dims?: number[];
            slice?: (...slices: (number | [number | null, number | null] | null)[]) => {
                data?: ArrayLike<bigint | number>;
            };
        };
        if (!Array.isArray(tensor.dims) || tensor.dims.length < 2 || typeof tensor.slice !== 'function') {
            return null;
        }

        const sequenceLength = tensor.dims[1] ?? 0;
        if (sequenceLength <= promptLength) {
            return '';
        }

        const generated = tensor.slice(0, [promptLength, null]);
        if (!generated.data) {
            return null;
        }

        const decode = (
            this.model.tokenizer as {
                decode?: (tokens: Array<bigint | number>, options?: unknown) => string;
            }
        ).decode;
        if (typeof decode !== 'function') {
            return null;
        }

        return decode.call(this.model.tokenizer, Array.from(generated.data), { skip_special_tokens: false });
    }

    private estimatePromptTokens(conversation: Array<Record<string, unknown>>): number {
        try {
            const rendered = this.model.tokenizer.apply_chat_template(
                conversation as never,
                {
                    tokenize: false,
                    add_generation_prompt: true,
                    tools: this.adapter.formatTools(this.tools),
                    enable_thinking: this.enableThinking,
                } as never,
            );
            const prompt = this.adapter.preparePromptForGeneration(String(rendered));
            const tokenized = (
                this.model.tokenizer as unknown as (
                    text: string[],
                    options: Record<string, unknown>,
                ) => {
                    input_ids?: { dims?: number[]; size?: number };
                }
            )([prompt], {
                add_special_tokens: false,
                padding: true,
                truncation: true,
                return_tensor: true,
                return_dict: true,
            });
            const ids = tokenized.input_ids;
            if (!ids) return 0;
            if (typeof ids.size === 'number') return ids.size;
            if (Array.isArray(ids.dims) && ids.dims.length > 1) return ids.dims[0] * ids.dims[1];
            return 0;
        } catch {
            return 0;
        }
    }

    private makeUsage(promptTokens: number, completionTokens: number): Usage {
        return {
            promptTokens,
            completionTokens,
            totalTokens: promptTokens + completionTokens,
        };
    }

    private nextItemId(prefix: string): string {
        this.itemIdCounter += 1;
        return `${prefix}_${this.itemIdCounter}`;
    }

    private resolveAdapter = (): ModelAdapter =>
        this.adapterRegistry.resolve({
            modelId: this.model.modelId,
            modelType: this.tryReadString(this.readModelConfig(), 'model_type'),
            chatTemplate: this.tryReadString(this.model.tokenizer, 'chat_template'),
            enableThinking: this.enableThinking,
        }) ?? new ModelAdapterBase();

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

    private async delay(ms: number): Promise<void> {
        await new Promise((resolve) => {
            setTimeout(resolve, ms);
        });
    }
}
