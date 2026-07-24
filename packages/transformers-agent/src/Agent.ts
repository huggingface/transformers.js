import { TextStreamer } from '@huggingface/transformers';
import { ModelAdapterBase, ModelAdapterRegistry } from './adapters';
import type { Model } from './Model';
import type { ToolList } from './Tool';
import type {
    AgentConfig,
    Message,
    MessageContent,
    ModelAdapter,
    Prompt,
    PromptResult,
    StreamChunk,
    ToolCall,
    Usage,
} from './types';

type ModelMessage = Message & { thinking?: string };

export class Agent {
    readonly model: Model;
    readonly tools: ToolList;
    readonly maxNewTokens: number;
    readonly temperature: number | undefined;
    readonly enableThinking: boolean;
    readonly adapter: ModelAdapter;

    private _history: Message[] = [];
    private _modelHistory: ModelMessage[] = [];
    private readonly _initialPrompts: Message[];
    private itemIdCounter = 0;
    private promptActive = false;
    private readonly adapterRegistry = new ModelAdapterRegistry();

    get history(): ReadonlyArray<Message> {
        return this.cloneMessages(this._history);
    }

    get initialPrompts(): ReadonlyArray<Message> {
        return this.cloneMessages(this._initialPrompts);
    }

    constructor(config: AgentConfig) {
        this.model = config.model;
        this._initialPrompts = this.cloneMessages(config.initialPrompts ?? []);
        this.tools = config.tools ?? [];
        this.maxNewTokens = config.maxNewTokens ?? 1024;
        this.temperature = config.temperature;
        this.enableThinking = config.enableThinking ?? false;
        this.adapter = config.adapter ?? this.resolveAdapter();
        this.clearHistory();
    }

    async prompt(input: Prompt): Promise<PromptResult> {
        let result: PromptResult | undefined;
        await this.generateTurn(input, (chunk) => {
            if (chunk.done) {
                result = this.toPromptResult(chunk);
            }
        });

        if (!result) {
            throw new Error('The model did not produce a result.');
        }
        return result;
    }

    async *promptStreaming(input: Prompt): AsyncIterable<StreamChunk> {
        const queue: StreamChunk[] = [];
        let done = false;
        let generationError: unknown;

        const generation = this.generateTurn(input, (chunk) => queue.push(chunk))
            .catch((error) => {
                generationError = error;
            })
            .finally(() => {
                done = true;
            });

        while (!done || queue.length > 0) {
            const chunk = queue.shift();
            if (chunk) {
                yield chunk;
            } else {
                await this.delay(0);
            }
        }

        await generation;
        if (generationError !== undefined) {
            throw generationError;
        }
    }

    clearHistory(): void {
        if (this.promptActive) {
            throw new Error('Cannot clear history while a prompt is running.');
        }
        this._history = this.cloneMessages(this._initialPrompts);
        this.validateHistory(this._history);
        this._modelHistory = this.cloneMessages(this._initialPrompts);
    }

    private async generateTurn(input: Prompt, onUpdate: (chunk: StreamChunk) => void): Promise<void> {
        if (this.promptActive) {
            throw new Error('Only one prompt can run at a time for an Agent.');
        }
        if (!this.model.isInitialized) {
            throw new Error('Model is not initialized. Call model.init() before prompting an Agent.');
        }

        this.promptActive = true;
        const historyLength = this._history.length;
        const modelHistoryLength = this._modelHistory.length;
        try {
            this.appendPrompt(input);
            const conversation = this.adapter.formatMessages(this._modelHistory);
            let previewRaw = '';

            const generated = await this.generateAssistantMessage(conversation, (delta) => {
                previewRaw += delta;
                const parsed = this.adapter.parseAssistantContent(previewRaw, this.createPreviewIdFactory());
                onUpdate({
                    done: false,
                    response: parsed.visibleText,
                    thinking: parsed.thinkingText,
                    toolCalls: parsed.toolCalls.map((call) => this.toPublicToolCall(call)),
                    usage: this.makeUsage(0, 0),
                });
            });

            const parsed = this.adapter.parseAssistantContent(generated.modelContent, (prefix) =>
                this.nextItemId(prefix),
            );
            const toolCalls = parsed.toolCalls.map((call) => this.toPublicToolCall(call));
            const result: StreamChunk = {
                done: true,
                response: parsed.visibleText,
                thinking: parsed.thinkingText,
                toolCalls,
                usage: this.makeUsage(generated.promptTokens, generated.completionTokens),
            };

            if (result.response || result.toolCalls.length > 0) {
                const assistantMessage = this.createAssistantMessage(result.response, result.toolCalls);
                this.validateHistory([...this._history, assistantMessage]);
                this._history.push(assistantMessage);
                this._modelHistory.push({
                    ...this.cloneMessage(assistantMessage),
                    thinking: result.thinking || undefined,
                });
            }
            onUpdate(result);
        } catch (error) {
            this._history.length = historyLength;
            this._modelHistory.length = modelHistoryLength;
            throw error;
        } finally {
            this.promptActive = false;
        }
    }

    private appendPrompt(input: Prompt): void {
        if (typeof input === 'string') {
            const message: Message = { role: 'user', content: input };
            this._history.push(message);
            this._modelHistory.push(this.cloneMessage(message));
            return;
        }
        const messages = this.cloneMessages(input);
        this.validateHistory([...this._history, ...messages]);
        this._history.push(...messages);
        this._modelHistory.push(...this.cloneMessages(messages));
    }

    private createAssistantMessage(response: string, toolCalls: ToolCall[]): Message {
        if (toolCalls.length === 0) {
            return { role: 'assistant', content: response };
        }

        const content: MessageContent[] = [
            ...(response ? [{ type: 'text' as const, value: response }] : []),
            ...toolCalls.map((call) => ({
                type: 'tool-call' as const,
                value: { ...call, arguments: this.cloneSerializable(call.arguments) },
            })),
        ];
        return { role: 'assistant', content };
    }

    private cloneMessages(messages: ReadonlyArray<Message>): Message[] {
        return messages.map((message) => this.cloneMessage(message));
    }

    private cloneMessage(message: Message): Message {
        return {
            ...message,
            content: Array.isArray(message.content)
                ? message.content.map((part) => this.cloneContentPart(part))
                : message.content,
        };
    }

    private cloneContentPart(part: MessageContent): MessageContent {
        if (part.type === 'tool-call') {
            return { ...part, value: { ...part.value, arguments: this.cloneSerializable(part.value.arguments) } };
        }
        if (part.type === 'tool-response') {
            return { ...part, value: this.cloneSerializable(part.value) };
        }
        return part.type === 'text' ? { ...part } : this.cloneSerializable(part);
    }

    private toPublicToolCall(call: { id: string; name: string; args: Record<string, unknown> }): ToolCall {
        return { callID: call.id, name: call.name, arguments: this.cloneSerializable(call.args) };
    }

    private toPromptResult(chunk: StreamChunk): PromptResult {
        return {
            response: chunk.response,
            thinking: chunk.thinking,
            toolCalls: chunk.toolCalls.map((call) => ({
                ...call,
                arguments: this.cloneSerializable(call.arguments),
            })),
            usage: { ...chunk.usage },
        };
    }

    private async generateAssistantMessage(
        conversation: Array<Record<string, unknown>>,
        onDelta?: (text: string) => void,
    ): Promise<{ modelContent: string; completionTokens: number; promptTokens: number }> {
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
            ) => { input_ids?: { dims?: number[]; size?: number } }
        )([prompt], {
            add_special_tokens: false,
            padding: true,
            truncation: true,
            return_tensor: true,
            return_dict: true,
        });
        const promptTokens = input.input_ids?.dims?.[1] ?? input.input_ids?.size ?? 0;
        const output = (await model.generate({
            ...input,
            max_new_tokens: this.maxNewTokens,
            ...(this.temperature !== undefined
                ? { temperature: this.temperature, do_sample: true }
                : { do_sample: false }),
            streamer,
        })) as { sequences?: unknown } | unknown;
        const sequences =
            typeof output === 'object' && output !== null && 'sequences' in output
                ? (output as { sequences?: unknown }).sequences
                : output;
        const modelRawText = this.decodeGeneratedContinuation(sequences, promptTokens) ?? streamedRawText;
        return {
            modelContent: this.adapter.normalizeAssistantContent(modelRawText),
            completionTokens,
            promptTokens,
        };
    }

    private decodeGeneratedContinuation(sequences: unknown, promptLength: number): string | null {
        if (!sequences || typeof sequences !== 'object') return null;
        const tensor = sequences as {
            dims?: number[];
            slice?: (...slices: (number | [number | null, number | null] | null)[]) => {
                data?: ArrayLike<bigint | number>;
            };
        };
        if (!Array.isArray(tensor.dims) || tensor.dims.length < 2 || typeof tensor.slice !== 'function') return null;
        if ((tensor.dims[1] ?? 0) <= promptLength) return '';
        const generated = tensor.slice(0, [promptLength, null]);
        if (!generated.data) return null;
        const decode = (
            this.model.tokenizer as { decode?: (tokens: Array<bigint | number>, options?: unknown) => string }
        ).decode;
        if (typeof decode !== 'function') return null;
        return decode.call(this.model.tokenizer, Array.from(generated.data), { skip_special_tokens: false });
    }

    private createPreviewIdFactory(): (prefix: string) => string {
        let offset = 0;
        return (prefix) => `${prefix}_${this.itemIdCounter + ++offset}`;
    }

    private validateHistory(messages: ReadonlyArray<Message>): void {
        const calls = new Map<string, { name: string; resolved: boolean }>();
        for (const message of messages) {
            if (typeof message.content === 'string') continue;
            const hasTextOrMedia = message.content.some(
                (part) => part.type === 'text' || part.type === 'image' || part.type === 'audio',
            );
            const toolCalls = message.content.filter((part) => part.type === 'tool-call');
            const toolResponses = message.content.filter((part) => part.type === 'tool-response');

            if (toolCalls.length > 0 && message.role !== 'assistant') {
                throw new Error('Tool calls are only valid in assistant messages.');
            }
            if (toolResponses.length > 0 && message.role !== 'user') {
                throw new Error('Tool responses are only valid in user messages.');
            }
            if (toolResponses.length > 0 && hasTextOrMedia) {
                throw new Error('Tool responses cannot be mixed with text, image, or audio content in one message.');
            }

            for (const part of toolCalls) {
                if (calls.has(part.value.callID)) {
                    throw new Error(`Duplicate tool call ID: ${part.value.callID}`);
                }
                calls.set(part.value.callID, { name: part.value.name, resolved: false });
            }
            for (const part of toolResponses) {
                const call = calls.get(part.value.callID);
                if (!call) throw new Error(`Unknown tool call ID: ${part.value.callID}`);
                if (call.resolved) throw new Error(`Tool call already has a response: ${part.value.callID}`);
                if (call.name !== part.value.name) {
                    throw new Error(`Tool response name does not match call ${part.value.callID}.`);
                }
                call.resolved = true;
            }
        }
    }

    private cloneSerializable<T>(value: T): T {
        return structuredClone(value);
    }

    private makeUsage(promptTokens: number, completionTokens: number): Usage {
        return { promptTokens, completionTokens, totalTokens: promptTokens + completionTokens };
    }

    private nextItemId(prefix: string): string {
        return `${prefix}_${++this.itemIdCounter}`;
    }

    private resolveAdapter = (): ModelAdapter =>
        this.adapterRegistry.resolve({
            modelId: this.model.modelId,
            modelType: this.tryReadString(this.readModelConfig(), 'model_type'),
            chatTemplate: this.tryReadString(this.model.tokenizer, 'chat_template'),
            enableThinking: this.enableThinking,
        }) ?? new ModelAdapterBase();

    private readModelConfig(): Record<string, unknown> {
        const config = (this.model.model as unknown as Record<string, unknown>)?.config;
        return config && typeof config === 'object' && !Array.isArray(config)
            ? (config as Record<string, unknown>)
            : {};
    }

    private tryReadString(value: unknown, key: string): string | undefined {
        if (!value || typeof value !== 'object') return undefined;
        const field = (value as Record<string, unknown>)[key];
        return typeof field === 'string' ? field : undefined;
    }

    private async delay(ms: number): Promise<void> {
        await new Promise((resolve) => setTimeout(resolve, ms));
    }
}
