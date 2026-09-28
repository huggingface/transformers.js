import { TextStreamer } from '@huggingface/transformers';
import { ModelAdapterBase, ModelAdapterRegistry } from './adapters';
import type { Model } from './Model';
import type { ToolList } from './Tool';
import type {
    AgentConfig,
    LanguageModelMessageContent,
    Message,
    MessageContent,
    ModelAdapter,
    Prompt,
    ToolCall,
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

    async prompt(input: Prompt): Promise<LanguageModelMessageContent[]> {
        return this.generateTurn(input);
    }

    promptStreaming(input: Prompt): ReadableStream<LanguageModelMessageContent> {
        return new ReadableStream({
            start: (controller) => {
                void this.generateTurn(input, (chunk) => controller.enqueue(chunk)).then(
                    () => controller.close(),
                    (error) => controller.error(error),
                );
            },
        });
    }

    clearHistory(): void {
        if (this.promptActive) {
            throw new Error('Cannot clear history while a prompt is running.');
        }
        this._history = this.cloneMessages(this._initialPrompts);
        this.validateHistory(this._history);
        this._modelHistory = this.cloneMessages(this._initialPrompts);
    }

    private async generateTurn(
        input: Prompt,
        onChunk?: (chunk: LanguageModelMessageContent) => void,
    ): Promise<LanguageModelMessageContent[]> {
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
            let streamedThinking = '';
            let streamedText = '';

            const generated = await this.generateAssistantMessage(
                conversation,
                onChunk
                    ? (delta) => {
                          previewRaw += delta;
                          const parsed = this.adapter.parseAssistantContent(previewRaw, this.createPreviewIdFactory());
                          streamedThinking = this.emitContentDelta(
                              'thinking',
                              streamedThinking,
                              parsed.thinkingText,
                              onChunk,
                          );
                          streamedText = this.emitContentDelta('text', streamedText, parsed.visibleText, onChunk);
                      }
                    : undefined,
            );

            const parsed = this.adapter.parseAssistantContent(generated.modelContent, (prefix) =>
                this.nextItemId(prefix),
            );
            const toolCalls = parsed.toolCalls.map((call) => this.toPublicToolCall(call));
            const result = this.createAssistantContent(parsed.thinkingText, parsed.visibleText, toolCalls);

            if (parsed.visibleText || toolCalls.length > 0) {
                const assistantMessage = this.createAssistantMessage(parsed.visibleText, toolCalls);
                this.validateHistory([...this._history, assistantMessage]);
                this._history.push(assistantMessage);
                this._modelHistory.push({
                    ...this.cloneMessage(assistantMessage),
                    thinking: parsed.thinkingText || undefined,
                });
            }

            for (const part of result) {
                if (part.type === 'tool-call') onChunk?.(this.cloneContentPart(part));
            }
            return result;
        } catch (error) {
            this._history.length = historyLength;
            this._modelHistory.length = modelHistoryLength;
            throw error;
        } finally {
            this.promptActive = false;
        }
    }

    private emitContentDelta(
        type: 'thinking' | 'text',
        previous: string,
        current: string,
        onChunk?: (chunk: LanguageModelMessageContent) => void,
    ): string {
        if (!current.startsWith(previous)) {
            throw new Error(`The model adapter produced non-incremental streaming ${type}.`);
        }
        const delta = current.slice(previous.length);
        if (delta) onChunk?.({ type, value: delta });
        return current;
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

    private createAssistantContent(
        thinking: string,
        response: string,
        toolCalls: ToolCall[],
    ): LanguageModelMessageContent[] {
        return [
            ...(thinking ? [{ type: 'thinking' as const, value: thinking }] : []),
            ...(response ? [{ type: 'text' as const, value: response }] : []),
            ...toolCalls.map((call) => ({
                type: 'tool-call' as const,
                value: { ...call, arguments: this.cloneSerializable(call.arguments) },
            })),
        ];
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

    private async generateAssistantMessage(
        conversation: Array<Record<string, unknown>>,
        onDelta?: (text: string) => void,
    ): Promise<{ modelContent: string }> {
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
}
