import { TextStreamer } from '@huggingface/transformers';
import { formatMessages, formatTools, getModelFamily } from './messageFormatting';
import type { Model } from './Model';
import type { ToolList } from './Tool';
import type {
    AgentConfig,
    LanguageModelMessageContent,
    Message,
    MessageContent,
    Prompt,
    ToolCall,
    Usage,
} from './types';

type ModelMessage = Message & { thinking?: string };
type ResponseEvent =
    | { type: 'region_open'; field: string }
    | { type: 'region_chunk'; field: string; text: string; dirty: boolean }
    | { type: 'region_close'; field: string; value: unknown };
type ParsedAssistantMessage = {
    thinking?: unknown;
    content?: unknown;
    tool_calls?: unknown;
};
type ResponseStreamState = { dirtyFields: Set<string> };

export class Agent {
    readonly model: Model;
    readonly tools: ToolList;
    readonly maxNewTokens: number;
    readonly temperature: number | undefined;
    readonly enableThinking: boolean;

    private _history: Message[] = [];
    private _modelHistory: ModelMessage[] = [];
    private readonly _initialPrompts: Message[];
    private _latestUsage: Usage | null = null;
    private itemIdCounter = 0;
    private promptActive = false;

    get history(): ReadonlyArray<Message> {
        return this.cloneMessages(this._history);
    }

    get initialPrompts(): ReadonlyArray<Message> {
        return this.cloneMessages(this._initialPrompts);
    }

    getLatestUsage(): Usage | null {
        return this._latestUsage ? { ...this._latestUsage } : null;
    }

    constructor(config: AgentConfig) {
        this.model = config.model;
        this._initialPrompts = this.cloneMessages(config.initialPrompts ?? []);
        this.tools = config.tools ?? [];
        this.maxNewTokens = config.maxNewTokens ?? 1024;
        this.temperature = config.temperature;
        this.enableThinking = config.enableThinking ?? false;
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
        this._latestUsage = null;
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
        const turnStartedAt = performance.now();
        const historyLength = this._history.length;
        const modelHistoryLength = this._modelHistory.length;
        try {
            this.appendPrompt(input);
            const conversation = formatMessages(this._modelHistory, this.getModelFamily());
            const generated = await this.generateAssistantMessage(conversation, onChunk);

            const { thinking, content, toolCalls } = this.readAssistantMessage(generated.message);
            const result = this.createAssistantContent(thinking, content, toolCalls);

            if (content || toolCalls.length > 0) {
                const assistantMessage = this.createAssistantMessage(content, toolCalls);
                this.validateHistory([...this._history, assistantMessage]);
                this._history.push(assistantMessage);
                this._modelHistory.push({
                    ...this.cloneMessage(assistantMessage),
                    thinking: thinking || undefined,
                });
            }

            for (const part of result) {
                if (part.type === 'tool-call') onChunk?.(this.cloneContentPart(part));
            }
            this._latestUsage = {
                ...generated.usage,
                totalTimeMs: performance.now() - turnStartedAt,
            };
            return result;
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

    private async generateAssistantMessage(
        conversation: Array<Record<string, unknown>>,
        onChunk?: (chunk: LanguageModelMessageContent) => void,
    ): Promise<{ message: ParsedAssistantMessage; usage: Usage }> {
        let completionTokens = 0;
        let firstTokenAt: number | undefined;
        let streamedRawText = '';
        const tokenizer = this.model.tokenizer;
        const model = this.model.model;
        const tools = formatTools(this.tools);
        const rawInput = tokenizer.apply_chat_template(
            conversation as never,
            {
                tools,
                add_generation_prompt: true,
                tokenize: false,
                enable_thinking: this.enableThinking,
            } as never,
        );
        const prompt = String(rawInput);
        const parser = onChunk ? tokenizer.get_response_parser({ prefix: prompt, tools }) : null;
        const responseStreamState: ResponseStreamState = { dirtyFields: new Set() };
        if (parser) this.emitResponseEvents(parser.initial_events as ResponseEvent[], responseStreamState, onChunk);
        const streamer = new TextStreamer(tokenizer, {
            skip_prompt: true,
            skip_special_tokens: false,
            callback_function: (text: string) => {
                if (text) firstTokenAt ??= performance.now();
                streamedRawText += text;
                if (parser) {
                    this.emitResponseEvents(parser.feed(text) as ResponseEvent[], responseStreamState, onChunk);
                }
            },
            token_callback_function: (tokens: bigint[]) => {
                if (tokens.length > 0) firstTokenAt ??= performance.now();
                completionTokens += tokens.length;
            },
        });

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
        const generationStartedAt = performance.now();
        const output = (await model.generate({
            ...input,
            max_new_tokens: this.maxNewTokens,
            ...(this.temperature !== undefined
                ? { temperature: this.temperature, do_sample: true }
                : { do_sample: false }),
            streamer,
        })) as { sequences?: unknown } | unknown;
        const generationEndedAt = performance.now();
        const generationTimeMs = generationEndedAt - generationStartedAt;
        const sequences =
            typeof output === 'object' && output !== null && 'sequences' in output
                ? (output as { sequences?: unknown }).sequences
                : output;
        if (completionTokens === 0 && sequences && typeof sequences === 'object') {
            const sequenceLength = (sequences as { dims?: number[] }).dims?.[1];
            if (sequenceLength !== undefined) completionTokens = Math.max(0, sequenceLength - promptTokens);
        }
        const timeToFirstTokenMs =
            firstTokenAt === undefined
                ? completionTokens > 0
                    ? generationTimeMs
                    : 0
                : firstTokenAt - generationStartedAt;
        const modelRawText = this.decodeGeneratedContinuation(sequences, promptTokens) ?? streamedRawText;
        if (parser) {
            const [, finalEvents] = parser.finalize();
            this.emitResponseEvents(finalEvents as ResponseEvent[], responseStreamState, onChunk);
        }
        return {
            message: tokenizer.parse_response(modelRawText, { prefix: prompt, tools }) as ParsedAssistantMessage,
            usage: {
                promptTokens,
                completionTokens,
                totalTokens: promptTokens + completionTokens,
                tokensPerSecond: generationTimeMs > 0 ? (completionTokens * 1000) / generationTimeMs : 0,
                timeToFirstTokenMs,
                generationTimeMs,
                totalTimeMs: generationEndedAt - generationStartedAt,
            },
        };
    }

    private emitResponseEvents(
        events: ResponseEvent[],
        state: ResponseStreamState,
        onChunk?: (chunk: LanguageModelMessageContent) => void,
    ): void {
        for (const event of events) {
            if (event.type === 'region_open') {
                state.dirtyFields.delete(event.field);
                continue;
            }
            if (event.type === 'region_chunk') {
                if (event.dirty) {
                    state.dirtyFields.add(event.field);
                } else {
                    this.emitResponseText(event.field, event.text, onChunk);
                }
                continue;
            }
            if (state.dirtyFields.delete(event.field) && typeof event.value === 'string') {
                this.emitResponseText(event.field, event.value, onChunk);
            }
        }
    }

    private emitResponseText(
        field: string,
        text: string,
        onChunk?: (chunk: LanguageModelMessageContent) => void,
    ): void {
        if (!text) return;
        if (field === 'thinking') onChunk?.({ type: 'thinking', value: text });
        if (field === 'content') onChunk?.({ type: 'text', value: text });
    }

    private readAssistantMessage(message: ParsedAssistantMessage): {
        thinking: string;
        content: string;
        toolCalls: ToolCall[];
    } {
        const toolCalls: ToolCall[] = [];
        if (Array.isArray(message.tool_calls)) {
            for (const value of message.tool_calls) {
                if (!value || typeof value !== 'object') continue;
                const call = value as Record<string, unknown>;
                const fn = call.function;
                if (!fn || typeof fn !== 'object') continue;
                const { name, arguments: args } = fn as Record<string, unknown>;
                if (typeof name !== 'string') continue;
                toolCalls.push({
                    callID: typeof call.id === 'string' ? call.id : this.nextItemId('toolcall'),
                    name,
                    arguments:
                        args && typeof args === 'object' && !Array.isArray(args)
                            ? this.cloneSerializable(args as Record<string, unknown>)
                            : {},
                });
            }
        }
        return {
            thinking: typeof message.thinking === 'string' ? message.thinking : '',
            content: typeof message.content === 'string' ? message.content : '',
            toolCalls,
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

    private getModelFamily() {
        return getModelFamily(this.model.modelId, this.tryReadString(this.readModelConfig(), 'model_type'));
    }

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
