/**
 * @file Type definitions for @huggingface/transformers-agent
 *
 * A high-level agent SDK built on top of @huggingface/transformers,
 * designed around the KV cache model: system prompt and tools are
 * fixed at agent construction time, only user input varies per turn.
 *
 * Tool definitions follow the W3C WebMCP ModelContextTool interface
 * (https://webmachinelearning.github.io/webmcp/), using JSON Schema
 * for input definitions and a structured content array for results —
 * the same shape used by MCP servers and the WebMCP browser API.
 */

// ---------------------------------------------------------------------------
// Re-exports from @huggingface/transformers
// ---------------------------------------------------------------------------

export type {
    DeviceType,
    DataType,
    ProgressInfo,
    ProgressCallback,
    InitiateProgressInfo,
    DownloadProgressInfo,
    ProgressStatusInfo,
    DoneProgressInfo,
    ReadyProgressInfo,
} from '@huggingface/transformers';

// ---------------------------------------------------------------------------
// JSON Schema (subset used by WebMCP / MCP inputSchema)
// ---------------------------------------------------------------------------

/** A JSON Schema v7 object describing a tool's input parameters. */
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

// ---------------------------------------------------------------------------
// WebMCP tool result content types
// ---------------------------------------------------------------------------
// Mirrors the MCP / WebMCP content block shape returned by tool execute().

export interface TextContent {
    type: 'text';
    text: string;
}

export interface ImageContent {
    type: 'image';
    /** Base64-encoded image data. */
    data: string;
    mimeType: 'image/png' | 'image/jpeg' | 'image/gif' | 'image/webp';
}

/** Structured data returned alongside text (WebMCP outputSchema extension). */
export interface StructuredContent {
    type: 'structured';
    data: Record<string, unknown>;
}

export type ContentBlock = TextContent | ImageContent | StructuredContent;

/**
 * The return value of a tool's `execute` function.
 * Matches the WebMCP / MCP tool result shape.
 *
 * @example
 * execute: async ({ query }) => ({
 *   content: [{ type: 'text', text: JSON.stringify(results) }]
 * })
 */
export interface ToolCallOutput {
    content: ContentBlock[];
    /**
     * Set to `true` if the tool encountered an error.
     * The model will see the content as an error message and can decide
     * how to handle it (retry, report to user, etc.).
     */
    isError?: boolean;
}

// ---------------------------------------------------------------------------
// Tool definition — WebMCP ModelContextTool shape
// ---------------------------------------------------------------------------

/**
 * A tool definition following the W3C WebMCP `ModelContextTool` interface.
 *
 * The shape is intentionally compatible with `navigator.modelContext.registerTool()`,
 * meaning tools written for the browser WebMCP API can be reused here directly.
 *
 * @example
 * const searchWeb: Tool = {
 *   description: 'Search the web for current information',
 *   inputSchema: {
 *     type: 'object',
 *     properties: {
 *       query: { type: 'string', description: 'The search query' },
 *     },
 *     required: ['query'],
 *   },
 *   execute: async ({ query }) => ({
 *     content: [{ type: 'text', text: await fetchSearchResults(query as string) }],
 *   }),
 * };
 */
export interface Tool {
    /** Human-readable description passed to the model as part of the tool schema. */
    description: string;

    /**
     * JSON Schema v7 describing the tool's input parameters.
     * Must be a top-level object schema. Sent verbatim to the model.
     */
    inputSchema: JSONSchemaObject;

    /**
     * The function invoked when the model calls this tool.
     * Receives parsed arguments conforming to `inputSchema`.
     * Returns a WebMCP-compatible content array.
     */
    execute: (args: Record<string, unknown>) => Promise<ToolCallOutput>;
}

/**
 * A map of named tools. Keys become the tool names visible to the model.
 *
 * @example
 * const tools: ToolMap = {
 *   searchWeb: { description: '...', inputSchema: { ... }, execute: async (...) => ... },
 *   readUrl:   { description: '...', inputSchema: { ... }, execute: async (...) => ... },
 * };
 */
export type ToolMap = Record<string, Tool>;

// ---------------------------------------------------------------------------
// Tool call events (emitted during streaming and stored in step results)
// ---------------------------------------------------------------------------

/**
 * A tool invocation emitted by the model during generation.
 */
export interface ToolCall {
    /** Matches a key in the `ToolMap` provided to the agent. */
    name: string;
    /** Parsed arguments conforming to the tool's `inputSchema`. */
    args: Record<string, unknown>;
    /** Opaque ID assigned by the model for this specific invocation. */
    id: string;
}

/**
 * A completed tool call, including the output returned by `execute`.
 */
export interface ToolCallResult extends ToolCall {
    output: ToolCallOutput;
    /** Duration of the `execute` call in milliseconds. */
    durationMs: number;
}

// ---------------------------------------------------------------------------
// Streaming chunk types
// ---------------------------------------------------------------------------

/** A token or text fragment has been generated. */
export interface TextDeltaChunk {
    type: 'text.delta';
    delta: string;
}

/**
 * The model has started invoking a tool.
 * Arguments are fully resolved at this point (not streamed).
 */
export interface ToolStartChunk {
    type: 'tool.start';
    name: string;
    args: Record<string, unknown>;
    /** Matches ToolCall.id for correlation with the subsequent tool.result chunk. */
    id: string;
}

/** A tool's `execute` function has returned. */
export interface ToolResultChunk {
    type: 'tool.result';
    name: string;
    output: ToolCallOutput;
    /** Matches ToolCall.id. */
    id: string;
    durationMs: number;
}

/** One full reasoning + (optional) tool loop step has completed. */
export interface StepDoneChunk {
    type: 'step.done';
    stepIndex: number;
    /** All tool calls made in this step. */
    toolCalls: ToolCallResult[];
    /** Partial text generated in this step. Empty if the step only called tools. */
    text: string;
    usage: StepUsage;
}

/**
 * The agent has finished — all steps done, no further tool calls.
 * Always the final chunk emitted by `agent.stream()`.
 */
export interface DoneChunk {
    type: 'done';
    /** Concatenated final assistant text across all steps. */
    text: string;
    steps: StepResult[];
    usage: TotalUsage;
}

/**
 * Discriminated union of all possible stream chunks.
 * Use a `switch` on `chunk.type` for full TypeScript type narrowing.
 *
 * @example
 * for await (const chunk of agent.stream('Hello')) {
 *   switch (chunk.type) {
 *     case 'text.delta':  ui.append(chunk.delta); break;
 *     case 'tool.start':  ui.showTool(chunk.name, chunk.args); break;
 *     case 'tool.result': ui.showResult(chunk.output); break;
 *     case 'step.done':   ui.markStep(chunk.stepIndex); break;
 *     case 'done':        ui.finalize(chunk.text, chunk.usage); break;
 *   }
 * }
 */
export type StreamChunk = TextDeltaChunk | ToolStartChunk | ToolResultChunk | StepDoneChunk | DoneChunk;

// ---------------------------------------------------------------------------
// Usage / token counts
// ---------------------------------------------------------------------------

export interface StepUsage {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
}

export interface TotalUsage {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
    /** Number of agentic loop iterations executed. */
    steps: number;
}

// ---------------------------------------------------------------------------
// Step and run results
// ---------------------------------------------------------------------------

export interface StepResult {
    stepIndex: number;
    /** Assistant text in this step. Empty if the step only made tool calls. */
    text: string;
    toolCalls: ToolCallResult[];
    usage: StepUsage;
}

export interface RunResult {
    /** Concatenated final assistant text across all steps. */
    text: string;
    steps: StepResult[];
    usage: TotalUsage;
}

// ---------------------------------------------------------------------------
// Conversation message types
// ---------------------------------------------------------------------------

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
    /**
     * Serialised ToolCallOutput.content passed back to the model.
     * Stored as a string so the history stays JSON-serialisable.
     */
    content: string;
}

export type Message = SystemMessage | UserMessage | AssistantMessage | ToolMessage;

// ---------------------------------------------------------------------------
// Lifecycle hook types
// ---------------------------------------------------------------------------

export type BeforeToolCallHook = (call: ToolCall) => void | Promise<void>;
export type AfterToolCallHook = (call: ToolCall, output: ToolCallOutput, durationMs: number) => void | Promise<void>;
export type OnStepHook = (step: StepResult) => void | Promise<void>;

/** Calling this removes the previously registered hook. */
export type Unsubscribe = () => void;

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

import type { DataType, DeviceType, ProgressCallback } from '@huggingface/transformers';

/**
 * Configuration provided at `Model` construction time.
 * No download or initialization happens here.
 */
export interface ModelConfig {
    modelId: string;
    device?: DeviceType;
    /** Quantization dtype, or a per-module dtype map for encoder-decoder models. */
    dtype?: DataType | Record<string, DataType>;
}

/**
 * Represents a model that may or may not be initialized.
 *
 * Introspection methods (`isCached`, `downloadSize`, `cachedSize`) are
 * available before `init()` and do not trigger a download — useful for
 * showing a confirmation dialog before committing to a large download.
 *
 * Only an initialized `Model` can be passed to `Agent`.
 *
 * @example
 * const model = new Model({ modelId: 'onnx-community/Qwen2.5-0.5B-Instruct', device: 'webgpu', dtype: 'q4' });
 *
 * if (!await model.isCached()) {
 *   const size = await model.downloadSize();
 *   const ok = await ui.confirm(`Download ${formatBytes(size)}?`);
 *   if (!ok) return;
 * }
 *
 * await model.init((info) => ui.updateProgress(info));
 *
 * const agent = new Agent({ model, system: '...' });
 */
export declare class Model {
    constructor(config: ModelConfig);

    readonly modelId: string;
    readonly device: DeviceType;
    readonly dtype: DataType | Record<string, DataType>;

    /** `true` once `init()` has completed successfully. */
    readonly isInitialized: boolean;

    // --- Pre-init introspection (no download required) ---

    /** Returns `true` if all model files are already in the local cache. */
    isCached(): Promise<boolean>;

    /** Bytes still needed to complete the download. `0` if fully cached. */
    downloadSize(): Promise<number>;

    /** Bytes already present in the local cache for this model. */
    cachedSize(): Promise<number>;

    // --- Initialization ---

    /**
     * Downloads (if needed) and initializes the model.
     * Must be called before passing this instance to `Agent`.
     *
     * @throws {Error} If initialization fails (network error, unsupported device, …)
     */
    init(progressCallback?: ProgressCallback): Promise<void>;

    // --- Static factory (when pre-init introspection is not needed) ---

    /**
     * Constructs a Model, calls `init()`, and returns the initialized instance.
     * Equivalent to `new Model(config)` followed by `await model.init(cb)`.
     */
    static load(config: ModelConfig, progressCallback?: ProgressCallback): Promise<Model>;

    // --- Agent factory shorthand ---

    /**
     * Creates an `Agent` backed by this model.
     * Shorthand for `new Agent({ model: this, ...options })`.
     *
     * @throws {Error} If the model has not been initialized.
     */
    agent(options: Omit<AgentConfig, 'model'>): Agent;
}

// ---------------------------------------------------------------------------
// Agent
// ---------------------------------------------------------------------------

/**
 * Configuration provided at `Agent` construction time.
 * All cache-relevant parameters live here and cannot change after construction.
 */
export interface AgentConfig {
    /** An initialized `Model` instance. */
    model: Model;

    /**
     * System prompt. Fixed for the lifetime of the agent (KV cache constraint).
     */
    system: string;

    /**
     * Named tools available to the model. Keys become tool names.
     * Each tool's `execute` is called automatically during the agentic loop.
     * Fixed for the lifetime of the agent (KV cache constraint).
     */
    tools?: ToolMap;

    /**
     * Maximum number of agentic loop iterations.
     * Prevents runaway loops. Defaults to `10`.
     */
    maxSteps?: number;

    /**
     * Maximum number of new tokens generated per step.
     * Maps to `max_new_tokens` in the underlying generation config.
     */
    maxNewTokens?: number;
}

/**
 * A stateful agent that maintains conversation history and KV cache across turns.
 *
 * The agent owns three immutable pieces of state:
 * - `system`  — the system prompt
 * - `tools`   — the tool registry
 * - `history` — append-only conversation log (readable, resettable)
 *
 * Only `input` (a plain string) varies per turn.
 *
 * @example
 * const agent = new Agent({
 *   model,
 *   system: 'You are a research assistant.',
 *   tools: { searchWeb, readUrl },
 *   maxSteps: 10,
 * });
 *
 * agent
 *   .onBeforeToolCall((call) => logger.info('calling tool', call.name))
 *   .onAfterToolCall((call, output) => metrics.record(call.name));
 *
 * const result = await agent.run('What are the latest Transformers.js updates?');
 * console.log(result.text, result.steps, result.usage);
 *
 * for await (const chunk of agent.stream('Compare with TensorFlow.js')) {
 *   switch (chunk.type) {
 *     case 'text.delta':  ui.append(chunk.delta); break;
 *     case 'tool.start':  ui.showTool(chunk.name); break;
 *     case 'tool.result': ui.showResult(chunk.output); break;
 *     case 'done':        ui.finalize(chunk.text); break;
 *   }
 * }
 */
export declare class Agent {
    constructor(config: AgentConfig);

    readonly system: string;
    readonly tools: ToolMap;

    /**
     * Read-only snapshot of the current conversation history.
     * Do not mutate — use `clearHistory()` to reset.
     */
    readonly history: ReadonlyArray<Message>;

    // --- Generation ---

    /**
     * Run the agent and await the full response.
     * Executes the agentic loop (model → tools → model → …) until
     * the model stops invoking tools or `maxSteps` is reached.
     *
     * @throws {Error} If the model has not been initialized.
     */
    run(input: string): Promise<RunResult>;

    /**
     * Run the agent and stream chunks as they are produced.
     * Yields `TextDeltaChunk`, `ToolStartChunk`, `ToolResultChunk`,
     * `StepDoneChunk`, and finally a `DoneChunk`.
     *
     * History is updated identically to `run()`.
     */
    stream(input: string): AsyncIterable<StreamChunk>;

    // --- History management ---

    /**
     * Clears conversation history and invalidates the KV cache.
     * The next `run()` / `stream()` call starts a fresh conversation.
     */
    clearHistory(): void;

    // --- Lifecycle hooks (chainable) ---

    /**
     * Registers a hook called immediately before each tool's `execute`.
     * Supports method chaining.
     */
    onBeforeToolCall(hook: BeforeToolCallHook): this;

    /**
     * Registers a hook called immediately after each tool's `execute` returns.
     * Supports method chaining.
     */
    onAfterToolCall(hook: AfterToolCallHook): this;

    /**
     * Registers a hook called after each complete agentic step.
     * Useful for showing intermediate progress inside a `run()` call.
     * Supports method chaining.
     */
    onStep(hook: OnStepHook): this;
}
