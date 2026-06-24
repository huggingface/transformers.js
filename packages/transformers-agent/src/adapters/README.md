# Model Adapters

`ModelAdapter` contains the model-specific details that the generic `Agent` loop should not know about.

Add a new adapter when a model family differs in any of these areas:

- chat-template message shape
- tool schema shape
- tool-call syntax in generated text
- special/control tokens that should be hidden from users
- thinking/reasoning text extraction
- KV-cache compatibility

## Contract

Implement `ModelAdapter` from `./types`.

```ts
export interface ModelAdapter {
    readonly id: string;
    supports(context: ModelAdapterContext): boolean;
    formatTools(tools: ToolList): Array<Record<string, unknown>>;
    formatMessages(messages: ReadonlyArray<Message>): Array<Record<string, unknown>>;
    normalizeAssistantContent(content: string): string;
    parseAssistantContent(content: string, nextId: (prefix: string) => string): ParseResult;
    useKvCache(enableThinking: boolean): boolean;
}
```

## Responsibilities

`supports(context)` decides whether the adapter applies to a loaded model. Use stable model config values first, such as `model_type`, and fall back to model id patterns only when needed.

`formatTools(tools)` converts `Tool` definitions into the shape expected by the model chat template.

`formatMessages(messages)` converts normalized agent messages into the exact chat-template message structure expected by the model. For example, Gemma 4 combines an assistant tool-call message and the following tool responses into one assistant message containing `tool_calls` and `tool_responses`.

`normalizeAssistantContent(content)` removes model-specific control tokens from generated assistant text. Do this in the adapter, not in `Agent.ts`.

`parseAssistantContent(content, nextId)` extracts visible text, thinking text, and tool calls from raw generated text. It should tolerate partial streaming chunks and avoid exposing incomplete tool calls as visible text.

`useKvCache(enableThinking)` reports whether the adapter can safely reuse the model KV cache across loop steps.

## Adding A New Adapter

1. Extend `ModelAdapterBase` unless the model needs a completely different implementation.
2. Override only the methods where the model differs.
3. Add tests for complete tool calls, partial streaming tool calls, plain answers with control tokens, and thinking text if the model supports it.
4. Export the adapter from `src/adapters/index.ts`.
5. Add it to `ModelAdapterRegistry` if it should be auto-selected.

Custom adapters can also be passed directly:

```ts
const agent = new Agent({
    model,
    adapter: new MyModelAdapter(),
    initialPrompts: [{ role: 'system', content: 'You are helpful.' }],
});
```
