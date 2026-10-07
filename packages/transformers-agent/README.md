# @huggingface/transformers-agent

Run local language models in the browser and Node.js with a Prompt-API-inspired session interface powered by [Transformers.js](https://github.com/huggingface/transformers.js).

```bash
npm install @huggingface/transformers-agent
```

## Model

```ts
import { Model } from "@huggingface/transformers-agent";

const model = await Model.load(
  {
    modelId: "onnx-community/gemma-4-E2B-it-ONNX",
    device: "webgpu",
    dtype: "q4f16",
  },
  (info) => info.status === "progress_total" && console.log(info.progress),
);
```

Use `new Model(config)` with `isCached()`, `downloadSize()`, `cachedSize()`, and `init()` when the application needs to manage download consent and progress separately.

## Agent

An `Agent` owns a serializable message history and the model's session state. Each call to `prompt()` performs exactly one model turn.

```ts
import { Agent } from "@huggingface/transformers-agent";

const agent = new Agent({
  model,
  initialPrompts: [
    { role: "system", content: "You are a helpful research assistant." },
  ],
  enableThinking: true,
});

const result = await agent.prompt("Who are you?");

for (const part of result) {
  if (part.type === "thinking") console.log(part.value);
  if (part.type === "text") console.log(part.value);
  if (part.type === "tool-call") console.log(part.value);
}

console.log(agent.getLatestUsage());
```

`prompt()` returns a `Promise<LanguageModelMessageContent[]>`. Model reasoning is returned as `{ type: "thinking", value }`, separately from visible `{ type: "text", value }` output and tool calls. After a turn completes, `getLatestUsage()` returns its token counts, `tokensPerSecond`, `timeToFirstTokenMs`, `generationTimeMs`, and `totalTimeMs`. TPS measures completion tokens over model generation time; total time also includes local prompt preparation and response parsing. It returns `null` before the first completed turn and after `clearHistory()`.

### Streaming

`promptStreaming()` returns a `ReadableStream<LanguageModelMessageContent>`. Thinking and text chunks are incremental deltas, not cumulative snapshots. Structured output such as a tool call is emitted as an individual content chunk. The stream closes when the turn completes; there is no `done` chunk.

```ts
let response = "";
let thinking = "";

for await (const chunk of agent.promptStreaming("Explain WebGPU briefly.")) {
  if (chunk.type === "thinking") {
    thinking += chunk.value;
    renderThinking(thinking);
  } else if (chunk.type === "text") {
    response += chunk.value;
    renderResponse(response);
  } else if (chunk.type === "tool-call") {
    console.log(chunk.value);
  }
}

console.log(agent.getLatestUsage());
```

### History

String prompts are appended as user messages. Message arrays can contain system, user, or assistant messages and are also appended before generation.

```ts
console.log(agent.history);
agent.clearHistory();
```

The message shape follows Chromium's Prompt API direction: content is either a string or typed text, image, audio, tool-call, and tool-response parts. Text and tool content are supported; unsupported multimodal content fails explicitly.

## Open-Loop Tools

`Tool` combines the declaration sent to the model with a typed application-owned `execute` function. `Agent` reads only the declaration fields: it never calls `execute` and never starts a follow-up model turn automatically.

```ts
import { Agent, Tool } from "@huggingface/transformers-agent";

const getWeatherTool = new Tool<{ location: string }>({
  name: "get_weather",
  description: "Get current weather for a location.",
  parameters: {
    location: Tool.string({ description: "City name" }),
  },
  execute: async ({ location }) => [
    {
      type: "object",
      value: await getWeather(location),
    },
  ],
});

const agent = new Agent({
  model,
  tools: [getWeatherTool],
  initialPrompts: [{ role: "system", content: "You are a concise assistant." }],
  enableThinking: true,
});

const first = await agent.prompt("What's the weather in London?");
const call = first.find((part) => part.type === "tool-call")?.value;

if (call?.name === "get_weather") {
  // prompt() returned the call without executing it. The application chooses
  // whether and when to invoke the previously declared tool.
  const result = await getWeatherTool.execute(
    call.arguments as { location: string },
  );

  const final = await agent.prompt([
    {
      role: "user",
      content: [
        {
          type: "tool-response",
          value: {
            callID: call.callID,
            name: call.name,
            result,
          },
        },
      ],
    },
  ]);

  console.log(final.find((part) => part.type === "text")?.value);
}
```

Failed executions use the same application-managed flow:

```ts
await agent.prompt([
  {
    role: "user",
    content: [
      {
        type: "tool-response",
        value: {
          callID: call.callID,
          name: call.name,
          errorMessage: "Weather service unavailable",
        },
      },
    ],
  },
]);
```

A tool-response message must contain only tool-response parts. The SDK validates its `callID` and name against an unresolved tool call in session history before prompting the model.

This open loop keeps tool execution inspectable, makes every request one prompt and one response, and leaves approval, retries, parallelism, and multi-turn orchestration in application code.
