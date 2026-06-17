# @huggingface/transformers-agent

The idea of @huggingface/transformers-agent is to create an easy to use Agent SDK for Transformers.js.

Using Transformers.js feels a lot like using the a bare engine. Other libraries (like WebLLM) feel way easier to use. This is also feedback I got a lot when doing workshops and talks about Transformers.js text-generation.

## Why a new API?

My original idea was to create a Transformers.js implementation of the [Responses API](https://huggingface.co/docs/inference-providers/en/guides/responses-api). But the main problem is that it's built around the assumption of a **remote model behind an HTTP API**, and does not have a concept of a locally loaded, cacheable, device-bound model with a persistent KV cache.

### Responses API

https://huggingface.co/docs/inference-providers/en/guides/responses-api

```typescript
import OpenAI from 'openai';

const client = new OpenAI({
    baseURL: 'https://router.huggingface.co/v1',
    apiKey: process.env.HF_TOKEN,
});

const tools = [
    {
        type: 'function',
        name: 'get_weather',
        description: 'Get the current weather for a location',
        parameters: {
            type: 'object',
            properties: { location: { type: 'string', description: 'City name' } },
            required: ['location'],
        },
    },
];

async function getWeather({ location }) {
    return { temperature: 18, condition: 'cloudy' };
}

const response = await client.responses.create({
    model: 'moonshotai/Kimi-K2-Instruct-0905:groq',
    input: "What's the weather in Geneva?",
    tools,
    tool_choice: 'auto',
});

const call = response.output.find((item) => item.type === 'function_call');

if (call) {
    const args = JSON.parse(call.arguments);
    const result = await getWeather(args);

    // Manual loop: send a fresh request with the tool result appended
    const final = await client.responses.create({
        model: 'moonshotai/Kimi-K2-Instruct-0905:groq',
        previous_response_id: response.id,
        input: [
            {
                type: 'function_call_output',
                call_id: call.call_id,
                output: JSON.stringify(result),
            },
        ],
    });

    console.log(final.output_text);
}
```

**Why it doesn't fit:**
- `new OpenAI({ baseURL: "https://router.huggingface.co/v1", apiKey })` is built around routing a request to a *remote* provider (GPT, Gemini, etc.) — there's no concept of a local model object, device selection (`webgpu`/`wasm`), or a download/cache lifecycle. `model` is just a string id passed over the wire.
- For client-defined function tools, the agentic loop is still **manual**: the response comes back with a `function_call` output item, and you must execute it yourself, then send a *new* `responses.create()` call with a `function_call_output` item referencing the `call_id` — there's no `Agent.run()`-style loop for your own tools (only `mcp`-type remote tools are auto-executed server-side).
- Every call is stateless from the client's perspective — there's no KV cache to preserve; `input` either takes a full message array each time or relies on the server retaining state via a `previous_response_id`, which doesn't map onto a tensor-level KV cache living in your own browser tab.

## Related APIs and alignment choices

`@huggingface/transformers-agent` intentionally overlaps with existing agent/tool ecosystems, but it does not mirror any one API 1:1.

### AI SDK (tools + tool calling)

The AI SDK offers a broad, provider-agnostic tool-calling layer with many advanced orchestration features (for example strict mode, approval flows, dynamic tools, repair hooks, and extensive step controls).

That is excellent for multi-provider/cloud-first apps, but `transformers-agent` focuses on a different center of gravity:

- local model lifecycle (`isCached()`, `downloadSize()`, `init()` progress)
- direct runtime knobs (`modelId`, `device`, `dtype`)
- cache-aware agent ergonomics for iterative on-device runs

In short, AI SDK optimizes for maximum backend flexibility, while `transformers-agent` optimizes for Transformers.js local inference constraints and UX.

### Chrome Prompt API

Chrome's Prompt API is a browser-native session API for Gemini Nano with built-in availability checks, hardware/storage constraints, and multimodal prompting.

It is a great fit when your target is specifically Chrome built-in AI, but it is not a portable Transformers.js abstraction across browser + Node.js runtimes, and it does not expose the same model/runtime controls (`modelId`/`dtype`/`device`) used in Transformers.js workflows.

### Why I do not fully align 1:1

I align where it helps interoperability, especially at the tool contract layer:

- tools follow the W3C WebMCP `ModelContextTool` shape (compatible with `navigator.modelContext.registerTool()`)

But I intentionally keep a Transformers.js-first API because on-device products are dominated by concerns that generic agent APIs do not prioritize:

- download and cache UX before first run
- deterministic local performance tuning
- preserving KV cache validity across turns
- runtime portability between browser and Node.js

## Goal

The goal is to make running an agent with Transformers.js feel like using a product-level SDK, not like assembling an inference engine by hand.

Transformers.js exposes powerful local-model controls, but building an agent still requires repeated glue code: model loading, cache checks, message formatting, tool schema handling, tool execution, loop orchestration, and conversation state.

`@huggingface/transformers-agent` should provide that missing layer:

- a `Model` object that makes the local model lifecycle explicit: model id, device, dtype, cache checks, download size, progress, and initialization
- an `Agent` object that owns the system prompt, tools, history, and KV cache so repeated turns can stay efficient
- a tool API that can be adapted into the function schema format expected by model chat templates
- a built-in agent loop for local tools, so callers can use `agent.run()` or `agent.stream()` instead of manually detecting tool calls and sending follow-up prompts
- browser and Node.js support without assuming a hosted model, API key, server-side session, or remote provider

The library is not trying to replace Transformers.js or hide that inference is happening locally. It should expose the local-model constraints that matter for user experience and make them easy to build around.

In short: make the simple path simple, keep the local controls visible, and provide enough structure that people can build useful local agents without rewriting the same orchestration loop every time.

## Examples

### Minimal

```typescript
const model = new Model({
  modelId: "onnx-community/gemma-4-E2B-it-ONNX",
  device: "webgpu",
  dtype: "q4f16",
});

await model.init(console.log);

const agent = new Agent({
  model,
  system: "You are a helpful research assistant.",
});

const response = await agent.run("Who are you?");
```

### Streaming

```typescript
const model = new Model({
  modelId: "onnx-community/gemma-4-E2B-it-ONNX",
  device: "webgpu",
  dtype: "q4f16",
});

await model.init(console.log);

const agent = new Agent({
  model,
  system: "You are a helpful research assistant.",
});

for await (const chunk of agent.stream("Who are you?")) {
  console.log(chunk.runs.at(-1).text);
}
```

### Tools
```typescript
const model = new Model({
  modelId: "onnx-community/gemma-4-E2B-it-ONNX",
  device: "webgpu",
  dtype: "q4f16",
});

await model.init(console.log);

const getWeatherTool = new Tool<{ location: string; unit: string }>({
  name: "get_weather",
  title: "Get weather",
  description: "Get current weather information for a location",
  parameters: {
    location: Tool.string({
      description: "he city and state, e.g. San Francisco, CA",
    }),
    unit: Tool.string({
      description: "The unit of temperature to use",
      enum: ["celsius", "fahrenheit"],
    }),
  },
  required: ["location"],
  execute: async ({ location, unit }) => {
    return {
      content: [
        {
          type: "text",
          text: `The weather in ${location} in Sunny, 20 degrees ${unit ?? "celsius"}.`,
        },
      ],
    };
  },
});

const agent = new Agent({
  model,
  system: "You are a helpful AI assistant.",
  tools: [getWeatherTool],
});

const result = await agent.run("Whats the weather in London?");
console.log(result.runs.at(-1).text);
```
