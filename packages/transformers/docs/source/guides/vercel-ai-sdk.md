# Using Transformers.js with the Vercel AI SDK

[Vercel AI SDK](https://ai-sdk.dev/) is a popular toolkit for building AI-powered applications. With [`@browser-ai/transformers-js`](https://www.browser-ai.dev/docs/ai-sdk-v6/transformers-js), you can use Transformers.js as a model provider for the AI SDK, enabling in-browser (and server-side) inference with a clean, declarative API.

This guide covers the core concepts and API patterns. For a full step-by-step project walkthrough, see the [Building a Next.js AI Chatbot](../tutorials/next-ai-sdk) tutorial.

## Why use the Vercel AI SDK with Transformers.js?

Building functional and production-ready applications using Hugging Face models in the browser with just `@huggingface/transformers` requires significant boilerplate: Web Worker setup, message passing, progress tracking, streaming, interrupt handling, error states, and state management. The `@browser-ai/transformers-js` provider abstracts all of this away, letting you use the same `streamText`, `generateText`, and `useChat` APIs you'd use with any other AI SDK provider. Read more about this [here](https://www.browser-ai.dev/docs/ai-sdk-v6/transformers-js/why).

## Installation

```bash
npm install @browser-ai/transformers-js @huggingface/transformers ai @ai-sdk/react
```

| @browser-ai/transformers-js | AI SDK | Notes |
|---|---|---|
| v2.0.0+ | v6.x | Current stable |
| v1.0.0 | v5.x | Legacy |

## Text generation

### Streaming text

```js
import { streamText } from "ai";
import { transformersJS } from "@browser-ai/transformers-js";

const result = streamText({
  model: transformersJS("HuggingFaceTB/SmolLM2-360M-Instruct"),
  prompt: "Invent a new holiday and describe its traditions.",
});

for await (const textPart of result.textStream) {
  console.log(textPart);
}
```

### Non-streaming text

```js
import { generateText } from "ai";
import { transformersJS } from "@browser-ai/transformers-js";

const result = await generateText({
  model: transformersJS("HuggingFaceTB/SmolLM2-360M-Instruct"),
  prompt: "Invent a new holiday and describe its traditions.",
});
console.log(result.text);
```

## Text embeddings

```js
import { embed, embedMany } from "ai";
import { transformersJS } from "@browser-ai/transformers-js";

// Single embedding
const { embedding } = await embed({
  model: transformersJS.embedding("Supabase/gte-small"),
  value: "Hello, world!",
});

// Multiple embeddings
const { embeddings } = await embedMany({
  model: transformersJS.embedding("Supabase/gte-small"),
  values: ["Hello", "World", "AI"],
});
```

## Audio transcription

```js
import { experimental_transcribe as transcribe } from "ai";
import { transformersJS } from "@browser-ai/transformers-js";

const transcript = await transcribe({
  model: transformersJS.transcription("Xenova/whisper-base"),
  audio: audioFile,
});
console.log(transcript.text);
console.log(transcript.segments); // segments with timestamps
```

## Vision models

```js
import { streamText } from "ai";
import { transformersJS } from "@browser-ai/transformers-js";

const result = streamText({
  model: transformersJS("HuggingFaceTB/SmolVLM-256M-Instruct", {
    isVisionModel: true,
    device: "webgpu",
  }),
  messages: [
    {
      role: "user",
      content: [
        { type: "text", text: "Describe this image" },
        { type: "image", image: someImageBlobOrUrl },
      ],
    },
  ],
});

for await (const chunk of result.textStream) {
  console.log(chunk);
}
```

## Web Worker offloading

For better performance, run model inference off the main thread with a Web Worker.

**1. Create `worker.ts`:**

```typescript
import { TransformersJSWorkerHandler } from "@browser-ai/transformers-js";

const handler = new TransformersJSWorkerHandler();
self.onmessage = (msg: MessageEvent) => {
  handler.onmessage(msg);
};
```

**2. Pass the worker when creating the model:**

```js
import { streamText } from "ai";
import { transformersJS } from "@browser-ai/transformers-js";

const model = transformersJS("HuggingFaceTB/SmolLM2-360M-Instruct", {
  device: "webgpu",
  worker: new Worker(new URL("./worker.ts", import.meta.url), {
    type: "module",
  }),
});

const result = streamText({
  model,
  messages: [{ role: "user", content: "Hello!" }],
});
```

## Download progress tracking

Models are downloaded on first use. Track progress to provide a better UX:

```js
import { streamText } from "ai";
import { transformersJS } from "@browser-ai/transformers-js";

const model = transformersJS("HuggingFaceTB/SmolLM2-360M-Instruct");
const availability = await model.availability();

if (availability === "unavailable") {
  console.log("Browser doesn't support Transformers.js");
} else if (availability === "downloadable") {
  await model.createSessionWithProgress(({ progress }) => {
    console.log(`Download progress: ${Math.round(progress * 100)}%`);
  });
}

// Model is ready
const result = streamText({ model, prompt: "Hello!" });
```

## Tool calling

<Tip>

For best tool calling results, use reasoning models like Qwen3 which handle multi-step reasoning well.

</Tip>

```js
import { streamText, tool, stepCountIs } from "ai";
import { transformersJS } from "@browser-ai/transformers-js";
import { z } from "zod";

const result = await streamText({
  model: transformersJS("onnx-community/Qwen3-0.6B-ONNX"),
  messages: [{ role: "user", content: "What's the weather in San Francisco?" }],
  tools: {
    weather: tool({
      description: "Get the weather in a location",
      inputSchema: z.object({
        location: z.string().describe("The location to get the weather for"),
      }),
      execute: async ({ location }) => ({
        location,
        temperature: 72 + Math.floor(Math.random() * 21) - 10,
      }),
    }),
  },
  stopWhen: stepCountIs(5),
});
```

Tool calling also supports [tool execution approval (`needsApproval`)](https://ai-sdk.dev/docs/ai-sdk-core/tools-and-tool-calling#tool-execution-approval) for human-in-the-loop workflows.

## `useChat` with custom transport

When using the `useChat` hook, you create a [custom transport](https://ai-sdk.dev/docs/ai-sdk-ui/transport) to handle client-side inference. Here's a minimal example:

```typescript
import {
  ChatTransport, UIMessageChunk, streamText,
  convertToModelMessages, ChatRequestOptions,
} from "ai";
import {
  TransformersJSLanguageModel,
  TransformersUIMessage,
} from "@browser-ai/transformers-js";

export class TransformersChatTransport
  implements ChatTransport<TransformersUIMessage>
{
  constructor(private readonly model: TransformersJSLanguageModel) {}

  async sendMessages(
    options: {
      chatId: string;
      messages: TransformersUIMessage[];
      abortSignal: AbortSignal | undefined;
    } & {
      trigger: "submit-message" | "submit-tool-result" | "regenerate-message";
      messageId: string | undefined;
    } & ChatRequestOptions,
  ): Promise<ReadableStream<UIMessageChunk>> {
    const prompt = await convertToModelMessages(options.messages);
    const result = streamText({
      model: this.model,
      messages: prompt,
      abortSignal: options.abortSignal,
    });
    return result.toUIMessageStream();
  }

  async reconnectToStream(): Promise<ReadableStream<UIMessageChunk> | null> {
    return null; // client-side AI doesn't support stream reconnection
  }
}
```

Then use it in your component:

```typescript
import { useChat } from "@ai-sdk/react";
import { transformersJS, TransformersUIMessage } from "@browser-ai/transformers-js";

const model = transformersJS("HuggingFaceTB/SmolLM2-360M-Instruct", {
  device: "webgpu",
  worker: new Worker(new URL("./worker.ts", import.meta.url), { type: "module" }),
});

const { sendMessage, messages, stop } = useChat<TransformersUIMessage>({
  transport: new TransformersChatTransport(model),
});
```

## Browser compatibility fallback

If the device doesn't support in-browser inference, you can fall back to a server-side model:

```typescript
import {
  transformersJS, TransformersUIMessage,
  doesBrowserSupportTransformersJS,
} from "@browser-ai/transformers-js";

const { sendMessage, messages, stop } = useChat<TransformersUIMessage>({
  transport: doesBrowserSupportTransformersJS()
    ? new TransformersChatTransport(model)
    : new DefaultChatTransport({ api: "/api/chat" }),
});
```

## Further reading

- [Building a Next.js AI Chatbot](../tutorials/next-ai-sdk) &mdash; a step-by-step tutorial building a full chatbot with tool calling
- [`@browser-ai/transformers-js` documentation](https://www.browser-ai.dev/docs/ai-sdk-v6/transformers-js)
- [Vercel AI SDK documentation](https://ai-sdk.dev/)
