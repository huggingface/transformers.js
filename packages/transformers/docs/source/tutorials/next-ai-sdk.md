# Building a Next.js AI Chatbot with Vercel AI SDK

In this tutorial, we'll build an in-browser AI chatbot using Next.js, Transformers.js, and the Vercel AI SDK v6. The chatbot runs entirely client-side with WebGPU acceleration &mdash; and supports tool calling with human approval.

Useful links:
- [Source code](https://github.com/huggingface/transformers.js-examples/tree/main/next-vercel-ai-sdk-v6-tool-calling)
- [`@browser-ai/transformers-js` docs](https://www.browser-ai.dev/docs/ai-sdk-v6/transformers-js)
- [Vercel AI SDK docs](https://ai-sdk.dev/)

## Prerequisites

- [Node.js](https://nodejs.org/en/) version 18+
- [npm](https://www.npmjs.com/) version 9+
- A browser with WebGPU support (Chrome 113+, Edge 113+, or Firefox/Safari with flags enabled)

## Step 1: Create the project

Create a new Next.js application:

```bash
npx create-next-app@latest next-ai-chatbot
cd next-ai-chatbot
```

Install the AI and Transformers.js dependencies:

```bash
npm install ai @ai-sdk/react @browser-ai/transformers-js @huggingface/transformers zod
```

## Step 2: Configure Next.js for browser inference

Transformers.js uses ONNX Runtime under the hood for both browser and server-side (Node.js) inference.  In our case we only need the browser runtime so we can tell Next.js to exclude the Node.js-specific packages when bundling for the browser. Update `next.config.ts`

```typescript
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export", // optional: export as a static site
  webpack: (config) => {
    config.resolve.alias = {
      ...config.resolve.alias,
      sharp$: false,
      "onnxruntime-node$": false,
    };
    return config;
  },
};

export default nextConfig;
```

## Step 3: Create the Web Worker

Running model inference on the main thread would block the UI. The `@browser-ai/transformers-js` package provides a ready-made worker handler that handles all the complexity for you.

Create `src/app/worker.ts`:

```typescript
import { TransformersJSWorkerHandler } from "@browser-ai/transformers-js";

const handler = new TransformersJSWorkerHandler();
self.onmessage = (msg: MessageEvent) => {
  handler.onmessage(msg);
};
```

That's it &mdash; the handler takes care of model loading, inference, streaming, and communication with the main thread.

## Step 4: Define the model configuration

Create `src/app/models.ts` to define which models are available. These are ONNX-format models from Hugging Face:

```typescript
import { WorkerLoadOptions } from "@browser-ai/transformers-js";

export interface ModelConfig extends Omit<WorkerLoadOptions, "modelId"> {
  id: string;
  name: string;
  supportsWorker?: boolean;
}

export const MODELS: ModelConfig[] = [
  {
    id: "onnx-community/Qwen3-0.6B-ONNX",
    name: "Qwen3 0.6B",
    device: "webgpu",
    dtype: "q4f16",
    supportsWorker: true,
  },
  {
    id: "onnx-community/granite-4.0-350m-ONNX-web",
    name: "Granite 4.0 350M",
    device: "webgpu",
    dtype: "fp16",
    supportsWorker: true,
  },
];
```

<Tip>

For tool calling, use reasoning models like Qwen3 which handle multi-step reasoning well, or fine-tuned model specifically for tool-calling capabilities. The `supportsWorker` flag controls whether the model is loaded in a Web Worker for better performance.

</Tip>

## Step 5: Define tools

Create `src/app/tools.ts` with tools the model can call. Each tool uses [Zod](https://zod.dev/) for input validation:

```typescript
import { tool } from "ai";
import z from "zod";

export const createTools = () => ({
  getCurrentTime: tool({
    description: "Get the current date and time.",
    inputSchema: z.object({}),
    execute: async () => {
      const now = new Date();
      return {
        timestamp: now.toISOString(),
        date: now.toLocaleDateString("en-US", {
          weekday: "long", year: "numeric", month: "long", day: "numeric",
        }),
        time: now.toLocaleTimeString("en-US", {
          hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: true,
        }),
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      };
    },
  }),
  randomNumber: tool({
    description: "Generate a random integer between min and max (inclusive).",
    inputSchema: z.object({
      min: z.number().describe("The minimum value (inclusive)"),
      max: z.number().describe("The maximum value (inclusive)"),
    }),
    execute: async ({ min, max }) => {
      return Math.floor(Math.random() * (Math.floor(max) - Math.ceil(min) + 1)) + Math.ceil(min);
    },
  }),
  getLocation: tool({
    description: "Get the user's current geographic location.",
    inputSchema: z.object({}),
    needsApproval: true, // requires user confirmation before executing
    execute: async () => {
      return new Promise((resolve, reject) => {
        navigator.geolocation.getCurrentPosition(
          (pos) => resolve({
            latitude: pos.coords.latitude,
            longitude: pos.coords.longitude,
          }),
          (err) => reject(err.message),
        );
      });
    },
  }),
});
```

The `getLocation` tool uses `needsApproval: true`, which means the AI SDK will pause execution and wait for the user to approve or reject the tool call before running it.

## Step 6: Create the chat transport

The Vercel AI SDK's `useChat` hook needs a [transport](https://ai-sdk.dev/docs/ai-sdk-ui/transport) that handles communication with the model. For client-side inference, we implement a custom `ChatTransport`.

Create `src/app/chat-transport.ts`:

```typescript
import {
  ChatTransport, UIMessageChunk, streamText,
  convertToModelMessages, ChatRequestOptions,
  createUIMessageStream, stepCountIs,
} from "ai";
import {
  TransformersJSLanguageModel,
  TransformersUIMessage,
  transformersJS,
} from "@browser-ai/transformers-js";
import { MODELS } from "./models";
import { createTools } from "./tools";

export class TransformersChatTransport
  implements ChatTransport<TransformersUIMessage>
{
  private model: TransformersJSLanguageModel;
  private tools: ReturnType<typeof createTools>;

  constructor() {
    const config = MODELS[0];
    this.model = transformersJS(config.id, {
      device: config.device,
      dtype: config.dtype,
      ...(config.supportsWorker
        ? {
            worker: new Worker(new URL("./worker.ts", import.meta.url), {
              type: "module",
            }),
          }
        : {}),
    });
    this.tools = createTools();
  }

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
    const { messages, abortSignal } = options;
    const prompt = await convertToModelMessages(messages);

    return createUIMessageStream<TransformersUIMessage>({
      execute: async ({ writer }) => {
        // Track download progress if the model hasn't been downloaded yet
        let downloadProgressId: string | undefined;
        const availability = await this.model.availability();

        if (availability !== "available") {
          await this.model.createSessionWithProgress(
            (progress: number) => {
              const percent = Math.round(progress * 100);

              if (progress >= 1) {
                if (downloadProgressId) {
                  writer.write({
                    type: "data-modelDownloadProgress",
                    id: downloadProgressId,
                    data: {
                      status: "complete", progress: 100,
                      message: "Model ready!",
                    },
                  });
                }
                return;
              }

              if (!downloadProgressId) {
                downloadProgressId = `download-${Date.now()}`;
              }

              writer.write({
                type: "data-modelDownloadProgress",
                id: downloadProgressId,
                data: {
                  status: "downloading", progress: percent,
                  message: `Downloading model... ${percent}%`,
                },
              });
            },
          );
        }

        const result = streamText({
          model: this.model,
          tools: this.tools,
          stopWhen: stepCountIs(5),
          messages: prompt,
          abortSignal,
        });

        writer.merge(result.toUIMessageStream({ sendStart: false }));
      },
    });
  }

  async reconnectToStream(): Promise<ReadableStream<UIMessageChunk> | null> {
    return null;
  }
}
```

Key parts of the transport:
- **Availability check**: Determines if the model needs downloading before inference.
- **Progress streaming**: Sends download progress as custom data parts (`data-modelDownloadProgress`) that the UI can render as a progress bar.
- **Tool support**: Passes the tools to `streamText()` so the model can call them.
- **Step limiting**: `stopWhen: stepCountIs(5)` prevents infinite tool-calling loops.

## Step 7: Build the chat UI

Now wire everything together in your page component. Create `src/app/page.tsx`:

```tsx
"use client";

import { useState } from "react";
import { useChat } from "@ai-sdk/react";
import { TransformersUIMessage } from "@browser-ai/transformers-js";
import { lastAssistantMessageIsCompleteWithApprovalResponses } from "ai";
import { TransformersChatTransport } from "./chat-transport";

export default function ChatPage() {
  const [input, setInput] = useState("");

  const {
    messages,
    sendMessage,
    status,
    stop,
    addToolApprovalResponse,
  } = useChat<TransformersUIMessage>({
    transport: new TransformersChatTransport(),
    experimental_throttle: 75,
    // Automatically resumes after tool approval responses are submitted
    sendAutomaticallyWhen: lastAssistantMessageIsCompleteWithApprovalResponses,
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && status === "ready") {
      sendMessage({ text: input });
      setInput("");
    }
  };

  return (
    <div style={{ maxWidth: 600, margin: "0 auto", padding: 24 }}>
      <h1>AI Chatbot</h1>

      <div>
        {messages.map((message) => (
          <div key={message.id} style={{ marginBottom: 16 }}>
            <strong>{message.role === "user" ? "You" : "Assistant"}:</strong>
            {message.parts.map((part, i) => {
              switch (part.type) {
                case "text":
                  return <p key={i}>{part.text}</p>;

                case "data-modelDownloadProgress":
                  if (!part.data.message) return null;
                  return (
                    <div key={i}>
                      <p>{part.data.message}</p>
                      {part.data.status === "downloading" && (
                        <progress value={part.data.progress} max={100} />
                      )}
                    </div>
                  );

                default:
                  // Handle tool parts
                  if (part.type.startsWith("tool-") && "state" in part) {
                    if (
                      part.state === "approval-requested" &&
                      "approval" in part
                    ) {
                      return (
                        <div key={i} style={{ border: "1px solid #ccc", padding: 8 }}>
                          <p>Tool <strong>{part.type.replace("tool-", "")}</strong> wants to run.</p>
                          <button onClick={() =>
                            addToolApprovalResponse({ id: part.approval!.id, approved: true })
                          }>
                            Approve
                          </button>
                          <button onClick={() =>
                            addToolApprovalResponse({
                              id: part.approval!.id, approved: false,
                              reason: "User denied",
                            })
                          }>
                            Deny
                          </button>
                        </div>
                      );
                    }
                    if ("output" in part && part.output) {
                      return (
                        <pre key={i} style={{ background: "#f5f5f5", padding: 8 }}>
                          {JSON.stringify(part.output, null, 2)}
                        </pre>
                      );
                    }
                  }
                  return null;
              }
            })}
          </div>
        ))}
      </div>

      {status === "submitted" && <p><em>Thinking...</em></p>}

      <form onSubmit={handleSubmit}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask something..."
          style={{ width: "100%", padding: 8 }}
        />
        <div style={{ marginTop: 8 }}>
          {status === "streaming" ? (
            <button type="button" onClick={stop}>Stop</button>
          ) : (
            <button type="submit" disabled={!input.trim()}>Send</button>
          )}
        </div>
      </form>
    </div>
  );
}
```

The component renders message parts based on their `type`:
- `text` &mdash; standard text output from the model.
- `data-modelDownloadProgress` &mdash; custom data parts sent by the transport during model download.
- `tool-*` &mdash; tool call parts with states like `approval-requested`, `output-available`, etc.

The `sendAutomaticallyWhen: lastAssistantMessageIsCompleteWithApprovalResponses` option tells `useChat` to automatically resume generation after the user responds to a tool approval request.

## Step 8: Run the application

Start the development server:

```bash
npm run dev
```

Open your browser and navigate to the URL shown in the terminal. The first time you send a message, the model will be downloaded and cached in the browser. Subsequent visits will load the cached model.

Try prompts like:
- "What time is it?"
- "Generate a random number between 1 and 100"
- "Where am I located?" (this will trigger a tool approval prompt)

## Next steps

- Add more models and a model selector &mdash; see the [full example source](https://github.com/huggingface/transformers.js-examples/tree/main/next-vercel-ai-sdk-v6-tool-calling) for a multi-model implementation with Zustand state management.
- Add a browser compatibility check with `doesBrowserSupportTransformersJS()` and fall back to a server-side route if WebGPU is unavailable.
- Explore the [Vercel AI SDK agents documentation](https://ai-sdk.dev/docs/agents/overview) for more complex agent patterns.
- See the [Vercel AI SDK guide](../guides/vercel-ai-sdk) for a reference of all supported features (embeddings, vision, transcription, etc.).
