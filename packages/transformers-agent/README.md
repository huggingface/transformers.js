# @huggingface/transformers-agent

Run local AI agents in the browser and Node.js, powered by [Transformers.js](https://github.com/huggingface/transformers.js).

```bash
npm install @huggingface/transformers-agent
```

---

## Model

```ts
import { Model } from '@huggingface/transformers-agent';

const model = new Model({
  modelId: 'onnx-community/gemma-4-E2B-it-ONNX',
  device: 'webgpu',
  dtype: 'q4f16',
});
```

Check cache state before downloading:

```ts
const cached = await model.isCached();
const size   = await model.downloadSize(); // bytes not yet cached

if (!cached) {
  const ok = await ui.confirm(`Download ${formatBytes(size)}?`);
  if (!ok) return;
}

await model.init((info) => {
  if (info.status === 'progress_total') ui.setProgress(info.progress);
});
```

Or use the static shorthand when you don't need pre-init introspection:

```ts
const model = await Model.load(
  { modelId: 'onnx-community/gemma-4-E2B-it-ONNX', device: 'webgpu', dtype: 'q4f16' },
  (info) => (info.status === 'progress_total') && console.log(info),
);
```

---

## Agent

The agent owns the initial prompts, tools, and conversation history. All three are
fixed at construction time so the KV cache stays valid across turns; only the
user input changes per call.

```ts
import { Agent } from '@huggingface/transformers-agent';

const agent = new Agent({
  model,
  initialPrompts: [{ role: 'system', content: 'You are a helpful research assistant.' }],
  tools: [searchWeb, readUrl],
  maxSteps: 10,
});
```

Model-specific details such as chat-template message structure, special tokens,
thinking text, tool-call syntax, and KV-cache behavior are handled by
`ModelAdapter` implementations. See `src/adapters/README.md` if you need to add
support for a new model family.

### Non-streaming

```ts
const result = await agent.run('What are the latest Transformers.js updates?');

console.log(result.runs.at(-1)?.text); // final answer
console.log(result.runs);             // per-step tool calls and text
console.log(result.usage);            // token counts across all steps
```

### Streaming

```ts
for await (const chunk of agent.stream('Compare that with TensorFlow.js')) {
  ui.renderRuns(chunk.runs);

  if (chunk.done) {
    ui.finalize(chunk.runs, chunk.usage);
  }
}
```

### History and cache

The agent appends messages automatically. Call `clearHistory()` to start a fresh
conversation (this also invalidates the KV cache):

```ts
console.log(agent.history); // ReadonlyArray<Message>
agent.clearHistory();
```

---

## Tools

Tools follow the [W3C WebMCP `ModelContextTool`](https://webmachinelearning.github.io/webmcp/)
interface — the same shape used by `navigator.modelContext.registerTool()` in the browser.

```ts
import { Tool } from '@huggingface/transformers-agent';

const searchWeb = new Tool<{ query: string }>({
  name: 'searchWeb',
  title: 'Search web',
  description: 'Search the web for current information.',
  parameters: {
    query: Tool.string({ description: 'The search query' }),
  },
  execute: async ({ query }) => ({
    content: [{ type: 'text', text: await fetchSearchResults(query) }],
  }),
});

const readUrl = new Tool<{ url: string }>({
  name: 'readUrl',
  title: 'Read URL',
  description: 'Fetch the text content of a URL.',
  parameters: {
    url: Tool.string({ description: 'The URL to read' }),
  },
  execute: async ({ url }) => ({
    content: [{ type: 'text', text: await fetch(url).then(r => r.text()) }],
  }),
});
```

`Tool` builds the WebMCP-compatible input schema from parameter helpers.

If you already have a WebMCP tool definition, use `Tool.fromWebMCP`:

```ts
const searchWeb = Tool.fromWebMCP<{ query: string }>({
  name: 'searchWeb',
  title: 'Search web',
  description: 'Search the web for current information.',
  inputSchema: {
    type: 'object',
    properties: {
      query: { type: 'string', description: 'The search query' },
    },
    required: ['query'],
  },
  execute: async ({ query }, client) => {
    return {
      content: [
        { type: 'text', text: await fetchSearchResults(query) },
      ],
    };
  },
  annotations: {
    readOnlyHint: true,
  },
});
```

You can also export the top-level WebMCP shape:

```ts
const webTool = searchWeb.toWebMCP();
```

The chat-template tool schema sent to the model is adapted internally from this
shape:

```ts
{
  type: 'function',
  function: {
    name: searchWeb.name,
    description: searchWeb.description,
    parameters: searchWeb.inputSchema,
  },
}
```

Return `isError: true` to let the model handle failures gracefully:

```ts
execute: async ({ url }) => {
  try {
    return { content: [{ type: 'text', text: await fetch(url).then(r => r.text()) }] };
  } catch (e) {
    return { content: [{ type: 'text', text: String(e) }], isError: true };
  }
},
```

---

## Lifecycle hooks

Hooks are registered after construction and support chaining. They have no effect
on the KV cache or generation.

```ts
agent
  .onBeforeToolCall((call) => {
    logger.info('tool call', call.name, call.args);
  })
  .onAfterToolCall((call, output, durationMs) => {
    metrics.record(call.name, durationMs);
  })
  .onStep((step) => {
    // fires after each agentic loop iteration inside run() or stream()
    ui.showProgress(`Step done, ${step.tools.length} tool(s) used`);
  });
```

---

## Full example

```ts
import { Model, Agent } from '@huggingface/transformers-agent';

const model = new Model({
  modelId: 'onnx-community/gemma-4-E2B-it-ONNX',
  device: 'webgpu',
  dtype: 'q4f16',
});

if (!await model.isCached()) {
  await confirmDownload(await model.downloadSize());
}

await model.init((info) => {
  if (info.status === 'progress_total') progressBar.update(info.progress);
});

const agent = new Agent({
  model,
  system: 'You are a helpful assistant.',
  tools: [searchWeb, readUrl],
  maxSteps: 5,
});

agent.onBeforeToolCall((call) => console.log(`-> ${call.name}`, call.args));

let renderedText = '';
let renderedRunIndex = -1;
for await (const chunk of agent.stream('What changed in Transformers.js v4?')) {
  const latestRunIndex = chunk.runs.length - 1;
  const latestRun = chunk.runs[latestRunIndex];
  if (latestRunIndex !== renderedRunIndex) {
    renderedText = '';
    renderedRunIndex = latestRunIndex;
  }

  if (latestRun && latestRun.text.length > renderedText.length) {
    process.stdout.write(latestRun.text.slice(renderedText.length));
    renderedText = latestRun.text;
  }

  if (chunk.done) console.log('\n\nTokens used:', chunk.usage.totalTokens);
}
```
