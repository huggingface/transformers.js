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

The agent owns the system prompt, tools, and conversation history. All three are
fixed at construction time so the KV cache stays valid across turns — only the
user input changes per call.

```ts
import { Agent } from '@huggingface/transformers-agent';

const agent = new Agent({
  model,
  system: 'You are a helpful research assistant.',
  tools: { searchWeb, readUrl },
  maxSteps: 10,
});
```

### Non-streaming

```ts
const result = await agent.run('What are the latest Transformers.js updates?');

console.log(result.text);          // final answer
console.log(result.steps);         // per-step tool calls and text
console.log(result.usage);         // token counts across all steps
```

### Streaming

```ts
for await (const chunk of agent.stream('Compare that with TensorFlow.js')) {
  switch (chunk.type) {
    case 'text.delta':  ui.appendText(chunk.delta);           break;
    case 'tool.start':  ui.showTool(chunk.name, chunk.args);  break;
    case 'tool.result': ui.showResult(chunk.output);          break;
    case 'step.done':   ui.markStep(chunk.stepIndex);         break;
    case 'done':        ui.finalize(chunk.text, chunk.usage); break;
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
import type { Tool } from '@huggingface/transformers-agent';

const searchWeb: Tool = {
  description: 'Search the web for current information.',
  inputSchema: {
    type: 'object',
    properties: {
      query: { type: 'string', description: 'The search query' },
    },
    required: ['query'],
  },
  execute: async ({ query }) => ({
    content: [{ type: 'text', text: await fetchSearchResults(query as string) }],
  }),
};

const readUrl: Tool = {
  description: 'Fetch the text content of a URL.',
  inputSchema: {
    type: 'object',
    properties: {
      url: { type: 'string', description: 'The URL to read' },
    },
    required: ['url'],
  },
  execute: async ({ url }) => ({
    content: [{ type: 'text', text: await fetch(url as string).then(r => r.text()) }],
  }),
};
```

Return `isError: true` to let the model handle failures gracefully:

```ts
execute: async ({ url }) => {
  try {
    return { content: [{ type: 'text', text: await fetch(url as string).then(r => r.text()) }] };
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
    // fires after each agentic loop iteration inside run()
    ui.showProgress(`Step ${step.stepIndex} done`);
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
  tools: { searchWeb, readUrl },
  maxSteps: 5,
});

agent.onBeforeToolCall((call) => console.log(`→ ${call.name}`, call.args));

for await (const chunk of agent.stream('What changed in Transformers.js v4?')) {
  if (chunk.type === 'text.delta') process.stdout.write(chunk.delta);
  if (chunk.type === 'done') console.log('\n\nTokens used:', chunk.usage.totalTokens);
}
```