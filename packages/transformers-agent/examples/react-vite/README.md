# React + Vite example

Simple playground app for manually testing the `@huggingface/transformers-agent` SDK.

## Run

From repository root:

```bash
pnpm install
pnpm --filter @huggingface/transformers-agent build
pnpm --dir packages/transformers-agent/examples/react-vite dev
```

Then open the local Vite URL shown in your terminal.

## What it tests

- `Model.isCached()`
- `Model.downloadSize()`
- `Model.cachedSize()`
- `Model.init(progressCallback)`
- `new Agent(...)` + `agent.promptStreaming(...)`
- open-loop tool calls returned to the application
