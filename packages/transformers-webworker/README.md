# @huggingface/transformers-webworker

Web Worker utilities for Transformers.js - a lightweight package designed to be used in Web Workers.

## Installation

You need to install both this package and the main transformers package:

```bash
npm install @huggingface/transformers @huggingface/transformers-webworker
```

Or with pnpm:

```bash
pnpm add @huggingface/transformers @huggingface/transformers-webworker
```

## Usage

This package provides utilities for using Transformers.js pipelines in Web Workers.

### In Your Main Thread

Use `webWorkerPipeline` to communicate with a Web Worker running a pipeline:

```typescript
import { webWorkerPipeline } from '@huggingface/transformers-webworker';

// Create your worker
const worker = new Worker('./worker.js');

// Initialize the pipeline
const classifier = await webWorkerPipeline(
  worker,
  'sentiment-analysis',
  'Xenova/distilbert-base-uncased-finetuned-sst-2-english'
);

// Use it like a regular pipeline
const result = await classifier('I love this!');
console.log(result);
```

### In Your Web Worker

Use `webWorkerPipelineHandler` to handle pipeline requests:

```typescript
// worker.js
import { webWorkerPipelineHandler } from '@huggingface/transformers-webworker';

const handler = webWorkerPipelineHandler();
self.onmessage = handler.onmessage;
```

> **Note:** The handler internally uses `pipeline` from `@huggingface/transformers` to create and cache pipeline instances.

## Development

```bash
# Install dependencies
pnpm install

# Build the package
pnpm build

# Development mode with watch
pnpm dev
```
