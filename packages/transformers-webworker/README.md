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

// Initialize the pipeline with options
const classifier = await webWorkerPipeline(
  worker,
  'sentiment-analysis',
  'Xenova/distilbert-base-uncased-finetuned-sst-2-english',
  {
    // Options like progress_callback are supported via callback bridge
    progress_callback: (progress) => {
      console.log('Progress:', progress);
    }
  }
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

## Options and Limitations

### Function Callbacks

Function callbacks like `progress_callback` are automatically handled via a callback bridge and will execute in the main thread:

```typescript
const pipe = await webWorkerPipeline(worker, 'text-generation', 'model', {
  progress_callback: (progress) => {
    console.log('Loading:', progress);
  }
});
```

### GPU Acceleration

Use the `device` parameter to enable GPU acceleration. The worker will handle GPU context creation:

```typescript
// ✅ Correct: Use device parameter
await webWorkerPipeline(worker, 'text-generation', 'model', {
  device: 'webgpu'  // or 'webnn'
});

// ❌ Incorrect: Don't pass GPU objects in session_options
await webWorkerPipeline(worker, 'text-generation', 'model', {
  session_options: {
    executionProviders: [{ name: 'webgpu', device: gpuDevice }]  // Won't work!
  }
});
```

**Note:** `session_options` cannot contain GPU devices, WebNN contexts, or typed arrays as these are not serializable across worker boundaries.

## Development

```bash
# Install dependencies
pnpm install

# Build the package
pnpm build

# Development mode with watch
pnpm dev
```
