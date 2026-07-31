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

// Release the model and callback resources when it is no longer needed
await classifier.dispose();
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

Top-level, fire-and-forget callbacks like `progress_callback` are automatically handled via a callback bridge and execute in the main thread:

```typescript
const pipe = await webWorkerPipeline(worker, 'text-generation', 'model', {
  progress_callback: (progress) => {
    console.log('Loading:', progress);
  }
});
```

Only callbacks in the pipeline initialization options are bridged. Callback return values, nested callback functions, call-time callbacks, and class-based streamers such as `TextStreamer` are not supported across the worker boundary. Use a custom worker protocol when token-by-token streaming is required.

### Structured Clone Boundary

Inputs, call-time options, and results use the browser's structured clone algorithm. Class instances lose their prototype when crossing the worker boundary. For example, a `Tensor` or `RawImage` result arrives as structured data without class methods such as `tolist()`. Convert results inside the worker or reconstruct the required type on the main thread when those methods are needed.

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

**Note:** GPU devices and WebNN contexts cannot be passed across the worker boundary. Other values supported by the browser's structured clone algorithm, including typed arrays, can be passed normally.

### Multiple Pipelines

Multiple pipeline proxies can share one worker. Requests are routed independently and each initialized pipeline remains loaded until its `dispose()` method is called or the worker terminates.

Call `dispose()` and await any in-flight pipeline requests before calling `worker.terminate()`. Browsers do not emit a termination event that this package can use to reject requests interrupted by `terminate()`.

## Development

```bash
# Install dependencies
pnpm install

# Build the package
pnpm build

# Development mode with watch
pnpm dev
```
