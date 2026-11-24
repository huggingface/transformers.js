# Building an Angular Application

In this tutorial, we'll be building a simple Angular application that performs multilingual translation using Transformers.js!

**Useful links:**

- [Transformers.js documentation](https://huggingface.co/docs/transformers.js)
- [Angular web workers guide](https://angular.dev/ecosystem/web-workers)

## Prerequisites

- [Node.js](https://nodejs.org/en/) version 20.19+
- [npm](https://www.npmjs.com/) version 10+

## Step 1: Initialize the project

We'll use the [Angular CLI](https://angular.dev/tools/cli) to create the boilerplate project. Run the following command in your terminal:

```bash
npx @angular/cli@latest new angular-sentiment --style=css --ssr=false
```

The flags make the project use plain CSS and turn off server-side rendering (SSR). SSR is not wanted because the model runs in the browser and web workers don't exist server-side. The CLI may ask a few more questions but the defaults are fine for this tutorial.

Next, enter the project directory and start the development server:

```bash
cd angular-sentiment
npm start
```

Open the URL shown in the terminal (e.g., [http://localhost:4200/](http://localhost:4200/)) in the browser, it should show the default Angular welcome page.

## Step 2: Install and configure Transformers.js

Install Transformers.js from [NPM](https://www.npmjs.com/package/@huggingface/transformers):

```bash
npm install @huggingface/transformers
```

For this tutorial, we will use [`Xenova/distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/Xenova/distilbert-base-uncased-finetuned-sst-2-english), a ~67M parameter model fine-tuned on the Stanford Sentiment Treebank dataset. Before we continue, there are two things to note:

1. ML inference is computationally intensive, so it's better to avoid running models on the main (UI) thread to keep the UI responsive.
2. The model files are downloaded the first time they are needed, so we only want to start that download when the user starts typing, instead of when the page loads.

We can achieve both of these goals with a [Web Worker](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Using_web_workers) and a lazily created worker instance. The Angular CLI has a command that sets up the build configuration for workers. Run:

```bash
ng generate web-worker sentiment --snippet=false
```

> **Tip:** The Angular builder bundles workers when using the `new Worker(new URL('./file', import.meta.url))` pattern, which we'll use in Step 4.

## Step 3: Write the worker

Replace the contents of `src/app/sentiment.worker.ts` with the code below. It defines a `SentimentPipelineSingleton` class to lazily create one pipeline the first time `getInstance` is called and reuse it for every later message:

```typescript
/// <reference lib="webworker" />
import { pipeline, env, type TextClassificationPipeline } from '@huggingface/transformers';

// Download from the Hugging Face Hub.
env.allowLocalModels = false;

class SentimentPipelineSingleton {
  private static instance: Promise<TextClassificationPipeline> | null = null;

  static getInstance(progressCallback?: (info: unknown) => void) {
    if (!instance) {
      this.instance = pipeline(
        'text-classification',
        'Xenova/distilbert-base-uncased-finetuned-sst-2-english',
        { progress_callback: progressCallback },
      )
    }
    return this.instance;
  }
}


addEventListener('message', async ({ data }: MessageEvent<{ id: number; text: string }>) => {
  // The first classifier call loads the model and susequent call using the cached pipeline
  // The progress callback passes download progress to the main thread.
  const classifier = await SentimentPipelineSingleton.getInstance((info) => postMessage(info));

  // Perform the classification
  const output = await classifier(data.text);

  // Send the result back, tagged with the id of the request
  postMessage({ status: 'complete', id: data.id, output });
});
```

The worker sends progress messages, and `complete` messages back to the main thread.

We add the `id` field because users type quickly which can cause several requests can be in flight at once, and we only want to track the latest answer.

## Step 4: Connect the worker to Angular with a service

We'll wrap the worker in an injectable service that exposes its state as [signals](https://angular.dev/guide/signals) to keep rendering logic decoupled from model logic. Components can read the `ready`, `progressEntries` and `result` signals directly in their templates, and Angular re-renders only occur when the values of those signals change.

Create a new file called `src/app/sentiment.service.ts` with the following contents:

```typescript
import { Injectable, inject, OnDestroy, signal } from '@angular/core';

export interface ProgressEntry {
  file: string;
  progress: number;
}

export interface Classification {
  label: string;
  score: number;
}

@Injectable({ providedIn: 'root' })
export class SentimentService implements OnDestroy  {
  private worker: Worker | null = null;
  private readonly lastRequestId = signal(0);

  /** null = not requested yet/initial state, false = loading model, true = ready */
  readonly ready = signal<boolean | null>(null);
  readonly progressEntries = signal<ProgressEntry[]>([]);
  readonly result = signal<Classification | null>(null);

  ngOnDestroy(): void {
    this.worker?.terminate();
  }

  classify(text: string): void {
    // Changing the id invalidates any answer not yet received
    this.lastRequestId.update((n) => n + 1);
    const id = this.lastRequestId();

    if (!text.trim()) {
      this.result.set(null);
      return;
    }
    this.getWorker().postMessage({ id, text });
  }

  private getWorker(): Worker {
    if (!this.worker) {
      this.worker = new Worker(new URL('./sentiment.worker', import.meta.url), {
        type: 'module',
      });
      this.worker.addEventListener('message', (e) => this.onMessage(e));
    }
    return this.worker;
  }

  private onMessage({ data }: MessageEvent): void {
    switch (data.status) {
      case 'initiate':
        this.ready.set(false);
        this.progressEntries.update((items) => [...items, { file: data.file, progress: 0 }]);
        break;

      case 'progress':
        this.progressEntries.update((items) =>
          items.map((item) =>
            item.file === data.file ? { ...item, progress: data.progress } : item,
          ),
        );
        break;

      case 'done':
        this.progressEntries.update((items) => items.filter((item) => item.file !== data.file));
        break;

      case 'ready':
        this.ready.set(true);
        break;

      case 'complete':
        if (this.lastRequestId() === data.id) {
          this.result.set(data.output[0]);
        }
        break;
    }
  }
}
```

The `new Worker(new URL('./sentiment.worker', import.meta.url), { type: 'module' })` call must keep this shape to be including in the Angular bundle.

The template in the next step uses the states of `ready`  to decide what to show.

  - `null` before the first request
  - `false` while the model downloads
  - `true` once it can classify

## Step 5: Design the user interface

The UI needs a text input, a set of progress bars to show the model progress, and a place to show the model output/result. We'll start with the root component. Replace the contents of `src/app/app.ts` with:

```typescript
import { Component, inject } from '@angular/core';
import { PercentPipe } from '@angular/common';
import { SentimentService } from './sentiment.service';

@Component({
  selector: 'app-root',
  imports: [PercentPipe],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  protected readonly sentiment = inject(SentimentService);

  protected onInput(event: Event): void {
    this.sentiment.classify((event.target as HTMLInputElement).value);
  }
}
```

Replace `src/app/app.html` with the template. It uses the service's `ready`, `progressEntries` and `result` signals and `@if` and `@for` blocks to manage the UI state:

```html
<main class="container">
  <h1>Transformers.js</h1>
  <h2>Angular template</h2>

  <input type="text" placeholder="Enter text here" (input)="onInput($event)" />

  @if (sentiment.ready() === false) {
    <p>Loading model... (only runs once)</p>

    @for (item of sentiment.progressEntries(); track item.file) {
      <div class="progress-container">
        <div class="progress-bar" [style.width.%]="item.progress">
          {{ item.file }} ({{ item.progress.toFixed(2) }}%)
        </div>
      </div>
    }
  }

  @if (sentiment.result(); as result) {
    <div class="result" [class.negative]="result.label === 'NEGATIVE'">
      <strong>{{ result.label }}</strong>
      <span>{{ result.score | percent: '1.0-2' }}</span>
    </div>
  }
</main>
```

Lastly, add a little CSS to `src/app/app.css` to make the app look a little nicer:

```css
.container {
  margin: 40px auto;
  width: min(90vw, 480px);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  font-family: system-ui, sans-serif;
}

h1,
h2 {
  margin: 0;
  text-align: center;
}

input {
  width: 100%;
  padding: 8px 12px;
  font-size: 1em;
  border: 1px solid #ccc;
  border-radius: 8px;
  box-sizing: border-box;
}

.progress-container {
  width: 100%;
  font-size: 14px;
  color: white;
  background-color: #e9ecef;
  border-radius: 8px;
  overflow: hidden;
}

.progress-bar {
  padding: 0 6px;
  min-width: 1%;
  background-color: #007bff;
  white-space: nowrap;
  overflow: hidden;
}

.result {
  display: flex;
  gap: 12px;
  padding: 8px 16px;
  border-radius: 8px;
  background-color: #d4edda;
  color: #155724;
}

.result.negative {
  background-color: #f8d7da;
  color: #721c24;
}
```

> **Note:** The project generated by `ng new` includes a default `src/app/app.spec.ts` that checks for the welcome-page heading. It will fail now that we've replaced the template, so either update it or delete it if you don't plan to run tests.

## Step 6: Run the app

Start the development server (restart it if it was already running when you generated the worker):

```bash
npm start
```

Visit [http://localhost:4200/](http://localhost:4200/) and type `I love Transformers.js` into the text box. The first time, you'll see progress bars while the model files download; after that the result appears, something like `POSITIVE 99.97%`. The files are cached by the browser, so later visits skip the download. Try a negative sentence to watch the label and color flip.

## (Optional) Step 7: Build and deploy

To build your application, run:

```bash
ng build
```

This bundles the app into static files in `dist/angular-sentiment/browser`. You can host those files anywhere that serves static content. For this demo we'll use a static [Hugging Face Space](https://huggingface.co/docs/hub/spaces). If you haven't already, you can create a free Hugging Face account [here](https://huggingface.co/join).

1. Visit [https://huggingface.co/new-space](https://huggingface.co/new-space) and fill in the form. Remember to select "Static" as the space type.
2. Go to "Files" &rarr;"Add file" &rarr; "Upload files". Drag the contents of the `dist/angular-sentiment/browser` folder into the upload box and click "Upload". After they have uploaded, scroll down to the button and click "Commit changes to main".

**That's it!** Your application should now be live at `https://huggingface.co/spaces/<your-username>/<your-space-name>`.
