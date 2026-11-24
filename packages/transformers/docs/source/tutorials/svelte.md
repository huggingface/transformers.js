# Building a Svelte application

In this tutorial, we'll be building a simple [Svelte](https://svelte.dev/) chat application that runs a small language model entirely in your browser with Transformers.js! The model's replies are streamed into the page token by token as they are generated.

Useful links:

- [SvelteKit server-side example](https://github.com/huggingface/transformers.js-examples/tree/main/sveltekit), if you want to run Transformers.js on the server instead of in the browser

## Prerequisites

- [Node.js](https://nodejs.org/en/) version 18+
- [npm](https://www.npmjs.com/) version 9+
- A browser with [WebGPU support](https://caniuse.com/webgpu) (e.g., a recent version of Chrome or Edge)

## Step 1: Initialise the project

For this tutorial, we will use the [Svelte CLI](https://svelte.dev/docs/cli/overview) to initialise our project. The CLI creates SvelteKit projects and uses Vite under the hood, so the development server command and localhost URL shown below still apply. Run the following command in your terminal:

```bash
npx sv create svelte-chat
```

If prompted to install `sv`, type <kbd>y</kbd> and press <kbd>Enter</kbd>. Then answer the prompts as follows:

1. Template: `SvelteKit minimal`
2. Type checking: `No` (plain JavaScript)
3. Add-ons: none
4. Dependency installation: no, because we will install dependencies in the next step

The minimal template creates a SvelteKit app with the page component in `src/routes/+page.svelte` and the HTML shell in `src/app.html`. Files in `src/lib` can be imported with the `#lib` alias (older SvelteKit projects use `$lib` instead).

Next, enter the project directory and install the necessary development dependencies:

```bash
cd svelte-chat
npm install
```

To test that our application is working, we can run the following command:

```bash
npm run dev
```

Visiting the URL shown in the terminal (e.g., [http://localhost:5173/](http://localhost:5173/)) should show the starter SvelteKit page.
You can stop the development server by pressing <kbd>Ctrl</kbd> + <kbd>C</kbd> in the terminal.

## Step 2: Install and configure Transformers.js

Now we get to the fun part: adding machine learning to our application! First, install Transformers.js from [NPM](https://www.npmjs.com/package/@huggingface/transformers) with the following command:

```bash
npm install @huggingface/transformers
```

For this application, we will use [onnx-community/LFM2.5-350M-ONNX](https://huggingface.co/onnx-community/LFM2.5-350M-ONNX), a small instruction-tuned language model from Liquid AI that is designed for on-device use. We will load its 4-bit quantized weights (`dtype: "q4"`) and run it on the GPU with WebGPU (`device: "webgpu"`). Before we start, there are 2 things we need to take note of:

1. ML inference can be quite computationally intensive, so it's better to load and run the models in a separate thread from the main (UI) thread.
2. Even quantized, the model weights are around 300 MB, so we don't want to download them until the user sends their first message.

We can achieve both of these goals by using a [Web Worker](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Using_web_workers).

Create a file called `worker.js` in the `src/routes` directory. This script will do all the heavy-lifting for us, including loading and running of the text generation pipeline. To ensure the model is only loaded once, we will create the `TextGenerationPipeline` class which uses the [singleton pattern](https://en.wikipedia.org/wiki/Singleton_pattern) to lazily create a single instance of the pipeline when `getInstance` is first called, and use this pipeline for all subsequent calls:

```javascript
import { pipeline, TextStreamer } from "@huggingface/transformers";

class TextGenerationPipeline {
  static task = "text-generation";
  static model = "onnx-community/LFM2.5-350M-ONNX";
  static instance = null;

  static async getInstance(progress_callback = null) {
    this.instance ??= pipeline(this.task, this.model, {
      dtype: "q4",
      device: "webgpu",
      progress_callback,
    });
    return this.instance;
  }
}
```

<Tip>

If your browser does not support WebGPU, you can change `device: "webgpu"` to `device: "wasm"` to run the model on the CPU instead. Generation will be slower, but the rest of the tutorial works the same way.

</Tip>

## Step 3: Design the user interface

<Tip>

We recommend starting the development server again with `npm run dev`
(if not already running) so that you can see your changes in real-time.

</Tip>

First, let's create some child components. Create the following files in the `src/lib` directory (the template already includes this folder; create it if it does not exist):

1. `ChatMessage.svelte`: This component will display a single message in the conversation, from either the user or the assistant.

   ```svelte
   <script>
     let { role = 'user', content = '' } = $props();
   </script>

   <div class="message {role}">
     <span class="role">{role === 'user' ? 'You' : 'Assistant'}</span>
     <p>{content}</p>
   </div>
   ```

2. `Progress.svelte`: This component will display the overall model download progress.

   ```svelte
   <script>
     let { text = '', percentage = 0 } = $props();
   </script>

   <div class="progress-container">
     <div class="progress-bar" style:width="{percentage}%">
       {text} ({percentage.toFixed(2)}%)
     </div>
   </div>
   ```

Now let's update `src/routes/+page.svelte`. Replace its contents with the following, which sets up our state variables, renders the UI, and adds the page styles:

```svelte
<script>
  import ChatMessage from '#lib/ChatMessage.svelte';
  import Progress from '#lib/Progress.svelte';

  // Model loading
  let ready = $state(null);
  let disabled = $state(false);
  let loadProgress = $state(0);

  // Conversation
  let input = $state('Tell me a short story about a brave knight.');
  let messages = $state([]);

  function send() {}
</script>

<main>
  <h1>Transformers.js</h1>
  <h2>In-browser chat with streaming text generation in Svelte!</h2>

  <div class="chat-container">
    {#each messages as message}
      <ChatMessage role={message.role} content={message.content} />
    {/each}
  </div>

  <form class="input-container" onsubmit={(e) => { e.preventDefault(); send(); }}>
    <textarea bind:value={input} rows={3} placeholder="Type a message..."></textarea>
    <button type="submit" disabled={disabled || !input.trim()}>Send</button>
  </form>

  <div class="progress-bars-container">
    {#if ready === false}
      <p>Loading model... (only run once)</p>
      <Progress text="Overall progress" percentage={loadProgress} />
    {/if}
  </div>
</main>

<style>
  :global(:root) {
    font-family: Inter, system-ui, Avenir, Helvetica, Arial, sans-serif;
    line-height: 1.5;
    font-weight: 400;
    color: #213547;
    background-color: #ffffff;

    font-synthesis: none;
    text-rendering: optimizeLegibility;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    -webkit-text-size-adjust: 100%;
  }

  :global(body) {
    margin: 0;
    display: flex;
    place-items: center;
    min-width: 320px;
    min-height: 100vh;
  }

  main {
    width: 800px;
    max-width: 100%;
    margin: 0 auto;
    padding: 2rem;
    text-align: center;
  }

  h1 {
    font-size: 3.2em;
    line-height: 1;
  }

  h1,
  h2 {
    margin: 8px;
  }

  textarea {
    padding: 0.6em;
  }

  button {
    padding: 0.6em 1.2em;
    cursor: pointer;
    font-weight: 500;
  }

  button[disabled] {
    cursor: not-allowed;
  }

  textarea,
  button {
    border-radius: 8px;
    border: 1px solid transparent;
    font-size: 1em;
    font-family: inherit;
    background-color: #f9f9f9;
    transition: border-color 0.25s;
  }

  textarea:hover,
  button:not([disabled]):hover {
    border-color: #646cff;
  }

  textarea:focus,
  textarea:focus-visible,
  button:focus,
  button:focus-visible {
    outline: 4px auto -webkit-focus-ring-color;
  }

  .chat-container {
    margin: 25px 0;
    display: flex;
    flex-direction: column;
    gap: 10px;
    text-align: left;
  }

  :global(.message) {
    padding: 0.6em 1em;
    border-radius: 8px;
    max-width: 80%;
  }

  :global(.message p) {
    margin: 0;
    white-space: pre-wrap;
  }

  :global(.message .role) {
    font-size: 0.8em;
    font-weight: 600;
  }

  :global(.message.user) {
    align-self: flex-end;
    background-color: #e7f0ff;
  }

  :global(.message.assistant) {
    align-self: flex-start;
    background-color: #f1f1f1;
  }

  .input-container {
    display: flex;
    gap: 10px;
  }

  .input-container > textarea {
    flex: 1;
  }

  :global(.progress-container) {
    position: relative;
    font-size: 14px;
    color: white;
    background-color: #e9ecef;
    border: solid 1px;
    border-radius: 8px;
    text-align: left;
    overflow: hidden;
  }

  :global(.progress-bar) {
    padding: 0 4px;
    z-index: 0;
    top: 0;
    width: 1%;
    overflow: hidden;
    background-color: #007bff;
    white-space: nowrap;
  }

  .progress-bars-container {
    padding: 8px;
    height: 140px;
  }
</style>
```

The `send` function is just a placeholder for now. We will replace it in the next section. Some styles use `:global(...)` because styles in Svelte components are scoped, and those selectors target elements inside the child components.

## Step 4: Connecting everything together

Now that we have a basic user interface set up, we can finally connect everything together.

First, let's set up the Web Worker and the `send` function. Replace the `<script>` section of `src/routes/+page.svelte` with the following:

```svelte
<script>
  import { onMount } from 'svelte';
  import ChatMessage from '#lib/ChatMessage.svelte';
  import Progress from '#lib/Progress.svelte';

  // Model loading
  let ready = $state(null);
  let disabled = $state(false);
  let loadProgress = $state(0);

  // Conversation
  let input = $state('Tell me a short story about a brave knight.');
  let messages = $state([]);

  let worker;

  onMount(() => {
    worker = new Worker(new URL('./worker.js', import.meta.url), {
      type: 'module',
    });

    worker.addEventListener('message', onMessageReceived);

    return () => {
      worker?.removeEventListener('message', onMessageReceived);
      worker?.terminate();
    };
  });

  function onMessageReceived(e) {
    switch (e.data.status) {
      case 'progress_total':
        ready = false;
        loadProgress = e.data.progress;
        break;

      case 'ready':
        ready = true;
        break;

      case 'update':
        // Append the newly generated text to the assistant's reply
        messages[messages.length - 1].content += e.data.output;
        break;

      case 'complete':
        disabled = false;
        break;

      case 'error':
        messages[messages.length - 1].content = `Error: ${e.data.error}`;
        disabled = false;
        break;
    }
  }

  function send() {
    const text = input.trim();
    if (!text) return;

    disabled = true;
    input = '';
    messages.push({ role: 'user', content: text });

    // Send the whole conversation so the model can see earlier turns
    worker.postMessage({ messages: $state.snapshot(messages) });

    // Add an empty assistant message that the streamed text will fill in
    messages.push({ role: 'assistant', content: '' });
  }
</script>
```

SvelteKit server-renders pages by default, so creating the worker inside `onMount` keeps it browser-only and avoids running Web Worker code during SSR. In Svelte 5, you can return a cleanup function from `onMount` instead of using `onDestroy`.

Because `messages` is declared with `$state`, it is a deeply reactive proxy: pushing a new message or appending to `content` updates the page automatically. Proxies cannot be sent to a worker with `postMessage`, so we use `$state.snapshot` to send a plain copy of the conversation instead.

Now, let's add an event listener in `src/routes/worker.js` to listen for messages from the main thread. We will send back messages (e.g., for model loading progress and text streaming) to the main thread with `self.postMessage`.

```javascript
// Listen for messages from the main thread
self.addEventListener("message", async (event) => {
  try {
    // Retrieve the text generation pipeline. When called for the first time,
    // this will load the pipeline and save it for future use.
    const generator = await TextGenerationPipeline.getInstance((x) => {
      // Forward overall download progress and the ready signal
      if (x.status === "progress_total" || x.status === "ready") {
        self.postMessage(x);
      }
    });

    // Send each new piece of text to the main thread as it is generated
    const streamer = new TextStreamer(generator.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: function (text) {
        self.postMessage({
          status: "update",
          output: text,
        });
      },
    });

    // Generate a reply to the conversation
    const output = await generator(event.data.messages, {
      max_new_tokens: 512,
      do_sample: false,
      streamer,
    });

    // Send the final reply back to the main thread
    self.postMessage({
      status: "complete",
      output: output[0].generated_text.at(-1).content,
    });
  } catch (error) {
    self.postMessage({
      status: "error",
      error: error.message,
    });
  }
});
```

The text generation pipeline accepts the conversation as a list of `{ role, content }` messages and applies the model's chat template for us. The returned `generated_text` is the conversation with the new assistant message appended, so `generated_text.at(-1).content` is the model's reply.

You can now run the application with `npm run dev` and chat with the model directly in your browser! The first message triggers the model download, and later messages reuse the loaded model.

## (Optional) Step 5: Build and deploy

To build your application, run `npm run build`. You can then run `npm run preview` to preview the production build locally.

For this demo, we will deploy our application as a static [Hugging Face Space](https://huggingface.co/docs/hub/spaces), but you can deploy it anywhere you like! If you haven't already, you can create a free Hugging Face account [here](https://huggingface.co/join).

1. Visit [https://huggingface.co/new-space](https://huggingface.co/new-space) and fill in the form. Remember to select "Static" as the space type.
2. SvelteKit's default adapter does not create a `dist` folder for static hosting. To deploy this as a Static Space, configure [`@sveltejs/adapter-static`](https://svelte.dev/docs/kit/adapter-static), then run `npm run build`.
3. Go to "Files" &rarr; "Add file" &rarr; "Upload files". Drag the files generated by the static adapter into the upload box and click "Upload". After they have uploaded, scroll down to the button and click "Commit changes to main".

**That's it!** Your application should now be live at `https://huggingface.co/spaces/<your-username>/<your-space-name>`!
