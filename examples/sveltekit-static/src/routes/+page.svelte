<script>
  import { onMount } from 'svelte';

  // Create a global variable to store the worker.
  let pipelineWorker;

  // reactive variables to store the result and the ready state.
  let result = null;
  let ready = null;

  onMount(async () => {
    // onMount set up the worker as soon as the Svelte page component is mounted.
    if (!pipelineWorker) {
      // Create the worker if it does not yet exist.
      const Worker = await import('$lib/worker.js?worker');
      pipelineWorker = new Worker.default();

      const onMessageReceived = (e) => {
        switch (e.data.status) {
          case 'initiate':
            ready = false;
            break;
          case 'ready':
            ready = true;
            break;
          case 'complete':
            result = e.data.output[0];
            break;
        }
      };

      // Attach the callback function as an event listener.
      pipelineWorker.addEventListener('message', onMessageReceived);
    }
  });

  // main classify function that sends messages with the text to the worker.
  function classify(text) {
    if (pipelineWorker) {
      pipelineWorker.postMessage({ text });
    }
  }
</script>

<main class="flex min-h-screen flex-col items-center justify-center p-12">
  <h1 class="text-5xl font-bold mb-2 text-center">Transformers.js</h1>
  <h2 class="text-2xl mb-4 text-center">SvelteKit Static template (client-side)</h2>
  <input
    type="text"
    class="w-full max-w-xs p-2 border border-gray-300 rounded mb-4 dark:text-black"
    placeholder="Enter text here"
    on:input={(e) => {
      classify(e.target.value);
    }}
  />

  {#if ready !== null}
    <pre class="bg-gray-100 dark:bg-gray-800 p-2 rounded">{!ready || !result
        ? 'Loading...'
        : JSON.stringify(result, null, 2)}</pre>
  {/if}
</main>
