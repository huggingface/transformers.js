<script>
  import { onMount } from 'svelte';

  let result = null;
  let ready = null;

  onMount(() => {
    // Make a request to the /classify routem, load the model
    fetch("/classify?text='Hello'");
  });

  async function classify(text) {
    if (!text) return;
    if (ready === null) {
      ready = false;
    }
    // Make a request to the /classify route on the server.
    const response = await fetch(`/classify?text=${encodeURIComponent(text)}`);

    // If this is the first time we've made a request, set the ready flag.
    if (!ready) {
      ready = true;
    }
    result = await response.json();
  }
</script>

<main class="flex min-h-screen flex-col items-center justify-center p-12">
  <h1 class="text-5xl font-bold mb-2 text-center">Transformers.js</h1>
  <h2 class="text-2xl mb-4 text-center">SvelteKit (server-side)</h2>
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
