/**
 * @module generation/grammar/llguidance
 */

let cachedRuntimePromise = null;
const normalizedRuntimeCache = new WeakMap();

function normalizeRuntime(runtime, options = {}) {
    const cached = normalizedRuntimeCache.get(runtime);
    if (cached) {
        return cached;
    }

    const createTokenizer = runtime.createTokenizer ?? runtime.create_tokenizer;
    const createInterpreter = runtime.createInterpreter ?? runtime.create_interpreter;

    if (typeof createTokenizer !== 'function' || typeof createInterpreter !== 'function') {
        throw new Error(
            'Invalid llguidance runtime: expected createTokenizer/createInterpreter functions exposed by the WASM module.',
        );
    }

    const normalized = { ...runtime, ...options, createTokenizer, createInterpreter };
    normalizedRuntimeCache.set(runtime, normalized);
    return normalized;
}

async function loadBundledRuntime() {
    const { loadBundledLLGuidance } = await import('llguidance');
    return normalizeRuntime(await loadBundledLLGuidance(), { acceptsTokenizerObjects: true });
}

/**
 * Loads the llguidance WASM runtime.
 *
 * Consumers can provide a compatible runtime on
 * `globalThis.__transformers_llguidance`; otherwise, the bundled `llguidance`
 * package runtime is loaded lazily.
 *
 * @param {Object|null} runtime Optional runtime, primarily useful for tests or custom integrations.
 * @returns {Promise<Object>}
 */
export async function loadLLGuidanceRuntime(runtime = null) {
    if (runtime) {
        return normalizeRuntime(runtime);
    }

    if (globalThis.__transformers_llguidance) {
        return normalizeRuntime(globalThis.__transformers_llguidance);
    }

    cachedRuntimePromise ??= loadBundledRuntime();
    return cachedRuntimePromise;
}
