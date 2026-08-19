import { pathToFileURL } from "node:url";

const runtime = pathToFileURL(new URL("./runtime.mjs", import.meta.url).pathname).href;
const helpers = pathToFileURL(new URL("./helpers.mjs", import.meta.url).pathname).href;

export async function resolve(specifier, context, nextResolve) {
  if (context.parentURL?.includes("/llguidance-js/test/upstream/") && specifier === "./helpers.mjs") {
    return { url: helpers, shortCircuit: true };
  }
  if (context.parentURL?.includes("/llguidance-js/test/") && specifier.endsWith("/src/llguidance.node.ts")) {
    return { url: runtime, shortCircuit: true };
  }
  return nextResolve(specifier, context);
}
