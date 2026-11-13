import { build } from "esbuild";
import { fileURLToPath } from "node:url";

// Exercise internal matchers without adding them to the package's public exports.
const { outputFiles } = await build({
  stdin: {
    contents: 'export { withResponseProtocol, responseProtocol } from "./src/engine/response_protocol.ts"; export { compileRegex } from "./src/engine/regex.ts";',
    resolveDir: fileURLToPath(new URL("../", import.meta.url)),
  },
  bundle: true,
  write: false,
  format: "esm",
  platform: "neutral",
});
const { withResponseProtocol, responseProtocol, compileRegex } = await import(`data:text/javascript;base64,${Buffer.from(outputFiles[0].text).toString("base64")}`);
const encoder = new TextEncoder();
const matcher = (source) => ({ machine: compileRegex(source), eager: false, source });

function initialize(pattern, text, onTransition = () => {}) {
  const anchor = matcher(pattern);
  const transition = anchor.machine.transition;
  anchor.machine.transition = (state, byte) => {
    onTransition();
    return transition(state, byte);
  };
  const payload = { initial: "", transition: (state, byte) => state + String.fromCharCode(byte), viable: () => true, accepting: () => true };
  const machine = withResponseProtocol(payload, { thinkingOpen: matcher("~"), thinkingClose: [encoder.encode("!")], startAnchor: anchor });
  const state = machine.initialize(encoder.encode(text));
  return state.phase === "start" ? state.structured : state.state;
}

// Reference behavior from the reviewed baseline: greedy, leftmost, non-overlapping matches.
function referenceLastEnd(pattern, text) {
  const machine = compileRegex(pattern);
  const bytes = encoder.encode(text);
  let last = -1;
  for (let start = 0; start < bytes.length; ) {
    let state = machine.initial;
    let end = -1;
    for (let i = start; i < bytes.length; ++i) {
      state = machine.transition(state, bytes[i]);
      if (!machine.viable(state)) break;
      if (machine.accepting(state)) end = i + 1;
    }
    if (end > start) {
      last = end;
      start = end;
    } else ++start;
  }
  return last;
}

describe("Response protocol prompt-anchor search", () => {
  it.each([4096, 8192, 16384])("bounds missing-anchor transition work linearly at %i bytes", (length) => {
    let work = 0;
    expect(
      initialize("a*b", "a".repeat(length), () => {
        if (++work > 3 * length) throw new Error("Anchor search exceeded its linear transition-work budget");
      }),
    ).toBe("");
    expect(work).toBeLessThanOrEqual(3 * length);
  });

  it.each([
    ["a*b", "aaaaaa"],
    ["ab", "abzab!"],
    ["aba", "ababa!"],
    ["a+b", "aaabxaaaaabz"],
    ["a.*b", "a1ba2b!"],
    ["ab", "zzab"],
    ["a*b", ""],
    ["a*b", "aaaabaaa"],
    ["aab", "aaabz"],
    ["a*", "aaa!aa"],
    ["ab|aab", "aabxab!"],
  ])("preserves greedy, non-overlapping selection for %j in %j", (pattern, text) => {
    const end = referenceLastEnd(pattern, text);
    expect(initialize(pattern, text)).toBe(end < 0 ? "" : text.slice(end));
  });

  it.each(["aaab<think>reasoning</think><final>o", "aaab<final>o", "aabignored aaab<final>o"])("replays thinking/content prefills after the last pattern anchor: %j", (prompt) => {
    const protocol = responseProtocol({
      response_template: {
        start_anchor_pattern: "a*b",
        fields: {
          reasoning_content: { open: "<think>", close: "</think>" },
          content: { open: "<final>", close: "</final>" },
        },
      },
    });
    const machine = withResponseProtocol(compileRegex("ok"), protocol);
    const state = machine.initialize(encoder.encode(prompt));
    expect(machine.accepting(machine.transition(state, "k".charCodeAt(0)))).toBe(true);
    expect(machine.viable(machine.transition(state, "x".charCodeAt(0)))).toBe(false);
  });
});
