import { build } from "esbuild";
import { fileURLToPath } from "node:url";
import { load_response_template, truncate_past_last_anchor } from "../../transformers/src/utils/chat_parsing/response_templates.js";

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

// Reference behavior for the supported anchor subset: greedy, leftmost, non-overlapping matches.
// Ordered alternatives and nullable anchors are tested as fallback, not against this reference.
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
  ])("preserves supported greedy, non-overlapping selection for %j in %j", (pattern, text) => {
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

describe("reassessment protocol regressions", () => {
  const template = (reasoning = {}, content = {}, anchor = { start_anchor: "<assistant>" }) => ({
    ...anchor,
    fields: {
      reasoning_content: { open: "<think>", close: "</think>", ...reasoning },
      content: { close: "<turn|>", ...content },
    },
  });
  const protocol = (value, pattern = "ok") => withResponseProtocol(compileRegex(pattern), responseProtocol({ response_template: value }));
  const feed = (machine, state, text) => encoder.encode(text).reduce(machine.transition, state);

  it.each([
    ["<assistant><think>\nreason", "</think>ok"],
    ["<assistant><think>\n\nreason", "</think>ok"],
    ["<assistant><think>", "reason</think>ok"],
  ])("replays accepted reasoning-opener suffixes: %s", (prefix, rest) => {
    const machine = protocol(template({ open: ["<think>", "<think>\n\n"] }));
    expect(machine.accepting(feed(machine, machine.initialize(encoder.encode(prefix)), rest))).toBe(true);
  });

  it.each(["<assistant><final>\no", "<assistant><think>r</think><final>\no", "<assistant><final>\n\no"])("replays content-opener suffixes: %s", (prefix) => {
    const machine = protocol(template({}, { open: ["<final>", "<final>\n\n"] }), "\\n?ok");
    const state = machine.initialize(encoder.encode(prefix));
    // Framing whitespace is optional; payload includes it only when its regex accepts it.
    expect(machine.accepting(feed(machine, state, "k"))).toBe(true);
  });

  it("rejects a prefill with no accepted opener", () => {
    const machine = protocol(template({}, { open: ["<final>", "<final>\n\n"] }));
    expect(() => machine.initialize(encoder.encode("<assistant><fX"))).toThrow();
  });

  it.each(["^<assistant>", "<assistant>$", "<assistant>|<assistant><think>", "<assistant><think>|<assistant>"])("explicitly falls back for unsupported anchor %s", (start_anchor_pattern) => {
    expect(responseProtocol({ response_template: template({}, {}, { start_anchor_pattern }) })).toBeUndefined();
  });

  it.each([
    ["^<assistant>", "x<assistant><think>", "x<assistant><think>"],
    ["<assistant>$", "<assistant><think>", "<assistant><think>"],
    ["<assistant>|<assistant><think>", "<assistant><think>", "<think>"],
    ["<assistant><think>|<assistant>", "<assistant><think>", ""],
  ])("leaves unsupported ordered/positional truncation to the parser: %s", (start_anchor_pattern, text, expected) => {
    const spec = template({}, {}, { start_anchor_pattern });
    expect(truncate_past_last_anchor(load_response_template(spec), text)).toBe(expected);
    expect(responseProtocol({ response_template: spec })).toBeUndefined();
  });

  it.each(["<assistant>first<assistant>last", "missing anchor", "<assistant>"])("preserves literal-anchor selection and the existing missing-anchor policy: %s", (text) => {
    const spec = template();
    const expected = truncate_past_last_anchor(load_response_template(spec), text);
    expect(initialize("<assistant>", text)).toBe(text.includes("<assistant>") ? expected : "");
  });

  it.each(["", "a*", "a?"])("allows nullable initial EOS for %j", (pattern) => {
    const machine = protocol(template(), pattern);
    expect(machine.acceptsEnd(machine.initial, encoder.encode("EOS"))).toBe(true);
  });

  it("allows a complete direct payload without committing an opener prefix", () => {
    const machine = protocol(template(), "<");
    const state = feed(machine, machine.initial, "<");
    expect(machine.accepting(state)).toBe(true);
    expect(machine.accepting(feed(machine, state, "think>r</think><"))).toBe(true);
    const required = protocol(template(), "<x");
    expect(required.accepting(feed(required, required.initial, "<"))).toBe(false);
    expect(required.accepting(required.initial)).toBe(false);
  });

  it.each([".+", "ok"])("allows only configured special closers after %s", (pattern) => {
    const machine = protocol(template(), pattern);
    const state = feed(machine, machine.initial, "ok");
    const closer = encoder.encode("<turn|>");
    expect(machine.allowsSpecial(state, closer.reduce(machine.transition, state), closer)).toBe(true);
    const unrelated = encoder.encode("<other>");
    expect(machine.allowsSpecial(state, unrelated.reduce(machine.transition, state), unrelated)).toBe(false);
    for (const text of ["payload<turn|>", "<turn|>payload"]) {
      const compound = encoder.encode(text);
      expect(machine.allowsSpecial(state, compound.reduce(machine.transition, state), compound)).toBe(false);
    }
    expect(machine.acceptsEnd(state, encoder.encode("EOS"))).toBe(true);
  });

  it("keeps closer-looking ordinary text in broad payloads", () => {
    const machine = protocol(template(), ".+");
    const state = feed(machine, machine.initial, "ok<turn|>ordinary");
    expect(machine.accepting(state)).toBe(true);
    expect(state.phase).toBe("structured");
  });
  it("keeps a closer branch when a bounded payload dies halfway through it", () => {
    const machine = protocol(template(), ".{1,4}");
    const state = feed(machine, machine.initial, "ok<turn|>");
    expect(machine.accepting(state)).toBe(true);
  });
});
