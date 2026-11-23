import { jest } from "@jest/globals";
import { build } from "esbuild";
import { fileURLToPath } from "node:url";
import { Tensor } from "@huggingface/transformers";
import { StructuredOutputProcessor } from "../dist/index.js";
import { load_response_template, truncate_past_last_anchor } from "../../transformers/src/utils/chat_parsing/response_templates.js";

// Exercise internal matchers without adding them to the package's public exports.
const { outputFiles } = await build({
  stdin: {
    contents: 'export { createTokenConstraint } from "./src/engine/constraint.ts"; export { withResponseProtocol, responseProtocol } from "./src/engine/response_protocol.ts"; export { compileRegex } from "./src/engine/regex.ts";',
    resolveDir: fileURLToPath(new URL("../", import.meta.url)),
  },
  bundle: true,
  write: false,
  format: "esm",
  platform: "neutral",
});
const { createTokenConstraint, withResponseProtocol, responseProtocol, compileRegex } = await import(`data:text/javascript;base64,${Buffer.from(outputFiles[0].text).toString("base64")}`);
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

describe("Response protocol templates and prefills", () => {
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

const encode = (text) => Array.from(new TextEncoder().encode(text));
const template = { start_anchor: "<assistant>", fields: { reasoning_content: { open: "<think>", close: "</think>" }, content: { close: "<turn|>" } } };
const source = {
  tokens: [...Array.from({ length: 256 }, (_, i) => [i]), [], ...["<turn|>", "<other>", "<turn|>", "<tu", "rn|>", "turn|>", "x<turn|>", "<turn|>x"].map(encode)],
  eos_token_id: 256,
  special_token_ids: [256, 257, 258, 260, 261, 262, 263, 264],
  response_template: template,
};
const ids = (text) => encode(text).map(BigInt);
const allowed = (mask, id) => Boolean(mask[id >>> 5] & (1 << (id & 31)));
const maskOf = (constraint) => {
  const mask = new Uint32Array(Math.ceil(constraint.vocabSize / 32));
  constraint.fillMask(mask);
  return mask;
};
const regex = String.raw`ok(?:<turn\|>x)?`;

describe("nullable protocol anchors", () => {
  it.each(["a*", "", "a?", "(?:a*)", "(?:a|)"])("falls back for nullable anchor %j without changing nullable payloads", (pattern) => {
    const spec = { ...template, start_anchor: undefined, start_anchor_pattern: pattern };
    delete spec.start_anchor;
    expect(truncate_past_last_anchor(load_response_template(spec), "aaa!")).toBe("");
    expect(truncate_past_last_anchor(load_response_template(spec), "")).toBe("");
    const warn = jest.spyOn(console, "warn").mockImplementation(() => {});
    try {
      expect(responseProtocol({ response_template: spec })).toBeUndefined();
      const count = warn.mock.calls.length;
      expect(count).toBe(1);
      expect(warn.mock.calls[0][0]).toContain("Falling back");
      expect(responseProtocol({ response_template: spec })).toBeUndefined();
      expect(warn).toHaveBeenCalledTimes(count);
    } finally {
      warn.mockRestore();
    }
    for (const payload of ["", "a*", "a?"]) {
      const machine = withResponseProtocol(compileRegex(payload), responseProtocol(source));
      expect(machine.accepting(machine.initial)).toBe(true);
    }
  });
  it.each(["a*b", "<assistant>"])("keeps non-nullable anchor %s", (pattern) => {
    expect(responseProtocol({ response_template: { ...template, start_anchor: undefined, start_anchor_pattern: pattern } })).toBeDefined();
  });
});

describe("special closer state selection through token constraints", () => {
  it.each(["after-mask", "without-mask", "repeated-mask", "shared-mask-cache"])("commits the terminating branch: %s", (path) => {
    const make = () => {
      const c = createTokenConstraint(source, { type: "regex", regex });
      c.initializePrompt(ids("<assistant>ok"));
      return c;
    };
    if (path === "shared-mask-cache") maskOf(make());
    const constraint = make();
    if (path !== "without-mask") {
      const mask = maskOf(constraint);
      expect(allowed(mask, 256)).toBe(true);
      expect(allowed(mask, 257)).toBe(true);
      if (path === "repeated-mask") expect(maskOf(constraint)).toEqual(mask);
    }
    expect(constraint.commit(257)).toBe(false);
    const after = maskOf(constraint);
    expect(allowed(after, 256)).toBe(true);
    expect(allowed(after, "x".charCodeAt(0))).toBe(false);
    expect(() => constraint.commit("x".charCodeAt(0))).toThrow();
  });
  it("keeps identical ordinary bytes in the payload language", () => {
    const c = createTokenConstraint(source, { type: "regex", regex });
    c.initializePrompt(ids("<assistant>ok"));
    maskOf(c);
    c.commit(259); // Ordinary token with exactly the special closer's bytes.
    const after = maskOf(c);
    expect(allowed(after, 256)).toBe(false);
    expect(allowed(after, 120)).toBe(true);
    c.commit(120);
    expect(allowed(maskOf(c), 256)).toBe(true);
  });
  it.each(["after-mask", "without-mask", "repeated-mask", "primed-constraint"])("retains a restarted closer when the payload dies: %s", (path) => {
    const make = () => {
      const c = createTokenConstraint(source, { type: "regex", regex: "ok<?" });
      c.initializePrompt(ids("<assistant>ok<"));
      return c;
    };
    // Partial delimiter states intentionally have no shared mask-cache key. Complete-payload
    // shared-cache paths are covered above; exercise repeated traversal and token-state caching here.
    if (path === "primed-constraint") maskOf(make());
    const c = make();
    if (path !== "without-mask") {
      const mask = maskOf(c);
      expect(allowed(mask, 257)).toBe(true);
      expect(allowed(mask, 258)).toBe(false);
      expect(allowed(mask, 263)).toBe(false);
      expect(allowed(mask, 264)).toBe(false);
      if (path === "repeated-mask") expect(maskOf(c)).toEqual(mask);
    }
    c.commit(257);
    const after = maskOf(c);
    expect(allowed(after, 256)).toBe(true);
    expect(allowed(after, 120)).toBe(false);
    expect(() => c.commit(120)).toThrow();
  });
  it("does not turn a narrow ordinary closer into a payload escape", () => {
    const c = createTokenConstraint(source, { type: "regex", regex: "ok<?" });
    c.initializePrompt(ids("<assistant>ok<"));
    const mask = maskOf(c);
    expect(allowed(mask, 264)).toBe(false); // Ordinary/special compound must not append payload after closing.
    expect(allowed(mask, 120)).toBe(false);
    // Ordinary bytes may finish a configured delimiter when they cannot extend the payload,
    // but must not permit any additional payload bytes. They never widen the payload regex.
    if (allowed(mask, 259)) {
      c.commit(259);
      expect(allowed(maskOf(c), 256)).toBe(true);
      expect(allowed(maskOf(c), 120)).toBe(false);
    } else {
      expect(() => c.commit(259)).toThrow();
    }
  });
  it.each([
    ["ok<", 257],
    ["ok<tu", 261],
    ["ok<", 262],
    ["ok", 257],
  ])("continues or restarts a closer after %s", (text, token) => {
    const c = createTokenConstraint(source, { type: "regex", regex: ".+" });
    c.initializePrompt(ids("<assistant>" + text));
    const mask = maskOf(c);
    expect(allowed(mask, token)).toBe(true);
    expect(allowed(mask, 258)).toBe(false);
    expect(allowed(mask, 263)).toBe(false);
    expect(allowed(mask, 264)).toBe(false);
    c.commit(token);
    expect(allowed(maskOf(c), 256)).toBe(true);
    expect(allowed(maskOf(c), 120)).toBe(false);
  });
  it("commits a split special closer and rejects incomplete required payloads", () => {
    const c = createTokenConstraint(source, { type: "regex", regex: ".+" });
    c.initializePrompt(ids("<assistant>ok"));
    expect(allowed(maskOf(c), 260)).toBe(true);
    c.commit(260);
    expect(allowed(maskOf(c), 256)).toBe(false);
    expect(allowed(maskOf(c), 261)).toBe(true);
    c.commit(261);
    expect(allowed(maskOf(c), 256)).toBe(true);
    const incomplete = createTokenConstraint(source, { type: "regex", regex: "okay" });
    incomplete.initializePrompt(ids("<assistant>ok"));
    expect(allowed(maskOf(incomplete), 256)).toBe(false);
    expect(allowed(maskOf(incomplete), 257)).toBe(false);
    expect(() => incomplete.commit(257)).toThrow();
  });
});

describe("processor commits special closer branches", () => {
  it.each([
    [regex, "ok", 257],
    [".+", "ok<", 257],
    [".+", "ok<tu", 261],
    ["ok<?", "ok<", 257],
  ])("terminates %s after %s", (pattern, prefix, token) => {
    const processor = new StructuredOutputProcessor(source, { type: "regex", regex: pattern });
    const input = ids("<assistant>" + prefix);
    const scores = () => new Tensor("float32", new Float32Array(source.tokens.length).fill(1), [1, source.tokens.length]);
    const initial = scores();
    processor([input], initial);
    expect(Number.isFinite(initial.data[token])).toBe(true);
    const after = scores();
    processor([[...input, BigInt(token)]], after);
    expect(Number.isFinite(after.data[256])).toBe(true);
    expect(Number.isFinite(after.data[120])).toBe(false);
  });
});

describe("regex anchors", () => {
  const accepts = (pattern, text) => {
    const machine = compileRegex(pattern);
    let state = machine.initial;
    for (const byte of encoder.encode(text)) {
      state = machine.transition(state, byte);
      if (!machine.viable(state)) return false;
    }
    return machine.accepting(state);
  };
  const candidates = ["yes", "no", "^yes", "yes$", "^no", "no$", "^yes$", "", "yesno"];

  it.each(["^yes$|^no$", "^yes|no$", "yes$|^no", "^(yes|no)$", "(?:yes|no)"])("%j accepts exactly yes and no", (pattern) => {
    expect(candidates.filter((text) => accepts(pattern, text))).toEqual(["yes", "no"]);
  });

  it("keeps escaped anchors literal", () => {
    expect(accepts("\\^yes\\$", "^yes$")).toBe(true);
    expect(accepts("\\^yes\\$", "yes")).toBe(false);
    expect(accepts("[$^]a", "$a")).toBe(true);
  });

  it.each(["a^b", "a$b", "(^a|b)+", "x(^a|b)", "(a$)b", "(a|b$)", "a$$"])("rejects the non-equivalent anchor in %j", (pattern) => {
    expect(() => compileRegex(pattern)).toThrow("anchors are only supported");
  });

  it("applies top-level alternative anchors through the processor", () => {
    const vocabulary = { tokens: [...Array.from({ length: 256 }, (_, id) => [id]), []], eos_token_id: 256, special_token_ids: [256] };
    const run = (text) => {
      const processor = new StructuredOutputProcessor(vocabulary, { type: "regex", regex: "^yes$|^no$" });
      const input = [0n];
      for (const byte of [...encoder.encode(text), 256]) {
        const scores = new Tensor("float32", new Float32Array(257).fill(1), [1, 257]);
        processor([input], scores);
        if (!Number.isFinite(scores.data[byte])) return false;
        input.push(BigInt(byte));
      }
      return true;
    };
    expect(run("yes")).toBe(true);
    expect(run("no")).toBe(true);
    expect(run("yes$")).toBe(false);
    expect(run("^no")).toBe(false);
  });
});
