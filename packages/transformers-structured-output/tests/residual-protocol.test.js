import { jest } from "@jest/globals";
import { build } from "esbuild";
import { fileURLToPath } from "node:url";
import { Tensor } from "@huggingface/transformers";
import { StructuredOutputProcessor } from "../dist/index.js";
import { load_response_template, truncate_past_last_anchor } from "../../transformers/src/utils/chat_parsing/response_templates.js";

const { outputFiles } = await build({
  stdin: {
    contents: 'export { createTokenConstraint } from "./src/engine/constraint.ts"; export { responseProtocol, withResponseProtocol } from "./src/engine/response_protocol.ts"; export { compileRegex } from "./src/engine/regex.ts";',
    resolveDir: fileURLToPath(new URL("../", import.meta.url)),
  },
  bundle: true,
  write: false,
  format: "esm",
  platform: "neutral",
});
const { createTokenConstraint, responseProtocol, withResponseProtocol, compileRegex } = await import(`data:text/javascript;base64,${Buffer.from(outputFiles[0].text).toString("base64")}`);
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
