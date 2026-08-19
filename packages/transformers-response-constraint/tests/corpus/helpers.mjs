import assert from "node:assert/strict";

import { loadLLGuidance } from "./runtime.mjs";

const encoder = new TextEncoder();

export const llguidance = await loadLLGuidance();
export const tokenizer = new llguidance.LLTokenizer({});

export function tokenAllowed(mask, token) {
  return Boolean(mask?.[token >>> 5] & (1 << (token & 31)));
}

export function resultField(result, field) {
  return result instanceof Map ? result.get(field) : result?.[field];
}

export function grammarAccepts(grammar, input) {
  const matcher = new llguidance.LLMatcher(tokenizer, grammar);
  for (const token of encoder.encode(input)) {
    const result = matcher._interpreter.computeMask();
    if (!tokenAllowed(resultField(result, "mask"), token)) return false;
    matcher._interpreter.commitToken(token);
  }
  const result = matcher._interpreter.computeMask();
  return tokenAllowed(resultField(result, "mask"), tokenizer.eos_token) || Boolean(resultField(result, "stop"));
}

export function jsonSchemaAccepts(schema, value, jsonText) {
  return grammarAccepts(llguidance.LLMatcher.grammar_from_json_schema(schema), jsonText ?? JSON.stringify(value));
}

export function assertJsonSchema(schema, value, expected, jsonText) {
  assert.equal(jsonSchemaAccepts(schema, value, jsonText), expected, `${jsonText ?? JSON.stringify(value)} should ${expected ? "match" : "not match"} ${JSON.stringify(schema)}`);
}

export function compileJsonSchema(schema) {
  return new llguidance.LLMatcher(tokenizer, llguidance.LLMatcher.grammar_from_json_schema(schema));
}

export function assertJsonSchemaCompileError(schema, message) {
  assert.throws(() => compileJsonSchema(schema), message ? (error) => String(error).includes(message) : undefined);
}
